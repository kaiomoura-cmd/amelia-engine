import discord
from discord.ext import commands
import os
import asyncio
import threading
import time
import queue
import logging
import logging.handlers
import re
import unicodedata
import numpy as np

import sounddevice as sd
import speech_recognition as sr
import subprocess
from groq import Groq
import tempfile

# ─── Módulos de interpretação ─────────────────────────
from .interpretacao import Transcricao, GeracaoVoz


# ─── Logger do Cog ───────────────────────────────────────────────
log = logging.getLogger("amelia.ia_voz")
log.setLevel(logging.DEBUG)

# Handler p/ arquivo (rotativo, 5MB max, 3 backups)
LOG_DIR = os.path.join(tempfile.gettempdir(), "amelia_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Guard contra import duplo: o cog é importado 2x (find_spec/import_module no check
# do projeto_bot + load_extension), o que duplicava os handlers e fazia cada linha
# de log aparecer 2x no arquivo.
if not log.handlers:
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "ia_voz.log"),
        maxBytes=5_242_880,  # 5 MB
        backupCount=3,
        encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    # Também joga no console (stderr)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[A.M.E.L.I.A] %(message)s"))
    log.addHandler(ch)

class MixedAudioSource(sr.AudioSource):
    """
    Fonte de áudio customizada para Linux que combina microfone
    e loopback do sistema (PulseAudio monitor).
    """

    # ─── Ganho do loopback (áudio do PC) ───
    # O Discord costuma vir bem mais baixo que o microfone. Sem reforço,
    # o detector de voz (energy_threshold do SpeechRecognition) nem dispara
    # pra voz dos players. 2.5x equilibra com o mic (mesmo valor do gravador).
    LOOPBACK_GAIN = 2.5

    def __init__(self):
        self.SAMPLE_RATE = 48000
        self.SAMPLE_WIDTH = 2   # 16-bit PCM
        self.CHANNELS = 1       # Mono (SpeechRecognition espera mono)
        self.CHUNK = 2048       # Frames por buffer
        
        self.mic_device = None       # Índice do microfone
        self.mic_channels = 0        # Canais do microfone
        
        # Filas para comunicação callback → main (sem tamanho máximo para não perder áudio durante a transcrição)
        self.mic_queue = queue.Queue(maxsize=0)
        self.loopback_queue = queue.Queue(maxsize=0)
        
        self.is_recording = False
        
        # Loopback via parec (subprocesso) — o monitor NÃO aparece no sounddevice
        self._parec_proc = None
        self._parec_thread = None
        
        self._discover_devices()
        self.stream = self  # Necessário para o SpeechRecognition
    
    def _discover_devices(self):
        """
        Encontra o microfone padrão via sounddevice.
        O loopback (monitor) é resolvido separadamente, via parec/pactl.
        """
        devices = sd.query_devices()
        log.info(f"Dispositivos de áudio disponíveis ({len(devices)}):")

        # sd.default.device pode ser int, _InputOutputPair ou None
        default_input_id = None
        try:
            d = sd.default.device
            default_input_id = d[0] if hasattr(d, '__getitem__') else d
        except Exception:
            default_input_id = None

        for i, dev in enumerate(devices):
            name = dev['name'].lower()

            # Microfone: dispositivo de entrada padrão
            if dev['max_input_channels'] > 0 and self.mic_device is None:
                if 'default' in name or i == default_input_id:
                    self.mic_device = i
                    self.mic_channels = min(dev['max_input_channels'], 2)  # Máximo 2 canais
                    log.info(f"  ✅ Microfone: [{i}] {dev['name']} ({self.mic_channels} canais)")

        # Fallback: qualquer dispositivo de entrada
        if self.mic_device is None:
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    self.mic_device = i
                    self.mic_channels = min(dev['max_input_channels'], 2)
                    log.warning(f"  ⚠️ Microfone (fallback): [{i}] {dev['name']}")
                    break

        if self.mic_device is None:
            log.warning("⚠️ Nenhum microfone encontrado!")
    
    def _mix_to_mono(self, data: np.ndarray, channels: int) -> np.ndarray:
        if channels == 1:
            return data.flatten()
        return data.mean(axis=1).astype(np.int16)
    
    def mic_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if self.is_recording and indata is not None:
            mono = self._mix_to_mono(indata, self.mic_channels)
            try:
                self.mic_queue.put_nowait(mono.tobytes())
            except queue.Full:
                pass

    def _get_monitor_source(self):
        """Obtém o nome da source de monitor do PipeWire/PulseAudio via pactl."""
        try:
            result = subprocess.run(
                ["pactl", "list", "sources", "short"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().split("\n"):
                if "monitor" in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]
        except Exception as e:
            log.warning(f"⚠️ Erro ao buscar monitor (pactl): {e}")
        return None

    def _parec_reader(self):
        """Thread: lê o monitor da saída via parec e alimenta loopback_queue.

        Converte stereo → mono em blocos de CHUNK frames (alinhado ao mic),
        para que read() misture as duas fontes sem dessincronizar.
        """
        monitor_source = self._get_monitor_source()
        if not monitor_source:
            log.warning("⚠️ Nenhuma source de monitor encontrada — loopback (áudio do PC) indisponível.")
            return

        log.info(f"  ✅ Loopback (monitor): {monitor_source}")

        cmd = [
            "parec",
            f"--device={monitor_source}",
            "--format=s16le",
            "--channels=2",
            f"--rate={self.SAMPLE_RATE}",
            "--latency=20"
        ]

        try:
            self._parec_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.CHUNK * 4
            )
        except FileNotFoundError:
            log.error("❌ parec não encontrado! Instale: sudo apt install pulseaudio-utils")
            self._parec_proc = None
            return

        proc = self._parec_proc
        if proc is None or proc.stdout is None:
            return

        bytes_per_frame = 4  # s16le stereo = 4 bytes/frame
        chunk_bytes = self.CHUNK * bytes_per_frame

        try:
            while self.is_recording and proc.poll() is None:
                raw = proc.stdout.read(chunk_bytes)
                if not raw:
                    break
                data = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
                mono = data.mean(axis=1).astype(np.int16)  # stereo → mono
                try:
                    self.loopback_queue.put(mono.tobytes())
                except Exception:
                    break
        except Exception as e:
            log.warning(f"⚠️ Erro lendo parec (loopback): {e}")
    
    def read(self, size):
        try:
            # Mic bloqueia por até 0.05s, garantindo o tempo real
            mic_bytes = self.mic_queue.get(timeout=0.05)
            mic_data = np.frombuffer(mic_bytes, dtype=np.int16)
        except queue.Empty:
            mic_data = np.zeros(self.CHUNK, dtype=np.int16)
        
        try:
            # Loopback não bloqueia (se não tiver, paciência, o mic já ditou o tempo)
            loop_bytes = self.loopback_queue.get_nowait()
            loop_data = np.frombuffer(loop_bytes, dtype=np.int16)
        except queue.Empty:
            loop_data = np.zeros(self.CHUNK, dtype=np.int16)
        
        min_len = min(len(mic_data), len(loop_data))
        if min_len == 0:
            return b'\x00' * (self.CHUNK * self.SAMPLE_WIDTH)
        
        # Reforça o loopback (áudio do PC) antes de misturar
        loop_gained = np.clip(
            loop_data[:min_len].astype(np.float32) * self.LOOPBACK_GAIN,
            -32768, 32767
        ).astype(np.int16)
        
        mixed = np.clip(
            mic_data[:min_len].astype(np.int32) +
            loop_gained.astype(np.int32),
            -32768, 32767
        ).astype(np.int16)
        
        if len(mixed) < self.CHUNK:
            mixed = np.pad(mixed, (0, self.CHUNK - len(mixed)), 'constant')
        
        return mixed.tobytes()
    
    def __enter__(self):
        self.is_recording = True
        
        if self.mic_device is not None:
            self.mic_stream = sd.InputStream(
                device=self.mic_device,
                channels=self.mic_channels,
                samplerate=self.SAMPLE_RATE,
                blocksize=self.CHUNK,
                callback=self.mic_callback,
                dtype='int16'
            )
            self.mic_stream.start()
        
        # Loopback via parec (subprocesso) — sounddevice NÃO expõe o monitor
        self._parec_thread = threading.Thread(target=self._parec_reader, daemon=True)
        self._parec_thread.start()
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.is_recording = False
        
        if hasattr(self, 'mic_stream'):
            self.mic_stream.stop()
            self.mic_stream.close()
        
        if self._parec_proc is not None and self._parec_proc.poll() is None:
            self._parec_proc.terminate()
            try:
                self._parec_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._parec_proc.kill()
        
        while not self.mic_queue.empty():
            self.mic_queue.get()
        while not self.loopback_queue.empty():
            self.loopback_queue.get()


class StreamingAudioSource(discord.AudioSource):
    """Fonte de áudio para Discord que recebe chunks de PCM em tempo real.
    
    O TTS Worker envia chunks de áudio em 24kHz mono float32.
    Esta classe converte para 48kHz stereo int16 (formato do Discord)
    e entrega exatamente 3840 bytes (20ms) por chamada de read().
    """
    
    DISCORD_RATE = 48000
    DISCORD_CHANNELS = 2
    DISCORD_SAMPLE_WIDTH = 2  # int16
    FRAME_SIZE = 3840  # 20ms @ 48kHz stereo int16 = 960 samples * 2 channels * 2 bytes
    TTS_RATE = 24000  # XTTSv2 gera a 24kHz
    
    def __init__(self):
        self._chunk_queue = queue.Queue()
        self._remainder = b''
        self._finished = False
        self._started = threading.Event()
    
    def feed(self, pcm_24k_float32: bytes):
        """Recebe um chunk de áudio do TTS (24kHz mono float32) e converte para Discord."""
        import struct
        
        # 1. Converte float32 → int16
        float_samples = np.frombuffer(pcm_24k_float32, dtype=np.float32)
        int16_samples = np.clip(float_samples * 32767, -32768, 32767).astype(np.int16)
        
        # 2. Resample 24kHz → 48kHz (duplica cada amostra — simples e eficiente)
        upsampled = np.repeat(int16_samples, 2)
        
        # 3. Mono → Stereo (duplica cada canal)
        stereo = np.column_stack([upsampled, upsampled]).flatten()
        
        pcm_bytes = stereo.astype(np.int16).tobytes()
        self._chunk_queue.put(pcm_bytes)
        self._started.set()
    
    def finish(self):
        """Sinaliza que não há mais chunks a enviar."""
        self._finished = True
        self._started.set()  # Desbloqueia read() caso esteja esperando
    
    def read(self) -> bytes:
        """Retorna exatamente 3840 bytes (20ms de áudio Discord).
        
        Chamado pelo discord.py a cada 20ms em uma thread separada.
        """
        while len(self._remainder) < self.FRAME_SIZE:
            try:
                chunk = self._chunk_queue.get(timeout=0.05)
                self._remainder += chunk
            except queue.Empty:
                if self._finished and self._chunk_queue.empty():
                    # Envia o que sobrou + padding de silêncio
                    if self._remainder:
                        padded = self._remainder + b'\x00' * (self.FRAME_SIZE - len(self._remainder))
                        self._remainder = b''
                        return padded[:self.FRAME_SIZE]
                    return b''  # Sinaliza fim do áudio para o Discord
                # Ainda não acabou, mas não tem dados — envia silêncio
                return b'\x00' * self.FRAME_SIZE
        
        # Extrai exatamente FRAME_SIZE bytes
        frame = self._remainder[:self.FRAME_SIZE]
        self._remainder = self._remainder[self.FRAME_SIZE:]
        return frame
    
    def is_opus(self) -> bool:
        return False
    
    def wait_for_first_chunk(self, timeout=30):
        """Bloqueia até o primeiro chunk de áudio chegar."""
        return self._started.wait(timeout=timeout)

class IAVoz(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.is_listening = False
        self.is_speaking = False  # Evita feedback loop enquanto fala
        self.tts_habilitado = True   # False desliga TTS (para debug de transcrição)
        self.listen_thread = None
        self.last_triggered_time = 0
        self.last_heartbeat = 0    # Para supervisão do worker
        self.last_audio_played = 0  # Timestamp do último áudio reproduzido com sucesso
        
        # ─── Módulos de interpretação ─────────────────────
        self.transcricao = Transcricao()
        self.voz = GeracaoVoz()
        
        # ─── Cliente Groq (LLM) ────────────────────────────
        self.groq_client = None
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            self.groq_client = Groq(api_key=groq_key)
        else:
            log.warning("GROQ_API_KEY não encontrada no .env!")
        
        # ─── Leitura da LORE (Personalidade) ──────────────
        lore_file = os.path.join(os.path.dirname(__file__), 'lore_amelia.md')
        if not os.path.exists(lore_file):
            default_lore = (
                "Você é a A.M.E.L.I.A., a inteligência artificial assistente de uma nave em uma campanha de RPG "
                "de ficção científica/terror. Responda de forma concisa, útil e levemente sintética. "
                "Como suas respostas serão lidas em voz alta por um sistema TTS, NUNCA use emojis, "
                "evite jargões de formatação ou asteriscos, e fale de maneira direta."
            )
            with open(lore_file, 'w', encoding='utf-8') as f:
                f.write(default_lore)
            self.system_instruction = default_lore
        else:
            with open(lore_file, 'r', encoding='utf-8') as f:
                self.system_instruction = f.read().strip()
        
        self.system_instruction += (
            "\n\nIMPORTANTE: Suas respostas serão lidas em voz alta por um sistema TTS. "
            "Fale de forma natural e fluida, sem formatação ou emojis. "
            "Seja SUCINTA: diga apenas o necessário, sem repetir a pergunta, "
            "sem enrolar e sem explicar o que não foi perguntado. "
            "Uma boa resposta é curta e afiada, como você."
        )
                
        # ─── Reconhecimento de fala ───────────────────────
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.7  # Espera 0.7s de silêncio para fechar a gravação (era 1.0s: -300ms em toda resposta)
        self.recognizer.non_speaking_duration = 0.5
        # energy_threshold: valor inicial. A calibração (adjust_for_ambient_noise)
        # vai ajustar dinamicamente baseado no ruído ambiente durante 2s.
        # ANTES era dynamic=True + restauração forçada pra 300 — isso quebrava
        # a escuta em ambientes silenciosos.
        self.recognizer.energy_threshold = 600
        # Reativando o ajuste dinâmico para adaptar aos ruídos constantes do PC e mic
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.8
        self.mixed_source = MixedAudioSource()
        
        self.voice_client = None
        
        # Limpa sujeira de sessões anteriores
        self._cleanup_stale_tempfiles()
        
        log.info("Cog inicializado. Modelos serão carregados sob demanda.")

    # ── Utilitários ────────────────────────────────────────────
    
    @staticmethod
    def _tem_ativacao(texto):
        """Detecta se o texto contém 'Amélia' ou variações comuns do Whisper.
        
        O Whisper frequentemente separa o 'A' do 'mélia' quando o usuário
        fala 'Amélia' de forma rápida, transcrevendo como 'a melia' ou 'a melha'.
        Esta função normaliza e compara palavra por palavra.
        """
        if not texto:
            return False
        
        # Remove acentos e converte para lowercase ASCII
        texto_norm = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII').lower()
        # Extrai palavras individuais (ignora pontuação)
        palavras = re.findall(r'[a-z0-9]+', texto_norm)
        
        # Lista de variações que o Whisper pode produzir para "Amélia"
        # O modelo medium produz variantes diferentes do small
        ativacoes = {
            "amelia", "melia", "melha", "mellia", "amilia",
            "emelia", "emilia", "amalia", "amelha", "ameilia",
            "amelia.", "amelia,", "amelia!", "amelia?",
        }
        
        for palavra in palavras:
            if palavra in ativacoes:
                return True
        
        # Verifica também combinações de 2 palavras ("a melia", "a mélia")
        texto_junto = "".join(palavras)
        for ativacao in ativacoes:
            if ativacao in texto_junto:
                return True
        
        return False

    def _cleanup_stale_tempfiles(self):
        """Apaga arquivos temporários abandonados de execuções anteriores."""
        temp_dir = tempfile.gettempdir()
        for fname in os.listdir(temp_dir):
            if fname.startswith("amelia_") and (fname.endswith(".wav") or fname.endswith(".mp3")):
                try:
                    os.remove(os.path.join(temp_dir, fname))
                    log.debug(f"Limpou tempfile esquecido: {fname}")
                except Exception:
                    pass

    def _load_whisper(self) -> bool:
        """Carrega APENAS o Whisper (rápido, ~5s)."""
        if not self.transcricao.modelo_carregado:
            ok = self.transcricao.carregar_modelo()
            if ok:
                log.info(
                    f"✅ Whisper carregado: {self.transcricao.modelo_nome} "
                    f"({self.transcricao.modelo_device}/{self.transcricao.modelo_compute})"
                )
            else:
                log.warning("Whisper não carregado — fallback Google STT.")
            return ok
        return True

    def _init_tts_background(self, ctx):
        """Inicia o TTS Worker em THREAD SEPARADA (não bloqueia).
        
        ⚡ Retorna IMEDIATAMENTE. O worker carrega em background.
        ⚡ Edge-TTS é usado como fallback até o worker ficar pronto.
        ⚡ Callbacks:
           - worker pronto: modelo carregado, pronto para receber requests
           - aquecimento completo: primeira inferência CUDA concluída
        """
        if not self.voz._voice_ref_existe():
            log.warning("Voice ref não encontrado. TTS via Edge-TTS.")
            return
        
        # ── Callback 1: Worker pronto (modelo carregado) ──
        async def avisar_worker_pronto():
            try:
                await ctx.send(
                    "🔊 **XTTSv2 carregado!** Aquecendo CUDA (primeira inferência)..."
                )
            except Exception as e:
                log.warning(f"Aviso worker pronto falhou: {e}")
        
        def callback_worker_pronto():
            asyncio.run_coroutine_threadsafe(avisar_worker_pronto(), self.bot.loop)
        
        # ── Callback 2: Aquecimento completo (primeira inferência CUDA) ──
        async def avisar_aquecimento():
            try:
                await ctx.send(
                    "🔥 **XTTSv2 aquecido e pronto!** "
                    "Todas as respostas serão geradas localmente com máxima qualidade."
                )
            except Exception as e:
                log.warning(f"Não foi possível enviar aviso de aquecimento: {e}")
        
        def callback_aquecimento():
            asyncio.run_coroutine_threadsafe(avisar_aquecimento(), self.bot.loop)
        
        # ═══ DISPARA EM THREAD SEPARADA ═══
        # ⚡ Isso permite que o carregamento do Whisper retorne IMEDIATAMENTE
        #    e a escuta comece sem esperar o TTS.
        def _iniciar_worker():
            self.voz.iniciar_worker_async(
                timeout=60,
                on_warmup_complete=callback_worker_pronto,
                on_aquecimento_completo=callback_aquecimento
            )
        
        t = threading.Thread(target=_iniciar_worker, daemon=True)
        t.start()
        log.info("Worker TTS iniciado em THREAD SEPARADA (não bloqueante).")

    @commands.command(name='amelia_on')
    async def amelia_on(self, ctx):
        """Ativa a escuta da Amélia no seu PC local (Mestre e Jogadores)."""
        if self.is_listening:
            await ctx.send("🔊 A.M.E.L.I.A. já está ouvindo.")
            return

        self.voice_client = ctx.voice_client or discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
        if not self.voice_client:
            await ctx.send("❌ A.M.E.L.I.A. não está conectada a um canal de voz. Use `!entrar_sessao` primeiro.")
            return
        
        if not self.voice_client.is_connected():
            await ctx.send("❌ Conexão de voz instável. Use `!entrar_sessao` novamente.")
            return

        if not self.groq_client:
            await ctx.send("❌ GROQ_API_KEY não configurada. Verifique o arquivo .env.")
            self.is_listening = False
            return

        # ════════════════════════════════════════════
        # Carregamento Gradual da IA (Evita OOM e Travamentos)
        # ════════════════════════════════════════════
        msg = await ctx.send("`[▬...................] 10% - Iniciando sequenciador de IA...`")
        
        await asyncio.sleep(0.5)
        await msg.edit(content="`[█████...............] 30% - Carregando Módulo de Ouvidos (Whisper)...`")
        
        # 1. Carrega Whisper (síncrono, ~5s) - O uso reduzido de memória previne travamentos
        whisper_ok = await self.bot.loop.run_in_executor(None, self._load_whisper)
        
        if not whisper_ok:
            await ctx.send("⚠️ **Whisper falhou.** Usarei Google STT como fallback.")

        await msg.edit(content="`[███████████.........] 60% - Whisper Carregado! Iniciando Motor de Voz (XTTSv2)...`")

        # 2. Inicia TTS Worker (em background mas AGORA, após o Whisper terminar)
        # O fato de não estarem rodando ao mesmo tempo salva a VRAM e impede o computador de congelar
        self._init_tts_background(ctx)
        
        await asyncio.sleep(1)
        await msg.edit(content="`[████████████████████] 100% - Sistemas Prontos e Sequenciados!`")

        # ════════════════════════════════════════════
        # 3. Começa a ESCUTAR IMEDIATAMENTE
        # ════════════════════════════════════════════
        self.is_listening = True
        self.last_heartbeat = time.time()
        
        self.listen_thread = threading.Thread(target=self._listen_worker, args=(ctx,))
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        await ctx.send(
            "🟢 **A.M.E.L.I.A. ativada e pronta.** Ouvindo a mesa inteira (Mestre e Jogadores).\n"
            "⚡ Diga o nome dela ('Amélia') para interagir.\n"
            "*(Nota: O XTTSv2 está aquecendo no fundo para usar a GPU local)*"
        )

    @commands.command(name='amelia_off')
    async def amelia_off(self, ctx):
        """Desativa a escuta da Amélia."""
        if not self.is_listening:
            await ctx.send("🔇 A.M.E.L.I.A. não estava ouvindo.")
            return
            
        log.info("Escuta desativada pelo usuário.")
        self.is_listening = False
        await ctx.send("🔴 **A.M.E.L.I.A. desativada.**")
        
        # Libera o MixedAudioSource
        try:
            if hasattr(self.mixed_source, '__exit__'):
                self.mixed_source.__exit__(None, None, None)
        except Exception:
            pass

    @commands.command(name='amelia_geracaoVozOff')
    async def amelia_geracao_voz_off(self, ctx):
        """Liga/desliga a geração de voz (TTS).
        
        Útil para testar a transcrição sem o delay do TTS.
        Quando desligado, as respostas são enviadas apenas como texto.
        """
        self.tts_habilitado = not self.tts_habilitado
        estado = "✅ **LIGADO**" if self.tts_habilitado else "🔇 **DESLIGADO**"
        await ctx.send(f"🎤 Geração de voz (TTS): {estado}")
        log.info(f"TTS {'ligado' if self.tts_habilitado else 'desligado'} por {ctx.author.name}")

    def _transcribe(self, audio):
        """Transcreve áudio usando o módulo Transcricao.
        
        Delega para self.transcricao.transcrever() que gerencia:
        - Whisper medium/small (GPU/CPU fallback)
        - Fallback Google STT
        - Logs de timing detalhados
        - Timeout na inferência
        """
        return self.transcricao.transcrever(audio)

    def _listen_worker(self, ctx):
        """Worker rodando em thread separada. Escuta o microfone e processa comandos de voz."""
        RECONNECT_COOLDOWN = 15  # segundos entre tentativas de reconexão
        HEARTBEAT_INTERVAL = 5    # segundos entre heartbeats
        
        try:
            with self.mixed_source as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                # Como ligamos o dynamic_energy_threshold, isso apenas define o ponto de partida.
                # Ele vai subir e descer conforme o som do jogo/PC, ignorando ruído constante.
                if self.recognizer.energy_threshold < 400:
                    self.recognizer.energy_threshold = 400
                elif self.recognizer.energy_threshold > 3000:
                    self.recognizer.energy_threshold = 3000
                log.info(f"Calibração base ajustada: energy_threshold={self.recognizer.energy_threshold:.0f}")
                log.info("Pronta para ouvir.")
                
                phrase_buffer = []
                last_speech_time = 0
                MERGE_WINDOW = 1.0  # janela de silêncio normal extra
                ACTIVATION_SILENCE = 0.0  # Sem silêncio extra, o pause_threshold de 0.7s já é suficiente
                COMMAND_MAX_DURATION = 20.0  # tempo máximo p/ completar um comando longo sem ser cortado
                activation_detected = False  # flag: "Amélia" foi detectada
                activation_time = 0
                last_reconnect_attempt = 0
                
                while self.is_listening:
                    # ── Heartbeat: verifica se o voice_client ainda está vivo ──
                    now = time.time()
                    if now - self.last_heartbeat >= HEARTBEAT_INTERVAL:
                        self.last_heartbeat = now
                        vc = self.voice_client or discord.utils.get(
                            self.bot.voice_clients, guild=getattr(ctx, 'guild', None)
                        )
                        if vc is None or not vc.is_connected():
                            log.warning("Voice client desconectou! Tentando reconectar...")
                            if now - last_reconnect_attempt >= RECONNECT_COOLDOWN:
                                last_reconnect_attempt = now
                                asyncio.run_coroutine_threadsafe(
                                    self._tentar_reconectar(ctx), self.bot.loop
                                )
                            else:
                                log.warning("Aguardando cooldown para reconectar...")
                                # Se ficou tempo demais desconectado, desliga
                                if now - self.last_heartbeat > 60:
                                    log.error("Worker encerrando: voz desconectada por >60s")
                                    self.is_listening = False
                                    break
                    
                    # ── Escuta ativa ──
                    try:
                        # ═══ PULA CAPTURA INTEIRA ENQUANTO FALA (economiza GPU) ═══
                        if self.is_speaking:
                            # Limpa buffer acumulado da própria voz do bot
                            if phrase_buffer:
                                phrase_buffer.clear()
                                last_speech_time = 0
                            
                            # Drena as filas para não processar áudio do passado (ex: a própria voz do bot ou pessoas falando)
                            try:
                                while True:
                                    source.mic_queue.get_nowait()
                            except queue.Empty:
                                pass
                            try:
                                while True:
                                    source.loopback_queue.get_nowait()
                            except queue.Empty:
                                pass
                                
                            time.sleep(0.05)  # pausa curta pra não girar CPU
                            continue
                        
                        audio = self.recognizer.listen(source, timeout=0.1, phrase_time_limit=15)
                        speech_ended_at = time.time()
                        
                        # ═══ DUPLA VERIFICAÇÃO: se começou a falar durante o listen ═══
                        if self.is_speaking:
                            continue  # descarta esse áudio sem transcrever
                        
                        texto = self._transcribe(audio)
                        
                        if not texto:
                            continue
                        
                        log.debug(f"Fragmento ouvido: {texto}")
                        
                        # ═══ ATIVAÇÃO INTELIGENTE ═══
                        # Detecta "Amélia" mas CONTINUA acumulando o comando
                        if self._tem_ativacao(texto) and not activation_detected:
                            activation_detected = True
                            activation_time = time.time()
                            log.info(f"⚡ Amélia detectada! Acumulando comando...")
                            # Não processa ainda — queremos ouvir o comando completo
                        
                        # Se ativação foi detectada, aceita fragmentos menores
                        if activation_detected:
                            phrase_buffer.append(texto)
                            last_speech_time = speech_ended_at
                        elif len(texto.split()) > 3:
                            # Conversa normal: só acumula fragmentos com contexto
                            phrase_buffer.append(texto)
                            last_speech_time = speech_ended_at
                        # else: fragmento curto sem ativação → descarta
                            
                    except sr.WaitTimeoutError:
                        # Se ainda tem áudio no buffer, estamos apenas tirando atraso. Não é um silêncio real.
                        if not source.mic_queue.empty() or not source.loopback_queue.empty():
                            continue
                            
                        # Timeout = silêncio real. Processa buffer acumulado.
                        if not phrase_buffer:
                            continue
                        
                        agora = time.time()
                        tempo_silencio = agora - last_speech_time
                        
                        # Decide qual janela de silêncio usar
                        if activation_detected:
                            # ═══ MODO COMANDO: silêncio mais curto (0.6s) ═══
                            if tempo_silencio >= ACTIVATION_SILENCE or (agora - activation_time) >= COMMAND_MAX_DURATION:
                                frase_completa = " ".join(phrase_buffer)
                                phrase_buffer.clear()
                                activation_detected = False
                                
                                if self.is_speaking:
                                    continue
                                
                                log.info(f"🎯 Comando: {frase_completa}")
                                if time.time() - self.last_triggered_time > 5:
                                    self.last_triggered_time = time.time()
                                    asyncio.run_coroutine_threadsafe(
                                        self.processar_resposta(ctx, frase_completa), self.bot.loop
                                    )
                        else:
                            # ═══ MODO NORMAL: silêncio padrão (1.2s) ═══
                            if tempo_silencio >= MERGE_WINDOW:
                                frase_completa = " ".join(phrase_buffer)
                                phrase_buffer.clear()
                                
                                if self.is_speaking:
                                    continue
                                
                                # Só processa se tiver a palavra mágica
                                if self._tem_ativacao(frase_completa):
                                    if time.time() - self.last_triggered_time > 10:
                                        self.last_triggered_time = time.time()
                                        log.info(f"🎯 Ativação tardia: {frase_completa}")
                                        asyncio.run_coroutine_threadsafe(
                                            self.processar_resposta(ctx, frase_completa), self.bot.loop
                                        )
                        continue
                    except sr.UnknownValueError:
                        pass
                    except Exception as e:
                        log.error(f"Erro no STT: {e}")
        except Exception as e:
            log.error(f"Erro ao acessar microfones: {e}")
            
        log.info("Worker de escuta encerrado.")

    async def _tentar_reconectar(self, ctx):
        """Tenta reconectar ao último canal de voz conhecido."""
        try:
            # Tenta pegar o autor original do comando
            author = getattr(ctx, 'author', None)
            if author and author.voice:
                canal = author.voice.channel
                if self.voice_client and self.voice_client.is_connected():
                    await self.voice_client.move_to(canal)
                else:
                    self.voice_client = await canal.connect(timeout=20.0, reconnect=True)
                log.info(f"Reconectado ao canal {canal.name}.")
                await ctx.send(f"🔄 A.M.E.L.I.A. reconectada ao canal `{canal.name}`.")
        except Exception as e:
            log.error(f"Falha na reconexão: {e}")

    async def _ensure_voice_alive(self, ctx):
        """Verifica se a conexão de voz está realmente funcional (não zumbi).
        
        Problema: O WebSocket pode ficar 'conectado' mas o canal UDP de áudio
        morre silenciosamente, especialmente com DAVE/E2EE após períodos ociosos.
        Resultado: ffmpeg roda, is_playing() retorna True, mas o Discord não
        recebe pacotes — sem anel verde, sem som.
        
        Solução: Se a conexão ficou ociosa por muito tempo, força reconexão
        via move_to() que re-estabelece o stream UDP.
        """
        MAX_IDLE = 120  # 2 minutos sem áudio = reconecta preventivamente
        
        vc = self.voice_client
        if not vc or not vc.is_connected():
            log.warning("Voice client desconectado antes do playback.")
            await self._tentar_reconectar(ctx)
            vc = self.voice_client
            if not vc or not vc.is_connected():
                return False
        
        # Se já faz muito tempo desde o último áudio, a conexão UDP pode estar morta
        agora = time.time()
        tempo_ocioso = agora - self.last_audio_played if self.last_audio_played > 0 else 9999
        
        # 🛡️ GUARD CLAUSE: primeira interação — não mexa na conexão!
        # Evita o disconnect agressivo que corta o áudio antes de tocar
        if self.last_audio_played == 0:
            return True
        
        if tempo_ocioso > MAX_IDLE:
            log.warning(f"Conexão de voz ociosa por {tempo_ocioso:.0f}s. Renovando stream UDP via move_to...")
            try:
                canal = vc.channel
                if canal:
                    await vc.move_to(canal)   # ← move_to, NÃO disconnect()!
                    await asyncio.sleep(0.3)  # Espera estabilizar o novo stream
                    log.info(f"Stream UDP renovado no canal {canal.name}.")
                else:
                    log.error("Não foi possível determinar o canal para reconexão.")
                    return False
            except Exception as e:
                log.error(f"Falha na renovação UDP: {e}")
                # Tenta reconectar via contexto como último recurso
                await self._tentar_reconectar(ctx)
                if not self.voice_client or not self.voice_client.is_connected():
                    return False
        
        return True

    async def processar_resposta(self, ctx, texto):
        """Pipeline de streaming verdadeiro: LLM → TTS → Playback em paralelo.
        
        O LLM roda em thread e emite sentenças para uma fila conforme são
        detectadas. O consumer puxa sentenças e inicia TTS imediatamente.
        Prefetch: TTS da sentença N+1 começa enquanto N ainda toca.
        """
        t_inicio = time.time()
        await ctx.send(f"🧠 *Processando...* (Ouvi: `{texto}`)")
        log.info(f"Processando: {texto}")
        
        try:
            # ════════════════════════════════════════════
            # 0. Calibração dinâmica — respostas proporcionais à pergunta
            # ════════════════════════════════════════════
            palavras_input = len(texto.split())
            if palavras_input <= 6:
                max_tok = 60
            elif palavras_input <= 12:
                max_tok = 150
            else:
                max_tok = 300
            
            instrucao_dinamica = (
                "\n\nSua resposta deve ser PROPORCIONAL à complexidade da pergunta. "
                "Perguntas simples como 'tá aí?' merecem respostas curtas de uma frase. "
                "Perguntas complexas podem ter respostas mais elaboradas. "
                "Nunca enrole — vá direto ao ponto."
            )
            log.info(f"  📏 Calibração: {palavras_input} palavras → max_tokens={max_tok}")
            
            # ════════════════════════════════════════════
            # 1. LLM Producer — Groq em thread, sentenças na fila em tempo real
            # ════════════════════════════════════════════
            fila_sentencas = queue.Queue()  # thread-safe
            resposta_parts = []
            groq_error = [None]  # mutable container for thread error
            
            t0 = time.time()
            log.info("Enviando para Groq Qwen 3.8-27B (streaming)...")
            
            def _groq_producer():
                """Thread: streaming Groq → sentenças na fila conforme detectadas."""
                buffer = ""
                try:
                    stream = self.groq_client.chat.completions.create(
                        model="qwen/qwen3.8-27b",
                        messages=[
                            {"role": "system", "content": self.system_instruction + instrucao_dinamica},
                            {"role": "user", "content": texto}
                        ],
                        temperature=0.7,
                        max_tokens=max_tok,
                        stream=True,
                        timeout=30,
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            buffer += delta.content
                            while True:
                                best = -1
                                for delim in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                                    pos = buffer.find(delim)
                                    if pos != -1 and (best == -1 or pos < best):
                                        best = pos + len(delim)
                                if best == -1:
                                    break
                                frase = buffer[:best].strip()
                                buffer = buffer[best:]
                                if len(frase) >= 10:
                                    fila_sentencas.put(frase)
                    
                    # Fecha o stream explicitamente (evita bloqueio httpx)
                    try:
                        stream.close()
                    except Exception:
                        pass
                    
                    rest = buffer.strip()
                    if rest:
                        fila_sentencas.put(rest)
                    
                    log.info(f"  ✅ LLM producer terminou ({time.time()-t0:.1f}s)")
                except Exception as e:
                    groq_error[0] = e
                    log.error(f"  ❌ LLM producer erro: {e}")
                finally:
                    fila_sentencas.put(None)  # Sentinel: LLM terminou
            
            prod_thread = threading.Thread(target=_groq_producer, daemon=True)
            prod_thread.start()
            
            # ════════════════════════════════════════════
            # 2. Consumer — TTS + Playback com prefetch
            # ════════════════════════════════════════════
            
            # Helper: pega próxima sentença da fila (async-safe)
            async def _next_sentence(timeout=3):
                try:
                    return await self.bot.loop.run_in_executor(
                        None, lambda: fila_sentencas.get(timeout=timeout)
                    )
                except queue.Empty:
                    return None
            
            # Helper: drena fila até esvaziar
            async def _drain_queue():
                while True:
                    s = await _next_sentence(0.5)
                    if s is None:
                        break
            
            use_streaming = self.tts_habilitado and self.voz.worker_pronto
            
            if use_streaming:
                # ═══ HEALTH CHECK ═══
                voz_ok = await self._ensure_voice_alive(ctx)
                if not voz_ok:
                    await _drain_queue()
                    await ctx.send("❌ Perdi a conexão com o canal de voz.")
                    return
                
                vc = self.voice_client
                if not vc or not vc.is_connected():
                    await _drain_queue()
                    await ctx.send("❌ Perdi a conexão com o canal de voz.")
                    return
                
                while vc.is_playing():
                    await asyncio.sleep(0.2)
                
                self.is_speaking = True
                t1 = time.time()
                idx = 0
                
                # ═══ LOOP PRINCIPAL: uma sentença por vez no worker ═══
                # O worker TTS é single-threaded — NÃO pode receber 2 requests
                # simultâneos, senão o pipe stdout/stdin corrompe.
                
                while True:
                    # Busca próxima sentença da fila (timeout curto)
                    t_wait = time.time()
                    sentenca = await _next_sentence(timeout=3)
                    dt_wait = time.time() - t_wait
                    
                    if dt_wait > 1.0:
                        log.warning(f"  ⏳ Esperou {dt_wait:.1f}s pela fila de sentenças")
                    
                    if sentenca is None:
                        # Fila vazia — verifica se o produtor terminou
                        if not prod_thread.is_alive():
                            # Produtor terminou — drena qualquer item restante
                            while not fila_sentencas.empty():
                                s = fila_sentencas.get_nowait()
                                if s is not None:
                                    resposta_parts.append(s)
                            break
                        # Produtor ainda vivo — tenta mais uma vez
                        sentenca = await _next_sentence(timeout=5)
                        if sentenca is None:
                            log.warning("  ⚠️ Produtor LLM parece travado. Encerrando.")
                            break
                    
                    if sentenca is None:
                        break
                    
                    # Checa erro do Groq
                    if not resposta_parts and groq_error[0]:
                        err_str = str(groq_error[0])
                        if '429' in err_str or 'rate_limit' in err_str.lower():
                            await ctx.send("⏳ **Cota do Groq esgotada.** Aguarde 30s.")
                            return
                        if '401' in err_str:
                            await ctx.send("❌ **Chave de API inválida.**")
                            return
                        raise groq_error[0]
                    
                    resposta_parts.append(sentenca)
                    idx += 1
                    log.info(f"  📝 Sentença {idx}: {sentenca[:50]}...")
                    
                    # Inicia TTS (UMA por vez — worker é single-threaded)
                    audio_source = StreamingAudioSource()
                    tts_task = asyncio.ensure_future(
                        self.voz.sintetizar_stream(sentenca, audio_source)
                    )
                    
                    # Espera primeiro chunk de áudio
                    got = await self.bot.loop.run_in_executor(
                        None, audio_source.wait_for_first_chunk, 15
                    )
                    if not got:
                        log.warning(f"  ⚠️ Timeout áudio sentença {idx}.")
                        await tts_task
                        continue
                    
                    # Espera reprodução anterior terminar
                    while vc.is_playing():
                        await asyncio.sleep(0.05)
                    
                    # Reproduz sentença atual
                    vc.play(discord.PCMVolumeTransformer(audio_source, volume=1.0))
                    
                    # Espera reprodução + TTS terminarem
                    while vc.is_playing():
                        await asyncio.sleep(0.1)
                    await tts_task
                
                if not resposta_parts:
                    if groq_error[0]:
                        raise groq_error[0]
                    await ctx.send("❌ Não consegui gerar uma resposta.")
                    return
                
                self.last_audio_played = time.time()
                resposta_completa = " ".join(resposta_parts)
                dt_total = time.time() - t1
                log.info(f"Groq: {len(resposta_parts)} sentenças ({len(resposta_completa)} chars)")
                log.info(f"🔊 Streaming completo ({dt_total:.1f}s). Escuta reativada.")
                await ctx.send(f"🗣️ **A.M.E.L.I.A:** {resposta_completa}")
            
            else:
                # ═══ FALLBACK: coleta tudo e processa sequencialmente ═══
                while True:
                    s = await _next_sentence()
                    if s is None:
                        break
                    resposta_parts.append(s)
                
                if groq_error[0]:
                    err_str = str(groq_error[0])
                    if '429' in err_str or 'rate_limit' in err_str.lower():
                        await ctx.send("⏳ **Cota do Groq esgotada.** Aguarde 30s.")
                        return
                    if '401' in err_str:
                        await ctx.send("❌ **Chave de API inválida.**")
                        return
                    raise groq_error[0]
                
                resposta_completa = " ".join(resposta_parts)
                log.info(f"Groq: {len(resposta_parts)} sentenças ({len(resposta_completa)} chars) em {time.time()-t0:.1f}s.")
                
                if not resposta_completa:
                    await ctx.send("❌ Não consegui gerar uma resposta.")
                    return
                
                await ctx.send(f"🗣️ **A.M.E.L.I.A:** {resposta_completa}")
                
                if self.tts_habilitado:
                    voz_ok = await self._ensure_voice_alive(ctx)
                    if not voz_ok:
                        await ctx.send("❌ Perdi a conexão com o canal de voz.")
                        return
                    vc = self.voice_client
                    if not vc or not vc.is_connected():
                        await ctx.send("❌ Perdi a conexão com o canal de voz.")
                        return
                    while vc.is_playing():
                        await asyncio.sleep(0.2)
                    
                    self.is_speaking = True
                    t1 = time.time()
                    log.info("Usando modo arquivo (fallback)...")
                    arquivo_saida = await self.voz.sintetizar(resposta_completa)
                    
                    if not arquivo_saida or not os.path.exists(arquivo_saida) or os.path.getsize(arquivo_saida) < 1000:
                        await ctx.send("❌ Não consegui gerar áudio.")
                        return
                    
                    import imageio_ffmpeg
                    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                    source = discord.FFmpegPCMAudio(arquivo_saida, executable=ffmpeg_path)
                    source = discord.PCMVolumeTransformer(source, volume=1.0)
                    vc.play(source)
                    
                    while vc.is_playing():
                        await asyncio.sleep(0.5)
                    
                    self.last_audio_played = time.time()
                    log.info(f"Áudio reproduzido ({time.time()-t1:.1f}s). Escuta reativada.")
                    
                    if os.path.exists(arquivo_saida):
                        try:
                            os.remove(arquivo_saida)
                        except Exception as e:
                            log.warning(f"NÃO limpou {arquivo_saida}: {e}")
                else:
                    log.info("🔇 TTS desabilitado — resposta em texto.")
                
        except Exception as e:
            await ctx.send(f"❌ Erro ao processar a resposta: {e}")
            log.exception("Erro no processamento da resposta")
        finally:
            self.is_speaking = False
            tempo_total = time.time() - t_inicio
            log.info(f"🏁 Resposta completa em {tempo_total:.1f}s (LLM + TTS + Playback)")

async def setup(bot):
    await bot.add_cog(IAVoz(bot))
