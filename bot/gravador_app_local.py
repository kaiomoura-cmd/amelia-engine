import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import subprocess
import time
import os
import sys
import logging
from datetime import datetime

# ─── Diretório de destino (ABSOLUTO, baseado na localização deste script) ───
# Isso evita duplicação de pastas quando o CWD muda
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(_SCRIPT_DIR, 'discord_logs', 'audio')
os.makedirs(AUDIO_DIR, exist_ok=True)

# ─── Logging ─────────────────────────────────────────────────────────────
# Tudo sai via stderr, que o bot (voz.py) redireciona para arquivo
# (gravador_<timestamp>.log / gravador_<timestamp>_err.log).
# NUNCA usar print() em hot path (callback PortAudio / save loop):
# com stdout/stderr em pipe não-lido, um print bloqueante congela a gravação.
log = logging.getLogger("gravador_app_local")
log.setLevel(logging.DEBUG)
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_sh)


class RateLimiter:
    """Loga no máximo 1 mensagem a cada `intervalo` segundos.

    O callback do PortAudio roda a ~85ms (blocksize 4096 @ 48kHz): se o status
    de overflow se repetir, um print por callback viraria spam e — com o pipe
    de stdout cheio — travaria a gravação inteira.
    """

    def __init__(self, intervalo=10.0):
        self.intervalo = intervalo
        self._ultima = 0.0
        self._suprimidas = 0

    def log(self, msg, nivel=logging.WARNING):
        agora = time.time()
        if agora - self._ultima >= self.intervalo:
            if self._suprimidas:
                msg = f"{msg} (+{self._suprimidas} suprimidas)"
                self._suprimidas = 0
            log.log(nivel, msg)
            self._ultima = agora
        else:
            self._suprimidas += 1


class DualRecorder:
    """
    Gravador de áudio dual para Linux (PulseAudio/PipeWire).

    Captura:
      - Microfone  → via sounddevice (dispositivo de entrada padrão)
      - Áudio do PC → via subprocess parec (monitor da saída padrão)

    Mixa ambos em tempo real e salva como OGG Vorbis 192kbps.
    Tamanho otimizado para sessões longas (ex: ~250MB em 3h).

    NÃO requer módulo loopback do PulseAudio (evita eco nos fones).
    """

    # ─── Ajuste de ganho individual ───
    # Microfone muitas vezes captura mais baixo que o áudio do PC.
    # Ajuste estas constantes conforme necessário:
    MIC_GAIN = 0.4      # Microfone (reduzido — estava 1.0, hardware já captura forte)
    PC_GAIN = 2.5       # Áudio do PC (reforçado — estava 1.5, Discord vinha baixo)

    def __init__(self):
        self.is_recording = False
        self.mic_queue = queue.Queue()
        self.pc_queue = queue.Queue()

        self.channels = 2       # Stereo
        self.rate = 48000       # 48kHz (ideal para transcrição)
        self.chunk = 4096       # Frames por buffer (~85ms)

        self.mic_device = None
        self.mic_channels = 0
        self.parec_process = None
        self.save_thread = None

        # Rate-limiter p/ o callback do PortAudio (hot path ~85ms)
        self._mic_status_limiter = RateLimiter(intervalo=10.0)

        self._find_mic_device()

    def _find_mic_device(self):
        """Localiza o dispositivo de microfone no sounddevice."""
        devices = sd.query_devices()
        default_input = sd.default.device

        # sd.default.device pode ser _InputOutputPair (acessa via índice)
        if hasattr(default_input, '__getitem__'):
            default_input_id = default_input[0]
        else:
            default_input_id = default_input

        log.debug(f"Dispositivos de áudio disponíveis ({len(devices)}):")

        for i, dev in enumerate(devices):
            name = dev['name']
            log.debug(f"  [{i}] {name} (in: {dev['max_input_channels']}, out: {dev['max_output_channels']})")

            if dev['max_input_channels'] > 0 and self.mic_device is None:
                if i == default_input_id:
                    self.mic_device = i
                    self.mic_channels = min(dev['max_input_channels'], self.channels)
                    log.info(f"  ✅ Microfone: [{i}] {name}")

        # Fallback
        if self.mic_device is None:
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    self.mic_device = i
                    self.mic_channels = min(dev['max_input_channels'], self.channels)
                    log.warning(f"  ⚠️ Microfone (fallback): [{i}] {dev['name']}")
                    break

        if self.mic_device is None:
            log.error("❌ Nenhum microfone encontrado!")
            sys.exit(1)

    def _get_source_name(self, keyword="monitor"):
        """Obtém o nome exato da source do PulseAudio por palavra-chave."""
        try:
            result = subprocess.run(
                ["pactl", "list", "sources", "short"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().split("\n"):
                if keyword in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]
        except Exception as e:
            log.warning(f"⚠️ Erro ao buscar source PulseAudio: {e}")
        return None

    def mic_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback do sounddevice para o microfone (thread de áudio, hot path!).

        ⚠️ NUNCA fazer I/O bloqueante aqui (print, file write, lock):
        roda a cada ~85ms e um bloqueio derruba a captura do mic.
        """
        if status:
            # Rate-limited: overflow pode se repetir a cada callback
            self._mic_status_limiter.log(f"Mic status: {status}")
        if self.is_recording and indata is not None:
            try:
                if indata.shape[1] == 1:
                    indata = np.repeat(indata, 2, axis=1)
                self.mic_queue.put(indata.copy())
            except Exception as e:
                self._mic_status_limiter.log(f"Erro no mic_callback: {e}")

    def _log_parec_stderr(self, process):
        """Thread: lê e exibe a stderr do parec (era DEVNULL — perdíamos erros!)."""
        if process.stderr is None:
            return
        try:
            for line in iter(process.stderr.readline, b''):
                if line:
                    log.warning(f"[parec stderr] {line.decode('utf-8', errors='replace').strip()}")
        except Exception:
            pass

    def _read_parec_loop(self):
        """Thread: lê áudio do PC via subprocess parec com reconexão automática.
        
        Se o parec morrer (PipeWire desconectar, timeout, etc), tenta reconectar
        até 10 vezes com backoff de 3s entre tentativas.
        Antes isso acontecia EM SILÊNCIO (stderr=DEVNULL) e a gravação
        continuava sem áudio do PC pelo resto da sessão.
        """
        max_reconnect = 10
        reconnect_delay = 3  # segundos entre tentativas

        for attempt in range(1, max_reconnect + 1):
            if not self.is_recording:
                break

            monitor_source = self._get_source_name("monitor")
            if not monitor_source:
                log.warning(f"⚠️ Source monitor não encontrada (tentativa {attempt}/{max_reconnect})")
                time.sleep(reconnect_delay)
                continue

            log.info(f"  ✅ Áudio do PC (monitor): {monitor_source}" +
                  (f" [reconexão {attempt}]" if attempt > 1 else ""))

            cmd = [
                "parec",
                f"--source={monitor_source}",
                "--format=s16le",
                "--channels=2",
                f"--rate={self.rate}",
                "--latency=20"
            ]

            try:
                self.parec_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=self.chunk * 4
                )
            except FileNotFoundError:
                log.error("❌ parec não encontrado! Instale: sudo apt install pulseaudio-utils")
                self.parec_process = None
                return

            # 📝 Thread pra logar stderr do parec (agora visível!)
            stderr_thread = threading.Thread(
                target=self._log_parec_stderr, args=(self.parec_process,), daemon=True
            )
            stderr_thread.start()

            bytes_per_frame = 4  # s16le stereo = 4 bytes/frame
            chunk_bytes = self.chunk * bytes_per_frame

            while self.is_recording and self.parec_process.poll() is None:
                try:
                    raw = self.parec_process.stdout.read(chunk_bytes)
                    if not raw:
                        break
                    data = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
                    self.pc_queue.put(data)
                except Exception as e:
                    log.warning(f"Erro lendo parec: {e}")
                    break

            # Se chegou aqui, o parec morreu — mas ainda estamos gravando
            if self.is_recording:
                exit_code = self.parec_process.poll()
                log.warning(f"⚠️ parec morreu (código {exit_code}, tentativa {attempt}/{max_reconnect}). "
                      f"Reconectando em {reconnect_delay}s...")
                time.sleep(reconnect_delay)

    def start(self, filename=None):
        """Inicia a gravação (microfone + áudio do PC).
        Se filename não for fornecido, gera um nome automático.
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            sessao_dir = os.path.join(AUDIO_DIR, f"sessao_{timestamp}")
            os.makedirs(sessao_dir, exist_ok=True)
            filename = os.path.join(sessao_dir, f"sessao_{timestamp}.ogg")
            self.flag_file = os.path.join(sessao_dir, "STOP_FLAG")
        else:
            self.flag_file = os.path.join(os.path.dirname(filename), "STOP_FLAG")

        self.is_recording = True

        # ─── Stream do microfone (sounddevice) ───
        if self.mic_device is not None:
            self.mic_stream = sd.InputStream(
                device=self.mic_device,
                channels=self.mic_channels,
                samplerate=self.rate,
                blocksize=self.chunk,
                callback=self.mic_callback,
                dtype='int16'
            )
            self.mic_stream.start()

        # ─── Subprocesso do parec (áudio do PC) ───
        self.parec_thread = threading.Thread(target=self._read_parec_loop, daemon=True)
        self.parec_thread.start()
        time.sleep(0.5)

        # ─── Thread de mixagem e salvamento ───
        self.save_thread = threading.Thread(target=self._save_loop, args=(filename,))
        self.save_thread.start()

        log.info(f"🎙️ Gravação iniciada → {filename}")
        log.info(f"   📌 Microfone (ganho {self.MIC_GAIN}x) + Áudio do PC (ganho {self.PC_GAIN}x)")
        log.info(f"   📌 Formato: OGG 192kbps 48kHz estéreo (~250MB em 3h de sessão)")
        log.info(f"   ⏹️  Para encerrar: Ctrl+C ou crie STOP_FLAG na pasta")

    def stop(self):
        """Para a gravação.
        
        Estratégia de shutdown rápido:
        1. Sinaliza is_recording = False (corta captura)
        2. Fecha fontes de áudio (mic + parec) IMEDIATAMENTE
        3. Drena filas por no máximo 2s
        4. Fecha ffmpeg na força bruta se necessário
        """
        self.is_recording = False

        # 1. Para as capturas IMEDIATAMENTE (corta a alimentação das filas)
        if hasattr(self, 'mic_stream'):
            self.mic_stream.stop()
            self.mic_stream.close()

        if self.parec_process and self.parec_process.poll() is None:
            self.parec_process.terminate()
            try:
                self.parec_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.parec_process.kill()

        # 2. Drena filas rapidamente (máx 2s) para não travar o shutdown
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join(timeout=2)

        # 3. Fecha ffmpeg na força bruta
        if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process is not None:
            try:
                self.ffmpeg_process.stdin.close()
            except:
                pass
            try:
                self.ffmpeg_process.wait(timeout=2)
            except:
                self.ffmpeg_process.kill()

        log.info("⏹️ Gravação encerrada.")

    def _save_loop(self, filename):
        """Thread de mixagem: combina microfone + áudio do PC e salva como OGG via ffmpeg.
        
        A thread para de escrever assim que is_recording=False E as filas estiverem vazias.
        Se demorar demais (>2s após stop), o stop() vai matar o ffmpeg externamente.
        """
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 's16le',
            '-ar', str(self.rate),
            '-ac', str(self.channels),
            '-i', 'pipe:0',
            '-c:a', 'libvorbis',
            '-b:a', '192k',
            filename
        ]

        try:
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            log.error("❌ ffmpeg não encontrado! Instale: sudo apt install ffmpeg")
            return

        empty_chunk = np.zeros((self.chunk, self.channels), dtype=np.int16)

        # 📝 Watchdog: detecta se uma fonte ficou silenciosa
        mic_silent_secs = 0.0
        pc_silent_secs = 0.0
        last_warn_time = 0.0

        try:
            while self.is_recording or not self.mic_queue.empty() or not self.pc_queue.empty():
                try:
                    mic_data = self.mic_queue.get(timeout=0.2)
                    mic_silent_secs = 0.0  # resetou, fonte voltou
                except queue.Empty:
                    mic_data = empty_chunk.copy()
                    mic_silent_secs += 0.2

                try:
                    pc_data = self.pc_queue.get(timeout=0.2)
                    pc_silent_secs = 0.0  # resetou, fonte voltou
                except queue.Empty:
                    pc_data = empty_chunk.copy()
                    pc_silent_secs += 0.2

                # Watchdog: avisa se uma fonte ficou muda por >5s (mas sem spam)
                now = time.time()
                if now - last_warn_time >= 30.0:  # no max 1 alerta a cada 30s
                    if mic_silent_secs > 5.0 and self.is_recording:
                        log.warning(f"⚠️ Microfone sem áudio por {mic_silent_secs:.0f}s (verifique o dispositivo)")
                        last_warn_time = now
                    if pc_silent_secs > 5.0 and self.is_recording:
                        log.warning(f"⚠️ Áudio do PC sem áudio por {pc_silent_secs:.0f}s (parec pode ter caído)")
                        last_warn_time = now

                min_frames = min(mic_data.shape[0], pc_data.shape[0])
                if min_frames == 0:
                    continue

                mic_float = mic_data[:min_frames].astype(np.float32) * self.MIC_GAIN
                pc_float = pc_data[:min_frames].astype(np.float32) * self.PC_GAIN

                mixed = np.clip(
                    mic_float + pc_float,
                    -32768.0, 32767.0
                ).astype(np.int16)

                if self.ffmpeg_process.stdin is not None and not self.ffmpeg_process.stdin.closed:
                    self.ffmpeg_process.stdin.write(mixed.tobytes())
                else:
                    break
        except BrokenPipeError:
            pass
        except ValueError:
            pass

        # Finaliza o ffmpeg
        try:
            self.ffmpeg_process.stdin.close()
        except:
            pass
        try:
            self.ffmpeg_process.wait(timeout=3)
        except:
            self.ffmpeg_process.kill()


if __name__ == "__main__":
    recorder = DualRecorder()
    recorder.start()  # Gera nome automático

    try:
        while True:
            if os.path.exists(recorder.flag_file):
                log.info("🚩 Flag de parada detectada. Encerrando...")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("🛑 Interrupção manual...")
    finally:
        recorder.stop()
        if hasattr(recorder, 'flag_file') and os.path.exists(recorder.flag_file):
            os.remove(recorder.flag_file)
