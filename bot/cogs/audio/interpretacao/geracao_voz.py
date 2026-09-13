"""
geracao_voz.py tudo aqui— Módulo de Geração de Voz (TTS)


Separa a lógica de síntese de fala do resto do bot.

Responsabilidades:
  - Gerenciamento do subprocesso XTTSv2 (isolamento cuDNN)
  - Lazy loading do worker
  - Fallback Edge-TTS (cloud gratuito)
  - Log detalhado de timing em cada etapa

Uso:
    voz = GeracaoVoz()
    voz.iniciar_worker()
    caminho = await voz.sintetizar("Olá, capitão.")
    # Reproduz caminho no Discord
"""

import os
import sys
import json
import time
import asyncio
import logging
import tempfile
import subprocess
import numpy as np

log = logging.getLogger("amelia.interpretacao.geracao_voz")


class GeracaoVoz:
    """Gerenciador de síntese de fala (TTS).

    O worker XTTSv2 roda em um SUBPROCESSO separado para evitar conflito
    de DLLs cuDNN entre PyTorch (CoquiTTS) e CTranslate2 (faster-whisper).

    ⚠ O worker DEMORA para iniciar (carregamento do modelo + torch.compile).
       O método `iniciar_worker_async()` permite começar sem travar o bot.
       Enquanto o worker não fica pronto, o fallback Edge-TTS é usado.
    """

    # ═══ CONSTANTES ═══
    WORKER_SCRIPT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "tts_worker.py")
    )
    VOICE_REF = os.path.join(
        os.path.dirname(__file__), "..", "amelia_voice_ref.wav"
    )

    TIMEOUT_WORKER_INIT = 120   # segundos p/ worker carregar modelo
    TIMEOUT_TTS = 120           # segundos p/ geração de audio (worker)
    TIMEOUT_EDGE = 30           # segundos p/ Edge-TTS

    def __init__(self):
        self.worker = None          # subprocess.Popen
        self.worker_lock = asyncio.Lock()
        self.worker_carregando = False  # True se está carregando em background
        self.worker_pronto = False      # True se worker respondeu "Pronto"
        self._warmup_callback = None       # callable() quando modelo carregar
        self._aquecimento_callback = None   # callable() quando aquecimento completar
        self._primeira_geracao_ok = False   # True após 1ª geração XTTS bem-sucedida
        self._on_primeira_geracao = None    # callback da 1ª geração (warmup real)
        self._worker_reiniciando = False    # True se reinício em andamento (evita loop)
        self._worker_restarts = 0           # contador de reinícios

    # ────────────────────────────────────────────────
    #  Gerenciamento do Worker (subprocesso)
    # ────────────────────────────────────────────────

    # ────────────────────────────────────────────────
    #  Carregamento ASSÍNCRONO do Worker (não bloqueia o bot)
    # ────────────────────────────────────────────────

    def _iniciar_stderr_reader(self):
        """Inicia thread que drena o stderr do worker (evita deadlock por buffer cheio)."""
        import threading
        
        def _drenar_stderr():
            """Lê stderr do worker em loop e faz log.
            
            ⚠ CRÍTICO: Sem isso, o buffer do stderr enche (4KB-64KB no Windows)
              e o worker TRAVA silenciosamente, nunca ficando "pronto".
            """
            try:
                for line in iter(self.worker.stderr.readline, ''):
                    if line:
                        log.debug(f"[TTS Worker stderr] {line.rstrip()}")
                    else:
                        break  # Fim do pipe
            except Exception as e:
                log.debug(f"Leitura stderr do worker encerrada: {e}")
        
        t = threading.Thread(target=_drenar_stderr, daemon=True)
        t.start()
        return t

    def iniciar_worker_async(self, timeout: int = 30, on_warmup_complete=None, on_aquecimento_completo=None) -> bool:
        """Inicia o worker TTS em background com timeout curto.
        
        Diferença do `iniciar_worker()`:
          - Espera no MÁXIMO `timeout` segundos (padrão 30s)
          - Se não ficar pronto a tempo, DEIXA CARREGANDO em background
          - O fallback Edge-TTS será usado até o worker ficar pronto
          - O worker continuará carregando e estará disponível na
            próxima requisição se terminar
        
        Args:
            timeout: segundos máximos para esperar o worker sinalizar "Pronto"
            on_warmup_complete: callable() chamado quando o worker carregar o modelo
            on_aquecimento_completo: callable() chamado quando a primeira
                inferência (warmup CUDA) terminar
        
        Returns:
            True se worker foi iniciado (mesmo se ainda carregando)
            False se não foi possível iniciar
        """
        self._warmup_callback = on_warmup_complete
        self._aquecimento_callback = on_aquecimento_completo
        
        if self.worker_pronto:
            return True
        
        if self.worker_carregando:
            log.debug("Worker já está carregando em background.")
            return True
        
        if not self._worker_script_existe():
            return False
        
        if not self._voice_ref_existe():
            return False
        
        log.info(f"Iniciando worker TTS (async, timeout={timeout}s)...")
        t0 = time.perf_counter()
        
        # Marca como carregando para não iniciar duas vezes
        self.worker_carregando = True
        
        try:
            self.worker = subprocess.Popen(
                [sys.executable, "-u", self.WORKER_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}
            )
            
            # ═══ INICIA DRENAGEM DE STDERR (ANTES de ler stdout!) ═══
            self._iniciar_stderr_reader()
            
            # ═══ FASE 1: Aguarda "Pronto para processar" (modelo carregado) ═══
            pronto_detected = False
            while time.perf_counter() - t0 < timeout:
                line = self.worker.stdout.readline()
                if line:
                    log.info(f"[TTS Worker] {line.strip()}")
                    if "Pronto para processar" in line:
                        dt = time.perf_counter() - t0
                        log.info(f"✅ Worker TTS pronto em {dt:.1f}s (async)")
                        self.worker_pronto = True
                        self.worker_carregando = False
                        pronto_detected = True
                        # 🔥 Dispara callback de modelo carregado
                        if self._warmup_callback:
                            try:
                                self._warmup_callback()
                            except Exception as e:
                                log.error(f"Warmup callback falhou: {e}")
                        break
            
            if not pronto_detected:
                # Timeout do async — mas worker CONTINUA carregando!
                dt = time.perf_counter() - t0
                log.warning(
                    f"⏳ Worker TTS ainda carregando após {dt:.1f}s "
                    f"(continuando em background). Edge-TTS será usado."
                )
                self._iniciar_monitor_background()
                return True
            
            # ═══ FASE 2: Aguarda "Aquecimento completo" (primeira inferência) ═══
            # O worker faz a primeira inferência após sinalizar "Pronto".
            # Isso pode levar 30-60s (torch.compile + alocação CUDA).
            timeout_aquecimento = 120  # segundos adicionais para o aquecimento
            t_aq = time.perf_counter()
            aquecimento_detected = False
            
            while time.perf_counter() - t_aq < timeout_aquecimento:
                # Usando select-like approach: verifica se tem dados no pipe
                line = self.worker.stdout.readline()
                if line:
                    log.info(f"[TTS Worker] {line.strip()}")
                    if "Aquecimento completo" in line:
                        dt = time.perf_counter() - t0
                        log.info(f"🔥 Worker TTS aquecido em {dt:.1f}s")
                        aquecimento_detected = True
                        if self._aquecimento_callback:
                            try:
                                self._aquecimento_callback()
                            except Exception as e:
                                log.error(f"Aquecimento callback falhou: {e}")
                        break
                    elif "Aquecimento FALHOU" in line:
                        log.warning("Worker TTS: aquecimento falhou, mas modelo está operacional.")
                        break
                else:
                    # readline retornou vazio = pipe fechado = worker morreu
                    if self.worker.poll() is not None:
                        log.error("Worker TTS morreu durante aquecimento!")
                        self._matar_worker()
                        self.worker_pronto = False
                        return False
            
            if not aquecimento_detected:
                log.warning(
                    f"⏳ Aquecimento do worker não completou em {timeout_aquecimento}s, "
                    f"mas modelo está carregado e operacional."
                )
            
            return True
        
        except Exception as e:
            log.error(f"Falha ao iniciar worker TTS: {e}")
            self._matar_worker()
            self.worker_carregando = False
            return False

    def _iniciar_monitor_background(self):
        """Monitora o worker em background até ficar pronto E aquecido, ou morrer."""
        import threading
        
        def _monitor():
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < self.TIMEOUT_WORKER_INIT:
                if not self.worker or self.worker.poll() is not None:
                    log.warning("Worker TTS morreu durante carregamento async.")
                    self.worker_pronto = False
                    self.worker_carregando = False
                    return
                
                # Tenta ler stdout
                try:
                    line = self.worker.stdout.readline()
                    if line:
                        log.info(f"[TTS Worker] {line.strip()}")
                        if "Pronto para processar" in line:
                            dt = time.perf_counter() - t0
                            log.info(f"✅ Worker TTS pronto (background) em {dt:.1f}s")
                            self.worker_pronto = True
                            self.worker_carregando = False
                            # 🔥 Dispara callback de modelo carregado
                            if self._warmup_callback:
                                try:
                                    self._warmup_callback()
                                except Exception as e:
                                    log.error(f"Warmup callback falhou: {e}")
                            # Continua monitorando para pegar "Aquecimento completo"
                            continue
                        elif "Aquecimento completo" in line:
                            log.info(f"🔥 Worker TTS aquecido (background) em {time.perf_counter()-t0:.1f}s")
                            if self._aquecimento_callback:
                                try:
                                    self._aquecimento_callback()
                                except Exception as e:
                                    log.error(f"Aquecimento callback falhou: {e}")
                            return  # Worker completamente pronto e aquecido!
                        elif "Aquecimento FALHOU" in line:
                            log.warning("Worker TTS: aquecimento falhou, mas modelo está operacional.")
                            return
                except Exception:
                    time.sleep(0.1)
            
            # Timeout total
            log.error(f"Worker TTS não ficou pronto em {self.TIMEOUT_WORKER_INIT}s (background)")
            self._matar_worker()
            self.worker_pronto = False
            self.worker_carregando = False
        
        t = threading.Thread(target=_monitor, daemon=True)
        t.start()

    def _worker_script_existe(self) -> bool:
        if not os.path.exists(self.WORKER_SCRIPT):
            log.error(f"Worker TTS não encontrado: {self.WORKER_SCRIPT}")
            return False
        return True

    def _voice_ref_existe(self) -> bool:
        if not os.path.exists(self.VOICE_REF):
            log.warning(f"Voice ref não encontrado: {self.VOICE_REF}. Edge-TTS será usado.")
            return False
        return True

    def iniciar_worker(self) -> bool:
        """Inicia o subprocesso do XTTSv2 (bloqueante, roda em executor).
        
        Returns:
            True se worker iniciou com sucesso, False se falhou.
        """
        if self.worker is not None and self.worker.poll() is None:
            log.debug("Worker TTS já está rodando.")
            return True

        if not self._worker_script_existe():
            return False

        if not self._voice_ref_existe():
            return False

        log.info("Iniciando worker TTS (subprocesso)...")
        t0 = time.perf_counter()

        try:
            self.worker = subprocess.Popen(
                [sys.executable, "-u", self.WORKER_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}
            )
            
            # ═══ INICIA DRENAGEM DE STDERR (ANTES de ler stdout!) ═══
            self._iniciar_stderr_reader()

            # Aguarda "Pronto para processar" na saída
            while time.perf_counter() - t0 < self.TIMEOUT_WORKER_INIT:
                line = self.worker.stdout.readline()
                if line:
                    log.info(f"[TTS Worker] {line.strip()}")
                    if "Pronto para processar" in line:
                        dt = time.perf_counter() - t0
                        log.info(f"Worker TTS pronto em {dt:.1f}s")
                        return True

            # Timeout
            log.error(f"Worker TTS não respondeu em {self.TIMEOUT_WORKER_INIT}s")
            self._matar_worker()
            return False

        except Exception as e:
            log.error(f"Falha ao iniciar worker TTS: {e}")
            self._matar_worker()
            return False

    def _matar_worker(self, auto_restart=False):
        """Mata o subprocesso do TTS.
        
        Args:
            auto_restart: Se True, agenda reinício automático do worker
                          em background (usado quando o worker morre
                          inesperadamente durante uso).
        """
        self.worker_pronto = False
        self.worker_carregando = False
        if self.worker:
            try:
                self.worker.stdin.close()
                self.worker.terminate()
                self.worker.wait(timeout=5)
            except Exception:
                try:
                    self.worker.kill()
                except Exception:
                    pass
            self.worker = None
            log.info("Worker TTS encerrado.")
        
        if auto_restart and not self._worker_reiniciando:
            self._reiniciar_worker_background()

    def _reiniciar_worker_background(self):
        # Reinicia o worker TTS em background (thread separada).
        # Usado quando o worker morre inesperadamente.
        import threading

        MAX_RESTARTS = 3
        if self._worker_reiniciando:
            log.debug("Reinicio do worker ja em andamento.")
            return

        if self._worker_restarts >= MAX_RESTARTS:
            log.error(f"Worker TTS atingiu limite de {MAX_RESTARTS} reinicios. Desistindo.")
            return

        self._worker_reiniciando = True
        self._worker_restarts += 1
        log.info(f"Reiniciando worker TTS (tentativa {self._worker_restarts}/{MAX_RESTARTS})... ")

        def _restart():
            try:
                ok = self.iniciar_worker_async(timeout=120)
                if ok:
                    log.info("Worker TTS reiniciado com sucesso (background).")
                    self._worker_restarts = 0
                else:
                    log.error("Falha ao reiniciar worker TTS (background).")
            except Exception as e:
                log.error(f"Excecao ao reiniciar worker TTS: {e}")
            finally:
                self._worker_reiniciando = False

        t = threading.Thread(target=_restart, daemon=True)
        t.start()

    # ────────────────────────────────────────────────
    #  Requisição Síncrona ao Worker (chamado em executor)
    # ────────────────────────────────────────────────

    def _requisitar_worker_sync(self, texto: str, file_path: str) -> tuple[bool, float]:
        """Envia requisição para o worker TTS (chamada BLOQUEANTE, usar em executor).
        
        ⚠ Falha RÁPIDO se worker não estiver pronto — sem tentar reiniciar.
           O fallback para Edge-TTS é gerenciado por `sintetizar()`.
        
        Args:
            texto: Texto a sintetizar
            file_path: Caminho do arquivo .wav de saída
        
        Returns:
            (sucesso: bool, tempo_generacao: float)
        """
        if not self.worker_pronto:
            log.warning("Worker TTS não está pronto — pulando XTTSv2.")
            return False, 0.0
        
        if not self.worker or self.worker.poll() is not None:
            log.warning("Worker TTS morreu — pulando XTTSv2. Edge-TTS será usado.")
            self.worker_pronto = False
            self.worker_carregando = False
            # Agenda reinício automático em background
            if not self._worker_reiniciando:
                self._reiniciar_worker_background()
            return False, 0.0

        try:
            req = json.dumps({"id": 1, "text": texto, "file_path": file_path})
            self.worker.stdin.write(req + "\n")
            self.worker.stdin.flush()

            # Lê resposta (com timeout manual — select não funciona com pipes no Windows)
            inicio = time.perf_counter()
            resposta = None
            while time.perf_counter() - inicio < self.TIMEOUT_TTS:
                if self.worker.stdout.readable():
                    resposta = self.worker.stdout.readline()
                    if resposta:
                        break
                    time.sleep(0.05)
                else:
                    time.sleep(0.05)

            if not resposta:
                log.error(f"Worker TTS timeout após {self.TIMEOUT_TTS}s")
                self._matar_worker(auto_restart=True)
                return False, 0.0

            resp = json.loads(resposta.strip())
            if resp.get("status") == "ok":
                # 🔥 Primeira geração bem-sucedida = warmup completo!
                if not self._primeira_geracao_ok:
                    self._primeira_geracao_ok = True
                    if self._on_primeira_geracao:
                        try:
                            self._on_primeira_geracao()
                        except Exception as e:
                            log.error(f"Callback primeira geração falhou: {e}")
                return True, resp.get("time", 0.0)
            else:
                log.error(f"Worker TTS erro: {resp.get('error', 'desconhecido')}")
                return False, 0.0

        except Exception as e:
            log.error(f"Erro comunicação worker TTS: {e}")
            self._matar_worker(auto_restart=True)
            return False, 0.0

    # ────────────────────────────────────────────────
    #  Limpeza de texto para TTS
    # ────────────────────────────────────────────────

    @staticmethod
    def _limpar_texto_tts(texto: str) -> str:
        """Remove formatação e corrige pontuação para o XTTSv2."""
        texto_limpo = texto.replace('*', '').replace('_', '')
        # O XTTSv2 tem um bug no idioma PT-BR onde ele frequentemente lê o caractere '.' 
        # como a palavra "ponto" (especialmente em siglas como A.M.E.L.I.A. ou finais de frase).
        # Substituir por vírgula mantém a pausa de respiração do TTS sem que ele fale a pontuação.
        texto_limpo = texto_limpo.replace('...', ', ').replace('.', ',')
        return texto_limpo.strip()

    # ────────────────────────────────────────────────
    #  API Pública: sintetizar texto em arquivo de áudio
    # ────────────────────────────────────────────────

    async def sintetizar(self, texto: str) -> str | None:
        """Gera áudio a partir do texto (modo arquivo — compatibilidade).
        
        Tenta:
          1. XTTSv2 via Worker (subprocesso) — melhor qualidade
          2. Edge-TTS (cloud) — fallback gratuito
        
        Args:
            texto: Texto a ser sintetizado
        
        Returns:
            str: Caminho do arquivo de áudio gerado, ou None se falhar.
        """
        texto_limpo = self._limpar_texto_tts(texto)
        
        if not texto_limpo:
            log.warning("Texto vazio após limpeza — nada a sintetizar.")
            return None

        # ── Tentativa 1: XTTSv2 via Worker ──
        if self._voice_ref_existe():
            arquivo = os.path.join(tempfile.gettempdir(), "amelia_tts_xtts.wav")
            t0 = time.perf_counter()

            try:
                sucesso, t_gen = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        self._requisitar_worker_sync,
                        texto_limpo,
                        arquivo
                    ),
                    timeout=self.TIMEOUT_TTS
                )

                if sucesso:
                    # Verifica se o arquivo é válido
                    if os.path.exists(arquivo) and os.path.getsize(arquivo) >= 1000:
                        dt = time.perf_counter() - t0
                        log.info(f"✅ XTTSv2 OK ({t_gen:.1f}s gen, {dt:.1f}s total)")
                        return arquivo
                    else:
                        log.warning(f"XTTSv2 gerou arquivo inválido: {arquivo}")

                else:
                    log.warning("XTTSv2 falhou na geração.")

            except asyncio.TimeoutError:
                log.error(f"XTTSv2 timeout ({self.TIMEOUT_TTS}s). Matando worker.")
                self._matar_worker()
            except Exception as e:
                log.error(f"XTTSv2 erro: {e}")

        # ── Tentativa 2: Edge-TTS (fallback cloud) ──
        import edge_tts
        arquivo = os.path.join(tempfile.gettempdir(), "amelia_tts_edge.mp3")
        t0 = time.perf_counter()

        try:
            communicate = edge_tts.Communicate(
                texto_limpo,
                "pt-BR-ThalitaMultilingualNeural",
                rate="-10%",
                pitch="-20Hz"
            )
            await asyncio.wait_for(
                communicate.save(arquivo),
                timeout=self.TIMEOUT_EDGE
            )

            if os.path.exists(arquivo) and os.path.getsize(arquivo) >= 1000:
                dt = time.perf_counter() - t0
                log.info(f"✅ Edge-TTS OK em {dt:.1f}s")
                return arquivo
            else:
                log.error("Edge-TTS gerou arquivo inválido.")
                return None

        except asyncio.TimeoutError:
            log.error(f"Edge-TTS timeout ({self.TIMEOUT_EDGE}s)")
            return None
        except Exception as e:
            log.error(f"Edge-TTS falhou: {e}")
            return None

    # ────────────────────────────────────────────────
    #  API Streaming: sintetizar com chunks em tempo real
    # ────────────────────────────────────────────────

    def _requisitar_worker_stream_sync(self, texto: str, audio_source) -> tuple[bool, float]:
        """Envia requisição streaming para o worker TTS (BLOQUEANTE, usar em executor).
        
        Lê chunks de áudio do stdout do worker e alimenta o audio_source em tempo real.
        
        Args:
            texto: Texto a sintetizar
            audio_source: StreamingAudioSource para alimentar com chunks
        
        Returns:
            (sucesso: bool, tempo_geracao: float)
        """
        import base64
        
        if not self.worker_pronto:
            log.warning("Worker TTS não está pronto — streaming indisponível.")
            return False, 0.0
        
        if not self.worker or self.worker.poll() is not None:
            log.warning("Worker TTS morreu — streaming indisponível.")
            self.worker_pronto = False
            if not self._worker_reiniciando:
                self._reiniciar_worker_background()
            return False, 0.0

        try:
            req = json.dumps({"id": 1, "text": texto, "mode": "stream"})
            self.worker.stdin.write(req + "\n")
            self.worker.stdin.flush()

            # Lê chunks até receber "done"
            inicio = time.perf_counter()
            chunk_count = 0
            import select
            while time.perf_counter() - inicio < self.TIMEOUT_TTS:
                # Verifica se worker morreu
                if self.worker.poll() is not None:
                    log.error("Worker TTS morreu durante streaming!")
                    audio_source.finish()
                    self._matar_worker(auto_restart=True)
                    return False, 0.0
                
                # select() com timeout de 0.5s — evita readline() bloqueante infinito
                ready, _, _ = select.select([self.worker.stdout], [], [], 0.5)
                if not ready:
                    continue  # Nada para ler, volta pro check de timeout
                
                line = self.worker.stdout.readline()
                if not line:
                    # Pipe fechado = worker morreu
                    log.error("Worker TTS: pipe stdout fechou durante streaming!")
                    audio_source.finish()
                    self._matar_worker(auto_restart=True)
                    return False, 0.0
                
                try:
                    resp = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                
                if resp.get("type") == "chunk":
                    audio_b64 = resp.get("audio", "")
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_source.feed(audio_bytes)
                        chunk_count += 1
                
                elif resp.get("type") == "done":
                    t_gen = resp.get("time", 0.0)
                    log.info(f"✅ XTTSv2 Streaming OK ({t_gen:.1f}s, {chunk_count} chunks)")
                    audio_source.finish()
                    return True, t_gen
                
                elif resp.get("status") == "error":
                    log.error(f"Worker TTS stream erro: {resp.get('error', '?')}")
                    audio_source.finish()
                    return False, 0.0

            log.error(f"Worker TTS stream timeout após {self.TIMEOUT_TTS}s")
            audio_source.finish()
            self._matar_worker(auto_restart=True)
            return False, 0.0

        except Exception as e:
            log.error(f"Erro comunicação worker TTS stream: {e}")
            audio_source.finish()
            self._matar_worker(auto_restart=True)
            return False, 0.0

    async def sintetizar_stream(self, texto: str, audio_source) -> bool:
        """Gera áudio em streaming e alimenta o audio_source em tempo real.
        
        Args:
            texto: Texto a sintetizar
            audio_source: StreamingAudioSource para alimentar com chunks
        
        Returns:
            True se geração com sucesso, False se falhou
        """
        texto_limpo = self._limpar_texto_tts(texto)
        if not texto_limpo:
            log.warning("Texto vazio — nada a sintetizar em stream.")
            audio_source.finish()
            return False

        if self._voice_ref_existe() and self.worker_pronto:
            try:
                sucesso, t_gen = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        self._requisitar_worker_stream_sync,
                        texto_limpo,
                        audio_source
                    ),
                    timeout=self.TIMEOUT_TTS
                )
                if sucesso:
                    return True
                else:
                    log.warning("Streaming XTTSv2 falhou.")
            except asyncio.TimeoutError:
                log.error("Streaming XTTSv2 timeout.")
                audio_source.finish()
            except Exception as e:
                log.error(f"Streaming XTTSv2 erro: {e}")
                audio_source.finish()
        
        # Fallback: gera arquivo completo e alimenta de uma vez
        log.info("Fallback: gerando arquivo completo para streaming...")
        arquivo = await self.sintetizar(texto)
        if arquivo:
            try:
                import wave
                with wave.open(arquivo, 'rb') as wf:
                    pcm_data = wf.readframes(wf.getnframes())
                    # O arquivo do XTTS é 24kHz mono int16, mas o do Edge-TTS pode ser mp3
                    # Para simplificar, alimentamos via feed que faz a conversão
                    float_data = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32767.0
                    audio_source.feed(float_data.tobytes())
                audio_source.finish()
                return True
            except Exception as e:
                log.error(f"Fallback stream falhou: {e}")
                audio_source.finish()
                return False
        
        audio_source.finish()
        return False

    # ────────────────────────────────────────────────
    #  Limpeza
    # ────────────────────────────────────────────────

    def limpar(self):
        """Libera recursos do worker TTS."""
        self._matar_worker()

