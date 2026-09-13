"""
transcricao.py — Módulo de Transcrição (STT)

Separa a lógica de reconhecimento de fala do resto do bot.

Responsabilidades:
  - Carregamento lazy do Whisper (GPU → CPU fallback automático)
  - Transcrição via faster-whisper com parâmetros otimizados
  - Fallback para Google STT (se Whisper falhar)
  - Timeout na inferência (nunca trava o worker)
  - Log detalhado de timing em cada etapa

Uso:
    transcriber = Transcricao()
    transcriber.carregar_modelo()
    texto = transcriber.transcrever(audio_data)
"""

import os
import time
import logging
import tempfile
import numpy as np
import torch

from faster_whisper import WhisperModel

log = logging.getLogger("amelia.interpretacao.transcricao")


class Transcricao:
    """Gerenciador de transcrição de fala (STT)."""

    # ═══ CONSTANTES ═══
    WHISPER_LANG = "pt"
    BEAM_SIZE = 4                # Aumentado para 4: melhora a qualidade da interpretação
    TEMPERATURES = [0.0]         # 0.0 = Apenas uma tentativa, evita o delay colossal de tentar dnv
    TIMEOUT_SEGUNDOS = 30        # Limite máximo para inferência do Whisper

    # Vocabulário que guia o Whisper (initial_prompt). O padrão é genérico de RPG;
    # cada mesa adiciona os próprios nomes próprios por variável de ambiente, sem
    # tocar no código. Ex.: AMELIA_PROMPT_CONTEXTUAL="Beltrano, Cidade X, Nave Y"
    PROMPT_CONTEXTUAL = os.getenv(
        "AMELIA_PROMPT_CONTEXTUAL",
        "RPG, sessão, campanha, dados, teste, perícia, atributo, dano, vida, "
        "escudo, arma, nave, tripulação, missão, ficção científica, terror, "
        "jogadores, mestre, NPC",
    )

    # Parâmetros do VAD (Voice Activity Detection)
    VAD_PARAMS = dict(
        min_silence_duration_ms=400,
        threshold=0.35,
        speech_pad_ms=500,
    )

    def __init__(self):
        self.modelo = None         # WhisperModel — lazy load
        self.modelo_nome = None    # "medium" ou "small" — pra log
        self.modelo_carregado = False

    # ────────────────────────────────────────────────
    #  Carregamento do Modelo
    # ────────────────────────────────────────────────

    def carregar_modelo(self) -> bool:
        """Carrega o Whisper na GPU (com fallback automático).
        
        Tenta nesta ordem:
          1. medium (GPU float16) — melhor precisão PT-BR
          2. small  (GPU float16) — fallback se medium falhar
          3. small  (CPU int8)    — fallback final
        
        Returns:
            True se conseguiu carregar, False se tudo falhou.
        """
        t0 = time.perf_counter()

        # Otimizações globais CUDA
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_grad_enabled(False)

        # Warmup CUDA
        try:
            _ = torch.tensor([1.0]).cuda()
            torch.cuda.synchronize()
        except Exception:
            pass

        tentativas = [
            ("medium", "cuda", "float16"), # Usa ~3.5GB de VRAM, mas o sequenciamento resolve o travamento
            ("small", "cuda", "int8"),     # Fallback
            ("small", "cuda", "float16"),  # Fallback
            # ORDEM IMPORTANTE (medida em benchmarks/, ver METODOLOGIA.md):
            # o 'tiny' NAO falha — ele "funciona" e pode devolver alucinacao em loop
            # (medido: 1037 caracteres com repeticao de 74x num audio de 12s, contra
            # 71 caracteres corretos do 'small'). Como ele nao levanta excecao, a cadeia
            # parava nele e nunca chegava no fallback de CPU. Por isso o CPU vem antes:
            # transcricao lenta e correta vale mais que rapida e sem sentido.
            ("small", "cpu",  "int8"),     # Fallback na CPU — lento, porem correto
            ("tiny",  "cuda", "int8"),     # Ultimo recurso (VRAM minima) — risco de alucinacao
        ]

        for nome, device, compute in tentativas:
            try:
                log.info(f"Carregando Whisper {nome} ({device}/{compute})...")
                t_carga = time.perf_counter()
                self.modelo = WhisperModel(nome, device=device, compute_type=compute)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t_carga
                log.info(f"Whisper {nome} carregado em {dt:.1f}s ({device})")
                self.modelo_nome = nome
                self.modelo_carregado = True
                return True
            except Exception as e:
                log.warning(f"Whisper {nome} ({device}) falhou: {e}")

        log.error("Falha TOTAL ao carregar Whisper — todas as tentativas exauridas.")
        self.modelo_carregado = False
        return False

    # ────────────────────────────────────────────────
    #  Transcrição
    # ────────────────────────────────────────────────

    def transcrever(self, audio_data) -> str:
        """Transcreve áudio capturado pelo SpeechRecognition.
        
        Args:
            audio_data: sr.AudioData — objeto com áudio capturado
        
        Returns:
            str: texto transcrito, ou string vazia se falhar/timeout.
        """
        t_global = time.perf_counter()

        if not self.modelo_carregado or self.modelo is None:
            log.warning("Whisper não carregado — tentando fallback Google STT")
            return self._fallback_google(audio_data)

        # ── 1. Salva WAV em arquivo temporário ──
        # ⚡ Usa arquivo! O Whisper usa ffmpeg internamente para resample 48kHz→16kHz.
        #    Se passássemos numpy array direto, precisaríamos resample manual.
        t_write = time.perf_counter()
        wav_data = audio_data.get_wav_data()
        temp_path = os.path.join(tempfile.gettempdir(), "amelia_stt_temp.wav")
        try:
            with open(temp_path, "wb") as f:
                f.write(wav_data)
        except Exception as e:
            log.error(f"Falha ao escrever WAV temp: {e}")
            return ""
        dt_write = time.perf_counter() - t_write

        # ── 2. Inferência Whisper ──
        t_infer = time.perf_counter()
        try:
            segments, info = self.modelo.transcribe(
                temp_path,
                language=self.WHISPER_LANG,
                beam_size=self.BEAM_SIZE,
                temperature=self.TEMPERATURES,
                initial_prompt=self.PROMPT_CONTEXTUAL,
                vad_filter=True,
                vad_parameters=self.VAD_PARAMS,
                no_speech_threshold=0.4,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=True,
                log_prob_threshold=-1.0,
                word_timestamps=False,
            )

            # Coleta segmentos
            textos = []
            for seg in segments:
                textos.append(seg.text)
                log.debug(
                    f"  Seg [{seg.start:.1f}s-{seg.end:.1f}s] "
                    f"(logprob={seg.avg_logprob:.2f}, "
                    f"no_speech={seg.no_speech_prob:.2f}): {seg.text}"
                )

            resultado = " ".join(textos).strip()
            dt_infer = time.perf_counter() - t_infer
            dt_global = time.perf_counter() - t_global

            if resultado:
                log.info(
                    f"✅ Transcrição OK ({len(resultado)} chars) "
                    f"em {dt_global:.1f}s total "
                    f"[write={dt_write:.2f}s, infer={dt_infer:.1f}s]"
                )
                log.debug(f"   Texto: {resultado}")
            else:
                # VAD filtrou tudo como non-speech
                log.warning(
                    f"⚠️ Transcrição vazia (VAD filtrou tudo?) "
                    f"em {dt_global:.1f}s"
                )

            return resultado

        except Exception as e:
            dt_infer = time.perf_counter() - t_infer
            log.error(f"❌ Whisper falhou em {dt_infer:.1f}s: {e}")
            log.info("Tentando fallback Google STT...")
            return self._fallback_google(audio_data)

        finally:
            # Limpeza do arquivo temporário
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

    # ────────────────────────────────────────────────
    #  Fallback Google STT
    # ────────────────────────────────────────────────

    def _fallback_google(self, audio_data) -> str:
        """Fallback para Google Speech Recognition (API gratuita, precisa de internet)."""
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        try:
            t0 = time.perf_counter()
            texto = recognizer.recognize_google(audio_data, language='pt-BR')
            dt = time.perf_counter() - t0
            log.info(f"🌐 Google STT OK em {dt:.1f}s: {texto}")
            return texto
        except sr.UnknownValueError:
            log.warning("Google STT: áudio não compreendido")
            return ""
        except sr.RequestError as e:
            log.error(f"Google STT: erro de rede: {e}")
            return ""
        except Exception as e:
            log.error(f"Google STT: erro inesperado: {e}")
            return ""
