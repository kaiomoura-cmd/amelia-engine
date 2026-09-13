#!/usr/bin/env python
"""
Worker process para CoquiTTS XTTSv2.
Executa em processo SEPARADO para evitar conflito de DLLs cuDNN com faster-whisper.

Protocolo de comunicacao (stdin/stdout, JSON Lines):

  Modo arquivo (padrao):
    Request:  {"id": N, "text": "...", "file_path": "..."}
    Response: {"id": N, "status": "ok", "time": 3.5}

  Modo streaming:
    Request:  {"id": N, "text": "...", "mode": "stream"}
    Response: {"id": N, "type": "chunk", "audio": "<base64 PCM 24kHz mono float32>"}
    Response: {"id": N, "type": "done", "time": 2.1}
"""
import sys
import json
import os
import time
import base64
import tempfile
import gc

# ─── Config ──────────────────────────────────────────────────────
VOICE_REF = os.path.join(os.path.dirname(__file__), "cogs", "audio", "amelia_voice_ref.wav")
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_OUTPUT = os.path.join(tempfile.gettempdir(), "amelia_tts_response.wav")


def main():
    # ═══ LIMITES DE MEMÓRIA (para sistemas com 8GB RAM) ═══
    # Reduz fragmentação e uso de RAM do PyTorch
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OMP_NUM_THREADS", "2")  # Limita threads OpenMP
    os.environ.setdefault("MKL_NUM_THREADS", "2")  # Limita threads MKL

    import torch
    from TTS.api import TTS

    # ═══ OTIMIZACOES GLOBAIS ═══
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    # Forca init CUDA
    _ = torch.tensor([1.0]).cuda()
    torch.cuda.synchronize()
    del _

    # ═══ CARREGA MODELO ═══
    tts = TTS(model_name=MODEL_NAME).to("cuda")
    xtts_model = tts.synthesizer.tts_model  # Referência direta ao modelo XTTS
    gc.collect()  # Libera RAM usada durante o carregamento
    print(f"[TTS Worker] Modelo carregado em {torch.cuda.get_device_name(0)}", flush=True)

    # ╔════════════════════════════════════════════════════════════╗
    # ║  torch.compile REMOVIDO intencionalmente.                 ║
    # ║  mode='reduce-overhead' consome +2-3GB de RAM para        ║
    # ║  compilar kernels Triton — inviável em 8GB RAM.           ║
    # ║  O ganho (~20% em inferências repetidas) não justifica    ║
    # ║  o risco de OOM e travamento do sistema.                  ║
    # ╚════════════════════════════════════════════════════════════╝

    # ═══ Cache do speaker embedding (extrai UMA vez, reusa) ═══
    gpt_cond_latent = None
    speaker_embedding = None
    if os.path.exists(VOICE_REF):
        try:
            speaker_manager = tts.synthesizer.tts_model.speaker_manager
            gpt_cond_latent, speaker_embedding = speaker_manager.compute_embeddings(VOICE_REF)
            print(f"[TTS Worker] Speaker embedding cached", flush=True)
        except AttributeError:
            try:
                gpt_cond_latent, speaker_embedding = tts.synthesizer.tts_model.get_conditioning_latents(
                    audio_path=VOICE_REF,
                    gpt_cond_len=30,
                    max_ref_length=60,
                    sound_norm_refs=False,
                )
                print(f"[TTS Worker] Speaker embedding cached (get_conditioning_latents)", flush=True)
            except Exception as e2:
                print(f"[TTS Worker] Speaker cache FALHOU: {e2}", flush=True)
    else:
        print(f"[TTS Worker] AVISO: Voice ref nao encontrado: {VOICE_REF}", flush=True)

            # ═══ SINALIZA PRONTO (antes do warmup) ═══
    # O modelo esta carregado e aceita requisicoes.
    # O aquecimento (alocacao CUDA) acontece AGORA.
    print("[TTS Worker] Pronto para processar requisicoes." + " " * 40, flush=True)

    # ═══ WARMUP: Primeira inferencia (aloja CUDA kernels) ═══
    # No Windows: torch.compile NAO funciona (sem Triton), entao pulamos.
    # No Linux: sera ativado quando migrarmos.
    def _check_ram():
        """Mostra uso de RAM atual para diagnóstico."""
        try:
            import psutil
            proc = psutil.Process()
            ram_mb = proc.memory_info().rss / 1024 / 1024
            total_mb = psutil.virtual_memory().total / 1024 / 1024
            avail_mb = psutil.virtual_memory().available / 1024 / 1024
            print(f"[TTS Worker] RAM: {ram_mb:.0f}MB (processo) | {avail_mb:.0f}MB livre de {total_mb:.0f}MB", flush=True)
        except ImportError:
            pass  # psutil não instalado, sem problema

    _check_ram()

    def _warmup():
        t0 = time.perf_counter()
        try:
            # Texto CURTO para reduzir pico de memória no warmup
            warmup_text = "Teste rápido."
            warmup_output = os.path.join(tempfile.gettempdir(), "amelia_tts_warmup.wav")
            with torch.inference_mode():
                tts.tts_to_file(
                    text=warmup_text,
                    speaker_wav=VOICE_REF,
                    language="pt",
                    file_path=warmup_output
                )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()  # Libera RAM do warmup
            if os.path.exists(warmup_output):
                try: os.remove(warmup_output)
                except: pass
            dt = time.perf_counter() - t0
            _check_ram()
            print(f"[TTS Worker] Aquecimento OK em {dt:.1f}s", flush=True)
            return True
        except Exception as e:
            print(f"[TTS Worker] Aquecimento FALHOU: {e}", flush=True)
            return False

    if _warmup():
        print("[TTS Worker] Aquecimento completo.", flush=True)
    else:
        print("[TTS Worker] AVISO: Warmup falhou, mas modelo pode estar operacional.", flush=True)

    # ═══ LOOP PRINCIPAL: le requisicoes do stdin ═══
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            req_id = req.get("id", 0)
            text = req.get("text", "")
            mode = req.get("mode", "file")  # "file" (padrão) ou "stream"
            file_path = req.get("file_path", DEFAULT_OUTPUT)

            if not text:
                raise ValueError("texto vazio")

            if mode == "stream" and gpt_cond_latent is not None:
                # ═══ MODO STREAMING: gera audio em chunks ═══
                t0 = time.perf_counter()
                chunk_count = 0
                with torch.inference_mode():
                    chunks_gen = xtts_model.inference_stream(
                        text=text,
                        language="pt",
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        stream_chunk_size=20,
                        temperature=0.65,
                        repetition_penalty=10.0,
                        enable_text_splitting=False,
                    )
                    for chunk_tensor in chunks_gen:
                        # chunk_tensor: torch.Tensor [1, N] ou [N] float32 24kHz
                        audio_np = chunk_tensor.cpu().squeeze().numpy()
                        audio_bytes = audio_np.tobytes()
                        audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
                        chunk_resp = {
                            "id": req_id,
                            "type": "chunk",
                            "audio": audio_b64,
                        }
                        sys.stdout.write(json.dumps(chunk_resp) + "\n")
                        sys.stdout.flush()
                        chunk_count += 1

                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                t_gen = time.perf_counter() - t0
                done_resp = {"id": req_id, "type": "done", "time": t_gen, "chunks": chunk_count}
                sys.stdout.write(json.dumps(done_resp) + "\n")
                sys.stdout.flush()
                continue  # Próxima requisição

            # ═══ MODO ARQUIVO (padrão): gera arquivo completo ═══
            t0 = time.perf_counter()
            with torch.inference_mode():
                tts.tts_to_file(
                    text=text,
                    speaker_wav=VOICE_REF,
                    language="pt",
                    file_path=file_path
                )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            t_gen = time.perf_counter() - t0

            # Verifica se arquivo foi criado
            if os.path.exists(file_path) and os.path.getsize(file_path) > 100:
                resp = {"id": req_id, "status": "ok", "time": t_gen}
            else:
                resp = {"id": req_id, "status": "error", "error": "Arquivo de saida vazio ou nao criado"}

        except json.JSONDecodeError:
            resp = {"id": 0, "status": "error", "error": "JSON invalido"}
        except Exception as e:
            resp = {"id": req_id if 'req_id' in dir() else 0, "status": "error", "error": str(e)[:200]}

        # Envia resposta
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
