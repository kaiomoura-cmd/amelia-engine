#!/usr/bin/env python3
"""Benchmark de STT — faster-whisper (o modelo de produção da A.M.E.L.I.A.).

Mede, para cada combinação de modelo/dispositivo/precisão:
  * tempo de CARGA do modelo (cold start)
  * latência de transcrição de áudio curto / médio / longo
  * RTF (real-time factor) = tempo de transcrição / duração do áudio

Reproduz o caminho EXATO de produção (cogs/audio/interpretacao/transcricao.py):
  áudio PCM 48 kHz mono 16-bit -> WAV temporário -> Whisper (ffmpeg resample 16 kHz)

Uso:
    python bench_stt.py                 # varredura completa
    python bench_stt.py --repeticoes 5  # ajusta o número de repetições
    python bench_stt.py --rapido        # só GPU float16 (tiny/small/medium)

Saída: benchmarks/dados/resultados_stt.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = os.path.join(RAIZ, "benchmarks", "audio")
SAIDA = os.path.join(RAIZ, "benchmarks", "dados", "resultados_stt.csv")

# Configurações a testar: (modelo, dispositivo, compute_type)
CONFIGURACOES_COMPLETAS = [
    ("tiny", "cuda", "int8"),
    ("small", "cuda", "float16"),
    ("medium", "cuda", "float16"),
    ("small", "cpu", "int8"),
]
CONFIGURACOES_RAPIDAS = [
    ("tiny", "cuda", "int8"),
    ("small", "cuda", "float16"),
    ("medium", "cuda", "float16"),
]

AMOSTRAS = ["teste_curto.wav", "teste_medio.wav", "teste_longo.wav"]

# Parâmetros idênticos aos de produção (transcricao.py)
IDIOMA = "pt"
BEAM_SIZE = 4
TEMPERATURAS = [0.0]
VAD_FILTER = True


def carregar_amostra(nome: str):
    """Lê o WAV como PCM 16-bit (formato que o bot entrega ao Whisper)."""
    caminho = os.path.join(AUDIO, nome)
    dados, taxa = sf.read(caminho, dtype="int16", always_2d=False)
    if dados.ndim > 1:
        dados = dados[:, 0]
    duracao = len(dados) / taxa
    return caminho, dados, taxa, duracao


def transcrever(modelo, caminho_wav: str):
    """Executa a inferência e devolve (tempo, texto, duração de fala detectada).

    A duração "depois do VAD" é o denominador honesto para o RTF: as gravações de
    sessão têm muito silêncio, então dividir pelo tamanho nominal do arquivo
    subestima o trabalho real do modelo.
    """
    t0 = time.perf_counter()
    segmentos, info = modelo.transcribe(
        caminho_wav,
        language=IDIOMA,
        beam_size=BEAM_SIZE,
        temperature=TEMPERATURAS,
        vad_filter=VAD_FILTER,
    )
    # O faster-whisper é lazy: a inferência real só acontece ao consumir o gerador.
    texto = " ".join(s.text for s in segmentos)
    try:
        import torch
        torch.cuda.synchronize()
    except Exception:
        pass
    dt = time.perf_counter() - t0
    fala = getattr(info, "duration_after_vad", None)
    if not fala or fala <= 0:
        fala = getattr(info, "duration", None)
    return dt, texto.strip(), fala


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeticoes", type=int, default=5, help="execuções por amostra (após warm-up)")
    ap.add_argument("--rapido", action="store_true", help="só as configurações de GPU float16")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    configuracoes = CONFIGURACOES_RAPIDAS if args.rapido else CONFIGURACOES_COMPLETAS

    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sem GPU"
    except Exception:
        gpu = "torch indisponível"

    amostras = {nome: carregar_amostra(nome) for nome in AMOSTRAS}
    temporario = os.path.join(tempfile.gettempdir(), "amelia_bench_stt.wav")
    # O caminho de produção escreve o PCM num WAV temporário antes de transcrever.
    caminho, dados, taxa, _ = amostras[AMOSTRAS[0]]
    sf.write(temporario, dados, taxa, subtype="PCM_16")

    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    linhas = []

    print("=" * 78)
    print("  BENCHMARK STT — faster-whisper (parâmetros de produção)")
    print(f"  GPU: {gpu}")
    print(f"  Repetições por amostra: {args.repeticoes} (+1 de warm-up)")
    print("=" * 78)

    for modelo_nome, device, compute in configuracoes:
        rotulo = f"{modelo_nome}/{device}/{compute}"
        print(f"\n▶ {rotulo}")
        try:
            t0 = time.perf_counter()
            modelo = WhisperModel(modelo_nome, device=device, compute_type=compute)
            try:
                import torch
                torch.cuda.synchronize()
            except Exception:
                pass
            t_carga = time.perf_counter() - t0
            print(f"  carga do modelo: {t_carga:.2f} s")
        except Exception as e:
            print(f"  ✗ falhou ao carregar: {e}")
            continue

        for nome_amostra in AMOSTRAS:
            caminho, dados, taxa, duracao = amostras[nome_amostra]
            sf.write(temporario, dados, taxa, subtype="PCM_16")

            # warm-up (a primeira inferência compensa alocação de buffers/CUDA)
            try:
                transcrever(modelo, temporario)
            except Exception as e:
                print(f"  ✗ {nome_amostra}: {e}")
                continue

            tempos, texto, fala = [], "", None
            for _ in range(args.repeticoes):
                dt, texto, fala = transcrever(modelo, temporario)
                tempos.append(dt * 1000)

            mediana = statistics.median(tempos)
            p95 = sorted(tempos)[max(0, int(round(0.95 * len(tempos))) - 1)]
            rtf = (mediana / 1000) / duracao if duracao else 0
            rtf_fala = (mediana / 1000) / fala if fala else 0
            print(f"  {nome_amostra:18} {duracao:6.1f}s de arquivo"
                  f" (fala {fala:6.1f}s) → mediana {mediana:7.0f} ms | "
                  f"p95 {p95:7.0f} ms | RTF {rtf:.4f} | RTF-sobre-fala {rtf_fala:.4f}")

            linhas.append({
                "modelo": modelo_nome,
                "dispositivo": device,
                "precisao": compute,
                "amostra": nome_amostra,
                "duracao_audio_s": round(duracao, 2),
                "duracao_fala_s": round(fala, 2) if fala else "",
                "carga_modelo_s": round(t_carga, 2),
                "repeticoes": args.repeticoes,
                "mediana_ms": round(mediana, 1),
                "p95_ms": round(p95, 1),
                "min_ms": round(min(tempos), 1),
                "max_ms": round(max(tempos), 1),
                "rtf": round(rtf, 5),
                "rtf_fala": round(rtf_fala, 5),
                "caracteres_transcritos": len(texto),
            })

        del modelo
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    if linhas:
        with open(SAIDA, "w", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
            escritor.writeheader()
            escritor.writerows(linhas)
        print(f"\nCSV gravado em {SAIDA}")
    else:
        print("\nNenhuma medição coletada.")
        sys.exit(1)


if __name__ == "__main__":
    main()
