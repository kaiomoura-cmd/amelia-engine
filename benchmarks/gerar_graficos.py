#!/usr/bin/env python3
"""Gera os gráficos do README a partir dos CSVs de benchmark.

Lê benchmarks/dados/*.csv e escreve benchmarks/graficos/*.png

    python gerar_graficos.py

Regra: todo gráfico carrega o hardware na legenda. Latência sem hardware é marketing.
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS = os.path.join(RAIZ, "benchmarks", "dados")
SAIDA = os.path.join(RAIZ, "benchmarks", "graficos")

# ─── Identidade visual ───────────────────────────────────────────────
TINTA = "#1f2937"
GRADE = "#e5e7eb"
CORES = {
    "tiny": "#94a3b8",
    "small": "#2563eb",
    "medium": "#7c3aed",
    "stream": "#059669",
    "arquivo": "#f59e0b",
    "carga": "#dc2626",
    "fixo": "#9ca3af",
    "medido": "#2563eb",
    "nao_medido": "#d1d5db",
}
HARDWARE = "RTX 3060 Ti (8 GB) · AMD Ryzen 7 2700X · 7,7 GB RAM"


def configurar():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRADE,
        "axes.labelcolor": TINTA,
        "text.color": TINTA,
        "xtick.color": TINTA,
        "ytick.color": TINTA,
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.dpi": 140,
    })


def rodape(fig, extra: str = ""):
    texto = HARDWARE + (f"  ·  {extra}" if extra else "")
    fig.text(0.5, 0.015, texto, ha="center", va="bottom", fontsize=8, color="#6b7280")


def ler_csv(nome: str):
    caminho = os.path.join(DADOS, nome)
    if not os.path.exists(caminho):
        return []
    with open(caminho, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def grafico_stt_latencia(linhas):
    if not linhas:
        return
    amostras = sorted({l["amostra"] for l in linhas},
                      key=lambda a: float(next(x["duracao_audio_s"] for x in linhas if x["amostra"] == a)))
    modelos = sorted({l["modelo"] for l in linhas}, key=lambda m: ["tiny", "small", "medium"].index(m) if m in ("tiny", "small", "medium") else 99)
    rotulos = [f"{a.replace('teste_','').replace('.wav','')}\n{float(next(x['duracao_audio_s'] for x in linhas if x['amostra']==a)):.0f}s"
               for a in amostras]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    largura = 0.8 / max(1, len(modelos))
    for i, modelo in enumerate(modelos):
        valores, p95s = [], []
        for a in amostras:
            linha = next((x for x in linhas if x["amostra"] == a and x["modelo"] == modelo), None)
            valores.append(float(linha["mediana_ms"]) if linha else 0)
            p95s.append(float(linha["p95_ms"]) if linha else 0)
        pos = [j + i * largura for j in range(len(amostras))]
        barras = ax.bar(pos, valores, largura, label=f"Whisper {modelo}", color=CORES.get(modelo, "#888"))
        ax.errorbar(pos, valores, yerr=[[0]*len(pos), [p - v for p, v in zip(p95s, valores)]],
                    fmt="none", ecolor="#374151", elinewidth=1, capsize=3)
        for b, v in zip(barras, valores):
            ax.text(b.get_x() + b.get_width()/2, v + 60, f"{v:.0f}", ha="center", fontsize=8)

    ax.set_xticks([j + largura * (len(modelos) - 1) / 2 for j in range(len(amostras))])
    ax.set_xticklabels(rotulos)
    ax.set_ylabel("latência mediana (ms)")
    ax.set_xlabel("amostra de áudio (duração do arquivo)")
    ax.set_title("Latência de transcrição por modelo e duração de áudio")
    ax.legend(frameon=False)
    ax.text(0.99, 0.97, "barras de erro = p95", transform=ax.transAxes, ha="right",
            va="top", fontsize=8, color="#6b7280")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    rodape(fig)
    fig.savefig(os.path.join(SAIDA, "stt_latencia.png"))
    plt.close(fig)


def grafico_stt_rtf(linhas):
    if not linhas:
        return
    # ordem cronológica pelo tamanho do áudio (não alfabética!)
    amostras = sorted({l["amostra"] for l in linhas},
                      key=lambda a: float(next(x["duracao_audio_s"] for x in linhas if x["amostra"] == a)))
    modelos = sorted({l["modelo"] for l in linhas}, key=lambda m: ["tiny", "small", "medium"].index(m) if m in ("tiny", "small", "medium") else 99)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    largura = 0.8 / max(1, len(modelos))
    valores_todos = []
    for i, modelo in enumerate(modelos):
        valores = []
        for a in amostras:
            linha = next((x for x in linhas if x["amostra"] == a and x["modelo"] == modelo), None)
            valores.append(float(linha.get("rtf_fala") or linha.get("rtf") or 0) if linha else 0)
        valores_todos += [v for v in valores if v > 0]
        pos = [j + i * largura for j in range(len(amostras))]
        barras = ax.bar(pos, valores, largura, label=f"Whisper {modelo}", color=CORES.get(modelo, "#888"))
        for b, v in zip(barras, valores):
            ax.text(b.get_x() + b.get_width()/2, v * 1.15, f"{v:.4f}",
                    ha="center", fontsize=8, color=TINTA)
    ax.set_yscale("log")
    topo = max(valores_todos) * 3.5 if valores_todos else 1.0
    ax.set_ylim(min(valores_todos) * 0.4 if valores_todos else 0.001, max(topo, 1.5))
    ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.2)
    ax.text(0.5, 1.0, " tempo real (RTF = 1,0) — acima disso a transcrição não acompanha a fala",
            color="#dc2626", fontsize=8, ha="left", va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xticks([j + largura * (len(modelos) - 1) / 2 for j in range(len(amostras))])
    ax.set_xticklabels([a.replace("teste_", "").replace(".wav", "") for a in amostras])
    ax.set_ylabel("RTF (tempo de transcrição ÷ fala parlada)")
    ax.set_title("Fator de tempo real — quanto menor, melhor   (eixo em escala logarítmica)", pad=26)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    rodape(fig)
    fig.savefig(os.path.join(SAIDA, "stt_rtf.png"))
    plt.close(fig)


def grafico_carga_modelos(stt, carga_tts):
    dados = [("Whisper tiny", next((float(x["carga_modelo_s"]) for x in stt if x["modelo"] == "tiny"), None), "#94a3b8"),
             ("Whisper small", next((float(x["carga_modelo_s"]) for x in stt if x["modelo"] == "small"), None), "#2563eb"),
             ("Whisper medium", next((float(x["carga_modelo_s"]) for x in stt if x["modelo"] == "medium"), None), "#7c3aed")]
    if carga_tts:
        c = {l["etapa"]: float(l["segundos"]) for l in carga_tts if l["segundos"]}
        dados.append(("XTTSv2 (worker completo)", c.get("pronto_para_requisicoes"), "#dc2626"))
        dados.append(("XTTSv2 (+ aquecimento)", (c.get("pronto_para_requisicoes") or 0) + (c.get("aquecimento") or 0), "#f59e0b"))
    dados = [d for d in dados if d[1] is not None]
    if not dados:
        return

    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    nomes = [d[0] for d in dados]
    valores = [d[1] for d in dados]
    barras = ax.barh(nomes[::-1], valores[::-1], color=[d[2] for d in dados][::-1], height=0.6)
    ax.set_xscale("log")
    ax.set_xlim(0.5, max(valores) * 6)   # espaço pro rótulo do maior valor
    for b, v in zip(barras, valores[::-1]):
        ax.text(v * 1.12, b.get_y() + b.get_height()/2, f"{v:.1f} s", va="center", fontsize=9)
    ax.set_xlabel("tempo até estar operacional (escala logarítmica)")
    ax.set_title("Por que o worker é um processo persistente", pad=30)
    ax.text(0.5, 1.015,
            "recarregar o modelo a cada frase custaria segundos por resposta;  "
            "carregando uma vez, cada resposta paga só a inferência",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color="#6b7280")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    rodape(fig)
    fig.savefig(os.path.join(SAIDA, "carga_modelos.png"))
    plt.close(fig)


def grafico_tts(linhas):
    if not linhas:
        return
    textos = ["curto", "medio", "longo"]
    textos = [t for t in textos if any(l["texto"] == t for l in linhas)]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    largura = 0.35
    for i, modo in enumerate(["stream", "arquivo"]):
        valores, chars = [], []
        for t in textos:
            linha = next((l for l in linhas if l["texto"] == t and l["modo"] == modo), None)
            if modo == "stream" and linha:
                valores.append(float(linha["ttfa_ms"]))
            else:
                valores.append(float(linha["mediana_ms"]) if linha else 0)
            chars.append(int(next((l["caracteres"] for l in linhas if l["texto"] == t), 0)))
        pos = [j + i * largura for j in range(len(textos))]
        rotulo = "STREAM — primeiro áudio (TTFA)" if modo == "stream" else "ARQUIVO — áudio completo"
        barras = ax.bar(pos, valores, largura, label=rotulo, color=CORES[modo])
        for b, v in zip(barras, valores):
            ax.text(b.get_x() + b.get_width()/2, v + 40, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks([j + largura/2 for j in range(len(textos))])
    ax.set_xticklabels([f"{t}\n{int(next((l['caracteres'] for l in linhas if l['texto']==t),0))} caracteres"
                        for t in textos])
    ax.set_ylabel("latência mediana (ms)")
    ax.set_xlabel("tamanho do texto gerado")
    ax.set_title("XTTSv2 — latência até o primeiro áudio vs. áudio completo")
    ax.legend(frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    rodape(fig)
    fig.savefig(os.path.join(SAIDA, "tts_latencia.png"))
    plt.close(fig)


def grafico_pipeline(stt, tts, carga_tts):
    """Waterfall: para onde vão os segundos, do fim da fala até o primeiro som."""
    stt_curto = next((float(l["mediana_ms"]) for l in stt if l["amostra"] == "teste_curto.wav" and l["modelo"] == "small"), None)
    if stt_curto is None:
        stt_curto = next((float(l["mediana_ms"]) for l in stt if l["amostra"] == "teste_curto.wav"), 0)
    ttfa = next((float(l["ttfa_ms"]) for l in tts if l["modo"] == "stream" and l["texto"] == "medio"), None)
    if ttfa is None:
        ttfa = next((float(l["ttfa_ms"] or 0) for l in tts if l["modo"] == "stream"), 0)

    etapas = [
        ("T1 · silêncio de fim de fala\n( pause_threshold = 0,7 s )", 700, CORES["fixo"], "medido na configuração"),
        ("T2 · transcrição (STT)\nWhisper small", stt_curto, CORES["medido"], "medido"),
        ("T3 · LLM via API\nGroq", None, CORES["nao_medido"], "não medido aqui"),
        ("T4 · primeiro áudio (TTS)\nXTTSv2 streaming", ttfa, CORES["medido"], "medido"),
    ]

    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    y = 0
    esquerda = 0
    for nome, valor, cor, obs in etapas:
        if valor is None:
            ax.barh(y, 650, left=esquerda, color=cor, height=0.55, hatch="//",
                    edgecolor="#9ca3af", linewidth=0.8)
            ax.text(esquerda + 325, y, "não medido", ha="center", va="center",
                    fontsize=8, color="#4b5563", style="italic")
            esquerda += 650
        else:
            ax.barh(y, valor, left=esquerda, color=cor, height=0.55)
            ax.text(esquerda + valor / 2, y, f"{valor:.0f} ms", ha="center", va="center",
                    fontsize=9, color="white", weight="bold")
            esquerda += valor
        y += 1
    ax.set_yticks(range(len(etapas)))
    ax.set_yticklabels([e[0] for e in etapas], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(esquerda * 1.08, 2600))
    ax.set_xlabel(
        f"latência acumulada (ms)   ·   medido aqui: {esquerda:.0f} ms  +  LLM (não medido)\n"
        f"com o LLM em streaming e o TTS por sentença, o primeiro áudio chega bem antes "
        f"deste total",
        fontsize=8.5, labelpad=8)
    ax.set_title("Onde vão os segundos: do fim da fala até a Amelia responder", pad=14)
    ax.legend(handles=[Patch(facecolor=CORES["medido"], label="medido neste benchmark"),
                       Patch(facecolor=CORES["fixo"], label="custo fixo de configuração"),
                       Patch(facecolor=CORES["nao_medido"], hatch="//", edgecolor="#9ca3af",
                             label="não medido")],
              frameon=False, fontsize=8.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    rodape(fig)
    fig.savefig(os.path.join(SAIDA, "pipeline_waterfall.png"))
    plt.close(fig)


def main():
    os.makedirs(SAIDA, exist_ok=True)
    configurar()
    stt = ler_csv("resultados_stt.csv")
    tts = ler_csv("resultados_tts.csv")
    carga = ler_csv("resultados_tts_carga.csv")

    gerados = []
    grafico_stt_latencia(stt); gerados.append("stt_latencia.png")
    grafico_stt_rtf(stt); gerados.append("stt_rtf.png")
    grafico_carga_modelos(stt, carga); gerados.append("carga_modelos.png")
    grafico_tts(tts); gerados.append("tts_latencia.png")
    grafico_pipeline(stt, tts, carga); gerados.append("pipeline_waterfall.png")

    for g in gerados:
        caminho = os.path.join(SAIDA, g)
        if os.path.exists(caminho):
            print(f"  ✅ {g} ({os.path.getsize(caminho)/1024:.0f} KB)")
        else:
            print(f"  ⚠️  {g} não gerado (faltam dados)")


if __name__ == "__main__":
    main()
