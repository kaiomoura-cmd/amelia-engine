#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║        A.M.E.L.I.A. — Diagnóstico de Sistema            ║
║  Verifica dependências, GPU, RAM e configs antes do boot ║
╚══════════════════════════════════════════════════════════╝

Uso: python diagnostico.py
"""

import sys
import os
import shutil
import importlib
import platform
from pathlib import Path

# ─── Cores ANSI ──────────────────────────────────────────
class Cor:
    VERDE   = "\033[92m"
    AMARELO = "\033[93m"
    VERMELHO= "\033[91m"
    CIANO   = "\033[96m"
    NEGRITO = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"

OK   = f"{Cor.VERDE}✔{Cor.RESET}"
WARN = f"{Cor.AMARELO}⚠{Cor.RESET}"
FAIL = f"{Cor.VERMELHO}✘{Cor.RESET}"

erros = 0
avisos = 0


def titulo(texto: str):
    print(f"\n{Cor.CIANO}{Cor.NEGRITO}── {texto} ──{Cor.RESET}")


def ok(msg: str):
    print(f"  {OK}  {msg}")


def warn(msg: str):
    global avisos
    avisos += 1
    print(f"  {WARN}  {Cor.AMARELO}{msg}{Cor.RESET}")


def fail(msg: str):
    global erros
    erros += 1
    print(f"  {FAIL}  {Cor.VERMELHO}{msg}{Cor.RESET}")


# ─── 1. Informações do Sistema ───────────────────────────
def checar_sistema():
    titulo("Sistema Operacional")
    ok(f"OS: {platform.system()} {platform.release()}")
    ok(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    ok(f"Arquitetura: {platform.machine()}")


# ─── 2. Memória RAM ──────────────────────────────────────
def checar_ram():
    titulo("Memória RAM")
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        disp_gb = mem.available / (1024 ** 3)
        uso_pct = mem.percent

        ok(f"Total: {total_gb:.1f} GB")
        if disp_gb < 2.0:
            fail(f"Disponível: {disp_gb:.1f} GB — CRÍTICO (mínimo recomendado: 4 GB)")
        elif disp_gb < 4.0:
            warn(f"Disponível: {disp_gb:.1f} GB — pouco (recomendado: 4+ GB)")
        else:
            ok(f"Disponível: {disp_gb:.1f} GB")
        ok(f"Uso atual: {uso_pct}%")
    except ImportError:
        warn("psutil não instalado — não foi possível checar RAM")


# ─── 3. GPU / CUDA ───────────────────────────────────────
def checar_gpu():
    titulo("GPU / CUDA")
    try:
        import torch
        if torch.cuda.is_available():
            nome = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            ok(f"CUDA disponível — {nome}")
            ok(f"VRAM: {vram:.1f} GB")
            ok(f"PyTorch: {torch.__version__}")
        else:
            warn("CUDA não disponível — TTS local usará CPU (lento)")
    except ImportError:
        fail("PyTorch não instalado — módulos de voz não funcionarão")


# ─── 4. Dependências Python ──────────────────────────────
def checar_dependencias():
    titulo("Dependências Python")
    deps = {
        "discord":    "py-cord (Bot Discord)",
        "dotenv":     "python-dotenv (.env)",
        "groq":       "Groq SDK (LLM)",
        "TTS":        "Coqui TTS (XTTSv2)",
        "whisper":    "OpenAI Whisper (STT)",
        "numpy":      "NumPy",
        "soundfile":  "SoundFile (áudio)",
        "pydub":      "PyDub (conversão áudio)",
    }
    for modulo, descricao in deps.items():
        try:
            m = importlib.import_module(modulo)
            versao = getattr(m, "__version__", "")
            extra = f" v{versao}" if versao else ""
            ok(f"{descricao}{extra}")
        except ImportError:
            fail(f"{descricao} — módulo '{modulo}' não encontrado")


# ─── 5. Libopus ──────────────────────────────────────────
def checar_opus():
    titulo("Libopus (Codec de Voz Discord)")
    if sys.platform == "win32":
        ok("Windows — opus geralmente empacotado com py-cord")
        return

    opus_encontrado = False
    for lib in ["libopus.so.0", "libopus.so"]:
        caminho = shutil.which(lib) or f"/usr/lib/{lib}"
        if Path(caminho).exists() or Path(f"/usr/lib/x86_64-linux-gnu/{lib}").exists():
            ok(f"{lib} encontrado")
            opus_encontrado = True
            break

    if not opus_encontrado:
        # Tenta via ldconfig
        try:
            import subprocess
            r = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=5)
            if "libopus" in r.stdout:
                ok("libopus encontrado via ldconfig")
                opus_encontrado = True
        except Exception:
            pass

    if not opus_encontrado:
        fail("libopus NÃO encontrado — instale: sudo apt install libopus0")


# ─── 6. Arquivo .env ─────────────────────────────────────
def checar_env():
    titulo("Configuração (.env)")
    env_path = Path(__file__).parent.parent / ".env"

    if not env_path.exists():
        fail(f".env não encontrado em {env_path}")
        return

    ok(f".env encontrado: {env_path}")

    from dotenv import dotenv_values
    config = dotenv_values(env_path)

    chaves_esperadas = [
        "DISCORD_BOT_TOKEN",
        "GROQ_API_KEY",
    ]
    for chave in chaves_esperadas:
        valor = config.get(chave, "")
        if valor:
            mascarado = valor[:4] + "…" + valor[-4:] if len(valor) > 12 else "****"
            ok(f"{chave} = {mascarado}")
        else:
            fail(f"{chave} — não definida ou vazia")


# ─── 7. FFmpeg ────────────────────────────────────────────
def checar_ffmpeg():
    titulo("FFmpeg")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        ok(f"ffmpeg encontrado: {ffmpeg}")
    else:
        fail("ffmpeg NÃO encontrado — instale: sudo apt install ffmpeg")


# ─── Relatório Final ─────────────────────────────────────
def relatorio():
    print(f"\n{'═' * 50}")
    if erros == 0 and avisos == 0:
        print(f"  {Cor.VERDE}{Cor.NEGRITO}★ Todos os sistemas operacionais! A.M.E.L.I.A. pronta. ★{Cor.RESET}")
    elif erros == 0:
        print(f"  {Cor.AMARELO}{Cor.NEGRITO}⚠ {avisos} aviso(s) — funcional, mas verifique os itens acima.{Cor.RESET}")
    else:
        print(f"  {Cor.VERMELHO}{Cor.NEGRITO}✘ {erros} erro(s) e {avisos} aviso(s) — corrija antes de iniciar.{Cor.RESET}")
    print(f"{'═' * 50}\n")


# ─── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{Cor.NEGRITO}╔══════════════════════════════════════════════════╗")
    print(f"║    🔍 A.M.E.L.I.A. — Diagnóstico de Sistema      ║")
    print(f"╚══════════════════════════════════════════════════╝{Cor.RESET}")

    checar_sistema()
    checar_ram()
    checar_gpu()
    checar_dependencias()
    checar_opus()
    checar_ffmpeg()
    checar_env()
    relatorio()

    sys.exit(1 if erros > 0 else 0)
