#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  🐧 Iniciando A.M.E.L.I.A. — Linux
# ═══════════════════════════════════════════════════════════════
#  Uso: chmod +x iniciar_amelia.sh && ./iniciar_amelia.sh
# ═══════════════════════════════════════════════════════════════

set -e

VERDE='\033[0;32m'
AZUL='\033[0;34m'
AMARELO='\033[1;33m'
VERMELHO='\033[0;31m'
RESET='\033[0m'

echo -e "${AZUL}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${AZUL}║   🐧 A.M.E.L.I.A. — Assistente de RPG     ║${RESET}"
echo -e "${AZUL}╚══════════════════════════════════════════════╝${RESET}"
echo ""

# ─── Verifica .env ─────────────────────────────────────────
cd "$(dirname "$0")"

if [ ! -f ".env" ]; then
    echo -e "${VERMELHO}❌ Arquivo .env não encontrado!${RESET}"
    echo -e "${AMARELO}   Crie o arquivo .env na raiz do projeto com:${RESET}"
    echo "   DISCORD_BOT_TOKEN=seu_token"
    echo "   CAIXA_DE_SOM_TOKEN=seu_token"
    echo "   GROQ_API_KEY=sua_chave"
    echo ""
    echo "   Modelo: copie de exemplo_env.txt"
    exit 1
fi

# ─── Verifica virtualenv ───────────────────────────────────
if [ ! -d "venv_linux" ]; then
    echo -e "${AMARELO}⚠️  Virtualenv 'venv_linux' não encontrada.${RESET}"
    echo -e "${AMARELO}   Criando com python3.11...${RESET}"
    python3.11 -m venv venv_linux
    source venv_linux/bin/activate
    pip install --upgrade pip
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    echo -e "${VERDE}✅ Virtualenv criada e dependências instaladas!${RESET}"
else
    source venv_linux/bin/activate
fi

# ─── Verifica PulseAudio e loopback ─────────────────────────
echo -e "${AZUL}[1/4] Verificando áudio...${RESET}"
if pulseaudio --check 2>/dev/null; then
    echo -e "${VERDE}  ✅ PulseAudio rodando${RESET}"
else
    echo -e "${AMARELO}  ⚠️ PulseAudio não está rodando. Tentando iniciar...${RESET}"
    pulseaudio --start 2>/dev/null || true
fi

# Verifica se o loopback (monitor) está disponível
if pactl list short sources 2>/dev/null | grep -qi monitor; then
    echo -e "${VERDE}  ✅ Loopback de áudio disponível${RESET}"
else
    echo -e "${AMARELO}  ⚠️ Monitor de áudio não encontrado. Carregando loopback...${RESET}"
    pactl load-module module-loopback latency_msec=10 2>/dev/null || true
    # Aguarda loopback carregar
    sleep 1
    if pactl list short sources 2>/dev/null | grep -qi monitor; then
        echo -e "${VERDE}  ✅ Loopback carregado!${RESET}"
    else
        echo -e "${AMARELO}  ⚠️ Loopback indisponível. A A.M.E.L.I.A. funcionará apenas com microfone.${RESET}"
    fi
fi

# ─── Verifica CUDA / GPU ───────────────────────────────────
echo -e "${AZUL}[2/4] Verificando GPU...${RESET}"
if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo -e "${VERDE}  ✅ GPU: $GPU_NAME${RESET}"
else
    echo -e "${AMARELO}  ⚠️ NVIDIA não detectado. Usando CPU (mais lento).${RESET}"
fi

# ─── Verifica Opus ─────────────────────────────────────────
echo -e "${AZUL}[3/4] Verificando codecs de áudio...${RESET}"
if ldconfig -p 2>/dev/null | grep -q libopus; then
    echo -e "${VERDE}  ✅ libopus instalado${RESET}"
else
    echo -e "${AMARELO}  ⚠️ libopus não encontrado. Instale: sudo apt install libopus0${RESET}"
fi

# ─── Inicia o Bot ──────────────────────────────────────────
echo -e "${AZUL}[4/4] Iniciando A.M.E.L.I.A...${RESET}"
echo ""
echo -e "${VERDE}╔══════════════════════════════════════════════╗${RESET}"
echo -e "${VERDE}║   🚀 A.M.E.L.I.A. Inicializando...         ║${RESET}"
echo -e "${VERDE}╚══════════════════════════════════════════════╝${RESET}"
echo ""

cd bot
python projeto_bot.py
