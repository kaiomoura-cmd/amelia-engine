"""
╔══════════════════════════════════════════════════════════════╗
║           CAIXA DE SOM WORKER — Subprocesso Leve           ║
║                                                             ║
║  Um bot Discord separado que funciona como "alto-falante"   ║
║  para a A.M.E.L.I.A.                                        ║
║                                                             ║
║  ─── Comunicação via JSON sobre stdin/stdout ───            ║
║                                                             ║
║  Recebe comandos pelo stdin (vindos da A.M.E.L.I.A.):       ║
║    {"action":"join",      "guild_id":..., "channel_id":...} ║
║    {"action":"play",      "url":"...", "title":"..."}       ║
║    {"action":"pause"}                                       ║
║    {"action":"resume"}                                      ║
║    {"action":"stop"}                                        ║
║    {"action":"volume",    "level":50}                        ║
║    {"action":"disconnect"}                                  ║
║    {"action":"status"}                                      ║
║                                                             ║
║  Envia status pelo stdout (para A.M.E.L.I.A.):              ║
║    {"type":"ready"}                                         ║
║    {"type":"playing",     "title":"..."}                    ║
║    {"type":"paused"}                                        ║
║    {"type":"finished",    "title":"..."}                    ║
║    {"type":"disconnected"}                                  ║
║    {"type":"error",       "message":"..."}                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import asyncio
import discord
from discord.ext import commands
from discord import PCMVolumeTransformer, FFmpegPCMAudio
import json
import sys
import os
import threading
import signal

# Para detectar se o processo pai morreu (Windows)
if sys.platform == 'win32':
    import ctypes

# ─── Configuração ───────────────────────────────────────────────
TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.getenv("CAIXA_DE_SOM_TOKEN")

# Se recebeu PID do pai como 2º argumento, guarda para watchdog
_PARENT_PID = int(sys.argv[2]) if len(sys.argv) > 2 else None
if not TOKEN:
    print(json.dumps({"type": "error", "message": "Nenhum token fornecido!"}))
    sys.exit(1)

# Política de event loop para Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ─── Intents mínimos ────────────────────────────────────────────
intents = discord.Intents.default()
intents.guilds = True
intents.voice_states = True

bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

# ─── Estado interno ─────────────────────────────────────────────
_current_volume = 0.5        # Volume padrão 50%
_voice_client = None
_current_title = ""
_current_url = ""
_guild_id = None
_channel_id = None
_shutdown = False

# Caminho do ffmpeg — prioriza o ffmpeg do sistema (funciona com HTTPS streaming)
# O ffmpeg estático do imageio_ffmpeg não suporta streaming HTTPS corretamente no Linux
import shutil
_system_ffmpeg = shutil.which("ffmpeg")
if _system_ffmpeg:
    FFMPEG_PATH = _system_ffmpeg
else:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        FFMPEG_PATH = get_ffmpeg_exe()
    except Exception:
        FFMPEG_PATH = "ffmpeg"


# ─── Utilitários de comunicação ────────────────────────────────
def _send(data: dict):
    """Envia JSON para a A.M.E.L.I.A. via stdout."""
    try:
        line = json.dumps(data, ensure_ascii=False)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except Exception:
        pass


# ─── Eventos do Bot ────────────────────────────────────────────
@bot.event
async def on_ready():
    _send({"type": "ready", "user": str(bot.user)})


@bot.event
async def on_voice_state_update(member, before, after):
    """Se formos desconectados manualmente, avisa A.M.E.L.I.A."""
    if member.id == bot.user.id:
        if before.channel and not after.channel:
            global _voice_client, _current_title
            _voice_client = None
            _current_title = ""
            _send({"type": "disconnected", "channel": before.channel.name if before.channel else None})


# ─── Processamento de Comandos ─────────────────────────────────
async def _cmd_join(guild_id: int, channel_id: int):
    """Conecta ao canal de voz especificado."""
    global _voice_client, _guild_id, _channel_id

    guild = bot.get_guild(guild_id)
    if not guild:
        _send({"type": "error", "message": f"Servidor {guild_id} não encontrado"})
        return

    channel = guild.get_channel(channel_id)
    if not channel:
        _send({"type": "error", "message": f"Canal {channel_id} não encontrado"})
        return

    try:
        if _voice_client and _voice_client.is_connected():
            await _voice_client.move_to(channel)
        else:
            _voice_client = await channel.connect(timeout=20.0, reconnect=True)

        _guild_id = guild_id
        _channel_id = channel_id
        _send({"type": "joined", "channel": channel.name})
    except Exception as e:
        _send({"type": "error", "message": f"Erro ao conectar: {type(e).__name__}: {e}"})


def _resolve_audio_url(url: str) -> tuple:
    """Resolve a URL de áudio usando yt-dlp (bloqueante — roda em thread).
    Retorna (audio_url, http_headers_dict) ou levanta exceção.
    """
    try:
        import yt_dlp
    except ImportError:
        # Se yt-dlp não estiver disponível, retorna a URL direta sem headers
        return url, {}

    # Caminho do cookies.txt na raiz do projeto
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cookies_path = os.path.join(project_root, "cookies.txt")
    node_path = os.path.join(project_root, "venv_linux", "bin", "node")

    ydl_opts = {
        'format': 'ba/b',
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
    }

    if os.path.exists(cookies_path):
        ydl_opts['cookiefile'] = cookies_path

    if os.path.exists(node_path):
        ydl_opts['js_runtimes'] = {'node': {'path': node_path}}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    audio_url = info.get('url', url)
    http_headers = info.get('http_headers', {})
    return audio_url, http_headers


async def _cmd_play(url: str, title: str = "", requester: str = ""):
    """Toca um áudio a partir de uma URL (streaming, não baixa arquivo)."""
    global _voice_client, _current_title, _current_url

    vc = _voice_client or discord.utils.get(bot.voice_clients)
    if not vc or not vc.is_connected():
        _send({"type": "error", "message": "Não estou conectada a nenhum canal de voz"})
        return

    # Se já estiver tocando, para primeiro
    if vc.is_playing():
        vc.stop()

    try:
        # Resolve a URL de áudio em tempo real usando yt-dlp (para YouTube e afins)
        is_youtube = 'youtube.com' in url or 'youtu.be' in url
        _send({"type": "debug", "message": f"URL recebida: {url[:100]}... | is_youtube={is_youtube}"})

        if is_youtube:
            _send({"type": "info", "message": f"Resolvendo URL de áudio via yt-dlp..."})
            loop = asyncio.get_event_loop()
            audio_url, http_headers = await loop.run_in_executor(None, _resolve_audio_url, url)
            _send({"type": "debug", "message": f"URL resolvida: {audio_url[:100]}... | headers={list(http_headers.keys())}"})
        else:
            audio_url = url
            http_headers = {}

        # Monta os before_options com reconexão + headers
        before_parts = ["-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"]

        if http_headers:
            # ffmpeg espera todos os headers em uma única flag -headers, separados por \r\n
            header_str = "".join(f"{k}: {v}\r\n" for k, v in http_headers.items())
            before_parts.append(f'-headers "{header_str}"')

        before_opts = " ".join(before_parts)

        source = FFmpegPCMAudio(
            audio_url,
            executable=FFMPEG_PATH,
            before_options=before_opts
        )
        source = PCMVolumeTransformer(source, volume=_current_volume)

        def _after_playing(error):
            """Callback quando a música termina."""
            global _current_title
            if error:
                _send({"type": "error", "message": f"Erro na reprodução: {error}"})
            else:
                _send({"type": "debug", "message": "after_playing chamado SEM erro"})
            title_atual = _current_title
            _send({"type": "finished", "title": title_atual})
            _current_title = ""

        vc.play(source, after=_after_playing)
        _current_title = title
        _current_url = url
        _send({"type": "playing", "title": title, "requester": requester})

        # Verifica se o ffmpeg realmente está rodando após 1 segundo
        await asyncio.sleep(1)
        if vc.is_playing():
            _send({"type": "debug", "message": "ffmpeg ainda rodando após 1s ✓"})
        else:
            _send({"type": "debug", "message": "ffmpeg PAROU antes de 1s!"})

    except Exception as e:
        import traceback
        _send({"type": "error", "message": f"Erro ao tocar: {type(e).__name__}: {e}\n{traceback.format_exc()}"})


async def _cmd_pause():
    """Pausa a reprodução."""
    vc = _voice_client or discord.utils.get(bot.voice_clients)
    if vc and vc.is_playing():
        vc.pause()
        _send({"type": "paused", "title": _current_title})
    else:
        _send({"type": "error", "message": "Nada está tocando no momento"})


async def _cmd_resume():
    """Continua a reprodução pausada."""
    vc = _voice_client or discord.utils.get(bot.voice_clients)
    if vc and vc.is_paused():
        vc.resume()
        _send({"type": "resumed", "title": _current_title})
    else:
        _send({"type": "error", "message": "Nada está pausado"})


async def _cmd_stop():
    """Para a reprodução completamente."""
    vc = _voice_client or discord.utils.get(bot.voice_clients)
    if vc and vc.is_playing():
        vc.stop()
    _current_title = ""
    _current_url = ""
    _send({"type": "stopped"})


async def _cmd_volume(level: int):
    """Ajusta o volume (0-100)."""
    global _current_volume
    if level < 0:
        level = 0
    if level > 100:
        level = 100

    _current_volume = level / 100.0

    # Aplica instantaneamente se estiver tocando
    vc = _voice_client or discord.utils.get(bot.voice_clients)
    if vc and vc.is_playing():
        try:
            source = vc.source
            if isinstance(source, PCMVolumeTransformer):
                source.volume = _current_volume
        except Exception:
            pass

    _send({"type": "volume", "level": level})


async def _cmd_disconnect():
    """Desconecta do canal de voz."""
    global _voice_client, _current_title
    vc = _voice_client or discord.utils.get(bot.voice_clients)
    if vc:
        if vc.is_playing():
            vc.stop()
        try:
            await vc.disconnect(force=True)
        except Exception:
            pass
        _voice_client = None
        _current_title = ""
        _send({"type": "disconnected"})
    else:
        _send({"type": "error", "message": "Não estou conectada"})


async def _cmd_status():
    """Envia status completo."""
    vc = _voice_client or discord.utils.get(bot.voice_clients)
    status_data = {
        "type": "status",
        "connected": vc is not None and vc.is_connected(),
        "playing": vc.is_playing() if vc else False,
        "paused": vc.is_paused() if vc else False,
        "channel": vc.channel.name if vc and vc.channel else None,
        "title": _current_title,
        "volume": int(_current_volume * 100),
    }
    _send(status_data)


# ─── Roteador de comandos ──────────────────────────────────────
async def _process_command(cmd: dict):
    """Roteia um comando recebido para o handler apropriado."""
    action = cmd.get("action", "")
    try:
        if action == "join":
            await _cmd_join(cmd["guild_id"], cmd["channel_id"])
        elif action == "play":
            await _cmd_play(cmd.get("url", ""), cmd.get("title", ""), cmd.get("requester", ""))
        elif action == "pause":
            await _cmd_pause()
        elif action == "resume":
            await _cmd_resume()
        elif action == "stop":
            await _cmd_stop()
        elif action == "volume":
            await _cmd_volume(cmd.get("level", 50))
        elif action == "disconnect":
            await _cmd_disconnect()
        elif action == "status":
            await _cmd_status()
        elif action == "shutdown":
            _send({"type": "shutdown"})
            await bot.close()
        else:
            _send({"type": "error", "message": f"Comando desconhecido: {action}"})
    except Exception as e:
        _send({"type": "error", "message": f"Erro: {type(e).__name__}: {e}"})


# ─── Leitor de stdin (thread separada) ─────────────────────────
def _processo_pai_vivo() -> bool:
    """Verifica se o processo pai (A.M.E.L.I.A.) ainda existe."""
    if _PARENT_PID is None:
        return True  # Não temos referência, assume vivo
    try:
        if sys.platform == 'win32':
            # No Windows: OpenProcess com PROCESS_QUERY_INFORMATION
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x0400, False, _PARENT_PID)  # PROCESS_QUERY_INFORMATION
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            # No Unix: signal 0 só verifica existência
            os.kill(_PARENT_PID, 0)
            return True
    except Exception:
        return False


def _watchdog_parent():
    '''Thread watchdog: monitora se o processo pai ainda vive.
    Se o pai morrer, inicia shutdown do worker.
    Verifica a cada 2s (era 10s — dava 10s de worker solto no ar).
    '''
    import time
    # Espera um pouco antes de começar a verificar (dá tempo do pai inicializar)
    time.sleep(5)
    while not _shutdown:
        if not _processo_pai_vivo():
            _send({'type': 'error', 'message': 'Processo pai (A.M.E.L.I.A.) morreu. Encerrando...'})
            # Agenda shutdown no event loop do bot
            asyncio.run_coroutine_threadsafe(
                _cmd_shutdown(), bot.loop
            )
            break
        time.sleep(2)  # Verifica a cada 2 segundos


async def _cmd_shutdown():
    """Desliga o worker completamente."""
    global _shutdown
    _shutdown = True
    _send({"type": "shutdown"})
    try:
        await bot.close()
    except Exception:
        pass
    # Força saída se bot.close() não funcionar
    os._exit(0)


def _stdin_reader():
    """Lê comandos JSON do stdin em uma thread e agenda no event loop do bot.
    
    Se o stdin fechar (pai morreu), inicia shutdown automático.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
                # Agenda no event loop principal do bot
                asyncio.run_coroutine_threadsafe(
                    _process_command(cmd), bot.loop
                )
            except json.JSONDecodeError:
                _send({"type": "error", "message": f"JSON inválido: {line[:100]}"})
    except EOFError:
        # stdin fechou = processo pai (A.M.E.L.I.A.) provavelmente morreu
        _send({"type": "error", "message": "Conexão com A.M.E.L.I.A. perdida. Encerrando..."})
        asyncio.run_coroutine_threadsafe(
            _cmd_shutdown(), bot.loop
        )
    except Exception as e:
        _send({"type": "error", "message": f"Stdin reader: {e}"})


# ─── Inicialização ─────────────────────────────────────────────
def main():
    # Thread para ler comandos do stdin
    reader_thread = threading.Thread(target=_stdin_reader, daemon=True)
    reader_thread.start()

    # Thread watchdog para monitorar processo pai
    if _PARENT_PID is not None:
        watchdog_thread = threading.Thread(target=_watchdog_parent, daemon=True)
        watchdog_thread.start()

    # Avisa que está iniciando
    print(json.dumps({"type": "booting"}), flush=True)

    # Trata SIGTERM/SIGINT para shutdown limpo
    def _signal_handler(sig, frame):
        asyncio.run_coroutine_threadsafe(_cmd_shutdown(), bot.loop)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Inicia o bot
    try:
        bot.run(TOKEN)
    except Exception as e:
        print(json.dumps({"type": "fatal", "message": str(e)}), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
