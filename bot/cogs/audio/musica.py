"""
╔══════════════════════════════════════════════════════════════╗
║   MÓDULO DE MÚSICA — A.M.E.L.I.A. (Cérebro do Caixa de Som)║
║                                                             ║
║  Gerencia o subprocesso Caixa de Som, fila de reprodução,  ║
║  playlists, buscas no YouTube e toda a lógica musical.     ║
║                                                             ║
║  O worker (caixa_de_som_worker.py) é um bot separado que   ║
║  só toca áudio — este cog é o cérebro.                     ║
╚══════════════════════════════════════════════════════════════╝
"""

import discord
from discord.ext import commands
import asyncio
import json
import os
import subprocess
import sys
import threading
import logging
import re
import time

log = logging.getLogger("amelia.musica")

# ─── Verifica yt-dlp ────────────────────────────────────────────
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    log.warning("yt-dlp não instalado. Buscas e extração de áudio desabilitadas.")

# ─── Constantes ──────────────────────────────────────────────────
PLAYLISTS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "playlists.json")
PID_FILE = os.path.join(os.path.dirname(__file__), "..", "..", ".caixa_de_som.pid")
DEFAULT_VOLUME = 50
MAX_QUEUE_SIZE = 100
SEARCH_TIMEOUT = 45  # segundos para usuário escolher resultado


class Musica(commands.Cog):
    """Sistema de música via Caixa de Som (subprocesso separado)."""

    def __init__(self, bot):
        self.bot = bot
        self.worker_process: subprocess.Popen = None
        self.worker_ready = False
        self.worker_lock = threading.Lock()

        # ─── Fila de reprodução ───
        self.queue: list[dict] = []           # Lista de músicas na fila
        self.current_song: dict = None        # Música atual
        self.is_playing = False
        self.volume = DEFAULT_VOLUME

        # ─── Estado do Caixa de Som ───
        self.som_channel_id: int = None
        self.som_guild_id: int = None
        self.som_connected = False

        # ─── Leituras de stdout do worker ───
        self._stdout_thread: threading.Thread = None
        self._shutdown = False

        # ─── Cache de buscas pendentes ───
        self._pending_searches: dict[int, list] = {}  # user_id -> [results]

        # ─── Playlists carregadas ───
        self.playlists: dict[str, list[dict]] = self._carregar_playlists()

        log.info("Módulo de música inicializado.")

    # ═══════════════════════════════════════════════════════════════
    #  GERENCIAMENTO DO WORKER
    # ═══════════════════════════════════════════════════════════════

    def _worker_path(self) -> str:
        """Caminho absoluto para o script do worker."""
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "caixa_de_som_worker.py")
        )

    @staticmethod
    def _matar_worker_por_pid():
        """Tenta matar um worker anterior usando o arquivo PID.
        Isso evita workers órfãos de execuções anteriores da A.M.E.L.I.A.
        """
        if not os.path.exists(PID_FILE):
            return
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            # Verifica se o processo existe
            try:
                if sys.platform == 'win32':
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    handle = kernel32.OpenProcess(0x0400, False, pid)
                    if handle:
                        kernel32.TerminateProcess(handle, 1)
                        kernel32.CloseHandle(handle)
                        log.warning(f"Worker órfão (PID {pid}) encontrado e morto.")
                else:
                    os.kill(pid, 9)
                    log.warning(f"Worker órfão (PID {pid}) encontrado e morto.")
            except (ProcessLookupError, PermissionError):
                pass
            except Exception as e:
                log.warning(f"Erro ao matar worker órfão PID {pid}: {e}")
        except (ValueError, FileNotFoundError):
            pass
        finally:
            try:
                os.remove(PID_FILE)
            except FileNotFoundError:
                pass

    async def _iniciar_worker(self):
        """Inicia o subprocesso do Caixa de Som."""
        if self.worker_process and self.worker_process.poll() is None:
            return  # Já está rodando

        worker_script = self._worker_path()
        if not os.path.exists(worker_script):
            await self._log_error(f"Worker não encontrado: {worker_script}")
            return False

        caixa_token = os.getenv("CAIXA_DE_SOM_TOKEN")
        if not caixa_token:
            await self._log_error("CAIXA_DE_SOM_TOKEN não definido no .env!")
            return False

        # ─── Antes de iniciar, mata qualquer worker órfão de execuções anteriores ───
        self._matar_worker_por_pid()

        log.info("Iniciando Caixa de Som Worker...")

        try:
            # Passa o PID do processo atual (A.M.E.L.I.A.) como 2º argumento
            # para o worker monitorar o pai e se auto-destruir se ele morrer
            parent_pid = os.getpid()

            self.worker_process = subprocess.Popen(
                [sys.executable, "-u", worker_script, caixa_token, str(parent_pid)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,  # Isola em grupo próprio — evita workers órfãos
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
            )

            # ─── Salva o PID do worker no arquivo ───
            try:
                with open(PID_FILE, 'w') as f:
                    f.write(str(self.worker_process.pid))
            except Exception as e:
                log.warning(f"Não foi possível salvar PID file: {e}")

            self.worker_ready = False

            # Inicia thread de leitura do stdout
            self._stdout_thread = threading.Thread(
                target=self._stdout_reader,
                daemon=True
            )
            self._stdout_thread.start()

            # Aguarda o worker ficar pronto
            timeout = 30  # segundos
            inicio = time.time()
            while time.time() - inicio < timeout:
                if self.worker_ready:
                    log.info(f"Caixa de Som pronta em {time.time()-inicio:.1f}s")
                    return True
                if self.worker_process.poll() is not None:
                    log.error("Worker morreu durante inicialização!")
                    stderr = self.worker_process.stderr.read()
                    log.error(f"Stderr do worker: {stderr}")
                    return False
                await asyncio.sleep(0.2)

            log.error("Timeout aguardando worker ficar pronto")
            self._matar_worker()
            return False

        except Exception as e:
            log.error(f"Erro ao iniciar worker: {e}")
            return False

    def _stdout_reader(self):
        """Lê as linhas de stdout do worker em uma thread separada."""
        try:
            for line in self.worker_process.stdout:
                if self._shutdown:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Agenda o processamento no event loop do bot
                    asyncio.run_coroutine_threadsafe(
                        self._processar_status(data), self.bot.loop
                    )
                except json.JSONDecodeError:
                    log.warning(f"JSON inválido do worker: {line[:100]}")
        except Exception as e:
            if not self._shutdown:
                log.error(f"Erro no stdout_reader: {e}")

    async def _processar_status(self, data: dict):
        """Processa mensagens de status vindas do worker."""
        msg_type = data.get("type", "")

        if msg_type == "ready":
            self.worker_ready = True
            log.info(f"Caixa de Som conectada como: {data.get('user', 'desconhecido')}")

        elif msg_type == "joined":
            self.som_connected = True
            log.info(f"Caixa de Som entrou no canal: {data.get('channel')}")

        elif msg_type == "playing":
            self.is_playing = True
            # O worker já começou a tocar
            log.info(f"▶️ Tocando: {data.get('title')}")

        elif msg_type == "paused":
            log.info(f"⏸️ Pausado: {data.get('title')}")

        elif msg_type == "resumed":
            log.info(f"▶️ Continuado: {data.get('title')}")

        elif msg_type == "finished":
            self.is_playing = False
            log.info(f"✅ Música terminou: {data.get('title')}")
            # Toca a próxima da fila automaticamente
            await self._tocar_proxima()

        elif msg_type == "stopped":
            self.is_playing = False

        elif msg_type == "disconnected":
            self.som_connected = False
            self.is_playing = False
            log.info("Caixa de Som desconectada do canal de voz")

        elif msg_type == "volume":
            log.info(f"Volume ajustado para: {data.get('level')}%")

        elif msg_type == "status":
            self.som_connected = data.get("connected", False)
            self.is_playing = data.get("playing", False)

        elif msg_type == "error":
            log.error(f"Erro no worker: {data.get('message')}")

        elif msg_type == "debug":
            log.warning(f"[Worker Debug] {data.get('message')}")

        elif msg_type == "info":
            log.info(f"[Worker] {data.get('message')}")

        elif msg_type == "booting":
            log.info("Worker está iniciando...")

    async def _enviar_comando(self, comando: dict):
        """Envia um comando JSON para o worker via stdin."""
        with self.worker_lock:
            if not self.worker_process or self.worker_process.poll() is not None:
                log.warning("Worker morto. Tentando reiniciar...")
                ok = await self._iniciar_worker()
                if not ok:
                    return False

            try:
                line = json.dumps(comando, ensure_ascii=False)
                self.worker_process.stdin.write(line + "\n")
                self.worker_process.stdin.flush()
                return True
            except Exception as e:
                log.error(f"Erro ao enviar comando: {e}")
                return False

    def _matar_worker(self):
        """Mata o processo do worker e limpa o arquivo PID."""
        with self.worker_lock:
            if self.worker_process:
                try:
                    self.worker_process.stdin.close()
                    self.worker_process.terminate()
                    self.worker_process.wait(timeout=5)
                except Exception:
                    try:
                        self.worker_process.kill()
                    except Exception:
                        pass
                self.worker_process = None
                self.worker_ready = False
                self.som_connected = False
                self.is_playing = False
                log.info("Worker encerrado.")

            # Limpa o arquivo PID
            try:
                if os.path.exists(PID_FILE):
                    os.remove(PID_FILE)
            except Exception as e:
                log.warning(f"Erro ao limpar PID file: {e}")

    async def _log_error(self, msg: str):
        """Loga erro e envia aviso interno."""
        log.error(msg)

    # ═══════════════════════════════════════════════════════════════
    #  FILA DE REPRODUÇÃO
    # ═══════════════════════════════════════════════════════════════

    async def _tocar_proxima(self):
        """Toca a próxima música na fila, se houver."""
        if not self.queue:
            self.current_song = None
            self.is_playing = False
            return

        # Pega a primeira da fila
        musica = self.queue.pop(0)
        self.current_song = musica

        # Envia comando de play para o worker com a URL original do YouTube
        # (não a URL de stream temporária, que expira e requer headers especiais)
        await self._enviar_comando({
            "action": "play",
            "url": musica["webpage_url"],
            "title": musica["title"],
            "requester": musica.get("requester", ""),
        })

    def _adicionar_fila(self, musica: dict):
        """Adiciona música à fila."""
        if len(self.queue) >= MAX_QUEUE_SIZE:
            return False
        self.queue.append(musica)
        return True

    # ═══════════════════════════════════════════════════════════════
    #  YT-DLP — Extração e Busca
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _is_url(texto: str) -> bool:
        """Verifica se o texto parece uma URL."""
        return bool(re.match(
            r'https?://(?:www\.)?(?:youtube\.com|youtu\.be|soundcloud\.com|spotify\.com|music\.youtube\.com)\S+',
            texto.strip()
        ))

    @staticmethod
    def _extrair_info(url: str) -> dict:
        """Extrai apenas metadados de um vídeo (título, duração, thumbnail).
        
        NÃO resolve a URL de áudio — isso é feito pelo worker no momento da reprodução.
        Isso torna a extração muito mais rápida, especialmente para vídeos longos.
        """
        if not YT_DLP_AVAILABLE:
            raise RuntimeError("yt-dlp não está instalado")

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'extract_flat': 'in_playlist',  # Não processa formatos de áudio
            'noplaylist': True,
        }
        
        # Procura por cookies.txt na raiz do projeto para evitar o erro de bot do YouTube
        cookies_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "cookies.txt")
        if os.path.exists(cookies_path):
            ydl_opts['cookiefile'] = cookies_path
        else:
            ydl_opts['cookiesfrombrowser'] = ('firefox',)
            
        # JS Runtime para resolver as chaves do YouTube
        node_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "venv_linux", "bin", "node")
        if os.path.exists(node_path):
            ydl_opts['js_runtimes'] = {'node': {'path': node_path}}

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        return {
            'title': info.get('title', 'Desconhecida'),
            'duration': info.get('duration', 0),
            'thumbnail': info.get('thumbnail', ''),
            'webpage_url': info.get('webpage_url', url),
            'uploader': info.get('uploader', 'Desconhecido'),
        }

    @staticmethod
    def _buscar_youtube(query: str) -> list[dict]:
        """Busca no YouTube e retorna top 3 resultados (bloqueante — usar em executor)."""
        if not YT_DLP_AVAILABLE:
            raise RuntimeError("yt-dlp não está instalado")

        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,   # Só metadados, não extrai áudio
        }

        # Procura por cookies.txt na raiz do projeto
        cookies_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "cookies.txt")
        if os.path.exists(cookies_path):
            ydl_opts['cookiefile'] = cookies_path
        else:
            # Fallback para navegador
            ydl_opts['cookiesfrombrowser'] = ('firefox',)
            
        # JS Runtime para resolver as chaves do YouTube
        node_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "venv_linux", "bin", "node")
        if os.path.exists(node_path):
            ydl_opts['js_runtimes'] = {'node': {'path': node_path}}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(f"ytsearch3:{query}", download=False)

        return [
            {
                'title': entry.get('title', 'Desconhecida'),
                'url': f"https://youtube.com/watch?v={entry['id']}",
                'duration': entry.get('duration', 0),
                'uploader': entry.get('uploader', 'Desconhecido'),
                'id': entry['id'],
            }
            for entry in results['entries']
        ]

    @staticmethod
    def _formatar_duracao(segundos) -> str:
        """Formata segundos em mm:ss ou h:mm:ss."""
        segundos = int(segundos or 0)
        if segundos >= 3600:
            h = segundos // 3600
            m = (segundos % 3600) // 60
            s = segundos % 60
            return f"{h}:{m:02d}:{s:02d}"
        m = segundos // 60
        s = segundos % 60
        return f"{m}:{s:02d}"

    # ═══════════════════════════════════════════════════════════════
    #  PLAYLISTS
    # ═══════════════════════════════════════════════════════════════

    def _carregar_playlists(self) -> dict:
        """Carrega playlists do arquivo JSON."""
        if not os.path.exists(PLAYLISTS_FILE):
            return {}
        try:
            with open(PLAYLISTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Erro ao carregar playlists: {e}")
            return {}

    def _salvar_playlists(self):
        """Salva playlists no arquivo JSON."""
        try:
            with open(PLAYLISTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.playlists, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"Erro ao salvar playlists: {e}")

    # ═══════════════════════════════════════════════════════════════
    #  COMANDOS DO DISCORD
    # ═══════════════════════════════════════════════════════════════

    async def _garantir_worker(self, ctx) -> bool:
        """Garante que o worker está rodando."""
        if not self.worker_process or self.worker_process.poll() is not None:
            await ctx.send("⏳ *Inicializando Caixa de Som...*")
            ok = await self._iniciar_worker()
            if not ok:
                await ctx.send(
                    "❌ **Não consegui iniciar a Caixa de Som.**\n"
                    "Verifique se o token `CAIXA_DE_SOM_TOKEN` está configurado no `.env`."
                )
                return False
        return True

    async def _garantir_conectado(self, ctx) -> bool:
        """Garante que o worker está conectado a um canal de voz.
        
        Se não estiver conectado, CONECTA AUTOMATICAMENTE ao canal
        do usuário — não precisa de !entrar_som separado.
        """
        if not self.som_connected:
            if not self.som_channel_id:
                # ─── Sem canal salvo: conecta no canal do usuário ───
                if not ctx.author.voice:
                    await ctx.send(
                        "❌ **Você precisa estar em um canal de voz** para eu tocar música!"
                    )
                    return False

                canal = ctx.author.voice.channel
                self.som_guild_id = ctx.guild.id
                self.som_channel_id = canal.id

                await self._enviar_comando({
                    "action": "join",
                    "guild_id": ctx.guild.id,
                    "channel_id": canal.id,
                })

                # Aguarda até 15s pela confirmação do worker
                for _ in range(75):
                    if self.som_connected:
                        break
                    await asyncio.sleep(0.2)

                if not self.som_connected:
                    await ctx.send(f"❌ Não consegui conectar a Caixa de Som em `{canal.name}`.")
                    return False

                await ctx.send(f"🔊 **Caixa de Som conectada em `{canal.name}`.**")
            else:
                # ─── Canal salvo: tenta reconectar ───
                await self._enviar_comando({
                    "action": "join",
                    "guild_id": self.som_guild_id,
                    "channel_id": self.som_channel_id,
                })

                # Aguarda até 15s pela confirmação
                for _ in range(75):
                    if self.som_connected:
                        break
                    await asyncio.sleep(0.2)

                if not self.som_connected:
                    await ctx.send("❌ Não consegui reconectar a Caixa de Som ao canal de voz.")
                    return False

        # Pequena pausa pra conexão de voz estabilizar antes de tocar
        await asyncio.sleep(1)
        return True

    # ── !entrar_som ────────────────────────────────────────────────

    @commands.command(name='entrar_som', aliases=['conectar_som'])
    async def entrar_som(self, ctx, *, canal: discord.VoiceChannel = None):
        """Faz a Caixa de Som entrar em um canal de voz.

        Uso: !entrar_som [#canal]
        Se não especificar canal, usa o canal que você está.
        """
        # Garante que o worker está rodando
        if not await self._garantir_worker(ctx):
            return

        # Descobre o canal
        if not canal:
            if not ctx.author.voice:
                await ctx.send("❌ Você precisa estar em um canal de voz ou mencionar um canal!")
                return
            canal = ctx.author.voice.channel

        self.som_guild_id = ctx.guild.id
        self.som_channel_id = canal.id

        await self._enviar_comando({
            "action": "join",
            "guild_id": ctx.guild.id,
            "channel_id": canal.id,
        })

        # Aguarda até 15s pela confirmação real
        for _ in range(75):
            if self.som_connected:
                break
            await asyncio.sleep(0.2)

        if self.som_connected:
            await ctx.send(f"🔊 **Caixa de Som conectada em `{canal.name}`.**")
        else:
            await ctx.send(f"❌ Não consegui conectar a Caixa de Som em `{canal.name}`.")

    # ── !sair_som ──────────────────────────────────────────────────

    @commands.command(name='sair_som', aliases=['desconectar_som'])
    async def sair_som(self, ctx):
        """Desconecta a Caixa de Som do canal de voz."""
        await self._enviar_comando({"action": "disconnect"})
        self.som_connected = False
        self.som_channel_id = None
        self.queue.clear()
        self.current_song = None
        self.is_playing = False
        await ctx.send("🔇 **Caixa de Som desconectada.**")

    # ── !play ──────────────────────────────────────────────────────

    @commands.command(name='play', aliases=['tocar', 'p'])
    async def play(self, ctx, *, query: str = None):
        """Toca uma música do YouTube.

        Uso:
          !play https://youtube.com/watch?v=...   → Toca URL direta
          !play interstellar soundtrack            → Busca no YouTube (top 3)
          !play                                    → Continua fila se pausada
        """
        if not query:
            # Se sem argumento, tenta continuar se estiver pausado
            if self.is_playing:
                await ctx.send("🎵 Já estou tocando! Use `!pausar`, `!pular` ou `!parar`.")
                return
            await self._enviar_comando({"action": "resume"})
            await ctx.send("▶️ **Continuando...**")
            return

        if not await self._garantir_worker(ctx):
            return
        if not await self._garantir_conectado(ctx):
            return

        # Verifica se é URL ou busca
        if self._is_url(query):
            await ctx.send(f"🔍 *Extraindo informações...*")
            try:
                info = await self.bot.loop.run_in_executor(
                    None, self._extrair_info, query.strip()
                )
            except Exception as e:
                await ctx.send(f"❌ Erro ao extrair áudio: {e}")
                return

            musica = {
                "title": info["title"],
                "duration": info["duration"],
                "thumbnail": info.get("thumbnail", ""),
                "webpage_url": info.get("webpage_url", query),
                "requester": str(ctx.author),
            }

            await self._adicionar_e_tocar(ctx, musica)

        else:
            # Busca no YouTube
            await ctx.send(f"🔍 *Buscando \"{query}\" no YouTube...*")
            try:
                resultados = await self.bot.loop.run_in_executor(
                    None, self._buscar_youtube, query
                )
            except Exception as e:
                await ctx.send(f"❌ Erro na busca: {e}")
                return

            if not resultados:
                await ctx.send(f"❌ Nenhum resultado encontrado para \"{query}\".")
                return

            # Mostra resultados
            msg_parts = [f"🎵 **Resultados para \"{query}\":**\n"]
            emojis = ["1️⃣", "2️⃣", "3️⃣"]

            for i, r in enumerate(resultados):
                dur = self._formatar_duracao(r.get("duration", 0))
                msg_parts.append(
                    f"{emojis[i]} **{r['title']}**\n"
                    f"   └ {r['uploader']} • `{dur}`"
                )

            msg_parts.append(f"\n⌨️ Digite o **número** (1, 2, 3) ou **cancelar**.")

            await ctx.send("\n".join(msg_parts))

            # Guarda resultados para o usuário
            self._pending_searches[ctx.author.id] = resultados

            # Aguarda resposta
            def check(m):
                return (m.author.id == ctx.author.id
                        and m.channel.id == ctx.channel.id
                        and m.content.strip() in ("1", "2", "3", "cancelar", "cancel"))

            try:
                resposta = await self.bot.wait_for('message', timeout=SEARCH_TIMEOUT, check=check)
            except asyncio.TimeoutError:
                self._pending_searches.pop(ctx.author.id, None)
                await ctx.send("⏰ Tempo esgotado. Use `!play` novamente.")
                return

            escolha = resposta.content.strip()
            if escolha in ("cancelar", "cancel"):
                self._pending_searches.pop(ctx.author.id, None)
                await ctx.send("❌ Busca cancelada.")
                return

            idx = int(escolha) - 1
            if idx < 0 or idx >= len(resultados):
                self._pending_searches.pop(ctx.author.id, None)
                await ctx.send("❌ Número inválido.")
                return

            selecionado = resultados[idx]
            self._pending_searches.pop(ctx.author.id, None)

            # Extrai URL de áudio do selecionado
            await ctx.send(f"🔍 *Extraindo áudio de \"{selecionado['title']}\"...*")
            try:
                info = await self.bot.loop.run_in_executor(
                    None, self._extrair_info, selecionado["url"]
                )
            except Exception as e:
                await ctx.send(f"❌ Erro ao extrair áudio: {e}")
                return

            musica = {
                "title": info["title"],
                "duration": info["duration"],
                "thumbnail": info.get("thumbnail", ""),
                "webpage_url": info.get("webpage_url", selecionado["url"]),
                "requester": str(ctx.author),
            }

            await self._adicionar_e_tocar(ctx, musica)

    async def _adicionar_e_tocar(self, ctx, musica: dict):
        """Adiciona à fila e toca se nada estiver tocando."""
        if not self._adicionar_fila(musica):
            await ctx.send("❌ **Fila cheia!** (máximo 100 músicas)")
            return

        dur = self._formatar_duracao(musica.get("duration", 0))
        embed = discord.Embed(
            title="🎵 Adicionada à fila",
            description=f"**{musica['title']}**",
            color=discord.Color.blue(),
        )
        embed.add_field(name="Duração", value=dur, inline=True)
        embed.add_field(name="Solicitada por", value=musica["requester"], inline=True)
        if musica.get("thumbnail"):
            embed.set_thumbnail(url=musica["thumbnail"])

        if self.is_playing or (self.current_song and self.current_song.get("webpage_url") == musica["webpage_url"]):
            # Já está tocando, só mostra na fila
            pos = len(self.queue)
            embed.add_field(name="Posição na fila", value=f"#{pos}" if pos > 0 else "Tocando agora", inline=True)
            await ctx.send(embed=embed)
        else:
            # Nada tocando, começa agora
            await ctx.send(embed=embed)
            await self._tocar_proxima()

    # ── !pausar ────────────────────────────────────────────────────

    @commands.command(name='pausar', aliases=['pause'])
    async def pausar(self, ctx):
        """Pausa a música atual."""
        if not await self._garantir_worker(ctx):
            return
        await self._enviar_comando({"action": "pause"})
        await ctx.send("⏸️ **Pausado.** Use `!continuar` para retomar.")

    # ── !continuar ─────────────────────────────────────────────────

    @commands.command(name='continuar', aliases=['resume', 'retomar'])
    async def continuar(self, ctx):
        """Continua a música pausada."""
        if not await self._garantir_worker(ctx):
            return
        await self._enviar_comando({"action": "resume"})
        await ctx.send("▶️ **Continuando...**")

    # ── !parar ─────────────────────────────────────────────────────

    @commands.command(name='parar', aliases=['stop', 'limpar'])
    async def parar(self, ctx):
        """Para a música e limpa a fila."""
        if not await self._garantir_worker(ctx):
            return
        await self._enviar_comando({"action": "stop"})
        self.queue.clear()
        self.current_song = None
        self.is_playing = False
        await ctx.send("⏹️ **Parado.** Fila limpa.")

    # ── !pular ─────────────────────────────────────────────────────

    @commands.command(name='pular', aliases=['skip', 'next', 'proxima'])
    async def pular(self, ctx):
        """Pula para a próxima música da fila."""
        if not await self._garantir_worker(ctx):
            return

        if not self.is_playing and not self.queue:
            await ctx.send("❌ Nada está tocando e a fila está vazia.")
            return

        # Para a atual — o callback "finished" vai disparar _tocar_proxima automaticamente
        await self._enviar_comando({"action": "stop"})

        if self.queue:
            await ctx.send("⏭️ **Pulando para a próxima...**")
        else:
            await ctx.send("⏭️ **Fila vazia.** Reprodução encerrada.")

    # ── !volume ────────────────────────────────────────────────────

    @commands.command(name='volume', aliases=['vol']) 
    async def volume(self, ctx, nivel: int = None):
        """Ajusta o volume da Caixa de Som (0-100).

        Uso: !volume 70
        """
        if not await self._garantir_worker(ctx):
            return

        if nivel is None:
            await ctx.send(f"🔊 **Volume atual:** `{self.volume}%`")
            return

        if nivel < 0 or nivel > 100:
            await ctx.send("❌ O volume deve estar entre **0** e **100**.")
            return

        self.volume = nivel
        await self._enviar_comando({"action": "volume", "level": nivel})
        await ctx.send(f"🔊 **Volume:** `{nivel}%`")

    # ── !fila ──────────────────────────────────────────────────────

    @commands.command(name='fila', aliases=['queue', 'lista'])
    async def fila(self, ctx):
        """Mostra a fila de reprodução atual."""
        if not self.current_song and not self.queue:
            await ctx.send("📭 **Fila vazia.** Use `!play` para adicionar músicas.")
            return

        embed = discord.Embed(
            title="🎵 Fila de Reprodução",
            color=discord.Color.blue(),
        )

        # Música atual
        if self.current_song:
            status = "▶️ Tocando agora" if self.is_playing else "⏸️ Pausado"
            dur = self._formatar_duracao(self.current_song.get("duration", 0))
            embed.add_field(
                name=f"{status}",
                value=f"**{self.current_song['title']}** (`{dur}`)",
                inline=False,
            )

        # Fila
        if self.queue:
            fila_texto = []
            for i, musica in enumerate(self.queue[:10], 1):  # Mostra só 10
                dur = self._formatar_duracao(musica.get("duration", 0))
                fila_texto.append(f"`{i}.` **{musica['title']}** (`{dur}`)")

            if len(self.queue) > 10:
                fila_texto.append(f"\n... e mais {len(self.queue) - 10} música(s)")

            embed.add_field(
                name=f"📋 Próximas ({len(self.queue)})",
                value="\n".join(fila_texto),
                inline=False,
            )
        else:
            embed.add_field(name="📋 Próximas", value="Nenhuma", inline=False)

        embed.set_footer(text=f"Volume: {self.volume}% | Caixa de Som: {'✅' if self.som_connected else '❌'}")

        await ctx.send(embed=embed)

    # ── !tocando ───────────────────────────────────────────────────

    @commands.command(name='tocando', aliases=['nowplaying', 'np'])
    async def tocando(self, ctx):
        """Mostra o que está tocando agora."""
        if not self.current_song:
            await ctx.send("📭 Nada está tocando no momento.")
            return

        musica = self.current_song
        dur = self._formatar_duracao(musica.get("duration", 0))

        embed = discord.Embed(
            title="🎵 Tocando Agora",
            description=f"**{musica['title']}**",
            color=discord.Color.green() if self.is_playing else discord.Color.gold(),
        )
        embed.add_field(name="Duração", value=dur, inline=True)
        embed.add_field(name="Solicitada por", value=musica.get("requester", "Desconhecido"), inline=True)
        embed.add_field(name="Status", value="▶️ Tocando" if self.is_playing else "⏸️ Pausado", inline=True)
        if musica.get("thumbnail"):
            embed.set_thumbnail(url=musica["thumbnail"])
        if musica.get("webpage_url"):
            embed.url = musica["webpage_url"]

        await ctx.send(embed=embed)

    # ── !playlist (gerenciamento) ─────────────────────────────────

    @commands.command(name='playlist', aliases=['pl'])
    async def playlist(self, ctx, *, args: str = None):
        """Gerencia playlists salvas.

        Uso:
          !playlist                         → Lista playlists disponíveis
          !playlist carregar <nome>         → Carrega uma playlist na fila
          !playlist salvar <nome>           → Salva a fila atual como playlist
          !playlist ver <nome>              → Mostra músicas de uma playlist
          !playlist deletar <nome>          → Deleta uma playlist
        """
        if not args:
            # Lista playlists disponíveis
            if not self.playlists:
                await ctx.send("📂 **Nenhuma playlist salva.** Use `!playlist salvar <nome>` para criar uma.")
                return

            embed = discord.Embed(
                title="📂 Playlists Disponíveis",
                color=discord.Color.purple(),
            )
            for nome, musicas in self.playlists.items():
                total = len(musicas)
                dur_total = sum(m.get("duration", 0) for m in musicas)
                embed.add_field(
                    name=nome,
                    value=f"{total} música(s) • {self._formatar_duracao(dur_total)}",
                    inline=False,
                )
            await ctx.send(embed=embed)
            return

        partes = args.strip().split(maxsplit=1)
        subcomando = partes[0].lower()
        nome = partes[1] if len(partes) > 1 else None

        if subcomando == "salvar" and nome:
            if not self.queue and not self.current_song:
                await ctx.send("❌ A fila está vazia. Adicione músicas primeiro!")
                return

            musicas_para_salvar = []
            if self.current_song:
                musicas_para_salvar.append({
                    "title": self.current_song["title"],
                    "webpage_url": self.current_song["webpage_url"],
                    "duration": self.current_song.get("duration", 0),
                })
            for m in self.queue:
                musicas_para_salvar.append({
                    "title": m["title"],
                    "webpage_url": m["webpage_url"],
                    "duration": m.get("duration", 0),
                })

            self.playlists[nome] = musicas_para_salvar
            self._salvar_playlists()
            await ctx.send(f"💾 **Playlist \"{nome}\" salva!** ({len(musicas_para_salvar)} música(s))")

        elif subcomando == "carregar" and nome:
            if nome not in self.playlists:
                await ctx.send(f"❌ Playlist \"{nome}\" não encontrada.")
                return

            if not await self._garantir_worker(ctx):
                return
            if not await self._garantir_conectado(ctx):
                return

            musicas = self.playlists[nome]
            qtd = 0
            for m in musicas:
                try:
                    info = await self.bot.loop.run_in_executor(
                        None, self._extrair_info, m["webpage_url"]
                    )
                    musica = {
                        "url": info["url"],
                        "title": info["title"],
                        "duration": info.get("duration", m.get("duration", 0)),
                        "thumbnail": info.get("thumbnail", ""),
                        "webpage_url": m["webpage_url"],
                        "requester": str(ctx.author),
                    }
                    if self._adicionar_fila(musica):
                        qtd += 1
                except Exception as e:
                    log.warning(f"Erro ao carregar '{m.get('title')}': {e}")

            if qtd > 0 and not self.is_playing:
                await self._tocar_proxima()

            await ctx.send(f"📂 **Playlist \"{nome}\" carregada!** ({qtd} música(s) adicionadas)")

        elif subcomando == "ver" and nome:
            if nome not in self.playlists:
                await ctx.send(f"❌ Playlist \"{nome}\" não encontrada.")
                return

            musicas = self.playlists[nome]
            embed = discord.Embed(
                title=f"📂 Playlist: {nome}",
                description=f"{len(musicas)} música(s)",
                color=discord.Color.purple(),
            )
            linhas = []
            for i, m in enumerate(musicas[:10], 1):
                dur = self._formatar_duracao(m.get("duration", 0))
                linhas.append(f"`{i}.` **{m['title']}** (`{dur}`)")
            if len(musicas) > 10:
                linhas.append(f"\n... e mais {len(musicas) - 10}")
            embed.description = "\n".join(linhas) if linhas else "Vazia"
            await ctx.send(embed=embed)

        elif subcomando == "deletar" and nome:
            if nome not in self.playlists:
                await ctx.send(f"❌ Playlist \"{nome}\" não encontrada.")
                return
            del self.playlists[nome]
            self._salvar_playlists()
            await ctx.send(f"🗑️ **Playlist \"{nome}\" deletada.**")

        else:
            await ctx.send(
                "❓ **Uso:**\n"
                "`!playlist` — Listar\n"
                "`!playlist salvar <nome>` — Salvar fila atual\n"
                "`!playlist carregar <nome>` — Carregar\n"
                "`!playlist ver <nome>` — Ver músicas\n"
                "`!playlist deletar <nome>` — Deletar"
            )

    # ── !status_som ───────────────────────────────────────────────

    @commands.command(name='status_som', aliases=['status_musica'])
    async def status_som(self, ctx):
        """Mostra o status completo do sistema de som."""
        embed = discord.Embed(
            title="🔊 Status da Caixa de Som",
            color=discord.Color.blue(),
        )

        worker_vivo = self.worker_process and self.worker_process.poll() is None
        embed.add_field(name="Worker", value="✅ Ativo" if worker_vivo else "❌ Inativo", inline=True)
        embed.add_field(name="Conectada", value="✅ Sim" if self.som_connected else "❌ Não", inline=True)
        embed.add_field(name="Volume", value=f"`{self.volume}%`", inline=True)

        if self.current_song:
            embed.add_field(
                name="▶️ Tocando",
                value=f"**{self.current_song['title']}**",
                inline=False,
            )

        embed.add_field(name="📋 Na fila", value=f"`{len(self.queue)}` música(s)", inline=True)
        embed.add_field(name="📂 Playlists", value=f"`{len(self.playlists)}` salvas", inline=True)

        await ctx.send(embed=embed)


    # ── !som ajuda ──────────────────────────────────────────────

    @commands.command(name='som', aliases=['musica', 'music', 'caixa'])
    async def som(self, ctx, *, subcomando: str = None):
        """Comando auxiliar de música. Use `!som ajuda` para ver o tutorial."""
        if subcomando and subcomando.strip().lower() in ('ajuda', 'help', 'tutorial', '?', 'manual'):
            await self._enviar_ajuda(ctx)
        else:
            await self._enviar_ajuda(ctx)

    async def _enviar_ajuda(self, ctx):
        embed = discord.Embed(
            title="🔊 Manual da Caixa de Som — A.M.E.L.I.A.",
            description="Sistema de música ambiente com Caixa de Som separada",
            color=discord.Color.purple()
        )

        embed.add_field(
            name="🎵 Tocando Música",
            value=(
                "`!play interstellar soundtrack` — Busca no YouTube (mostra 3 opções)\n"
                "`!play https://youtube.com/...` — Toca URL direta\n"
                "`!play` — Continua se estiver pausado\n"
                "*Se o S.O.M. não estiver na call, entra automaticamente no seu canal!*"
            ),
            inline=False
        )
        embed.add_field(
            name="⏯️ Controles",
            value=(
                "`!pausar` — Pausa a música\n"
                "`!continuar` — Retoma a música\n"
                "`!pular` — Pula para a próxima\n"
                "`!parar` — Para e limpa a fila"
            ),
            inline=False
        )
        embed.add_field(
            name="🔊 Conexão & Volume",
            value=(
                "`!entrar_som [#canal]` — S.O.M. entra no canal\n"
                "`!sair_som` — Desconecta\n"
                "`!volume 70` — Ajusta volume (0-100)\n"
                "`!status_som` — Status completo do sistema"
            ),
            inline=False
        )
        embed.add_field(
            name="📋 Fila & Informações",
            value=(
                "`!fila` — Mostra a fila de reprodução\n"
                "`!tocando` — O que está tocando agora"
            ),
            inline=False
        )
        embed.add_field(
            name="📂 Playlists",
            value=(
                "`!playlist` — Lista playlists salvas\n"
                "`!playlist salvar missao3` — Salva fila atual como playlist\n"
                "`!playlist carregar missao3` — Carrega uma playlist na fila\n"
                "`!playlist ver missao3` — Ver músicas de uma playlist\n"
                "`!playlist deletar missao3` — Deleta uma playlist"
            ),
            inline=False
        )
        embed.add_field(
            name="💡 Dicas",
            value=(
                "• O S.O.M. é um **segundo bot** que só toca áudio\n"
                "• A A.M.E.L.I.A. e o S.O.M. ficam em **canais separados**\n"
                "• A música **nunca atrapalha** a fala da A.M.E.L.I.A.\n"
                "• Use `!status_som` pra ver se o S.O.M. está online"
            ),
            inline=False
        )

        await ctx.send(embed=embed)


    # ═══════════════════════════════════════════════════════════════
    #  LIMPEZA
    # ═══════════════════════════════════════════════════════════════

    def cog_unload(self):
        """Limpeza quando o cog é descarregado."""
        self._shutdown = True
        self._matar_worker()


async def setup(bot):
    await bot.add_cog(Musica(bot))
