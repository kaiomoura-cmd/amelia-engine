import discord
from discord.ext import commands
import subprocess
import sys
import os
import glob
from datetime import datetime

# Gravação nativa via discord.sinks removida devido à migração para discord.py (DAVE E2EE)

class Voz(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.active_recordings = {}
        # Guarda referência ao sink e canal para uso no callback
        self._recording_sinks = {}
        self._recording_channels = {}
        
        # Referência ao processo de gravação local
        self.local_recording_process = None
        
        # ─── Caminho absoluto para a pasta de áudio ───
        # Baseado na localização DESTE arquivo (voz.py → Bot_Discord/discord_logs/audio)
        # Isso evita duplicação de pastas quando o CWD muda
        self._audio_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'discord_logs', 'audio'
        )

    @commands.command(name='entrar_sessao', aliases=['entrar_sessão', 'conectar'])
    async def entrar_sessao(self, ctx):
        """Conecta ao canal de voz do usuário."""
        if not ctx.author.voice:
            await ctx.send("❌ Você precisa estar em um canal de voz!")
            return

        canal_voz = ctx.author.voice.channel

        try:
            if ctx.voice_client:
                await ctx.voice_client.move_to(canal_voz)
            else:
                await canal_voz.connect(timeout=20.0, reconnect=True)

            await ctx.send(f"🎙️ **Conectada em `{canal_voz.name}`.**")
        except Exception as e:
            await ctx.send(f"⚠️ Erro de conexão: `{type(e).__name__}`: {e}")
            print(f"[VOZ] Erro ao conectar: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    @commands.command(name='status_voz')
    async def status_voz(self, ctx):
        """Verifica o estado atual da conexão de voz."""
        voice_client = ctx.voice_client or discord.utils.get(self.bot.voice_clients, guild=ctx.guild)

        if not voice_client:
            await ctx.send("❌ Não estou conectada a nenhum canal de voz.")
            return

        status = [
            f"📍 **Canal:** `{voice_client.channel.name}`",
            f"🔗 **Conectado:** `{voice_client.is_connected()}`",
            f"🔴 **Gravando:** `{ctx.guild.id in self.active_recordings}`",
        ]

        try:
            lat = voice_client.latency
            if lat and lat != float('inf'):
                status.append(f"⚡ **Latência:** `{lat * 1000:.2f}ms`")
            else:
                status.append("⚡ **Latência:** `N/A`")
        except Exception:
            status.append("⚡ **Latência:** `N/A`")

        try:
            if hasattr(voice_client, 'is_dave_connection'):
                status.append(f"🛡️ **DAVE (E2EE):** `{voice_client.is_dave_connection()}`")
        except Exception:
            pass

        await ctx.send("\n".join(status))

    def _audio_path(self, *subpath):
        """Retorna caminho absoluto dentro da pasta de áudio."""
        return os.path.join(self._audio_dir, *subpath)

    @commands.command(name='gravar_sessao', aliases=['iniciar_gravacao', 'iniciar_gravação', 'iniciar_gracacao', 'gravarsessao'])
    async def gravar_sessao(self, ctx):
        """Inicia a gravação da sessão (microfone + áudio do PC)."""
        if self.local_recording_process is not None and self.local_recording_process.poll() is None:
            await ctx.send("❌ Já existe uma gravação em andamento. Use `!parar_gravacao` primeiro.")
            return

        try:
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            script_path = os.path.join(script_dir, "gravador_app_local.py")

            # Garante que a pasta de áudio existe
            os.makedirs(self._audio_dir, exist_ok=True)

            # ⚠️ NUNCA redirecionar para PIPE sem leitor: o buffer de 64KB enche,
            # o print() do gravador bloqueia e a gravação congela em ~30 min.
            # Logs vão para arquivo com timestamp — diagnóstico real em caso de bug.
            log_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_out = open(os.path.join(self._audio_dir, f"gravador_{log_stamp}.log"), "a", encoding="utf-8")
            log_err = open(os.path.join(self._audio_dir, f"gravador_{log_stamp}_err.log"), "a", encoding="utf-8")

            self.local_recording_process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=log_out,
                stderr=log_err
            )
            await ctx.send(
                "🔴 **Gravação da Sessão Iniciada!**\n"
                "📌 Capturando **microfone** + **áudio do PC** simultaneamente.\n"
                "📁 Formato: `WAV 48kHz 16-bit` (qualidade máxima para transcrição)\n"
                "⏹️ Para parar: `!parar_gravacao`"
            )
        except Exception as e:
            await ctx.send(f"❌ Erro ao iniciar gravação: `{e}`")

    @commands.command(name='parar_gravacao', aliases=['parar_sessao', 'parar_sessão', 'parar_gravação', 'parargravacao'])
    async def parar_gravacao(self, ctx):
        """Para a gravação da sessão atual."""
        if self.local_recording_process is None or self.local_recording_process.poll() is not None:
            await ctx.send("❌ Não há nenhuma gravação em andamento. Use `!gravar_sessao` para iniciar.")
            return

        proc = self.local_recording_process
        self.local_recording_process = None  # Limpa ref antes de qualquer erro

        try:
            # 1. Escreve a STOP_FLAG para o script parar gracefulmente
            audio_dir = self._audio_dir
            flag_escrita = False
            if os.path.exists(audio_dir):
                pastas = [os.path.join(audio_dir, d) for d in os.listdir(audio_dir) if d.startswith('sessao_')]
                if pastas:
                    pasta_recente = max(pastas, key=os.path.getmtime)
                    flag_file = os.path.join(pasta_recente, "STOP_FLAG")
                    with open(flag_file, 'w') as f:
                        f.write('stop')
                    flag_escrita = True
                    print(f"[VOZ] STOP_FLAG escrita em: {flag_file}")

            if not flag_escrita:
                print(f"[VOZ] Aviso: Nenhuma pasta de sessão encontrada em {audio_dir}")

            # 2. Tenta parada graceful: SIGTERM primeiro, espera 5s
            proc.terminate()
            try:
                proc.wait(timeout=5)
                print("[VOZ] Processo de gravação finalizado via SIGTERM")
            except subprocess.TimeoutExpired:
                # 3. Se não respondeu ao SIGTERM, força SIGKILL
                print("[VOZ] SIGTERM não funcionou, enviando SIGKILL...")
                proc.kill()
                proc.wait(timeout=3)

            # 4. Busca o arquivo gerado
            wav_path = "desconhecido"
            if os.path.exists(audio_dir):
                pastas = [os.path.join(audio_dir, d) for d in os.listdir(audio_dir) if d.startswith('sessao_')]
                if pastas:
                    pasta_recente = max(pastas, key=os.path.getmtime)
                    # Pode ser .ogg ou .wav
                    arquivos = glob.glob(os.path.join(pasta_recente, "*.wav")) + \
                               glob.glob(os.path.join(pasta_recente, "*.ogg"))
                    if arquivos:
                        wav_path = arquivos[0]

            await ctx.send(
                "⏹️ **Gravação Encerrada!**\n"
                f"📁 Arquivo salvo em: `{wav_path}`\n"
                "📌 Pronto para transcrição!"
            )
        except Exception as e:
            await ctx.send(f"❌ Erro ao parar gravação: `{e}`")
            try:
                proc.kill()
            except Exception:
                pass

    @commands.command(name='sair_sessao', aliases=['sair_sessão', 'desconectar'])
    async def sair_sessao(self, ctx):
        """Desconecta do canal de voz."""
        voice_client = ctx.voice_client or discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
        if voice_client:
            # Gravação nativa removida
            self.active_recordings.pop(ctx.guild.id, None)
            self._recording_sinks.pop(ctx.guild.id, None)
            self._recording_channels.pop(ctx.guild.id, None)

            await voice_client.disconnect(force=True)
            await ctx.send("🎙️ Sessão encerrada.")
        else:
            await ctx.send("Não estou em nenhuma chamada.")

    @commands.command(name='gravar_pc')
    async def gravar_pc(self, ctx):
        """Alias para !gravar_sessao."""
        await self.gravar_sessao(ctx)

    @commands.command(name='parar_pc')
    async def parar_pc(self, ctx):
        """Alias para !parar_gravacao."""
        await self.parar_gravacao(ctx)

async def setup(bot):
    await bot.add_cog(Voz(bot))
