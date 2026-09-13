import discord
from discord.ext import commands
import os
import json

# Caminho absoluto baseado na localização deste arquivo
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(_BASE, 'discord_logs')

class Admin(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name='mapear_servidor')
    async def mapear_servidor(self, ctx):
        """Mapeia categorias, canais e permissões dos jogadores."""
        await ctx.send("Iniciando varredura da estrutura do servidor...")
        
        estrutura = {}
        for cat in ctx.guild.categories:
            estrutura[cat.name] = []
            for ch in cat.channels:
                # Pega as permissões do @everyone para descobrir se o canal é público ou restrito ao Mestre
                everyone_perms = ch.permissions_for(ctx.guild.default_role)
                visivel_para_todos = everyone_perms.read_messages
                
                tipo = "Texto" if isinstance(ch, discord.TextChannel) else "Voz" if isinstance(ch, discord.VoiceChannel) else "Outro"
                
                estrutura[cat.name].append({
                    "nome": ch.name,
                    "tipo": tipo,
                    "acesso_jogadores": visivel_para_todos
                })
                
        # Salva o arquivo json
        file_path = os.path.join(LOG_DIR, "estrutura_servidor.json")
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(estrutura, f, indent=4, ensure_ascii=False)
            
        await ctx.send(f"✅ Mapeamento concluído! Estrutura e permissões salvas em `{file_path}` para a IA (A.M.E.L.I.A) processar.")

async def setup(bot):
    await bot.add_cog(Admin(bot))



