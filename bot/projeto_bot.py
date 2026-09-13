import discord
from discord.ext import commands
import os
import asyncio
import sys
from dotenv import load_dotenv
import logging

# Configuração de logs para depuração de rede/voz
logging.basicConfig(level=logging.INFO)
logging.getLogger('discord.voice_client').setLevel(logging.DEBUG)

# No Linux, o event loop padrão funciona sem problemas.
# Apenas no Windows precisávamos do SelectorEventLoopPolicy.
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

# Aceitar termos do Coqui TTS automaticamente para evitar travamento em background
os.environ["COQUI_TOS_AGREED"] = "1"

# ─── Single Instance (PID Lock) ──────────────────────────────────
PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".amelia.pid")


def _single_instance_check():
    """Verifica se já existe outra instância da A.M.E.L.I.A. rodando.
    
    Se encontrar uma instância anterior:
      - Por padrão: mata a anterior e assume o lugar ("modo substituir")
      - Com flag --allow-multiple: permite múltiplas instâncias
      - Com flag --abort-if-running: apenas avisa e sai
    """
    allow_multiple = "--allow-multiple" in sys.argv
    abort_if_running = "--abort-if-running" in sys.argv
    
    if allow_multiple:
        print("[SISTEMA] Modo multi-instância ativado. Ignorando lock.")
        return
    
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            
            # Verifica se o processo antigo ainda existe
            processo_existe = False
            if sys.platform == 'win32':
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x0400, False, old_pid)
                if handle:
                    processo_existe = True
                    kernel32.CloseHandle(handle)
            else:
                try:
                    os.kill(old_pid, 0)
                    processo_existe = True
                except (ProcessLookupError, PermissionError):
                    pass
            
            if processo_existe:
                if abort_if_running:
                    print(f"[SISTEMA] Já existe uma A.M.E.L.I.A. rodando (PID {old_pid}).")
                    print(f"[SISTEMA] Use --allow-multiple se quiser rodar várias, ou feche a outra primeiro.")
                    sys.exit(1)
                else:
                    print(f"[SISTEMA] Outra A.M.E.L.I.A. encontrada (PID {old_pid}). Substituindo...")
                    if sys.platform == 'win32':
                        kernel32 = ctypes.windll.kernel32
                        handle = kernel32.OpenProcess(0x0400, False, old_pid)
                        if handle:
                            kernel32.TerminateProcess(handle, 1)
                            kernel32.CloseHandle(handle)
                            print(f"[SISTEMA] Instância anterior (PID {old_pid}) encerrada.")
                    else:
                        os.kill(old_pid, 9)
                        print(f"[SISTEMA] Instância anterior (PID {old_pid}) encerrada.")
            else:
                print(f"[SISTEMA] PID file órfão (PID {old_pid}) limpo.")
        except (ValueError, FileNotFoundError):
            pass
        except Exception as e:
            print(f"[AVISO] Erro ao verificar PID anterior: {e}")
        finally:
            try:
                os.remove(PID_FILE)
            except FileNotFoundError:
                pass
    
    # Escreve o PID atual
    try:
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
        print(f"[SISTEMA] PID lock: {os.getpid()}")
    except Exception as e:
        print(f"[AVISO] Não foi possível criar PID file: {e}")


def _limpar_pid_file():
    """Remove o arquivo PID na saída."""
    try:
        if os.path.exists(PID_FILE):
            with open(PID_FILE, 'r') as f:
                saved_pid = f.read().strip()
            # Só limpa se o PID for o nosso (evita matar lock de outra instância)
            if saved_pid == str(os.getpid()) or not saved_pid:
                os.remove(PID_FILE)
                print("[SISTEMA] PID lock liberado.")
    except Exception:
        pass

def carregar_opus():
    if discord.opus.is_loaded():
        return

    if sys.platform != 'win32':
        # Linux: tenta carregar libopus do sistema
        tentativas = ['libopus.so.0', 'libopus.so']
        for lib in tentativas:
            try:
                discord.opus.load_opus(lib)
                print(f"[SISTEMA] Opus carregado: {lib}")
                return
            except Exception:
                pass
        
        # Verifica se o libopus está instalado
        try:
            import subprocess
            result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True, timeout=5)
            if 'libopus' in result.stdout:
                print(f"[AVISO] libopus encontrado no sistema mas não foi carregado.")
            else:
                print(f"[AVISO] libopus não encontrado. Instale: sudo apt install libopus0")
        except Exception:
            print("[AVISO] libopus pode não estar instalado. Instale: sudo apt install libopus0")
    else:
        # Windows: caminhos das DLLs
        paths = [
            os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "discord", "bin", "libopus-0.x64.dll"),
            os.path.join(sys.prefix, "Lib", "site-packages", "discord", "bin", "libopus-0.x64.dll"),
        ]
        for path in paths:
            if os.path.exists(path):
                try:
                    discord.opus.load_opus(path)
                    print(f"[SISTEMA] Opus carregado de: {path}")
                    return
                except Exception as e:
                    print(f"[ERRO] Falha ao carregar Opus de {path}: {e}")
                    continue
    print("[AVISO] Opus não foi carregado. Funções de voz podem falhar.")

async def main():
    intents = discord.Intents.all()
    bot = commands.Bot(command_prefix='!', intents=intents)

    @bot.event
    async def on_ready():
        print(f"---")
        print(f"A.M.E.L.I.A. Online | Usuário: {bot.user}")
        print(f"py-cord: {discord.__version__}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"---")

    @bot.event
    async def on_command_error(ctx, error):
        """Handler global de erros de comando."""
        if isinstance(error, commands.CommandNotFound):
            cmd = ctx.message.content.split()[0][1:].lower()  # Remove o prefixo '!'
            
            # Mapeamento de typos/variações comuns → comandos reais
            sugestoes = {
                'entrar_sessão': '!entrar_sessao',
                'sessao': '!entrar_sessao',
                'sessão': '!entrar_sessao',
                'iniciar_gravação': '!gravar_sessao',
                'iniciar_gracacao': '!gravar_sessao',
                'gravarsessao': '!gravar_sessao',
                'parar_sessão': '!parar_gravacao',
                'parar_gravação': '!parar_gravacao',
                'parargravacao': '!parar_gravacao',
                'sair_sessão': '!sair_sessao',
                'status': '!status_voz',
                'help': '!ajuda',
            }
            
            if cmd in sugestoes:
                await ctx.send(f"⚡ Comando reconhecido! Tente `{sugestoes[cmd]}`")
            else:
                # Tenta encontrar comando similar por Levenshtein
                todos_comandos = [c.name for c in bot.commands]
                import difflib
                matches = difflib.get_close_matches(cmd, todos_comandos, n=3, cutoff=0.5)
                if matches:
                    sugestao = ', '.join([f'`!{m}`' for m in matches])
                    await ctx.send(f"❓ Comando `!{cmd}` não encontrado. Você quis dizer {sugestao}?")
                else:
                    await ctx.send(f"❓ Comando `!{cmd}` não encontrado. Use `!ajuda` para ver os comandos disponíveis.")
        else:
            # Outros erros: repassa para o handler padrão
            raise error

    carregar_opus()

    print("Carregando módulos...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    COGS_DIR = os.path.join(BASE_DIR, 'cogs')
    # Garante que o diretório do script está no path para importar módulos
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
    for root, dirs, files in os.walk(COGS_DIR):
        for filename in files:
            if filename.endswith('.py') and not filename.startswith('__'):
                # Exemplo: cogs/rpg/dados.py -> cogs.rpg.dados
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, BASE_DIR)
                module_name = rel_path.replace(os.sep, '.')[:-3]
                # Pula módulos utilitários que não são cogs (ex: transcricao.py, geracao_voz.py)
                try:
                    import importlib
                    spec = importlib.util.find_spec(module_name)
                    if spec and not hasattr(importlib.import_module(module_name), 'setup'):
                        continue
                except Exception:
                    pass

                try:
                    await bot.load_extension(module_name)
                    print(f"  > {module_name} OK")
                except Exception as e:
                    print(f"  > Erro em {module_name}: {e}")

    TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    async with bot:
        await bot.start(TOKEN)

if __name__ == '__main__':
    # ─── Verificação de single instance ───
    _single_instance_check()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        _limpar_pid_file()
