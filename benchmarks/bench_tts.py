#!/usr/bin/env python3
"""Benchmark de TTS — XTTSv2 via worker de PRODUÇÃO.

Não reimplementa nada: sobe o MESMO `tts_worker.py` que o bot usa em produção,
com o MESMO protocolo (JSON por linha em stdin/stdout) e os MESMOS parâmetros,
e mede:

  * tempo de carregamento do modelo (criação + transferência pra VRAM)
  * tempo de aquecimento (primeira inferência — aloca kernels CUDA)
  * modo ARQUIVO: texto -> arquivo completo
  * modo STREAM: TTFA (time to first audio chunk) + tempo total

O TTFA é a métrica que importa pra latência percebida: é o tempo entre o fim da
fala e o primeiro som sair.

Notas de robustez (aprendidas na marra):
  * a leitura do stdout do worker passa por uma thread + fila. Ler `readline()`
    direto misturado com `select` é armadilha: o buffer do Python pode já ter
    consumido dados que o `select` não vê.
  * TODA resposta tem timeout. Sem isso, uma dessincronização de protocolo deixa
    o benchmark pendurado pra sempre em vez de falhar com erro claro.
  * toda linha crua do worker vai para /tmp/bench_tts_raw.log (post-mortem).

Uso:
    python bench_tts.py --repeticoes 2 --amostras benchmarks/audio/geradas

Saída: benchmarks/dados/resultados_tts.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import queue
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAIDA = os.path.join(RAIZ, "benchmarks", "dados", "resultados_tts.csv")
LOG_CRU = "/tmp/bench_tts_raw.log"
LOG_ERR = "/tmp/bench_tts_stderr.log"

VAULT = os.path.expanduser("~/Documentos/Projetos/Amelia2.0-master")
WORKER = os.path.join(VAULT, "Bot_Discord", "tts_worker.py")
VENV_PYTHON = os.path.join(VAULT, "venv_linux", "bin", "python")

# Textos genéricos de propósito (nada de conteúdo de campanha)
TEXTOS = {
    "curto": "A neblina desce sobre o porto e o alarme começa a soar.",
    "medio": (
        "A neblina desce sobre o porto enquanto o alarme soa ao longe. "
        "Vocês percebem que a comporta do setor sul está travada e que há "
        "duas silhuetas se movendo entre os contêineres. O cheiro de óleo "
        "queimado vem do cais, e a luz vermelha pisca sobre a água parada."
    ),
    "longo": (
        "A neblina desce sobre o porto enquanto o alarme soa ao longe. "
        "Vocês percebem que a comporta do setor sul está travada e que há "
        "duas silhuetas se movendo entre os contêineres. O cheiro de óleo "
        "queimado vem do cais, e a luz vermelha pisca sobre a água parada. "
        "Do alto da torre de controle, uma voz distorcida repete o mesmo "
        "aviso a cada quinze segundos. O rádio do grupo estala uma única vez "
        "e alguém respira do outro lado. Vocês têm três minutos antes que a "
        "maré suba e feche a única rota de saída que ainda resta."
    ),
}


def _mostrar_erro_do_worker():
    """Imprime o fim do stderr do worker — é onde vive o traceback."""
    try:
        with open(LOG_ERR, encoding="utf-8") as f:
            linhas = f.readlines()[-12:]
        if linhas:
            print("  --- stderr do worker (fim) ---", flush=True)
            for l in linhas:
                print(f"    {l.rstrip()[:150]}", flush=True)
    except FileNotFoundError:
        pass


class WorkerTTS:
    """Sobe e conversa com o worker de produção."""

    def __init__(self, python=VENV_PYTHON, worker=WORKER, timeout_resposta=240):
        self.python = python
        self.worker = worker
        self.proc = None
        self.t_carga = None
        self.t_pronto = None
        self.t_warmup = None
        self.timeout_resposta = timeout_resposta
        self.fila: queue.Queue = queue.Queue()
        self.t_inicio = time.perf_counter()
        self.raw = open(LOG_CRU, "w", encoding="utf-8")

    # ─── leitura ────────────────────────────────────────────────────
    def _ler_stdout(self):
        """Thread leitora: registra a linha crua e enfileira."""
        try:
            for linha in self.proc.stdout:
                self.raw.write(f"[{time.perf_counter() - self.t_inicio:8.3f}s] {linha}")
                self.raw.flush()
                self.fila.put(linha)
        except Exception:
            pass
        finally:
            self.fila.put(None)  # EOF

    def _drenar_stderr(self):
        """Guarda o stderr do worker em arquivo.

        Não joga fora: se o worker morre na subida, é aqui que está o traceback.
        """
        try:
            with open(LOG_ERR, "a", encoding="utf-8") as f:
                for linha in self.proc.stderr:
                    f.write(linha)
                    f.flush()
        except Exception:
            pass

    def _linha(self, timeout):
        """Próxima linha crua. '' = nada no tempo; None = worker encerrou."""
        try:
            return self.fila.get(timeout=timeout)
        except queue.Empty:
            return ""

    def _json(self):
        """Próximo objeto JSON. Ignora (e reporta) linhas de diagnóstico."""
        linha = self._linha(self.timeout_resposta)
        if linha is None:
            raise RuntimeError("worker encerrou a conexão (EOF)")
        if linha == "":
            raise TimeoutError(
                f"worker silencioso por {self.timeout_resposta}s — "
                f"conversa completa em {LOG_CRU}"
            )
        linha = linha.strip()
        if not linha:
            return None
        if not linha.startswith("{"):
            print(f"      (worker) {linha[:110]}", flush=True)
            return None
        try:
            return json.loads(linha)
        except json.JSONDecodeError:
            print(f"      (linha ilegível) {linha[:90]}", flush=True)
            return None

    # ─── subida ─────────────────────────────────────────────────────
    def subir(self, timeout=420):
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            [self.python, "-u", self.worker],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"},
        )
        threading.Thread(target=self._ler_stdout, daemon=True).start()
        threading.Thread(target=self._drenar_stderr, daemon=True).start()

        pronto = False
        while time.perf_counter() - t0 < timeout:
            linha = self._linha(min(30, timeout))
            if linha is None:
                print("  ✗ worker encerrou durante a subida", flush=True)
                _mostrar_erro_do_worker()
                break
            if linha == "":
                if self.proc.poll() is not None:
                    break
                continue
            linha = linha.strip()
            if "Modelo carregado" in linha:
                self.t_carga = time.perf_counter() - t0
            if "Pronto para processar" in linha:
                self.t_pronto = time.perf_counter() - t0
                pronto = True
            if "Aquecimento OK em" in linha:
                try:
                    self.t_warmup = float(linha.split("Aquecimento OK em")[1].split("s")[0])
                except Exception:
                    self.t_warmup = time.perf_counter() - t0
                break
            if "Aquecimento completo" in linha:
                break
        return pronto

    # ─── requisições ────────────────────────────────────────────────
    def enviar(self, requisicao: dict) -> dict:
        self.proc.stdin.write(json.dumps(requisicao) + "\n")
        self.proc.stdin.flush()

        if requisicao.get("mode") == "stream":
            t0 = time.perf_counter()
            ttfa, chunks, audio = None, 0, bytearray()
            while True:
                resp = self._json()
                if resp is None:
                    continue
                if resp.get("type") == "chunk":
                    chunks += 1
                    if ttfa is None:
                        ttfa = time.perf_counter() - t0
                        print(f"      primeiro chunk em {ttfa*1000:.0f} ms", flush=True)
                    audio.extend(base64.b64decode(resp.get("audio", "")))
                elif resp.get("type") == "done":
                    total = time.perf_counter() - t0
                    return {"ttfa_ms": ttfa * 1000 if ttfa else None,
                            "total_ms": total * 1000,
                            "chunks": resp.get("chunks", chunks),
                            "time_reportado_ms": resp.get("time", 0) * 1000,
                            "pcm": bytes(audio)}

                elif resp.get("status") == "error":
                    # O worker reporta erro de forma estruturada quando o modo
                    # não está disponível (ex.: enable_text_splitting sem SpaCy).
                    # Sem tratar isso, o benchmark esperaria chunks pra sempre.
                    raise RuntimeError(
                        f"worker recusou a requisição stream: {resp.get('error', '?')}"
                    )

        t0 = time.perf_counter()
        while True:
            resp = self._json()
            if resp is None:
                continue
            if resp.get("status") == "error":
                raise RuntimeError(
                    f"worker recusou a requisição arquivo: {resp.get('error', '?')}"
                )
            if resp.get("status") == "ok" or "time" in resp:
                return {"total_ms": (time.perf_counter() - t0) * 1000,
                        "time_reportado_ms": resp.get("time", 0) * 1000}

    def encerrar(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=15)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        try:
            self.raw.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeticoes", type=int, default=5)
    ap.add_argument("--textos", default="curto,medio,longo")
    ap.add_argument("--amostras", default="",
                    help="pasta onde salvar os WAV gerados (pra ouvir o resultado)")
    ap.add_argument("--worker", default="",
                    help="caminho alternativo para o tts_worker.py (testar variantes sem "
                         "tocar no arquivo de produção)")
    args = ap.parse_args()

    nomes = [t.strip() for t in args.textos.split(",") if t.strip() in TEXTOS]
    if not nomes:
        print("Nenhum texto válido. Use: curto,medio,longo")
        sys.exit(1)

    print("=" * 78, flush=True)
    print("  BENCHMARK TTS — XTTSv2 via worker de produção", flush=True)
    print(f"  repeticoes={args.repeticoes}  textos={nomes}", flush=True)
    print(f"  worker: {args.worker or WORKER}", flush=True)
    print("=" * 78, flush=True)

    w = WorkerTTS(worker=args.worker) if args.worker else WorkerTTS()
    print("\n▶ Subindo o worker (carrega o modelo na VRAM)...", flush=True)
    t0 = time.perf_counter()
    try:
        if not w.subir():
            print("  ✗ worker não ficou pronto", flush=True)
            w.encerrar()
            sys.exit(1)
    except Exception as e:
        print(f"  ✗ falha na subida: {e}", flush=True)
        w.encerrar()
        sys.exit(1)

    print(f"  modelo carregado   : {w.t_carga:.1f} s" if w.t_carga else "  modelo carregado: ?", flush=True)
    print(f"  PRONTO p/ requisição: {w.t_pronto:.1f} s" if w.t_pronto else "  pronto: ?", flush=True)
    print(f"  aquecimento        : {w.t_warmup:.1f} s" if w.t_warmup else "  aquecimento: ?", flush=True)
    print(f"  operacional em     : {time.perf_counter() - t0:.1f} s", flush=True)

    linhas = []
    destino = os.path.join(tempfile.gettempdir(), "amelia_bench_tts.wav")

    try:
        for nome in nomes:
            texto = TEXTOS[nome]
            print(f"\n▶ texto '{nome}' ({len(texto)} caracteres)", flush=True)

            # ── ARQUIVO ──
            tempos = []
            for i in range(args.repeticoes):
                r = w.enviar({"id": 100 + i, "text": texto, "file_path": destino})
                tempos.append(r["total_ms"])
            mediana = statistics.median(tempos)
            p95 = sorted(tempos)[max(0, int(round(0.95 * len(tempos))) - 1)]
            print(f"  ARQUIVO  mediana {mediana:7.0f} ms | p95 {p95:7.0f} ms", flush=True)
            if args.amostras:
                os.makedirs(args.amostras, exist_ok=True)
                shutil.copy(destino, os.path.join(args.amostras, f"saida_arquivo_{nome}.wav"))
            linhas.append({"modo": "arquivo", "texto": nome, "caracteres": len(texto),
                           "repeticoes": args.repeticoes, "mediana_ms": round(mediana, 1),
                           "p95_ms": round(p95, 1), "min_ms": round(min(tempos), 1),
                           "max_ms": round(max(tempos), 1), "ttfa_ms": "", "chunks": ""})

            # ── STREAM ──
            ttfas, totais, chunks, pcm_final = [], [], [], None
            for i in range(args.repeticoes):
                r = w.enviar({"id": 200 + i, "text": texto, "mode": "stream"})
                if r["ttfa_ms"]:
                    ttfas.append(r["ttfa_ms"])
                totais.append(r["total_ms"])
                chunks.append(r["chunks"])
                pcm_final = r.get("pcm")
            if ttfas:
                msg = statistics.median(ttfas)
                mtotal = statistics.median(totais)
                print(f"  STREAM   TTFA mediana {msg:7.0f} ms | total mediana {mtotal:7.0f} ms "
                      f"| {chunks[0]} chunks", flush=True)
                if args.amostras and pcm_final:
                    import numpy as np
                    import soundfile as sf
                    audio = np.frombuffer(pcm_final, dtype=np.float32)
                    sf.write(os.path.join(args.amostras, f"saida_stream_{nome}.wav"), audio, 24000)
                linhas.append({"modo": "stream", "texto": nome, "caracteres": len(texto),
                               "repeticoes": args.repeticoes, "mediana_ms": round(msg, 1),
                               "p95_ms": round(sorted(ttfas)[max(0, int(round(0.95*len(ttfas)))-1)], 1),
                               "min_ms": round(min(ttfas), 1), "max_ms": round(max(ttfas), 1),
                               "ttfa_ms": round(msg, 1), "chunks": chunks[0]})
            else:
                print("  STREAM   ✗ nenhum chunk recebido", flush=True)
    finally:
        w.encerrar()

    if not linhas:
        print("Nenhuma medição coletada.", flush=True)
        sys.exit(1)

    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    with open(SAIDA, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)

    meta = os.path.join(os.path.dirname(SAIDA), "resultados_tts_carga.csv")
    with open(meta, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["etapa", "segundos"])
        escritor.writerow(["carga_modelo", round(w.t_carga, 2) if w.t_carga else ""])
        escritor.writerow(["pronto_para_requisicoes", round(w.t_pronto, 2) if w.t_pronto else ""])
        escritor.writerow(["aquecimento", round(w.t_warmup, 2) if w.t_warmup else ""])

    print(f"\nCSV: {SAIDA}", flush=True)
    print(f"Carga: {meta}", flush=True)
    print(f"Conversa crua: {LOG_CRU}", flush=True)


if __name__ == "__main__":
    main()
