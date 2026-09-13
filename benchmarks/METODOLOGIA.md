# Metodologia dos benchmarks

Este documento existe porque **número sem método não vale nada**. Todo resultado publicado no
README foi produzido pelos scripts desta pasta, nesta máquina, com o procedimento descrito aqui.

## Hardware e software

| Componente | Especificação |
|---|---|
| CPU | AMD Ryzen 7 2700X — 8 núcleos / 16 threads, até 3,7 GHz |
| GPU | NVIDIA RTX 3060 Ti — 8 GB de VRAM |
| RAM | 7,7 GB |
| SO | Linux (Ubuntu 24.04, kernel 6.14) — versão de desenvolvimento do projeto |
| Whisper | faster-whisper 1.2.1 (ctranslate2 4.7.1) |
| TTS | coqui-tts (XTTSv2), torch 2.5.1+cu121 |

Toda latência publicada carrega este hardware na legenda. Um número de latência sem o hardware
é marketing, não engenharia — o mesmo código roda 10× mais devagar numa máquina sem GPU.

## O que compõe a latência percebida

A métrica que importa é o tempo entre **parar de falar** e **o primeiro som sair**:

| # | Etapa | Medido? | Como |
|---|---|---|---|
| T1 | Detecção de fim de fala | ⚙️ não medido (configuração) | `pause_threshold = 0,7 s` em `ia_voz.py` — **não** é custo de máquina. O ganho de 1,0 → 0,7 s (~0,3 s) é aritmética de **configuração**, não medição de ponta a ponta |
| T2 | Transcrição (STT) | ✅ `bench_stt.py` | Whisper × modelo × duração de áudio |
| T3 | LLM (API) | ❌ não medido | depende de rede e de serviço externo |
| T4 | Síntese de voz (TTS) | ✅ `bench_tts.py` | XTTSv2 via worker de produção |
| T5 | Playback no Discord | ❌ não medido | depende da conexão de voz |

O que não foi medido está **marcado como não medido** nos gráficos (hachurado), nunca preenchido
com estimativa apresentada como medição.

## Procedimento

1. **Warm-up obrigatório.** A primeira inferência de cada configuração carrega o modelo e aloca
   kernels CUDA. Ela é executada e descartada; só depois começam as medições.
2. **Repetições.** Padrão de 2 a 5 execuções por configuração.
3. **Agregação.** Reportamos **mediana** e **p95** — nunca só a média. A média esconde a cauda, e
   cauda é o que o usuário sente.
4. **Caminho de produção.** Os scripts chamam o **mesmo código** que o bot usa:
   * `bench_stt.py` usa os parâmetros idênticos de `cogs/audio/interpretacao/transcricao.py`
     (`language="pt"`, `beam_size=4`, `temperature=[0.0]`, `vad_filter=True`) e alimenta o áudio
     no formato exato que o bot entrega (PCM 48 kHz mono 16-bit gravado num WAV temporário, para
     o ffmpeg interno do Whisper fazer o resample para 16 kHz).
   * `bench_tts.py` **sobe o `tts_worker.py` de produção** como subprocesso e conversa pelo
     protocolo real (JSON por linha em stdin/stdout). Não é uma reimplementação.
5. **Denominador honesto (RTF).** As gravações de sessão têm muito silêncio (~33% do arquivo). O
   RTF é calculado sobre a **duração de fala detectada pelo VAD**, não sobre o tamanho do arquivo
   — senão o fator de tempo real sai artificialmente bonito.

## Áudio de teste

As medições de STT usam trechos de gravações de sessão reais do projeto. Essas gravações são
material privado e **não estão neste repositório**. Para reproduzir:

```bash
# qualquer arquivo .wav serve; o script espera PCM 48 kHz mono 16-bit
ffmpeg -i seu_audio.ogg -ac 1 -ar 48000 -c:a pcm_s16le benchmarks/audio/teste_curto.wav
```

Três perfis de amostra: `teste_curto.wav` (12 s), `teste_medio.wav` (60 s) e `teste_longo.wav`
(300 s).

## Scripts

| Script | O que mede | Saída |
|---|---|---|
| `bench_stt.py` | carga do modelo, latência de transcrição, RTF | `dados/resultados_stt.csv` |
| `bench_tts.py` | carga do worker, TTFA (streaming), áudio completo | `dados/resultados_tts.csv` + `resultados_tts_carga.csv` |
| `gerar_graficos.py` | — | `graficos/*.png` |

## Achados que a medição revelou

1. **O modo streaming só funcionava com texto de uma frase.** Com múltiplas frases o worker devolvia
   `{"status": "error", "error": "enable_text_splitting=True requires Spacy..."}`. Como as
   respostas do LLM são parágrafos, na prática o streaming caía para o modo arquivo — pagando
   segundos onde poderia pagar centenas de milissegundos. Este bug de produção foi encontrado
   pelo benchmark, não pelo uso.
2. **`select()` + `readline()` misturados são armadilha.** Se uma leitura do sistema trouxer duas
   linhas, a segunda fica no buffer do Python e o `select` reporta "nada pronto" — a linha nunca é
   lida e o loop gira até o timeout. Este harness usa **thread leitora + fila**, que é imune.
3. **O filtro de silêncio (VAD) é praticamente grátis** e corta entre 19% e 38% do áudio de uma
   sessão (medido: 12 s de arquivo têm 9,7 s de fala; 300 s têm 200,5 s).
4. **O modelo `tiny` alucina em áudio curto.** Medido num trecho de 12 s: o `tiny` produziu
   **1.037 caracteres** com a palavra mais frequente repetida **74 vezes** (*"é um pouco mais um
   pouco mais..."*), enquanto o `small` produziu 71 caracteres corretos. No trecho de 60 s os dois
   se comportaram bem — ou seja, é um risco **condicional**, que aparece justamente quando o áudio
   é curto ou fragmentado. Como o `tiny` não levanta exceção, uma cadeia de fallback que o coloca
   antes da CPU aceita o resultado e para nele. Correção: reordenar (CPU antes de `tiny`).
5. **Denominador de RTF importa.** Dividir o tempo de transcrição pelo tamanho do arquivo faz o
   modelo parecer mais rápido: 0,009 contra 0,014 (RTF sobre fala) no trecho de 300 s. O número
   honesto usa a duração de fala detectada pelo VAD.
