# YouTube Whisper Transcriber

Transcreve os vídeos de um canal ou playlist do YouTube localmente e permite
busca semântica sobre as transcrições. Usa [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
com Vulkan em GPUs AMD e [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
como modo CPU.

O processo é incremental: vídeos concluídos não são processados de novo, tanto
na transcrição quanto na indexação.

Por padrão, o script usa o canal **Edson Burger - S. I. M.**, mas qualquer
canal, playlist ou vídeo pode ser informado com `--url`.

## O que ele faz

- encontra os vídeos do canal com `yt-dlp`;
- baixa apenas um áudio por vez;
- transcreve localmente com o modelo Whisper;
- gera arquivos TXT, VTT e JSON;
- apaga o áudio temporário depois da transcrição;
- registra os vídeos concluídos para permitir retomadas;
- filtra transmissões ao vivo e vídeos curtos;
- opcionalmente reaproveita as legendas que o YouTube já publica (`--preferir-legendas`).

## Requisitos

- Python 3.10 ou superior;
- [FFmpeg](https://ffmpeg.org/download.html) disponível no `PATH`;
- espaço para o modelo Whisper e para um áudio temporário;
- Node.js recomendado para a extração atual do YouTube pelo `yt-dlp`.

### GPU AMD no Windows

Nesta máquina, o script usa automaticamente o `whisper.cpp` compilado com
Vulkan para a Radeon RX 580. Para exigir a GPU explicitamente:

```powershell
.\.venv\Scripts\python.exe .\transcrever_canal.py --dispositivo gpu --threads 8
```

Use `--dispositivo cpu` para voltar ao `faster-whisper` em CPU. O modo `auto`
(padrão) escolhe a GPU quando o executável e o modelo Vulkan estão disponíveis.

### Legendas em vez de Whisper

```bash
./.venv/bin/python transcrever_canal.py --preferir-legendas
```

Tenta baixar as legendas que o YouTube já publica antes de transcrever o áudio.
Quando não há legenda, cai no Whisper normalmente.

Vale a pena porque as duas coisas somam: é ordens de grandeza mais rápido
(341 vídeos em vez de horas), e o texto de origem costuma ser melhor que o
reconhecimento de voz — o Whisper `small` troca *"saciedade"* por
*"sociedade"* e *"momento"* por *"gerginha"*.

O parser (`legendas.py`) remove a sobreposição que as legendas automáticas
fazem na emenda das frases, para a mesma frase não entrar duas vezes no índice.

O caminho GPU usa filtro de atividade de voz (VAD). Sem ele, o Whisper
transcreve intros de música e silêncio e inventa texto — foi a origem de
trechos corrompidos como *"uma palavra constante na sua gerginha"*. Para
desligar: `--sem-vad`.

## Instalação

```bash
python3 -m venv .venv
```

O `torch` é instalado separado, porque o índice depende da sua GPU:

```bash
# CPU
./.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
# NVIDIA
./.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Depois, o resto:

```bash
./.venv/bin/pip install -r requirements.txt
```

No Windows, use `.\.venv\Scripts\python.exe` no lugar de `./.venv/bin/python`.

## Uso

```bash
./.venv/bin/python ./transcrever_canal.py --threads 8
```

Para outro canal ou playlist:

```bash
./.venv/bin/python ./transcrever_canal.py --url "URL_DO_CANAL" --threads 8
```

Opções úteis:

```text
--modelo small                Modelo Whisper. O padrão é small.
--idioma pt                   Idioma do áudio. O padrão é pt; use 'auto' para detectar.
--limite 5                    Processa somente os primeiros cinco itens.
--saida pasta                 Altera a pasta de resultados.
--cookies arquivo.txt         Arquivo cookies.txt (detectado automaticamente se estiver na pasta).
--cookies-from-browser firefox  Usa cookies direto do Firefox.
--delay 2.0                   Pausa em segundos entre downloads para evitar detecção de bot.
--duracao-minima 60           Ignora vídeos menores que N segundos (0 desliga).
--preferir-legendas           Usa as legendas do YouTube antes do Whisper.
--sem-vad                     Desliga o filtro de voz no caminho GPU.
--manter-audio                Não apaga o áudio depois da transcrição.
--threads 14                  Quantidade de threads de CPU.
--dispositivo gpu             Exige GPU/Vulkan (auto é o padrão; cpu força a CPU).
```

Use `Ctrl+C` para interromper. Na próxima execução, o script lê
`transcricoes/.concluidos.txt` e continua pelos vídeos pendentes. Um lock
(`.transcrevendo.lock`) impede que duas instâncias transcrevam o mesmo canal ao
mesmo tempo — sem ele, o YouTube responde *"Sign in to confirm you're not a
bot"* por causa do tráfego em paralelo.

## Resolvendo o erro de cookies

O YouTube exige autenticação quando detecta múltiplos downloads em sequência. No
Windows, Chrome e Edge usam criptografia *App-Bound*, que impede a extração
direta da base do navegador.

1. Instale a extensão [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbngbenkjcffliehddfaccca)
   ou [Cookie-Editor](https://cookie-editor.com/).
2. Acesse o [YouTube](https://www.youtube.com), de preferência logado.
3. Clique em **Export** (formato Netscape).
4. Salve como `cookies.txt` na pasta do projeto.
5. Teste com `python testar_cookies.py`.

O `transcrever_canal.py` detecta o arquivo automaticamente. Se você usa
**Firefox**, pode usar `--cookies-from-browser firefox` sem exportar nada.

## Arquivos gerados

Para cada vídeo, a pasta `transcricoes` recebe:

- `.txt`: texto corrido;
- `.vtt`: legendas com marcação de tempo;
- `.json`: metadados, idioma detectado e segmentos temporizados.

Transcrições, banco vetorial, modelos, logs, ambientes virtuais e cookies são
ignorados pelo Git: são dados gerados, e 30 MB de transcrição versionada
disputa espaço no histórico com qualquer mudança real no código.

---

# Busca semântica sobre as transcrições

## Como funciona

O pipeline tem quatro etapas, cada uma em um módulo com uma responsabilidade:

1. **`rag_chunker.py`** divide cada transcrição em blocos de ~160 palavras,
   cortando em fim de frase. Cada bloco guarda o texto puro e metadados com a
   minutagem exata e o link direto no YouTube.

   Descarta artefatos do Whisper — `[MÚSICA DE FUNDO]`, `(aplausos)`,
   `. . . .` e repetições. No corpus atual, isso remove 28% dos segmentos sem
   perder nenhuma fala real.

2. **`embedding_engine.py`** gera os vetores. É um módulo reaproveitável, sem
   dependência de CLI: expõe `embed_documents()` e `embed_query()` como
   operações distintas, porque os modelos de instrução tratam as duas
   diferente.

3. **`chroma_store.py`** guarda no ChromaDB e valida que a base foi construída
   com o mesmo modelo usado na busca.

4. **`busca_lexical.py`** mantém um índice BM25 em disco. A busca é híbrida:
   vetorial + BM25, fundidos por RRF.

## Modelos

| Modelo | Dim. | Observação |
| --- | --- | --- |
| `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | 1536 | Multilíngue, bom em português |
| `infly/inf-retriever-v1-1.5b` | 1536 | Exige prefixos `query:`/`passage:` |
| `dunzhang/stella_en_1.5B_v5` | 1536 | Exige prefixos `s2p_` |
| `BAAI/bge-m3` | 1024 | Denso e esparso |
| `text-embedding-3-small` | 1536 | Via API OpenAI |
| `.../paraphrase-multilingual-MiniLM-L12-v2` | 384 | Leve, para testes rápidos |

Cada modelo tem um contrato registrado em `MODELOS` (dimensão, prefixo de
instrução, normalização). **Usar o prefixo errado não gera erro, apenas
degrada a recuperação** — por isso o contrato fica no código, e não na
documentação.

### A base guarda o modelo

A coleção grava um *fingerprint* do contrato nos metadados. Abrir a base com um
modelo diferente é recusado com uma mensagem explicativa:

```text
A coleção foi indexada com outro modelo de embeddings:
  - fingerprint 5c0a45719eeccf97 != e934663d506a3fa8
```

Isso importa porque `infly` e `Alibaba` geram as mesmas 1536 dimensões: sem a
checagem, trocar de modelo devolveria resultados sem sentido nenhum, sem
nenhum aviso.

## Gerando os embeddings

```bash
# Teste rápido com modelo leve, para validar o pipeline
./.venv/bin/python gerar_embeddings.py \
    --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Base de produção
./.venv/bin/python gerar_embeddings.py \
    --model "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

# Via API
./.venv/bin/python gerar_embeddings.py --provider api --model text-embedding-3-small
```

Opções uteis:

```text
--reindex             Apaga a coleção e recria do zero. Use ao trocar de modelo.
--force               Reindexa só os vídeos que já estão na base.
--limit 2             Processa apenas N vídeos.
--target-words 160    Tamanho alvo de cada bloco.
--overlap-words 30    Sobreposição entre blocos vizinhos.
--min-chunk-words 25  Blocos menores que isso são fundidos ao vizinho.
--device cuda         Força a GPU no modelo local.
```

O processo é incremental: um manifesto em `chroma_db/.manifesto.json` registra
o que já foi indexado. Vídeos cuja transcrição ficou mais nova no disco são
reprocessados automaticamente.

Antes de reinserir um vídeo, os blocos antigos dele são apagados. Sem isso, um
`upsert` deixaria para trás os blocos de índice maior quando a transcrição
passasse a gerar menos blocos.

## Buscando

```bash
./.venv/bin/python buscar_rag.py "Como funciona a técnica 369?"
```

```text
[1] cosseno 0.412 | RRF 0.0310 | vetorial #2 + lexical #1
  Vídeo:   Portal 12 12 - Ativação da Abundância
  Data:    20241213
  Momento: 01:15 → 02:48  (bloco 3 de 7)
  Link:    https://www.youtube.com/watch?v=abc&t=75s
  Trecho:
    A partir do segundo dia você começa a sentir o que está denso...
```

Opções:

```text
--top-k 5                Quantos resultados exibir.
--peso-lexical 1.0       Peso do BM25 na fusão. 0 = só vetorial, 2 = dobra.
--candidatos 50          Quantos cada modalidade considera antes da fusão.
--min-score 0.005        Descarta resultados fracos.
--por-video 2            Máximo de trechos de cada vídeo no resultado.
--video-id ABC123        Restringe a um vídeo.
```

A distância cosseno é mostrada como número, não como porcentagem de
"similaridade": em texto, valores de 0.6 a 0.9 são normais, e converter para
percentual daria uma leitura enganosa.

### Por que BM25 e não só vetorial

Termos como `369`, `Portal 12 12` e `Grabovoi` são densos nesse corpus, mas
diluem-se entre centenas de blocos do mesmo tema quando a busca é só semântica.
O BM25 acha o termo literal; o vetor acha o assunto. RRF combina os dois
rankings sem precisar normalizar escores incomparáveis.

O BM25 é implementado localmente (`busca_lexical.py`) porque o
`query_texts` do Chroma não faz BM25: ele baixa um modelo ONNX de 384
dimensões e faz busca vetorial, o que falha numa coleção de 1536. A API
`collection.search()` com RRF nativo existe, mas lança `NotImplementedError`
no Chroma local.

## Testes

```bash
./.venv/bin/python -m pytest
```

138 testes cobrindo o chunker (incluindo as regressões de `KeyError` e de URL
quebrada), o contrato de cada modelo, o BM25, a validação de compatibilidade do
banco, o parser de legendas, o lock e a montagem do comando do whisper.cpp.
Nenhum teste baixa modelo: a integração usa embeddings sintéticos.

## Estrutura

```text
transcrever_canal.py    Transcrição (yt-dlp + Whisper)
testar_cookies.py       Diagnóstico de cookies
legendas.py             Parser de WebVTT/SRT das legendas do YouTube
rag_chunker.py          Divisão das transcrições em blocos
embedding_engine.py     Contrato e execução dos modelos de embedding
chroma_store.py         Banco vetorial, validação e busca
busca_lexical.py        Índice BM25
gerar_embeddings.py     Indexação
buscar_rag.py           Busca no terminal
tests/                  Testes automatizados
```
