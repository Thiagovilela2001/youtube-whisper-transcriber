# YouTube Whisper Transcriber

Transcreve todos os videos de um canal ou playlist do YouTube localmente. Usa
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) com Vulkan em GPUs AMD e
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) como modo CPU. O
processo e incremental: videos concluidos nao sao processados novamente.

Por padrao, o script usa o canal **Edson Burger - S. I. M.**, mas qualquer canal,
playlist ou video pode ser informado com `--url`.

## O que ele faz

- encontra os videos do canal com `yt-dlp`;
- baixa apenas um audio por vez;
- transcreve localmente com o modelo Whisper;
- gera arquivos TXT, VTT e JSON;
- apaga o audio temporario depois da transcricao;
- registra os videos concluidos para permitir retomadas;
- detecta automaticamente o idioma de cada video.

## Requisitos

- Python 3.10 ou superior;
- [FFmpeg](https://ffmpeg.org/download.html) disponivel no `PATH`;
- espaco para o modelo Whisper e para um audio temporario;
- Node.js recomendado para a extracao atual do YouTube pelo `yt-dlp`.

### GPU AMD no Windows

Nesta maquina, o script usa automaticamente o `whisper.cpp` compilado com
Vulkan para a Radeon RX 580. Para exigir a GPU explicitamente:

```powershell
.\.venv\Scripts\python.exe .\transcrever_canal.py --dispositivo gpu --threads 8
```

Use `--dispositivo cpu` para voltar ao `faster-whisper` em CPU. O modo `auto`
(padrao) escolhe a GPU quando o executavel e o modelo Vulkan estao disponiveis.

## Instalacao

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux ou macOS

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

## Uso

No Windows:

```powershell
.\.venv\Scripts\python.exe .\transcrever_canal.py --threads 8
```

Em Linux ou macOS:

```bash
./.venv/bin/python ./transcrever_canal.py --threads 8
```

Para outro canal ou playlist:

```powershell
.\.venv\Scripts\python.exe .\transcrever_canal.py --url "URL_DO_CANAL_OU_PLAYLIST" --threads 8
```

Algumas opcoes uteis:

```text
--modelo small                Modelo Whisper. O padrao e small.
--idioma pt                   Idioma do audio (padrao fixo: pt). Use 'auto' para deteccao automatica.
--limite 5                    Processa somente os primeiros cinco itens.
--saida pasta                 Altera a pasta de resultados.
--cookies arquivo.txt         Arquivo cookies.txt (detectado automaticamente se estiver na pasta).
--cookies-from-browser firefox Usa cookies direto do Firefox.
--delay 2.0                   Pausa em segundos entre downloads para evitar deteccao de bot.
--manter-audio                Nao apaga o audio depois da transcricao.
--threads 14                  Define a quantidade de threads de CPU (recomendado: 14 para o Xeon).
--dispositivo gpu             Exige GPU/Vulkan (auto e o padrao; cpu forca a CPU).
```

### Resolvendo o erro de Cookies / "Sign in to confirm you're not a bot"

O YouTube frequentemente exige autenticação quando detecta múltiplos downloads em sequência. No Windows, o Chrome e Edge utilizam criptografia *App-Bound*, o que impede a extração direta da base do navegador.

**Solução recomendada (cookies.txt):**
1. Instale a extensão [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbngbenkjcffliehddfaccca) ou [Cookie-Editor](https://cookie-editor.com/) no seu navegador (Chrome ou Edge).
2. Acesse o [YouTube](https://www.youtube.com) (de preferência logado na sua conta).
3. Abra a extensão e clique em **Export** (formato Netscape / cookies.txt).
4. Salve o arquivo com o nome `cookies.txt` diretamente na pasta do projeto:
   `c:\Users\Thiago\Documents\traducao\youtube-whisper-transcriber\cookies.txt`
5. Teste se está tudo funcionando com:
   ```powershell
   .\.venv\Scripts\python.exe .\testar_cookies.py
   ```
6. O `transcrever_canal.py` detectará o arquivo `cookies.txt` automaticamente sem precisar de parâmetros adicionais!

*(Alternativa: se você utiliza o **Mozilla Firefox**, pode apenas rodar com `--cookies-from-browser firefox` sem precisar exportar nenhum arquivo.)*

Use `Ctrl+C` para interromper. Na proxima execucao, o script le
`transcricoes/.concluidos.txt` e continua pelos videos pendentes.

## Arquivos gerados

Para cada video, a pasta `transcricoes` recebe:

- `.txt`: texto corrido;
- `.vtt`: legendas com marcacao de tempo;
- `.json`: metadados, idioma detectado e segmentos temporizados.

Modelos, audios, transcricoes, logs, ambientes virtuais e arquivos de cookies
sao ignorados pelo Git.

---

## Embeddings (1536 dimensões) e Busca Semântica (RAG)

O projeto inclui um pipeline completo de embeddings em **1536 dimensões** e banco vetorial **ChromaDB** local para RAG (Retrieval-Augmented Generation).

### Abordagem Otimizada para RAG de Vídeos

1. **Chunking Temporal Semântico (`rag_chunker.py`)**:
   - Agrupa os segmentos Whisper em blocos de ~250 palavras com sobreposição (overlap de ~50 palavras) para preservar o contexto.
   - Mantém timestamps exatos (`start_time` e `end_time`).
   - Gera automaticamente o link com minutagem exata no YouTube (`https://youtube.com/watch?v=...&t=123s`).
   - Enriquece o texto do embedding com o contexto do título do vídeo e o intervalo de tempo correspondente.

2. **Modelos em 1536 dimensões (`embedding_engine.py`)**:
   - **Local (Multilíngue/Português):** `Alibaba-NLP/gte-Qwen2-1.5B-instruct` ou `infly/inf-retriever-v1-1.5b` (família Stella / Infini-AI, nativo em 1536d).
   - **Local (Stella v5):** `dunzhang/stella_en_1.5B_v5`.
   - **API (Nuvem):** OpenAI `text-embedding-3-small` (1536d) ou Infini-AI Cloud API (`https://cloud.infini-ai.com/maas/v1`).

### Como gerar os embeddings

**1. Modo Local (padrão):**
```powershell
.\.venv\Scripts\python.exe .\gerar_embeddings.py
```
*(Para testar apenas os primeiros 2 vídeos antes de rodar os 341: use `--limit 2`)*

**2. Modo Local especificando modelo:**
```powershell
.\.venv\Scripts\python.exe .\gerar_embeddings.py --model "infly/inf-retriever-v1-1.5b"
```

**3. Modo API (OpenAI ou Infini-AI):**
```powershell
.\.venv\Scripts\python.exe .\gerar_embeddings.py --provider openai --api-key "SUA_CHAVE"
```
*(Ou definindo a variável de ambiente `$env:OPENAI_API_KEY = "sua_chave"`)*

O processo é incremental: se você interromper, ele retoma de onde parou sem reprocessar vídeos já indexados.

### Como pesquisar no banco (Busca RAG)

Depois de gerados os embeddings, você pode fazer perguntas ou buscas semânticas diretamente pelo terminal:

```powershell
.\.venv\Scripts\python.exe .\buscar_rag.py "Como funciona a técnica 369 da lei da atração?"
```

O script retorna os trechos mais relevantes, a pontuação de similaridade, a minutagem exata e o **link clicável direto para o momento exato do vídeo no YouTube**!
