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
