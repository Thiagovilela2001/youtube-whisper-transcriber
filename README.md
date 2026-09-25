# YouTube Whisper Transcriber

Transcreve todos os videos de um canal ou playlist do YouTube localmente usando
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). O processo e
incremental: videos concluidos nao sao processados novamente.

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
--modelo small       Modelo Whisper. O padrao e small.
--idioma pt          Fixa o idioma; sem a opcao, a deteccao e automatica.
--limite 5           Processa somente os primeiros cinco itens.
--saida pasta        Altera a pasta de resultados.
--cookies arquivo    Usa um arquivo cookies.txt quando o YouTube exigir login.
--manter-audio       Nao apaga o audio depois da transcricao.
--threads 8          Define a quantidade de threads de CPU.
```

Use `Ctrl+C` para interromper. Na proxima execucao, o script le
`transcricoes/.concluidos.txt` e continua pelos videos pendentes.

## Arquivos gerados

Para cada video, a pasta `transcricoes` recebe:

- `.txt`: texto corrido;
- `.vtt`: legendas com marcacao de tempo;
- `.json`: metadados, idioma detectado e segmentos temporizados.

Modelos, audios, transcricoes, logs, ambientes virtuais e arquivos de cookies
sao ignorados pelo Git.
