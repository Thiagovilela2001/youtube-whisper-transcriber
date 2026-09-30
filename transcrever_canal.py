#!/usr/bin/env python3
"""Baixa o audio de um canal e cria transcricoes locais com Whisper."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, sanitize_filename

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment,misc]


CANAL_PADRAO = "https://www.youtube.com/@EdsonBurger/videos"


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcreve os videos de um canal com faster-whisper."
    )
    parser.add_argument("--url", default=CANAL_PADRAO, help="URL do canal ou playlist")
    parser.add_argument("--saida", type=Path, default=Path("transcricoes"))
    parser.add_argument("--modelo", default="small", help="Modelo Whisper (padrao: small)")
    parser.add_argument(
        "--idioma",
        default="pt",
        help="Idioma fixo do audio (padrao: pt). Use 'auto' para deteccao automatica.",
    )
    parser.add_argument("--limite", type=int, default=None, help="Limita os videos da playlist")
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="Arquivo cookies.txt (se omitido, busca automaticamente cookies.txt na pasta)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Navegador para carregar cookies (ex: firefox). No Windows, Chrome/Edge requerem cookies.txt.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Pausa em segundos entre downloads para evitar bloqueios do YouTube (padrao: 2.0)",
    )
    parser.add_argument(
        "--max-falhas-bot",
        type=int,
        default=3,
        help="Limite de bloqueios consecutivos antes de pausar a execucao com instrucoes (padrao: 3)",
    )
    parser.add_argument("--manter-audio", action="store_true")
    parser.add_argument("--threads", type=int, default=0, help="Threads de CPU; 0 usa o automatico")
    parser.add_argument(
        "--dispositivo",
        choices=("auto", "gpu", "cpu"),
        default="auto",
        help="Processador para o Whisper (padrao: auto; GPU usa Vulkan)",
    )
    return parser.parse_args()


def timestamp_vtt(segundos: float) -> str:
    milissegundos = max(0, round(segundos * 1000))
    horas, resto = divmod(milissegundos, 3_600_000)
    minutos, resto = divmod(resto, 60_000)
    segundos_int, milissegundos = divmod(resto, 1000)
    return f"{horas:02d}:{minutos:02d}:{segundos_int:02d}.{milissegundos:03d}"


def gravar_atomico(caminho: Path, conteudo: str) -> None:
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    temporario.write_text(conteudo, encoding="utf-8", newline="\n")
    os.replace(temporario, caminho)


def ids_concluidos(arquivo: Path) -> set[str]:
    if not arquivo.exists():
        return set()
    return {
        linha.strip()
        for linha in arquivo.read_text(encoding="utf-8").splitlines()
        if linha.strip()
    }


def firefox_instalado_com_cookies() -> bool:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return False
    perfis = Path(appdata) / "Mozilla" / "Firefox" / "Profiles"
    if not perfis.exists():
        return False
    for arquivo_sqlite in perfis.glob("*/cookies.sqlite"):
        try:
            with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as temporario:
                temporario_nome = temporario.name
            shutil.copy2(arquivo_sqlite, temporario_nome)
            conexao = sqlite3.connect(temporario_nome)
            cursor = conexao.cursor()
            cursor.execute("SELECT 1 FROM moz_cookies WHERE host LIKE '%youtube%' LIMIT 1")
            tem = cursor.fetchone() is not None
            conexao.close()
            Path(temporario_nome).unlink(missing_ok=True)
            if tem:
                return True
        except Exception:
            continue
    return False


def obter_opcoes_ytdlp(
    pasta: Path | None = None,
    video_id: str | None = None,
    cookies: Path | None = None,
    cookies_from_browser: str | None = None,
    extra_opcoes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    opcoes: dict[str, Any] = {
        "restrictfilenames": True,
        "windowsfilenames": True,
        "retries": 10,
        "fragment_retries": 10,
        "sleep_interval": 2,
        "max_sleep_interval": 5,
        "remote_components": ["ejs:github"],
    }

    node_caminho = shutil.which("node")
    if node_caminho:
        opcoes["js_runtimes"] = {"node": {"path": node_caminho}}
    else:
        opcoes["js_runtimes"] = {"node": {}}

    if cookies and Path(cookies).is_file():
        opcoes["cookiefile"] = str(cookies)
    elif cookies_from_browser:
        opcoes["cookiesfrombrowser"] = (cookies_from_browser, None, None, None)

    if pasta and video_id:
        opcoes["outtmpl"] = str(pasta / f"{video_id}.%(ext)s")
        opcoes["noplaylist"] = True
        opcoes["format"] = "bestaudio[abr<=96]/bestaudio/best"
        opcoes["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ]
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg and os.name == "nt":
            pacotes = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages"
            candidatos = list(pacotes.glob("Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe"))
            if candidatos:
                ffmpeg = str(max(candidatos, key=lambda caminho: caminho.stat().st_mtime))
        if ffmpeg:
            opcoes["ffmpeg_location"] = str(Path(ffmpeg).parent)

    if extra_opcoes:
        opcoes.update(extra_opcoes)

    return opcoes


def listar_videos(
    url: str,
    limite: int | None,
    cookies: Path | None = None,
    cookies_from_browser: str | None = None,
) -> list[dict[str, Any]]:
    extra: dict[str, Any] = {
        "extract_flat": "in_playlist",
        "ignoreerrors": True,
        "quiet": False,
    }
    if limite is not None:
        extra["playlistend"] = limite

    opcoes = obter_opcoes_ytdlp(
        cookies=cookies,
        cookies_from_browser=cookies_from_browser,
        extra_opcoes=extra,
    )

    with YoutubeDL(opcoes) as ydl:
        dados = ydl.extract_info(url, download=False)

    if not dados:
        raise RuntimeError("Nao foi possivel ler o canal.")
    if dados.get("_type") != "playlist":
        return [dados]
    return [entrada for entrada in dados.get("entries") or [] if entrada]


def localizar_audio(pasta: Path, video_id: str) -> Path:
    candidatos = [
        caminho
        for caminho in pasta.glob(f"{video_id}.*")
        if caminho.suffix not in {".part", ".ytdl", ".tmp"}
    ]
    if not candidatos:
        raise FileNotFoundError(f"Audio baixado nao encontrado para {video_id}")
    return max(candidatos, key=lambda caminho: caminho.stat().st_mtime)


def baixar_audio(
    url: str,
    video_id: str,
    pasta: Path,
    cookies: Path | None,
    cookies_from_browser: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    opcoes = obter_opcoes_ytdlp(
        pasta=pasta,
        video_id=video_id,
        cookies=cookies,
        cookies_from_browser=cookies_from_browser,
    )

    with YoutubeDL(opcoes) as ydl:
        info = ydl.extract_info(url, download=True)
    if not info:
        raise DownloadError(f"Nao foi possivel baixar {video_id}")
    return localizar_audio(pasta, video_id), info


def transcrever_cpu(
    modelo: Any,
    audio: Path,
    idioma: str | None,
) -> tuple[list[dict[str, Any]], str, float | None]:
    lang = None if (not idioma or idioma.lower() == "auto") else idioma.lower()
    segmentos_iter, info = modelo.transcribe(
        str(audio),
        language=lang,
        beam_size=3,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    segmentos: list[dict[str, Any]] = []
    ultimo_aviso = -30.0
    duracao = float(info.duration or 0)

    for segmento in segmentos_iter:
        texto = segmento.text.strip()
        if texto:
            segmentos.append(
                {"start": float(segmento.start), "end": float(segmento.end), "text": texto}
            )
        if segmento.end - ultimo_aviso >= 30:
            percentual = (segmento.end / duracao * 100) if duracao else 0
            print(
                f"  Whisper: {timestamp_vtt(segmento.end)} / "
                f"{timestamp_vtt(duracao)} ({percentual:.1f}%)",
                flush=True,
            )
            ultimo_aviso = segmento.end

    return segmentos, info.language, float(info.language_probability)


def transcrever_gpu(
    executavel: Path,
    modelo: Path,
    audio: Path,
    idioma: str | None,
    threads: int,
) -> tuple[list[dict[str, Any]], str, float | None]:
    prefixo = audio.with_name(audio.stem + ".whisper")
    json_temporario = Path(f"{prefixo}.json")
    lang = "auto" if (not idioma or idioma.lower() == "auto") else idioma.lower()
    comando = [
        str(executavel),
        "-m",
        str(modelo),
        "-f",
        str(audio),
        "-l",
        lang,
        "-bs",
        "3",
        "-oj",
        "-of",
        str(prefixo),
        "-pp",
    ]
    if threads > 0:
        comando.extend(["-t", str(threads)])

    probabilidade: float | None = None
    try:
        processo = subprocess.Popen(
            comando,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert processo.stdout is not None
        for linha in processo.stdout:
            print("  " + linha.rstrip(), flush=True)
            deteccao = re.search(r"auto-detected language: \w+ \(p = ([0-9.]+)\)", linha)
            if deteccao:
                probabilidade = float(deteccao.group(1))
        codigo = processo.wait()
        if codigo != 0:
            raise RuntimeError(f"whisper.cpp terminou com codigo {codigo}")
        dados = json.loads(json_temporario.read_text(encoding="utf-8", errors="replace"))
    finally:
        json_temporario.unlink(missing_ok=True)

    segmentos = []
    for item in dados.get("transcription", []):
        offsets = item.get("offsets", {})
        texto = str(item.get("text", "")).strip()
        if texto:
            segmentos.append(
                {
                    "start": float(offsets.get("from", 0)) / 1000,
                    "end": float(offsets.get("to", 0)) / 1000,
                    "text": texto,
                }
            )
    idioma_detectado = str(dados.get("result", {}).get("language") or idioma or "auto")
    return segmentos, idioma_detectado, probabilidade


def arquivos_gpu(modelo: str) -> tuple[Path, Path]:
    executavel = Path(".tools/whisper-vulkan/whisper-cli.exe")
    nomes_quantizados = {
        "tiny": "tiny-q5_1",
        "base": "base-q5_1",
        "small": "small-q5_1",
        "medium": "medium-q5_0",
        "large-v2": "large-v2-q5_0",
        "large-v3": "large-v3-q5_0",
        "large-v3-turbo": "large-v3-turbo-q5_0",
    }
    caminho_informado = Path(modelo)
    if caminho_informado.suffix.lower() == ".bin":
        caminho_modelo = caminho_informado
    else:
        nome = nomes_quantizados.get(modelo, modelo)
        caminho_modelo = Path("modelos_whisper_cpp") / f"ggml-{nome}.bin"
    return executavel, caminho_modelo


def salvar_transcricao(
    saida: Path,
    info_video: dict[str, Any],
    segmentos: Iterable[dict[str, Any]],
    idioma: str,
    probabilidade: float | None,
) -> tuple[Path, Path, Path]:
    segmentos = list(segmentos)
    video_id = str(info_video["id"])
    data = str(info_video.get("upload_date") or "SEM_DATA")
    titulo_original = str(info_video.get("title") or video_id)
    titulo = sanitize_filename(titulo_original, restricted=True).strip(" ._")[:110] or video_id
    base = saida / f"{data} - {titulo} [{video_id}]"

    texto = "\n".join(segmento["text"] for segmento in segmentos).strip() + "\n"
    vtt_linhas = ["WEBVTT", ""]
    for segmento in segmentos:
        vtt_linhas.extend(
            [
                f"{timestamp_vtt(segmento['start'])} --> {timestamp_vtt(segmento['end'])}",
                segmento["text"].replace("-->", "->"),
                "",
            ]
        )

    metadados = {
        "id": video_id,
        "title": titulo_original,
        "upload_date": info_video.get("upload_date"),
        "webpage_url": info_video.get("webpage_url"),
        "duration": info_video.get("duration"),
        "detected_language": idioma,
        "language_probability": probabilidade,
        "segments": segmentos,
    }

    # Anexa a extensao em vez de substituir um possivel ponto existente no titulo.
    txt = Path(f"{base}.txt")
    vtt = Path(f"{base}.vtt")
    json_saida = Path(f"{base}.json")
    gravar_atomico(txt, texto)
    gravar_atomico(vtt, "\n".join(vtt_linhas))
    gravar_atomico(json_saida, json.dumps(metadados, ensure_ascii=False, indent=2) + "\n")
    return txt, vtt, json_saida


def registrar_conclusao(arquivo: Path, video_id: str) -> None:
    with arquivo.open("a", encoding="utf-8", newline="\n") as destino:
        destino.write(video_id + "\n")
        destino.flush()
        os.fsync(destino.fileno())


def registrar_erro(arquivo: Path, video_id: str, erro: Exception) -> None:
    momento = datetime.now().astimezone().isoformat(timespec="seconds")
    mensagem = str(erro).replace("\r", " ").replace("\n", " ")
    with arquivo.open("a", encoding="utf-8", newline="\n") as destino:
        destino.write(f"{momento}\t{video_id}\t{type(erro).__name__}: {mensagem}\n")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = argumentos()
    args.saida.mkdir(parents=True, exist_ok=True)
    pasta_audio = args.saida / ".audio_temporario"
    pasta_audio.mkdir(parents=True, exist_ok=True)
    pasta_modelos = Path("modelos_whisper")
    pasta_modelos.mkdir(parents=True, exist_ok=True)
    arquivo_concluidos = args.saida / ".concluidos.txt"
    arquivo_erros = args.saida / "erros.log"

    cookies_from_browser = args.cookies_from_browser
    arquivo_cookies = args.cookies
    if arquivo_cookies:
        if not arquivo_cookies.is_file():
            print(f"[ERRO] Arquivo de cookies nao encontrado: {arquivo_cookies}", file=sys.stderr)
            return 1
        print(f"[Cookies] Usando arquivo: {arquivo_cookies}", flush=True)
    elif not cookies_from_browser:
        candidatos_cookies = [
            Path("cookies.txt"),
            Path("youtube_cookies.txt"),
            Path("www.youtube.com_cookies.txt"),
            args.saida / "cookies.txt",
        ]
        for candidato in candidatos_cookies:
            if candidato.is_file() and candidato.stat().st_size > 0:
                arquivo_cookies = candidato
                print(f"[Cookies] Arquivo detectado automaticamente: {candidato}", flush=True)
                break

        if not arquivo_cookies and firefox_instalado_com_cookies():
            cookies_from_browser = "firefox"
            print("[Cookies] Detectados cookies do YouTube no Firefox. Usando Firefox automaticamente!", flush=True)

    if cookies_from_browser and os.name == "nt":
        nav_lower = cookies_from_browser.lower()
        if any(nav in nav_lower for nav in ("chrome", "edge", "brave", "opera", "vivaldi")):
            print(
                f"[Aviso Cookies] O navegador '{cookies_from_browser}' no Windows 11/10 bloqueia "
                f"leitura direta pelo yt-dlp (App-Bound Encryption / DPAPI).\n"
                f"Recomendado: exporte um arquivo 'cookies.txt' ou utilize o Firefox (--cookies-from-browser firefox).\n",
                file=sys.stderr,
                flush=True,
            )

    concluidos = ids_concluidos(arquivo_concluidos)
    videos = listar_videos(
        args.url,
        args.limite,
        cookies=arquivo_cookies,
        cookies_from_browser=cookies_from_browser,
    )
    pendentes = [video for video in videos if str(video.get("id")) not in concluidos]
    print(
        f"Canal: {len(videos)} videos encontrados; "
        f"{len(concluidos)} concluidos; {len(pendentes)} pendentes.",
        flush=True,
    )
    if not pendentes:
        return 0

    executavel_gpu, modelo_gpu = arquivos_gpu(args.modelo)
    usar_gpu = args.dispositivo == "gpu" or (
        args.dispositivo == "auto" and executavel_gpu.exists() and modelo_gpu.exists()
    )
    if usar_gpu:
        if not executavel_gpu.exists():
            raise FileNotFoundError(f"Executavel Vulkan nao encontrado: {executavel_gpu}")
        if not modelo_gpu.exists():
            raise FileNotFoundError(f"Modelo GPU nao encontrado: {modelo_gpu}")
        print(f"Usando Whisper {args.modelo} na GPU/Vulkan: {modelo_gpu}", flush=True)
        modelo_cpu = None
    else:
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper nao esta instalado para o modo CPU. "
                "Instale requirements.txt ou use --dispositivo gpu."
            )
        print(f"Carregando Whisper {args.modelo} em CPU/int8...", flush=True)
        modelo_cpu = WhisperModel(
            args.modelo,
            device="cpu",
            compute_type="int8",
            cpu_threads=args.threads,
            download_root=str(pasta_modelos),
        )

    falhas = 0
    falhas_consecutivas_bot = 0
    for indice, entrada in enumerate(pendentes, start=1):
        video_id = str(entrada.get("id") or "")
        if not video_id:
            continue
        url = entrada.get("webpage_url") or entrada.get("url")
        if not str(url).startswith("http"):
            url = f"https://www.youtube.com/watch?v={video_id}"
        titulo = entrada.get("title") or video_id
        audio: Path | None = None
        print(f"[{indice}/{len(pendentes)}] {titulo} [{video_id}]", flush=True)
        try:
            audio, info_video = baixar_audio(
                str(url),
                video_id,
                pasta_audio,
                arquivo_cookies,
                cookies_from_browser,
            )
            falhas_consecutivas_bot = 0
            if usar_gpu:
                segmentos, idioma, probabilidade = transcrever_gpu(
                    executavel_gpu, modelo_gpu, audio, args.idioma, args.threads
                )
            else:
                segmentos, idioma, probabilidade = transcrever_cpu(
                    modelo_cpu, audio, args.idioma
                )
            arquivos = salvar_transcricao(
                args.saida, info_video, segmentos, idioma, probabilidade
            )
            registrar_conclusao(arquivo_concluidos, video_id)
            concluidos.add(video_id)
            print("  Salvo: " + ", ".join(str(a) for a in arquivos), flush=True)
            if args.delay > 0 and indice < len(pendentes):
                time.sleep(args.delay)
        except KeyboardInterrupt:
            print("Interrompido. O progresso concluido foi preservado.", flush=True)
            return 130
        except Exception as erro:
            falhas += 1
            registrar_erro(arquivo_erros, video_id, erro)
            msg_erro = str(erro)
            print(f"  ERRO: {type(erro).__name__}: {erro}", file=sys.stderr, flush=True)

            eh_erro_bot = (
                "Sign in to confirm you’re not a bot" in msg_erro
                or "Sign in to confirm you're not a bot" in msg_erro
                or "Failed to decrypt with DPAPI" in msg_erro
                or "HTTP Error 403" in msg_erro
            )
            if eh_erro_bot:
                falhas_consecutivas_bot += 1
                if falhas_consecutivas_bot >= args.max_falhas_bot:
                    print(
                        "\n" + "=" * 72 + "\n"
                        "[BLOQUEIO DE BOT / ERRO DE COOKIES DETECTADO]\n"
                        "O YouTube esta bloqueando as requisicoes e exigindo cookies/login.\n\n"
                        "Como resolver:\n"
                        " 1. Instale no seu navegador a extensao 'Get cookies.txt LOCALLY' ou 'Cookie-Editor'.\n"
                        " 2. Acesse https://www.youtube.com logado em sua conta.\n"
                        " 3. Clique na extensao e exporte os cookies como 'cookies.txt'.\n"
                        " 4. Salve o arquivo 'cookies.txt' na pasta deste projeto:\n"
                        f"    {Path.cwd().resolve()}\n"
                        " 5. Execute novamente o comando. O script detectara o cookies.txt automaticamente!\n\n"
                        " (Alternativa: Se tiver o Mozilla Firefox instalado, use: --cookies-from-browser firefox)\n"
                        "=" * 72 + "\n",
                        flush=True,
                    )
                    break
        finally:
            if audio and audio.exists() and not args.manter_audio:
                audio.unlink()

    if not args.manter_audio and pasta_audio.exists() and not any(pasta_audio.iterdir()):
        pasta_audio.rmdir()
    print(f"Finalizado com {falhas} falha(s).", flush=True)
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
