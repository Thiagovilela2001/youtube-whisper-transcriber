#!/usr/bin/env python3
"""Baixa o audio de um canal e cria transcricoes locais com Whisper."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from faster_whisper import WhisperModel
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, sanitize_filename


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
        default=None,
        help="Idioma fixo, como pt ou en. O padrao detecta automaticamente.",
    )
    parser.add_argument("--limite", type=int, default=None, help="Limita os videos da playlist")
    parser.add_argument("--cookies", type=Path, default=None, help="Arquivo cookies.txt")
    parser.add_argument("--manter-audio", action="store_true")
    parser.add_argument("--threads", type=int, default=0, help="Threads de CPU; 0 usa o automatico")
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


def listar_videos(url: str, limite: int | None) -> list[dict[str, Any]]:
    opcoes: dict[str, Any] = {
        "extract_flat": "in_playlist",
        "ignoreerrors": True,
        "quiet": False,
        "js_runtimes": {"node": {}},
    }
    if limite is not None:
        opcoes["playlistend"] = limite

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
) -> tuple[Path, dict[str, Any]]:
    opcoes: dict[str, Any] = {
        "format": "bestaudio[abr<=96]/bestaudio/best",
        "outtmpl": str(pasta / f"{video_id}.%(ext)s"),
        "noplaylist": True,
        "restrictfilenames": True,
        "windowsfilenames": True,
        "retries": 10,
        "fragment_retries": 10,
        "sleep_interval": 2,
        "max_sleep_interval": 5,
        "js_runtimes": {"node": {}},
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ],
    }
    if cookies:
        opcoes["cookiefile"] = str(cookies)

    with YoutubeDL(opcoes) as ydl:
        info = ydl.extract_info(url, download=True)
    if not info:
        raise DownloadError(f"Nao foi possivel baixar {video_id}")
    return localizar_audio(pasta, video_id), info


def transcrever(
    modelo: WhisperModel,
    audio: Path,
    idioma: str | None,
) -> tuple[list[dict[str, Any]], str, float]:
    segmentos_iter, info = modelo.transcribe(
        str(audio),
        language=idioma,
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


def salvar_transcricao(
    saida: Path,
    info_video: dict[str, Any],
    segmentos: Iterable[dict[str, Any]],
    idioma: str,
    probabilidade: float,
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

    concluidos = ids_concluidos(arquivo_concluidos)
    videos = listar_videos(args.url, args.limite)
    pendentes = [video for video in videos if str(video.get("id")) not in concluidos]
    print(
        f"Canal: {len(videos)} videos encontrados; "
        f"{len(concluidos)} concluidos; {len(pendentes)} pendentes.",
        flush=True,
    )
    if not pendentes:
        return 0

    print(f"Carregando Whisper {args.modelo} em CPU/int8...", flush=True)
    modelo = WhisperModel(
        args.modelo,
        device="cpu",
        compute_type="int8",
        cpu_threads=args.threads,
        download_root=str(pasta_modelos),
    )

    falhas = 0
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
            audio, info_video = baixar_audio(str(url), video_id, pasta_audio, args.cookies)
            segmentos, idioma, probabilidade = transcrever(modelo, audio, args.idioma)
            arquivos = salvar_transcricao(
                args.saida, info_video, segmentos, idioma, probabilidade
            )
            registrar_conclusao(arquivo_concluidos, video_id)
            concluidos.add(video_id)
            print("  Salvo: " + ", ".join(str(a) for a in arquivos), flush=True)
        except KeyboardInterrupt:
            print("Interrompido. O progresso concluido foi preservado.", flush=True)
            return 130
        except Exception as erro:
            falhas += 1
            registrar_erro(arquivo_erros, video_id, erro)
            print(f"  ERRO: {type(erro).__name__}: {erro}", file=sys.stderr, flush=True)
        finally:
            if audio and audio.exists() and not args.manter_audio:
                audio.unlink()

    if not args.manter_audio and pasta_audio.exists() and not any(pasta_audio.iterdir()):
        pasta_audio.rmdir()
    print(f"Finalizado com {falhas} falha(s).", flush=True)
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
