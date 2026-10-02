#!/usr/bin/env python3
"""Baixa só as legendas publicadas pelo YouTube, sem áudio e sem Whisper.

Existe porque o `transcrever_canal.py --preferir-legendas` cai no Whisper
quando a legenda não vem — e aqui o Whisper não roda, porque não há ffmpeg
nem binário Vulkan. Pior: quando o YouTube responde 429 à legenda, o script
mesmo baixa o áudio e só depois descobre o erro, gastando centenas de MB por
vídeo. Este script faz só o que interessa e trata o 429 como o que é: uma
limite de taxa, para esperar e tentar de novo.

    ./.venv/bin/python baixar_legendas_canal.py --url "URL_DO_CANAL"

Repetições são seguras: o `.concluidos.txt` da pasta de saída é lido no início
e vídeos já concluídos são pulados. Pode interrupting e rodar de novo.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from transcrever_canal import (  # noqa: E402
    baixar_legendas,
    ids_concluidos,
    listar_videos,
    LockDeExecucao,
    registrar_conclusao,
    salvar_transcricao,
)


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa as legendas de um canal, sem audio e sem Whisper."
    )
    parser.add_argument("--url", required=True, help="URL do canal, playlist ou video")
    parser.add_argument("--saida", type=Path, default=RAIZ / "transcricoes")
    parser.add_argument("--limite", type=int, default=None)
    parser.add_argument("--cookies", type=Path, default=None, help="Arquivo cookies.txt")
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Navegador para ler cookies direto (ex: firefox)",
    )
    parser.add_argument(
        "--tentativas", type=int, default=6, help="Tentativas por video (padrao: 6)"
    )
    parser.add_argument(
        "--espera",
        type=float,
        default=6.0,
        help="Pausa base entre videos, em segundos (padrao: 6)",
    )
    parser.add_argument(
        "--duracao-minima",
        type=int,
        default=60,
        help="Ignora videos com menos de N segundos (padrao: 60; 0 desliga)",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = argumentos()
    args.saida.mkdir(parents=True, exist_ok=True)
    temporario = args.saida / ".legendas_temporario"
    temporario.mkdir(parents=True, exist_ok=True)
    arquivo_concluidos = args.saida / ".concluidos.txt"
    arquivo_lock = args.saida / ".transcrevendo.lock"

    cookies = args.cookies
    if cookies and not cookies.is_file():
        print(f"[ERRO] Arquivo de cookies nao encontrado: {cookies}", file=sys.stderr)
        return 1
    if not cookies and not args.cookies_from_browser:
        for candidato in (
            Path("cookies.txt"),
            RAIZ / "cookies.txt",
            args.saida / "cookies.txt",
        ):
            if candidato.is_file() and candidato.stat().st_size > 0:
                cookies = candidato
                print(f"[Cookies] Arquivo detectado: {candidato}", flush=True)
                break

    videos = listar_videos(
        args.url,
        args.limite,
        cookies=cookies,
        cookies_from_browser=args.cookies_from_browser,
    )
    if args.duracao_minima > 0:
        videos = [
            v
            for v in videos
            if not v.get("is_live")
            and (not v.get("duration") or float(v["duration"]) >= args.duracao_minima)
        ]

    concluidos = ids_concluidos(arquivo_concluidos)
    pendentes = [v for v in videos if str(v.get("id")) not in concluidos]
    print(f"{len(videos)} videos; {len(concluidos)} concluidos; {len(pendentes)} pendentes.")
    if not pendentes:
        return 0

    sem_legenda: list[str] = []
    with LockDeExecucao(arquivo_lock):
        for indice, entrada in enumerate(pendentes, start=1):
            video_id = str(entrada.get("id") or "")
            if not video_id:
                continue
            url = entrada.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"
            print(
                f"[{indice}/{len(pendentes)}] {entrada.get('title') or video_id} [{video_id}]",
                flush=True,
            )

            resultado = None
            for tentativa in range(1, args.tentativas + 1):
                resultado = baixar_legendas(
                    str(url), video_id, temporario, cookies, args.cookies_from_browser
                )
                if resultado is not None:
                    break
                # O YouTube devolve 429 por rajada. A espera cresce por tentativa
                # e tem um componente aleatorio, para as varias instancias nao
                # voltarem a bater no mesmo instante.
                espera = min(90.0, 5.0 * tentativa) + random.uniform(0, 5)
                print(f"  429 ou sem legenda; tentativa {tentativa}, {espera:.1f}s", flush=True)
                time.sleep(espera)

            if resultado is None:
                sem_legenda.append(video_id)
                print(f"  SEM LEGENDA apos {args.tentativas} tentativas", flush=True)
            else:
                segmentos, info = resultado
                salvar_transcricao(
                    args.saida,
                    info,
                    segmentos,
                    str(info.get("language") or "pt"),
                    None,
                )
                registrar_conclusao(arquivo_concluidos, video_id)
                concluidos.add(video_id)
                print(f"  Salvo (legendas, {len(segmentos)} segmentos)", flush=True)

            time.sleep(args.espera + random.uniform(0, 3))

    if temporario.exists() and not any(temporario.iterdir()):
        temporario.rmdir()

    if sem_legenda:
        print(f"\n{len(sem_legenda)} video(s) sem legenda mesmo apos retentativas:")
        for video_id in sem_legenda:
            print(f"  https://www.youtube.com/watch?v={video_id}")
        print("\nEstes precisam de Whisper, que aqui exige ffmpeg instalado.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())