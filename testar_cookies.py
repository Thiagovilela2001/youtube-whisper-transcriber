#!/usr/bin/env python3
"""Script para testar a validade do arquivo cookies.txt ou dos cookies do navegador."""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

VIDEO_TESTE = "https://www.youtube.com/watch?v=CztYHonupnY"


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


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Testa a autenticacao de cookies com o YouTube via yt-dlp.")
    parser.add_argument("--cookies", type=Path, default=None, help="Caminho do arquivo cookies.txt")
    parser.add_argument("--cookies-from-browser", default=None, help="Navegador (ex: firefox)")
    parser.add_argument("--url", default=VIDEO_TESTE, help="URL de video para testar")
    args = parser.parse_args()

    arquivo_cookies = args.cookies
    cookies_from_browser = args.cookies_from_browser

    if not arquivo_cookies and not cookies_from_browser:
        for candidato in [
            Path("cookies.txt"),
            Path("youtube_cookies.txt"),
            Path("www.youtube.com_cookies.txt"),
            Path("transcricoes/cookies.txt"),
        ]:
            if candidato.is_file() and candidato.stat().st_size > 0:
                arquivo_cookies = candidato
                print(f"[Detectado] Usando arquivo: {candidato.resolve()}", flush=True)
                break

        if not arquivo_cookies and firefox_instalado_com_cookies():
            cookies_from_browser = "firefox"
            print("[Detectado] Cookies do YouTube encontrados no Firefox! Usando Firefox automaticamente.", flush=True)

    opcoes: dict[str, Any] = {
        "quiet": True,
        "no_warnings": False,
        "restrictfilenames": True,
        "windowsfilenames": True,
        "remote_components": ["ejs:github"],
    }

    node_caminho = shutil.which("node")
    if node_caminho:
        opcoes["js_runtimes"] = {"node": {"path": node_caminho}}

    if arquivo_cookies:
        if not arquivo_cookies.is_file():
            print(f"❌ [ERRO] O arquivo especificado nao existe: {arquivo_cookies}", file=sys.stderr)
            return 1
        opcoes["cookiefile"] = str(arquivo_cookies)
        print(f"Testando com arquivo de cookies: {arquivo_cookies.resolve()} ...")
    elif cookies_from_browser:
        if os.name == "nt" and any(b in cookies_from_browser.lower() for b in ("chrome", "edge", "brave")):
            print(
                f"⚠️ [Aviso] No Windows, o navegador '{cookies_from_browser}' costuma falhar "
                f"devido a App-Bound Encryption (erro DPAPI).",
                file=sys.stderr,
            )
        opcoes["cookiesfrombrowser"] = (cookies_from_browser, None, None, None)
        print(f"Testando com cookies do navegador: {cookies_from_browser} ...")
    else:
        print("⚠️ [Aviso] Nenhum cookie fornecido e nenhum arquivo 'cookies.txt' ou sessao do Firefox encontrada.")
        print("Testando conexao anonima (sem cookies)...")

    try:
        with YoutubeDL(opcoes) as ydl:
            info = ydl.extract_info(args.url, download=False)
        titulo = info.get("title") if info else "Desconhecido"
        duracao = int(info.get("duration") or 0) if info else 0
        minutos, segundos = divmod(duracao, 60)
        print("\n" + "=" * 60)
        print("✅ SUCESSO! A conexao com o YouTube funcionou perfeitamente.")
        print(f"   Video: {titulo}")
        print(f"   Duracao: {minutos:02d}:{segundos:02d}")
        print("=" * 60 + "\n")
        return 0
    except DownloadError as erro:
        msg = str(erro)
        print("\n" + "=" * 60, file=sys.stderr)
        print("❌ FALHA AO AUTENTICAR:", file=sys.stderr)
        if "Sign in to confirm you’re not a bot" in msg or "Sign in to confirm you're not a bot" in msg:
            print(" -> O YouTube esta exigindo login/cookies para este video/IP.", file=sys.stderr)
            print(" -> Se voce usou o Firefox, certifique-se de ter aberto o site do YouTube nele logado.", file=sys.stderr)
            print(" -> Se voce usou cookies.txt, verifique se o arquivo nao esta vazio.", file=sys.stderr)
        elif "Failed to decrypt with DPAPI" in msg:
            print(" -> O Windows impediu a leitura direta do Chrome/Edge (App-Bound Encryption).", file=sys.stderr)
            print(" -> Use o Firefox ou exporte o arquivo 'cookies.txt'.", file=sys.stderr)
        else:
            print(f" -> Detalhe: {msg}", file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)
        return 1
    except Exception as erro:
        print(f"\n❌ Erro inesperado: {type(erro).__name__}: {erro}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
