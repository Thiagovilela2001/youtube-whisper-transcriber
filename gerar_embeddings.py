#!/usr/bin/env python3
"""
Gera embeddings das transcrições e indexa no ChromaDB.

O processo é incremental: um manifesto em disco registra o que já foi
indexado, evitando tanto reprocessar tudo quanto varrer a coleção inteira a
cada execução.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from busca_lexical import invalidar_cache
from chroma_store import (
    IncompatibilidadeDeModelo,
    abrir_colecao,
    ler_manifesto,
    reconstruir_manifesto,
    salvar_manifesto,
)
from embedding_engine import create_engine
from rag_chunker import chunk_transcript

VERSAO_CHUNKER = 2


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera embeddings das transcrições do YouTube e salva no ChromaDB."
    )
    parser.add_argument(
        "--provider",
        default="local",
        help="'local' (sentence-transformers) ou 'api'/'openai'.",
    )
    parser.add_argument("--model", default=None, help="Nome do modelo de embeddings.")
    parser.add_argument("--api-key", default=None, help="Chave de API, se provider for api.")
    parser.add_argument("--base-url", default=None, help="Base URL compatível com OpenAI.")
    parser.add_argument("--device", default=None, help="'cpu' ou 'cuda' para o modelo local.")
    parser.add_argument("--transcricoes-dir", default="transcricoes")
    parser.add_argument("--chroma-dir", default="chroma_db")
    parser.add_argument("--collection-name", default="youtube_transcricoes")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Processa só N vídeos (teste).")
    parser.add_argument("--export-jsonl", default=None, help="Exporta os embeddings em JSONL.")
    parser.add_argument(
        "--force", action="store_true", help="Reindexa os vídeos que já estão na base."
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Apaga a coleção e recria do zero (use ao trocar de modelo ou de chunking).",
    )
    parser.add_argument("--target-words", type=int, default=160, help="Palavras alvo por bloco.")
    parser.add_argument("--overlap-words", type=int, default=30, help="Sobreposição entre blocos.")
    parser.add_argument(
        "--min-chunk-words",
        type=int,
        default=25,
        help="Blocos com menos palavras são fundidos ao vizinho.",
    )
    return parser.parse_args()


def _mtime(caminho: Path) -> float:
    try:
        return caminho.stat().st_mtime
    except OSError:
        return 0.0


def selecionar_videos(
    arquivos: List[Path],
    manifesto: Dict[str, Any],
    forcar: bool,
) -> Tuple[List[Tuple[Path, Dict[str, Any]]], int, int]:
    """
    Decide o que processar. Um vídeo é reprocessado se ainda não existir no
    manifesto, se a transcrição for mais nova que o registro, ou se `--force`.
    """
    conhecidos: Dict[str, Any] = manifesto.get("videos") or {}
    a_processar: List[Tuple[Path, Dict[str, Any]]] = []
    alterados = 0
    ignorados = 0

    for arquivo in arquivos:
        try:
            with arquivo.open("r", encoding="utf-8") as ponte:
                dados = json.load(ponte)
        except (OSError, json.JSONDecodeError) as erro:
            print(f"[Aviso] Ignorando {arquivo.name}: {erro}")
            continue

        video_id = str(dados.get("id") or arquivo.stem)
        registro = conhecidos.get(video_id)
        desatualizado = registro is not None and registro.get("chunker") != VERSAO_CHUNKER

        if registro and not forcar and not desatualizado:
            if _mtime(arquivo) <= float(registro.get("mtime") or 0.0):
                ignorados += 1
                continue
            alterados += 1
        elif desatualizado:
            alterados += 1

        a_processar.append((arquivo, dados))

    return a_processar, alterados, ignorados


def _remover_chunks_antigos(collection: Any, video_id: str) -> None:
    """
    Apaga todos os blocos de um vídeo antes de reinseri-lo.

    Sem isso, um `upsert` por id deixaria para trás os blocos de índices
    maiores quando a transcrição passasse a gerar menos blocos.
    """
    try:
        collection.delete(where={"video_id": video_id})
    except Exception as erro:
        print(f"[Aviso] Não foi possível limpar os blocos antigos de {video_id}: {erro}")


def main() -> int:
    args = argumentos()

    pasta_transcricoes = Path(args.transcricoes_dir)
    if not pasta_transcricoes.exists():
        print(f"Erro: '{args.transcricoes_dir}' não encontrado.")
        return 1

    arquivos = sorted(pasta_transcricoes.glob("*.json"))
    print(f"Transcrições encontradas: {len(arquivos)}")
    if not arquivos:
        print("Nenhum .json para processar.")
        return 0

    try:
        engine = create_engine(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            device=args.device,
        )
    except Exception as erro:
        print(f"Falha ao iniciar o motor de embeddings: {erro}")
        return 1

    spec = engine.spec
    print(f"Modelo: {spec.resumo()}")
    if spec.nota:
        print(f"Contrato: {spec.nota}")

    pasta_chroma = Path(args.chroma_dir)
    if args.reindex:
        import chromadb

        cliente = chromadb.PersistentClient(path=str(pasta_chroma))
        try:
            cliente.delete_collection(args.collection_name)
            print(f"[Reindex] Coleção '{args.collection_name}' removida.")
        except Exception:
            pass
        invalidar_cache(pasta_chroma)
        try:
            (pasta_chroma / ".manifesto.json").unlink(missing_ok=True)
        except OSError:
            pass

    try:
        collection = abrir_colecao(
            pasta_chroma,
            args.collection_name,
            spec,
            criar=True,
            permitir_reindexacao=args.reindex,
        )
    except IncompatibilidadeDeModelo as erro:
        print(f"\n{erro}")
        return 2

    manifesto = ler_manifesto(pasta_chroma)
    if not manifesto.get("videos") and collection.count() > 0:
        print("[Manifesto] Ausente; reconstruindo a partir da coleção existente...")
        manifesto = reconstruir_manifesto(collection)
        salvar_manifesto(pasta_chroma, manifesto)

    a_processar, alterados, ignorados = selecionar_videos(arquivos, manifesto, args.force)
    print(f"A indexar: {len(a_processar)} | já atuais: {ignorados} | transcrição alterada: {alterados}")
    if args.limit and args.limit > 0:
        a_processar = a_processar[: args.limit]
    if not a_processar:
        print("Nada a fazer. Use --force ou --reindex para reprocessar.")
        return 0

    jsonl = None
    if args.export_jsonl:
        jsonl = Path(args.export_jsonl).open("a", encoding="utf-8")

    videos_registro: Dict[str, Any] = dict(manifesto.get("videos") or {})
    total_chunks = 0
    inicio = time.time()
    processados = 0

    def gravar_manifesto() -> None:
        salvar_manifesto(
            pasta_chroma,
            {
                "videos": videos_registro,
                "modelo": spec.nome,
                "fingerprint": spec.fingerprint,
                "dimensao": spec.dimensao,
                "chunker": VERSAO_CHUNKER,
                "atualizado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )

    try:
        with tqdm(total=len(a_processar), desc="Indexando", unit="vídeo") as barra:
            for arquivo, dados in a_processar:
                video_id = str(dados.get("id") or arquivo.stem)
                titulo = str(dados.get("title") or arquivo.stem)

                blocos = chunk_transcript(
                    dados,
                    target_words=args.target_words,
                    overlap_words=args.overlap_words,
                    min_chunk_words=args.min_chunk_words,
                )
                if not blocos:
                    barra.set_postfix_str("sem texto")
                    barra.update(1)
                    continue

                embeddings = engine.embed_documents(
                    [b["document"] for b in blocos], batch_size=args.batch_size
                )

                _remover_chunks_antigos(collection, video_id)
                collection.upsert(
                    ids=[b["id"] for b in blocos],
                    embeddings=embeddings,
                    documents=[b["document"] for b in blocos],
                    metadatas=[b["metadata"] for b in blocos],
                )

                if jsonl:
                    for bloco, vetor in zip(blocos, embeddings):
                        jsonl.write(
                            json.dumps(
                                {
                                    "id": bloco["id"],
                                    "embedding": vetor,
                                    "document": bloco["document"],
                                    "metadata": bloco["metadata"],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    jsonl.flush()

                videos_registro[video_id] = {
                    "chunks": len(blocos),
                    "mtime": _mtime(arquivo),
                    "chunker": VERSAO_CHUNKER,
                    "fingerprint": spec.fingerprint,
                }
                total_chunks += len(blocos)
                processados += 1
                barra.set_postfix_str(f"+{len(blocos)} | {titulo[:24]}")
                barra.update(1)

                # Grava o manifesto a cada 25 vídeos: um reindex completo leva
                # mais de uma hora, e perdê-lo significa refazer tudo. O upsert é
                # idempotente, então o pior caso é trabalho repetido, não
                # base corrompida.
                if processados % 25 == 0:
                    gravar_manifesto()
                    invalidar_cache(pasta_chroma)
    finally:
        if jsonl:
            jsonl.close()

    gravar_manifesto()
    invalidar_cache(pasta_chroma)

    print("\n" + "=" * 60)
    print(f"Vídeos processados:   {len(a_processar)}")
    print(f"Blocos indexados:     {total_chunks}")
    print(f"Total na coleção:     {collection.count()}")
    print(f"Modelo:               {spec.nome} ({spec.dimensao}d, fp={spec.fingerprint})")
    print(f"Banco:                {pasta_chroma.resolve()}")
    print(f"Tempo:                {time.time() - inicio:.1f}s")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
