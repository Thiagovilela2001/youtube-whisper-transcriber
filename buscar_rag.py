#!/usr/bin/env python3
"""
Busca semântica (e lexical) nas transcrições indexadas.

Combina similaridade de cosseno com BM25 e funde os dois rankings por RRF.
Mostra o trecho, o vídeo, a minutagem e o link direto para o momento no YouTube.
"""

from __future__ import annotations

import argparse
from typing import List

from chroma_store import (
    IncompatibilidadeDeModelo,
    ResultadoBusca,
    abrir_colecao,
    buscar,
    diversificar,
)
from embedding_engine import create_engine


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Busca híbrida (vetorial + BM25) nas transcrições do YouTube."
    )
    parser.add_argument("query", help="Texto da pesquisa ou pergunta")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--provider", default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--chroma-dir", default="chroma_db")
    parser.add_argument("--collection-name", default="youtube_transcricoes")
    parser.add_argument("--video-id", default=None, help="Restringe a um vídeo específico")
    parser.add_argument(
        "--peso-lexical",
        type=float,
        default=1.0,
        help="Peso do BM25 na fusão RRF. 0 = só vetorial, 2 = dobra o peso.",
    )
    parser.add_argument(
        "--candidatos",
        type=int,
        default=50,
        help="Quantos resultados cada modality considera antes da fusão.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Descarta resultados com score RRF abaixo deste valor.",
    )
    parser.add_argument(
        "--por-video",
        type=int,
        default=0,
        help="Máximo de trechos de cada vídeo no resultado (0 = sem limite).",
    )
    return parser.parse_args()


def _origem(resultado: ResultadoBusca) -> str:
    if resultado.ranque_vetorial is not None and resultado.ranque_lexical is not None:
        return f"vetorial #{resultado.ranque_vetorial + 1} + lexical #{resultado.ranque_lexical + 1}"
    if resultado.ranque_vetorial is not None:
        return f"vetorial #{resultado.ranque_vetorial + 1}"
    return f"lexical #{resultado.ranque_lexical + 1}"


def mostrar(
    consulta: str,
    resultados: List[ResultadoBusca],
    total_base: int,
    modelo: str,
) -> None:
    print("\n" + "=" * 78)
    print(f'CONSULTA: "{consulta}"')
    print(f"Base: {total_base} blocos | modelo: {modelo}")
    print("=" * 78)

    if not resultados:
        print("Nenhum resultado acima do limiar. Tente --min-score 0 ou outras palavras.")
        return

    for posicao, resultado in enumerate(resultados, 1):
        meta = resultado.metadados
        # Distância cosseno: 0 = idêntico, 2 = oposto. Em texto, valores abaixo
        # de ~0.4 já indicam combinações fracas; por isso não se converte em
        # "percentual de similaridade", que daria uma leitura enganosa.
        distancia = (
            f"cosseno {resultado.distancia:.3f}" if resultado.distancia is not None else "lexical"
        )
        print(f"\n[{posicao}] {distancia} | RRF {resultado.score_rrf:.4f} | {_origem(resultado)}")
        print(f"  Vídeo:  {meta.get('title', '(sem título)')}")
        if meta.get("upload_date"):
            print(f"  Data:   {meta['upload_date']}")
        print(
            f"  Momento: {meta.get('start_str', '??:??')} → {meta.get('end_str', '??:??')}"
            f"  (bloco {int(meta.get('chunk_index', 0)) + 1}"
            f" de {int(meta.get('total_chunks', 1))})"
        )
        print(f"  Link:   {meta.get('timestamp_url', '')}")
        print("  Trecho:")
        print(f"    {resultado.documento.strip()}")
        print("-" * 78)


def main() -> int:
    args = argumentos()

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

    try:
        collection = abrir_colecao(
            args.chroma_dir, args.collection_name, engine.spec, criar=False
        )
    except IncompatibilidadeDeModelo as erro:
        print(f"\n{erro}")
        return 2
    except FileNotFoundError as erro:
        print(f"Erro: {erro}")
        return 1

    total = collection.count()
    if not total:
        print(f"A coleção '{args.collection_name}' está vazia.")
        return 1

    vetor = engine.embed_query(args.query)

    resultados = buscar(
        collection,
        vetor,
        args.query,
        candidatos=args.candidatos,
        hibrido=args.peso_lexical > 0,
        peso_lexical=args.peso_lexical,
        min_score=args.min_score,
        where={"video_id": args.video_id} if args.video_id else None,
        chroma_dir=args.chroma_dir,
    )

    if args.por_video > 0:
        resultados = diversificar(resultados, args.por_video)

    mostrar(args.query, resultados[: args.top_k], total, engine.spec.nome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
