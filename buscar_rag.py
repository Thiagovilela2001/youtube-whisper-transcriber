"""
Script para realizar buscas semânticas (RAG) no banco vetorial ChromaDB com transcrições do YouTube.
Exibe trecho encontrado, vídeo correspondente, minutagem exata e link com timestamp direto no YouTube.
"""

import argparse
import sys
from pathlib import Path
import chromadb

from embedding_engine import create_engine


def search_transcripts(
    query: str,
    top_k: int = 5,
    provider: str = "local",
    model: str = None,
    api_key: str = None,
    base_url: str = None,
    chroma_dir: str = "chroma_db",
    collection_name: str = "youtube_transcricoes",
    video_id_filter: str = None,
):
    chroma_path = Path(chroma_dir)
    if not chroma_path.exists():
        print(f"Erro: Banco ChromaDB não encontrado em '{chroma_dir}'.")
        print("Execute primeiro: python gerar_embeddings.py")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Erro ao acessar coleção '{collection_name}': {e}")
        sys.exit(1)

    total_docs = collection.count()
    if total_docs == 0:
        print(f"A coleção '{collection_name}' está vazia.")
        return

    # Inicializa engine para converter a query de busca no mesmo espaço vetorial 1536d
    engine = create_engine(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    query_embedding = engine.get_embeddings([query])[0]

    where_filter = None
    if video_id_filter:
        where_filter = {"video_id": video_id_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total_docs),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    print("\n" + "=" * 80)
    print(f"PERGUNTA / BUSCA: \"{query}\"")
    print(f"Total de registros na base: {total_docs} chunks")
    print("=" * 80)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        print("Nenhum resultado relevante encontrado.")
        return

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        # Distância cosseno: quanto menor a distância, maior a similaridade
        similarity = max(0.0, 1.0 - dist) * 100

        title = meta.get("title", "Sem título")
        upload_date = meta.get("upload_date", "")
        start_str = meta.get("start_str", "00:00")
        end_str = meta.get("end_str", "00:00")
        timestamp_url = meta.get("timestamp_url", "")
        chunk_idx = meta.get("chunk_index", 0)
        total_chunks = meta.get("total_chunks", 1)

        print(f"\n[Resultado #{rank}] - Similaridade: {similarity:.1f}%")
        print(f"  Vídeo:     {title}")
        if upload_date:
            print(f"  Data:      {upload_date}")
        print(f"  Momento:   {start_str} até {end_str} (Bloco {chunk_idx + 1} de {total_chunks})")
        print(f"  Link Direto: {timestamp_url}")
        print("  Trecho Transcrito:")
        print(f"    \"{doc.strip()}\"")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Busca semântica no banco vetorial ChromaDB com transcrições do YouTube."
    )
    parser.add_argument("query", type=str, help="Texto da pesquisa ou pergunta")
    parser.add_argument("--top-k", type=int, default=5, help="Quantidade de resultados a retornar")
    parser.add_argument(
        "--provider",
        type=str,
        default="local",
        choices=["local", "api", "openai", "infini-cloud", "stella"],
        help="Provedor de embedding para a query",
    )
    parser.add_argument("--model", type=str, default=None, help="Nome do modelo de embedding")
    parser.add_argument("--api-key", type=str, default=None, help="Chave de API (se usar provider 'api')")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL para API compatível com OpenAI")
    parser.add_argument("--chroma-dir", type=str, default="chroma_db", help="Diretório do ChromaDB")
    parser.add_argument("--collection-name", type=str, default="youtube_transcricoes", help="Nome da coleção")
    parser.add_argument("--video-id", type=str, default=None, help="Filtrar por ID específico de vídeo")

    args = parser.parse_args()

    search_transcripts(
        query=args.query,
        top_k=args.top_k,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        video_id_filter=args.video_id,
    )


if __name__ == "__main__":
    main()
