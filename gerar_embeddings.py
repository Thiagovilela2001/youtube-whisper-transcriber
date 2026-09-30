"""
Script principal para gerar embeddings de 1536 dimensões e indexar no banco vetorial ChromaDB.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Set

import chromadb
from tqdm import tqdm

from rag_chunker import chunk_transcript
from embedding_engine import create_engine, EXPECTED_DIMENSION


def get_existing_video_ids(collection: chromadb.Collection) -> Set[str]:
    """Obtém conjunto de IDs de vídeos que já foram indexados no ChromaDB."""
    try:
        results = collection.get(include=["metadatas"])
        existing_ids = set()
        for meta in results.get("metadatas", []):
            if meta and "video_id" in meta:
                existing_ids.add(meta["video_id"])
        return existing_ids
    except Exception as e:
        print(f"[Aviso] Não foi possível verificar vídeos existentes: {e}")
        return set()


def main():
    parser = argparse.ArgumentParser(
        description="Gera embeddings de 1536 dimensões para transcrições do YouTube e salva no ChromaDB."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="local",
        choices=["local", "api", "openai", "infini-cloud", "stella"],
        help="Provedor de embeddings ('local' com SentenceTransformers ou 'api' / 'openai').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Nome do modelo (ex: 'Alibaba-NLP/gte-Qwen2-1.5B-instruct', 'infly/inf-retriever-v1-1.5b', 'text-embedding-3-small').",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Chave de API (se usar provider 'api' ou 'openai').",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL para API compatível com OpenAI (ex: 'https://cloud.infini-ai.com/maas/v1').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Dispositivo para execução local ('cpu' ou 'cuda').",
    )
    parser.add_argument(
        "--transcricoes-dir",
        type=str,
        default="transcricoes",
        help="Diretório onde estão os arquivos .json das transcrições.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default="chroma_db",
        help="Diretório local para persistência do banco ChromaDB.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="youtube_transcricoes",
        help="Nome da coleção no ChromaDB.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Tamanho do lote para inferência de embeddings.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite de vídeos a processar (útil para testes rápidos).",
    )
    parser.add_argument(
        "--export-jsonl",
        type=str,
        default=None,
        help="Caminho opcional para exportar os embeddings gerados em formato JSONL.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Força reprocessamento de vídeos que já estejam no banco.",
    )

    args = parser.parse_args()

    transcricoes_path = Path(args.transcricoes_dir)
    if not transcricoes_path.exists():
        print(f"Erro: Diretório de transcrições '{args.transcricoes_dir}' não encontrado.")
        sys.exit(1)

    # 1. Encontrar todos os arquivos JSON
    json_files = sorted(list(transcricoes_path.glob("*.json")))
    print(f"Total de arquivos de transcrição encontrados: {len(json_files)}")

    if not json_files:
        print("Nenhum arquivo .json encontrado para processar.")
        sys.exit(0)

    # 2. Inicializar banco vetorial ChromaDB
    chroma_path = Path(args.chroma_dir)
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))

    # Cria ou obtém coleção configurada com distância cosseno
    collection = client.get_or_create_collection(
        name=args.collection_name,
        metadata={"hnsw:space": "cosine", "embedding_dimensions": EXPECTED_DIMENSION},
    )

    existing_videos = set() if args.force else get_existing_video_ids(collection)
    if existing_videos:
        print(f"Vídeos já indexados anteriormente no ChromaDB: {len(existing_videos)}")

    # Filtra vídeos que ainda precisam ser processados
    files_to_process = []
    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                vid_id = data.get("id", f.stem)
                if args.force or vid_id not in existing_videos:
                    files_to_process.append((f, data))
        except Exception as e:
            print(f"Erro ao ler {f.name}: {e}")

    if args.limit and args.limit > 0:
        files_to_process = files_to_process[: args.limit]

    print(f"Vídeos selecionados para processamento nesta rodada: {len(files_to_process)}")
    if not files_to_process:
        print("Todos os vídeos já foram indexados! Use --force para reprocessar se desejar.")
        return

    # 3. Inicializar Engine de Embeddings
    print("\nInicializando motor de embeddings (1536 dimensões)...")
    try:
        engine = create_engine(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            device=args.device,
        )
    except Exception as e:
        print(f"Falha ao inicializar o motor de embeddings: {e}")
        sys.exit(1)

    # 4. Processar vídeos e gerar embeddings
    total_chunks_indexed = 0
    start_time = time.time()

    jsonl_fp = None
    if args.export_jsonl:
        jsonl_fp = open(args.export_jsonl, "a", encoding="utf-8")

    print("\nIniciando geração de embeddings e inserção no ChromaDB...\n")

    try:
        with tqdm(total=len(files_to_process), desc="Processando vídeos", unit="vídeo") as pbar:
            for file_path, data in files_to_process:
                vid_id = data.get("id", file_path.stem)
                title = data.get("title", file_path.stem)

                # Divide o vídeo em blocos ideais para RAG com timestamps
                chunks = chunk_transcript(data)
                if not chunks:
                    pbar.set_postfix_str(f"Ignorado (sem texto): {title[:30]}")
                    pbar.update(1)
                    continue

                texts_to_embed = [c["embedding_input"] for c in chunks]
                chunk_ids = [c["id"] for c in chunks]
                documents = [c["document"] for c in chunks]
                metadatas = [c["metadata"] for c in chunks]

                # Gera os vetores de 1536 dimensões
                embeddings = engine.get_embeddings(texts_to_embed, batch_size=args.batch_size)

                # Se force=True, remove eventuais chunks antigos deste vídeo no ChromaDB
                if args.force:
                    try:
                        collection.delete(where={"video_id": vid_id})
                    except Exception:
                        pass

                # Insere no ChromaDB
                collection.upsert(
                    ids=chunk_ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )

                # Exporta para JSONL se solicitado
                if jsonl_fp:
                    for cid, emb, doc, meta in zip(chunk_ids, embeddings, documents, metadatas):
                        line = {
                            "id": cid,
                            "embedding": emb,
                            "document": doc,
                            "metadata": meta,
                        }
                        jsonl_fp.write(json.dumps(line, ensure_ascii=False) + "\n")
                    jsonl_fp.flush()

                total_chunks_indexed += len(chunks)
                pbar.set_postfix_str(f"+{len(chunks)} chunks | {title[:25]}...")
                pbar.update(1)

    finally:
        if jsonl_fp:
            jsonl_fp.close()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f"- Total de vídeos processados: {len(files_to_process)}")
    print(f"- Total de chunks indexados: {total_chunks_indexed}")
    print(f"- Dimensão dos vetores: {EXPECTED_DIMENSION} dimensões")
    print(f"- Banco ChromaDB salvo em: {chroma_path.resolve()}")
    print(f"- Coleção ChromaDB: {args.collection_name}")
    print(f"- Total de itens na coleção agora: {collection.count()}")
    if args.export_jsonl:
        print(f"- Arquivo JSONL exportado em: {Path(args.export_jsonl).resolve()}")
    print(f"- Tempo decorrido: {elapsed:.1f} segundos")
    print("=" * 60)


if __name__ == "__main__":
    main()
