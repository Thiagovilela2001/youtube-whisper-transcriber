"""
Módulo de divisão semântica temporal (Chunking) otimizado para RAG com transcrições de vídeos.
"""

from typing import List, Dict, Any


def format_seconds(seconds: float) -> str:
    """Converte segundos para formato MM:SS ou HH:MM:SS."""
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def create_youtube_timestamp_url(webpage_url: str, start_seconds: float) -> str:
    """Gera a URL do YouTube com o parâmetro de tempo exato."""
    sec = int(start_seconds)
    if "watch?v=" in webpage_url:
        return f"{webpage_url}&t={sec}s"
    elif "youtu.be/" in webpage_url:
        return f"{webpage_url}?t={sec}s"
    return f"{webpage_url}&t={sec}s"


def chunk_transcript(
    transcript_data: Dict[str, Any],
    target_words: int = 250,
    overlap_words: int = 50,
) -> List[Dict[str, Any]]:
    """
    Divide os segmentos de uma transcrição em blocos semânticos temporais ideais para RAG.
    
    Estratégia:
    - Agrupa segmentos contínuos respeitando o fluxo natural da fala.
    - Mantém marcação precisa de início (start_time) e fim (end_time).
    - Aplica sobreposição (overlap) para que contextos que cruzam as fronteiras não se percam.
    - Enriquece o texto de entrada do embedding com título e intervalo temporal.
    """
    video_id = transcript_data.get("id", "")
    title = transcript_data.get("title", "Vídeo sem título")
    upload_date = transcript_data.get("upload_date", "")
    webpage_url = transcript_data.get("webpage_url", f"https://www.youtube.com/watch?v={video_id}")
    duration = transcript_data.get("duration", 0)
    segments = transcript_data.get("segments", [])

    if not segments:
        return []

    chunks = []
    current_segments = []
    current_word_count = 0
    seg_idx = 0
    n = len(segments)

    while seg_idx < n:
        seg = segments[seg_idx]
        text = seg.get("text", "").strip()
        words = text.split()
        num_words = len(words)

        current_segments.append(seg)
        current_word_count += num_words

        # Se atingiu o tamanho alvo ou chegou ao último segmento
        if current_word_count >= target_words or seg_idx == n - 1:
            chunk_start = current_segments[0]["start"]
            chunk_end = current_segments[-1]["end"]
            chunk_text = " ".join(s["text"].strip() for s in current_segments if s.get("text", "").strip())

            start_str = format_seconds(chunk_start)
            end_str = format_seconds(chunk_end)
            timestamp_url = create_youtube_timestamp_url(webpage_url, chunk_start)

            # Texto enriquecido para geração de embedding: inclui título e janela de tempo
            embedding_input = f"Título do Vídeo: {title}\nMomento: {start_str} até {end_str}\nConteúdo: {chunk_text}"

            chunks.append({
                "chunk_text": chunk_text,
                "embedding_input": embedding_input,
                "start_time": float(chunk_start),
                "end_time": float(chunk_end),
                "start_str": start_str,
                "end_str": end_str,
                "timestamp_url": timestamp_url,
                "word_count": len(chunk_text.split()),
                "char_count": len(chunk_text),
            })

            # Se chegamos ao fim dos segmentos, encerramos
            if seg_idx == n - 1:
                break

            # Lógica de overlap: retroceder alguns segmentos para manter sobreposição
            if overlap_words > 0:
                overlap_count = 0
                rewind_steps = 0
                for s in reversed(current_segments):
                    s_words = len(s.get("text", "").split())
                    overlap_count += s_words
                    rewind_steps += 1
                    if overlap_count >= overlap_words:
                        break

                # Garante que sempre avançamos pelo menos 1 segmento para evitar loop infinito
                if rewind_steps >= len(current_segments):
                    rewind_steps = len(current_segments) - 1

                current_segments = current_segments[-rewind_steps:] if rewind_steps > 0 else []
                current_word_count = sum(len(s.get("text", "").split()) for s in current_segments)
            else:
                current_segments = []
                current_word_count = 0

        seg_idx += 1

    # Atribui IDs e metadados finais com total de chunks
    total_chunks = len(chunks)
    result = []
    for i, c in enumerate(chunks):
        chunk_id = f"{video_id}_chunk_{i:04d}"
        metadata = {
            "video_id": str(video_id),
            "title": str(title),
            "upload_date": str(upload_date),
            "webpage_url": str(webpage_url),
            "timestamp_url": str(c["timestamp_url"]),
            "start_time": float(c["start_time"]),
            "end_time": float(c["end_time"]),
            "start_str": str(c["start_str"]),
            "end_str": str(c["end_str"]),
            "chunk_index": int(i),
            "total_chunks": int(total_chunks),
            "word_count": int(c["word_count"]),
            "char_count": int(c["char_count"]),
        }
        result.append({
            "id": chunk_id,
            "document": c["chunk_text"],
            "embedding_input": c["embedding_input"],
            "metadata": metadata,
        })

    return result
