"""
Módulo de divisão semântica temporal (Chunking) otimizado para RAG com transcrições de vídeos.

O chunker é agnóstico ao modelo de embedding: devolve o texto puro do bloco e
deixa o enriquecimento (prefixos de instrução) para `embedding_engine`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# Pontuação que indica fim de frase em português.
FIM_DE_FRASE = re.compile(r"[.!?…]+[\"'”’)\]]*\s*$")

# Marcadores de som/não-fala inseridos pelo Whisper, sem conteúdo útil para o RAG.
# A busca é por palavra-chave: o Whisper emite variações como "[MÚSICA DE FUNDO]",
# "(música instrumental)", "[trilha sonora]" e "[efeito sonoro]".
PALAVRA_NAO_FALA = re.compile(
    r"\b(m[úu]sica|music|aplaussos|aplausos|ris[ao]s?|laughter|"
    r"sil[êe]ncio|silence|barulho|noise|som|[âa]udio|intro|outro|vinheta|jingle|"
    r"cantarola|cantarolando|humming|hum|murm[úu]rio|murmur|shhh|"
    r"efeito|trilha|ambiente|transi[çc][ãa]o|sonora|sonoro|instrumental|fundo)\b",
    re.IGNORECASE,
)

# Delimitadores que marcam um trecho como anotação, não como fala.
ENVOLTIO = re.compile(r"^[\[\(*{<]+\s*(.*?)\s*[\]\)}*>]+$")

# Palavras que nunca iniciam um bloco: quebram a leitura e poluem a busca.
PALAVRAS_DE_LIGACAO = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "para", "pra", "com", "sem",
    "que", "se", "ao", "aos", "à", "às", "e", "é", "mas", "ou", "onde", "quando", "como",
    "porque", "pois", "então", "já", "mais", "muito", "também", "você", "eu",
    "isso", "isto", "aquilo", "entender",
}


def format_seconds(seconds: float) -> str:
    """Converte segundos para formato MM:SS ou HH:MM:SS."""
    segundos = int(round(max(0.0, float(seconds or 0.0))))
    horas, resto = divmod(segundos, 3600)
    minutos, segs = divmod(resto, 60)
    if horas > 0:
        return f"{horas:02d}:{minutos:02d}:{segs:02d}"
    return f"{minutos:02d}:{segs:02d}"


def create_youtube_timestamp_url(webpage_url: str, start_seconds: float) -> str:
    """
    Gera a URL do YouTube com o tempo exato, funcionando para `watch?v=`,
    `youtu.be/`, `/shorts/` e URLs com query string já existente.
    """
    url = str(webpage_url or "").strip()
    segundos = max(0, int(float(start_seconds or 0.0)))

    if not url:
        return f"https://www.youtube.com/watch?v=&t={segundos}s"

    partes = urlparse(url)
    parametros = dict(parse_qsl(partes.query, keep_blank_values=True))
    parametros["t"] = f"{segundos}s"

    return urlunparse(partes._replace(query=urlencode(parametros)))


def _e_ruido(texto: str) -> bool:
    """
    Detecta artefatos típicos de alucinação do Whisper em trechos com música
    ou silêncio: ". . . . .", "[Música]", "Thank you." repetido, etc.
    """
    limpo = texto.strip()
    if not limpo:
        return True
    if not any(caractere.isalnum() for caractere in limpo):
        return True
    # Anotação entre colchetes/parênteses que cita som ou música: "[MÚSICA DE
    # FUNDO]", "(música instrumental)", "[trilha sonora]". Uma frase real como
    # "A música é linda." não é anotação e precisa sobreviver.
    involve = ENVOLTIO.match(limpo)
    if involve and PALAVRA_NAO_FALA.search(involve.group(1)):
        return True
    # Sequência do mesmo token repetida ("muito muito muito" ou ". . . .").
    tokens = limpo.split()
    if len(tokens) >= 4 and len(set(tokens)) == 1:
        return True
    return False


def _segmentos_validos(transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extrai os segmentos com texto, normalizando `start`/`end` e descartando
    entradas incompletas. Sem isso, um único segmento sem `start` derrubava
    todo o processamento do vídeo com KeyError.
    """
    saida: List[Dict[str, Any]] = []
    for indice, seg in enumerate(transcript_data.get("segments") or []):
        if not isinstance(seg, dict):
            continue
        texto = str(seg.get("text") or "").strip()
        if not texto or _e_ruido(texto):
            continue
        try:
            inicio = float(seg.get("start", 0.0) or 0.0)
        except (TypeError, ValueError):
            inicio = 0.0
        try:
            fim = float(seg.get("end", inicio) or inicio)
        except (TypeError, ValueError):
            fim = inicio
        if fim < inicio:
            inicio, fim = fim, inicio
        saida.append({"start": inicio, "end": fim, "text": texto, "ordem": indice})
    return saida


def _corte_de_sentenca(segmentos: List[Dict[str, Any]], alvo: int, inicio: int) -> int:
    """
    Encontra o índice (exclusivo) mais próximo de `alvo` em que o bloco pode
    ser cortado sem quebrar uma frase.
    """
    total = len(segmentos)
    acumula = 0
    ultimo_ponto = -1

    for pos in range(inicio, total):
        texto = segmentos[pos]["text"]
        palavras = texto.split()
        # Nunca deixa um bloco com um único segmento gigante sem ponto final.
        if pos > inicio and acumula >= alvo:
            return pos
        acumula += len(palavras)
        if pos > inicio and FIM_DE_FRASE.search(texto):
            ultimo_ponto = pos + 1
            if acumula >= alvo:
                return ultimo_ponto

    # Nenhum ponto de corte satisfatório: volta ao último ponto de frase válido.
    if ultimo_ponto > inicio:
        return ultimo_ponto
    return min(inicio + 1, total)


def _fundir_blocos_curtos(
    brutos: List[Dict[str, Any]], min_palavras: int
) -> List[Dict[str, Any]]:
    """
    Junta blocos curtos demais ao vizinho mais próximo.

    Resíduos de transcrição ("2026", "No.", "[cantarola]") viravam chunks de
    1 palavra que poluem o índice e não ajudam nenhuma busca. O bloco é
    absorvido pelo anterior quando existe, ou pelo seguinte quando é o primeiro.
    """
    if min_palavras <= 0 or len(brutos) < 2:
        return brutos

    resultado: List[Dict[str, Any]] = []
    for bloco in brutos:
        palavras = len(bloco["chunk_text"].split())
        if palavras < min_palavras and resultado:
            anterior = resultado[-1]
            anterior["chunk_text"] = f"{anterior['chunk_text']} {bloco['chunk_text']}"
            anterior["end_time"] = float(bloco["end_time"])
            anterior["end_seg"] = bloco["end_seg"]
            continue
        resultado.append(bloco)

    # Se o primeiro bloco ficou curto demais, ele é absorvido pelo segundo.
    while len(resultado) > 1 and len(resultado[0]["chunk_text"].split()) < min_palavras:
        primeiro = resultado.pop(0)
        segundo = resultado[0]
        segundo["chunk_text"] = f"{primeiro['chunk_text']} {segundo['chunk_text']}"
        segundo["start_time"] = float(primeiro["start_time"])
        segundo["start_seg"] = primeiro["start_seg"]

    return resultado


def chunk_transcript(
    transcript_data: Dict[str, Any],
    target_words: int = 160,
    overlap_words: int = 30,
    min_chunk_words: int = 25,
) -> List[Dict[str, Any]]:
    """
    Divide os segmentos de uma transcrição em blocos semânticos temporais para RAG.

    Estratégia:
    - Agrupa segmentos contínuos respeitando o fluxo natural da fala.
    - Corta preferencialmente em fim de frase, evitando blocos quebrados no meio.
    - Mantém marcação precisa de início (`start_time`) e fim (`end_time`).
    - Aplica sobreposição mínima, descartando palavras de ligação na fronteira
      para não repetir o mesmo texto em dois vetores.
    - Devolve o texto puro; o prefixo de instrução fica a cargo do motor de
      embeddings, que é quem conhece as exigências do modelo.
    """
    video_id = str(transcript_data.get("id") or "")
    title = str(transcript_data.get("title") or "Vídeo sem título")
    upload_date = str(transcript_data.get("upload_date") or "")
    webpage_url = str(
        transcript_data.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"
    )

    segmentos = _segmentos_validos(transcript_data)
    if not segmentos:
        return []

    target_words = max(20, int(target_words))
    overlap_words = max(0, int(overlap_words))

    brutos: List[Dict[str, Any]] = []
    pos = 0
    while pos < len(segmentos):
        corte = _corte_de_sentenca(segmentos, target_words, pos)
        bloco = segmentos[pos:corte]
        texto = " ".join(s["text"] for s in bloco)
        if texto:
            brutos.append(
                {
                    "chunk_text": texto,
                    "start_time": float(bloco[0]["start"]),
                    "end_time": float(bloco[-1]["end"]),
                    "start_seg": pos,
                    "end_seg": corte,
                }
            )
        if corte <= pos:
            corte = pos + 1
        pos = corte

    if not brutos:
        return []

    brutos = _fundir_blocos_curtos(brutos, min_palavras=min_chunk_words)

    # Sobreposição: reaproveita o final do bloco anterior como abertura do próximo,
    # mas descarta a primeira palavra de cada segmento herdado quando ela for uma
    # palavra de ligação, evitando "e a gente" colado no fim do trecho anterior.
    if overlap_words > 0 and len(brutos) > 1:
        for indice in range(1, len(brutos)):
            atual = brutos[indice]
            inicio_herdado = atual["start_seg"]

            palavras_herdadas = 0
            novo_inicio = inicio_herdado
            for pos_seg in range(atual["start_seg"], atual["end_seg"]):
                palavras_herdadas += len(segmentos[pos_seg]["text"].split())
                novo_inicio = pos_seg + 1
                if palavras_herdadas >= overlap_words:
                    break

            if novo_inicio <= inicio_herdado or novo_inicio >= atual["end_seg"]:
                continue

            herdado = [dict(s) for s in segmentos[inicio_herdado:novo_inicio]]
            if not herdado:
                continue

            # Corta a palavra de ligação que ficou pendurada na fronteira.
            primeira_palavra = herdado[0]["text"].split()
            if (
                len(herdado) > 1
                and primeira_palavra
                and primeira_palavra[0].lower().strip(",.;:") in PALAVRAS_DE_LIGACAO
            ):
                herdado = herdado[1:]
            if not herdado:
                continue

            combinado = herdado + segmentos[novo_inicio:atual["end_seg"]]
            atual["chunk_text"] = " ".join(s["text"] for s in combinado)
            atual["start_time"] = float(herdado[0]["start"])
            atual["start_seg"] = inicio_herdado

    total_chunks = len(brutos)
    resultado: List[Dict[str, Any]] = []

    for indice, bloco in enumerate(brutos):
        texto = bloco["chunk_text"]
        inicio = bloco["start_time"]
        fim = bloco["end_time"]
        inicio_str = format_seconds(inicio)
        fim_str = format_seconds(fim)

        resultado.append(
            {
                "id": f"{video_id}_chunk_{indice:04d}",
                "document": texto,
                "embedding_input": texto,
                "metadata": {
                    "video_id": video_id,
                    "title": title,
                    "upload_date": upload_date,
                    "webpage_url": webpage_url,
                    "timestamp_url": create_youtube_timestamp_url(webpage_url, inicio),
                    "start_time": float(inicio),
                    "end_time": float(fim),
                    "start_str": inicio_str,
                    "end_str": fim_str,
                    "chunk_index": int(indice),
                    "total_chunks": int(total_chunks),
                    "word_count": int(len(texto.split())),
                    "char_count": int(len(texto)),
                },
            }
        )

    return resultado
