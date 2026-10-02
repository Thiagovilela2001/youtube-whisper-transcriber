"""Testes do chunker: cobertura dos bugs corrigidos e das invariantes do RAG."""

from __future__ import annotations

import pytest

from rag_chunker import chunk_transcript, create_youtube_timestamp_url, format_seconds


def _segmento(texto: str, inicio: float, fim: float) -> dict:
    return {"start": inicio, "end": fim, "text": texto}


# --------------------------------------------------------------------------- #
# create_youtube_timestamp_url
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("url", "segundos", "esperado"),
    [
        ("https://www.youtube.com/watch?v=abc", 90, "https://www.youtube.com/watch?v=abc&t=90s"),
        # Antes gerava ".../shorts/abc&t=90s", sem "?", ou seja, um link quebrado.
        ("https://www.youtube.com/shorts/abc", 90, "https://www.youtube.com/shorts/abc?t=90s"),
        ("https://youtu.be/abc", 3725, "https://youtu.be/abc?t=3725s"),
        # Não pode duplicar nem perder parâmetros já existentes.
        (
            "https://www.youtube.com/watch?v=abc&list=PL1",
            125,
            "https://www.youtube.com/watch?v=abc&list=PL1&t=125s",
        ),
        # Um "t" antigo deve ser substituído, não accumulating.
        (
            "https://www.youtube.com/watch?v=abc&t=10s",
            60,
            "https://www.youtube.com/watch?v=abc&t=60s",
        ),
    ],
)
def test_timestamp_url(url: str, segundos: int, esperado: str) -> None:
    assert create_youtube_timestamp_url(url, segundos) == esperado


def test_timestamp_url_aceita_lista_python_de_valores() -> None:
    """segmentos do Whisper trazem float; o id não pode virar '90.0s'."""
    assert create_youtube_timestamp_url("https://youtu.be/x", 90.7).endswith("t=90s")


def test_timestamp_url_sem_url() -> None:
    assert create_youtube_timestamp_url("", 5) == "https://www.youtube.com/watch?v=&t=5s"


# --------------------------------------------------------------------------- #
# format_seconds
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("segundos", "esperado"),
    [
        (0, "00:00"),
        (59, "00:59"),
        (60, "01:00"),
        (3661, "01:01:01"),
        (7325, "02:02:05"),
        # Antes produzia "00:-1" emyar negative.
        (-5, "00:00"),
    ],
)
def test_format_seconds(segundos: float, esperado: str) -> None:
    assert format_seconds(segundos) == esperado


# --------------------------------------------------------------------------- #
# Robustez: segmentos incompletos
# --------------------------------------------------------------------------- #

def test_segmento_sem_timestamps_nao_quebra() -> None:
    """Regressão: `seg["start"]` levantava KeyError e derrubava o vídeo inteiro."""
    dados = {"id": "abc", "title": "t", "segments": [{"text": "primeiro"}, {"text": "segundo"}]}
    blocos = chunk_transcript(dados)
    assert len(blocos) == 1
    assert "primeiro" in blocos[0]["document"]


def test_timestamps_invertidos_sao_normalizados() -> None:
    dados = {"id": "abc", "segments": [_segmento("texto aqui", 50.0, 10.0)]}
    bloco = chunk_transcript(dados)[0]
    assert bloco["metadata"]["start_time"] <= bloco["metadata"]["end_time"]


def test_timestamps_ilegiveis_caem_para_zero() -> None:
    dados = {"id": "abc", "segments": [{"text": "oi", "start": "invalido", "end": None}]}
    assert chunk_transcript(dados)[0]["metadata"]["start_time"] == 0.0


def test_entradas_invalidas_sao_descartadas() -> None:
    dados = {
        "id": "abc",
        "segments": [None, 123, "texto solto", {"text": "   "}, {"text": "conteúdo real"}],
    }
    blocos = chunk_transcript(dados)
    assert len(blocos) == 1
    assert blocos[0]["document"] == "conteúdo real"


def test_video_sem_segmentos() -> None:
    assert chunk_transcript({"id": "abc", "segments": []}) == []
    assert chunk_transcript({"id": "abc"}) == []


# --------------------------------------------------------------------------- #
# Ruído de transcrição
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "texto",
    [
        ". . . .",
        "[MÚSICA]",
        "[MÚSICA DE FUNDO]",
        "(música instrumental)",
        "[musica ao fundo]",
        "[TRILHA SONORA]",
        "[Efeito sonoro]",
        "(aplausos)",
        "*risos*",
        "[cantarola]",
        "[SILÊNCIO]",
        "[VINHETA]",
        "  ",
        "muito muito muito muito",
    ],
)
def test_ruido_whisper_e_descartado(texto: str) -> None:
    dados = {"id": "abc", "segments": [_segmento(texto, 0.0, 1.0), _segmento("fala real", 1.0, 2.0)]}
    blocos = chunk_transcript(dados)
    assert len(blocos) == 1
    assert blocos[0]["document"] == "fala real"


@pytest.mark.parametrize(
    "texto",
    [
        # Fala que menciona música/silêncio não pode ser descartada.
        "A música é linda.",
        "O som do violão era lovely.",
        "Silêncio é resposta.",
        "Ele subiu as escadas do riso.",
        "369",
        "Portal 12 12",
        # Anotações entre colchetes que não são som.
        "[BLÁ BLÁ]",
        "(2 + 2) = 4",
    ],
)
def test_texto_legitimo_nao_e_confundido_com_ruido(texto: str) -> None:
    dados = {
        "id": "abc",
        "segments": [_segmento(texto, 0.0, 5.0), _segmento(".", 5.0, 6.0), _segmento("mais", 6.0, 9.0)],
    }
    blocos = chunk_transcript(dados)
    assert texto in blocos[0]["document"]


# --------------------------------------------------------------------------- #
# Invariantes dos blocos
# --------------------------------------------------------------------------- #

def _transcricao_longa(segmentos: int = 60) -> dict:
    return {
        "id": "video1",
        "title": "Vídeo de teste",
        "upload_date": "20240101",
        "webpage_url": "https://www.youtube.com/watch?v=video1",
        "segments": [
            _segmento(f"Frase número {i} com conteúdo suficiente para o teste.", i * 10.0, i * 10.0 + 9.0)
            for i in range(segmentos)
        ],
    }


def test_blocos_tem_metadados_completos() -> None:
    blocos = chunk_transcript(_transcricao_longa())
    assert blocos
    for indice, bloco in enumerate(blocos):
        meta = bloco["metadata"]
        assert meta["video_id"] == "video1"
        assert meta["chunk_index"] == indice
        assert meta["total_chunks"] == len(blocos)
        assert meta["start_str"] and meta["end_str"]
        assert meta["timestamp_url"].startswith("https://www.youtube.com/watch?v=video1&t=")
        assert meta["word_count"] == len(bloco["document"].split())


def test_ids_sao_estaveis_e_unicos() -> None:
    blocos = chunk_transcript(_transcricao_longa())
    ids = [b["id"] for b in blocos]
    assert len(set(ids)) == len(ids)
    assert ids[0] == "video1_chunk_0000"
    # Reexecutar sem mudar a entrada precisa produzir os mesmos ids (upsert seguro).
    assert [b["id"] for b in chunk_transcript(_transcricao_longa())] == ids


def test_tempo_avanca_sem_inverter() -> None:
    blocos = chunk_transcript(_transcricao_longa())
    for anterior, seguinte in zip(blocos, blocos[1:]):
        assert anterior["metadata"]["end_time"] <= seguinte["metadata"]["start_time"] + 1e-6


def test_blocos_respeitam_o_tamanho_alvo() -> None:
    blocos = chunk_transcript(_transcricao_longa(segmentos=120), target_words=80, overlap_words=0)
    for bloco in blocos:
        assert bloco["metadata"]["word_count"] <= 140


def test_blocos_curtos_sao_fundidos() -> None:
    """Resíduo de transcrição não pode virar um bloco de uma palavra."""
    segmentos = [_segmento(f"Frase {i} com palavras.", i * 5.0, i * 5.0 + 4.0) for i in range(20)]
    segmentos.append(_segmento("369", 100.0, 101.0))
    blocos = chunk_transcript({"id": "v", "segments": segmentos}, min_chunk_words=25)
    assert all(bloco["metadata"]["word_count"] >= 25 for bloco in blocos)


def test_sobreposicao_nao_repeti_o_bloco_inteiro() -> None:
    """A sobreposição existe para dar contexto, não para duplicar o índice."""
    blocos = chunk_transcript(_transcricao_longa(), overlap_words=30)
    for anterior, seguinte in zip(blocos, blocos[1:]):
        palavras_anteriores = set(anterior["document"].split())
        palavras_seguintes = set(seguinte["document"].split())
        # No máximo ~30 palavras podem se repetir entre blocos vizinhos.
        assert len(palavras_anteriores & palavras_seguintes) <= 35


def test_texto_do_embedding_nao_tem_prefixo_de_modelo() -> None:
    """
    O chunker é agnóstico: o prefixo de instrução é do motor de embeddings.
    Repetir o título aqui diluiria o sinal da busca.
    """
    bloco = chunk_transcript(_transcricao_longa())[0]
    assert bloco["embedding_input"] == bloco["document"]
    assert "Título do Vídeo" not in bloco["document"]


def test_sem_overlap_gera_blocos_independentes() -> None:
    """
    Sem sobreposição, cada segmento deve aparecer em exatamente um bloco.
    Usa um marcador único por segmento, senão as palavras comuns se confundem.
    """
    segmentos = [
        _segmento(f"marcador{i:03d} texto adicional para o bloco.", i * 10.0, i * 10.0 + 9.0)
        for i in range(60)
    ]
    blocos = chunk_transcript(
        {"id": "v", "segments": segmentos}, overlap_words=0, min_chunk_words=0
    )
    assert len(blocos) > 1
    for i in range(60):
        ocorrencias = sum(f"marcador{i:03d}" in b["document"] for b in blocos)
        assert ocorrencias == 1, f"marcador{i:03d} apareceu {ocorrencias}x"
