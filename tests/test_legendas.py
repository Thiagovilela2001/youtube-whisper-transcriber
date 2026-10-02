"""Testes do parser de legendas WebVTT/SRT."""

from __future__ import annotations

import pytest

from legendas import _sem_repeticao, limpar_texto, ler_legenda, parse_vtt, tempo_para_segundos

VTT_SIMPLES = """WEBVTT

00:00:01.000 --> 00:00:04.000
Primeira frase da legenda.

00:00:04.000 --> 00:00:08.500
Segunda frase bem mais longa
que ocupa duas linhas.
"""


# --------------------------------------------------------------------------- #
# Tempo
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("00:00:01.500", 1.5),
        ("01:02:03.250", 3723.25),
        ("00:30.000", 30.0),
        ("01:00:00,000", 3600.0),  # vírgula em SRT
        ("72:00.000", 4320.0),
        ("lixo", 0.0),
    ],
)
def test_tempo_para_segundos(texto: str, esperado: float) -> None:
    assert tempo_para_segundos(texto) == pytest.approx(esperado)


# --------------------------------------------------------------------------- #
# Limpeza
# --------------------------------------------------------------------------- #

def test_limpar_remove_tags_e_entidades() -> None:
    assert limpar_texto("<c>Lei da &amp; Atra&ccedil;&atilde;o</c>") == "Lei da & Atração"


def test_limpar_remove_cursores_e_timestamps() -> None:
    assert limpar_texto("00:00:01.000 --> 00:00:02.000") == ""
    assert limpar_texto("{\\an8}texto") == "texto"


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def test_parse_vtt_basico() -> None:
    segmentos = parse_vtt(VTT_SIMPLES)
    assert len(segmentos) == 2
    assert segmentos[0]["text"] == "Primeira frase da legenda."
    assert segmentos[0]["start"] == pytest.approx(1.0)
    assert segmentos[0]["end"] == pytest.approx(4.0)
    assert "Segunda frase" in segmentos[1]["text"]


def test_parse_vtt_aceita_srt() -> None:
    srt = "1\n00:00:01,000 --> 00:00:04,000\nTexto do SRT.\n"
    segmentos = parse_vtt(srt)
    assert len(segmentos) == 1
    assert segmentos[0]["text"] == "Texto do SRT."

def test_parse_vtt_remove_repeticao_do_youtube() -> None:
    """
    As legendas automáticas do YouTube se sobrepõem na emenda. Sem remover a
    sobreposição, a mesma frase entra duas vezes no índice.
    """
    vtt = """WEBVTT

00:00:00.000 --> 00:00:03.000
o amor é a força que move

00:00:03.000 --> 00:00:06.000
o amor é a força que move todas as coisas

00:00:06.000 --> 00:00:09.000
para frente sem medo nenhum
"""
    segmentos = parse_vtt(vtt)
    assert [s["text"] for s in segmentos] == [
        "o amor é a força que move",
        "todas as coisas",
        "para frente sem medo nenhum",
    ]


def test_parse_vtt_remove_sobreposicao_parcial_na_emenda() -> None:
    """O caso real: a legenda nova começa repetindo o fim da anterior."""
    vtt = """WEBVTT

00:00:00.000 --> 00:00:03.000
a lei da atração funciona

00:00:03.000 --> 00:00:06.000
da atração funciona através das nossas escolhas
"""
    assert [s["text"] for s in parse_vtt(vtt)] == [
        "a lei da atração funciona",
        "através das nossas escolhas",
    ]


def test_legenda_100_por_cento_repetida_vira_nada() -> None:
    vtt = "WEBVTT\n\n" + "".join(
        f"00:00:{i:02d}.000 --> 00:00:{i + 1:02d}.000\nfrase igual\n\n" for i in range(5)
    )
    assert len(parse_vtt(vtt)) == 1


def test_parse_vtt_preserva_as_legendas_individualmente() -> None:
    """
    O parser não decide tamanho de bloco: quem faz isso é o `rag_chunker`.
    Aqui só importa que cada cue vire um segmento, na ordem.
    """
    vtt = "WEBVTT\n\n" + "".join(
        f"00:00:{i:02d}.000 --> 00:00:{i + 1:02d}.000\npalavra{i}\n\n" for i in range(20)
    )
    segmentos = parse_vtt(vtt)
    assert len(segmentos) == 20
    assert segmentos[0]["text"] == "palavra0"


def test_parse_vtt_ignora_blocos_note_e_style() -> None:
    vtt = """WEBVTT

NOTE isto é um comentário
que pode ocupar várias linhas

STYLE
::cue { color: red }

00:00:01.000 --> 00:00:02.000
Fala real.
"""
    assert [s["text"] for s in parse_vtt(vtt)] == ["Fala real."]


def test_parse_vtt_remove_indice_da_legenda() -> None:
    """No SRT o número vem na linha antes do tempo; no VTT, antes ou depois."""
    srt = "1\n00:00:01,000 --> 00:00:02,000\nTexto.\n"
    assert parse_vtt(srt)[0]["text"] == "Texto."

    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n1\nTexto.\n"
    assert parse_vtt(vtt)[0]["text"] == "Texto."


def test_parse_vtt_entradas_vazias() -> None:
    assert parse_vtt("") == []
    assert parse_vtt("WEBVTT\n") == []


def test_parse_vtt_descarta_blocos_vazios() -> None:
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n.\n\n00:00:02.000 --> 00:00:03.000\n.\n"
    assert parse_vtt(vtt) == []


def test_tempo_fim_nunca_menor_que_inicio() -> None:
    vtt = "WEBVTT\n\n00:00:05.000 --> 00:00:02.000\nTexto.\n"
    assert parse_vtt(vtt)[0]["end"] >= parse_vtt(vtt)[0]["start"]


def test_sem_repeticao() -> None:
    assert _sem_repeticao("primeira frase", "") == "primeira frase"
    assert _sem_repeticao("o amor move tudo", "o amor move tudo") is None
    assert _sem_repeticao("totalmente diferente", "o amor move tudo") == "totalmente diferente"


def test_ler_legenda_de_arquivo(tmp_path) -> None:
    arquivo = tmp_path / "legenda.vtt"
    arquivo.write_text(VTT_SIMPLES, encoding="utf-8")
    assert len(ler_legenda(arquivo)) == 2


def test_ler_legenda_com_bom(tmp_path) -> None:
    arquivo = tmp_path / "legenda.vtt"
    arquivo.write_bytes("\ufeffWEBVTT\n\n00:00:01.000 --> 00:00:02.000\nCom BOM.\n".encode())
    assert ler_legenda(arquivo)[0]["text"] == "Com BOM."
