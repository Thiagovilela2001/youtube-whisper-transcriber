"""Testes do BM25 local e da fusão RRF da busca híbrida."""

from __future__ import annotations

import pytest

from busca_lexical import IndiceLexical, normalizar, tokenizar

DOCUMENTOS = {
    "a": "A tecnica 369 ativa a lei da atracao e traz abundancia",
    "b": "Portal 12 12 para atrair riqueza saude e felicidade",
    "c": "Como dormir dormindo para mudar sua vida",
    "d": "Portal 99 da sabedoria da riqueza",
}


@pytest.fixture
def indice() -> IndiceLexical:
    return IndiceLexical(DOCUMENTOS)


# --------------------------------------------------------------------------- #
# Normalização e tokenização
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("PÓRTAL", "portal"),
        ("Áudio", "audio"),
        ("Lei da Atração", "lei da atracao"),
    ],
)
def test_normalizar_remove_acentos_e_caixa(entrada: str, esperado: str) -> None:
    assert normalizar(entrada) == esperado


def test_tokenizar_mantem_numeros() -> None:
    """'369' e '12 12' são os termos que mais importam neste corpus."""
    assert "369" in tokenizar("A técnica 369.")
    assert tokenizar("Portal 12 12") == ["portal", "12", "12"]


def test_tokenizar_separa_pontuacao() -> None:
    assert tokenizar("riqueza-saúde/felicidade!") == ["riqueza", "saude", "felicidade"]


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #

def test_termo_raro_tem_mais_peso_que_termo_comum() -> None:
    """
    IDF: um termo presente em 1 de 4 documentos pesa mais que um presente nos 4.

    Compara o mesmo documento ('raro') sob as duas queries, com todos os
    documentos do mesmo tamanho, para isolar o IDF da normalização por
    comprimento.
    """
    indice = IndiceLexical(
        {
            "raro": "a tecnica 369 traz portal",
            "comum1": "o portal traz a tecnica",
            "comum2": "o portal traz a wealth",
            "comum3": "o portal traz a money",
        }
    )
    assert set(indice.tamanhos) == {5}

    escore_raro = dict(indice.buscar("369", limite=4))["raro"]
    escore_comum = dict(indice.buscar("portal", limite=4))["raro"]
    assert escore_raro > escore_comum


def test_busca_encontra_o_documento_certo(indice: IndiceLexical) -> None:
    resultados = indice.buscar("tecnica 369", limite=3)
    assert resultados
    assert resultados[0][0] == "a"


def test_busca_por_termo_exato_encontra_portais(indice: IndiceLexical) -> None:
    """É o caso que a busca vetorial sozinha costuma diluir."""
    resultados = indice.buscar("portal", limite=5)
    encontrados = {i for i, _ in resultados}
    assert {"b", "d"} <= encontrados


def test_stopwords_sao_descartadas(indice: IndiceLexical) -> None:
    assert indice.buscar("a de da que e") == []


def test_termo_inexistente_nao_quebra(indice: IndiceLexical) -> None:
    assert indice.buscar("xyzzy") == []


def test_consulta_vazia(indice: IndiceLexical) -> None:
    assert indice.buscar("") == []
    assert indice.buscar("   ") == []


def test_limite_e_respeitado(indice: IndiceLexical) -> None:
    assert len(indice.buscar("portal lei atracao", limite=2)) == 2


def test_scores_vencem_desc(indice: IndiceLexical) -> None:
    pontuacoes = [score for _, score in indice.buscar("portal 12 12", limite=4)]
    assert pontuacoes == sorted(pontuacoes, reverse=True)


def test_indice_vazio() -> None:
    vazio = IndiceLexical({})
    assert vazio.buscar("qualquer coisa") == []


def test_acento_na_query_encontra_documento_sem_acento(indice: IndiceLexical) -> None:
    assert indice.buscar("atração")
