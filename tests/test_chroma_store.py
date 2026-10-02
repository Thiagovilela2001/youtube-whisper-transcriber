"""
Testes de integração do `chroma_store`: validação de modelo, manifesto e fusão.

Usa uma coleção Chroma temporária com embeddings determinísticos, então roda
sem baixar modelo nenhum.
"""

from __future__ import annotations

import math

import chromadb
import pytest

from busca_lexical import IndiceLexical
from chroma_store import (
    IncompatibilidadeDeModelo,
    abrir_colecao,
    buscar,
    ler_manifesto,
    reconstruir_manifesto,
    salvar_manifesto,
)
from embedding_engine import MODELOS, ModelSpec

SPEC_A = MODELOS["infly/inf-retriever-v1-1.5b"]
SPEC_B = MODELOS["Alibaba-NLP/gte-Qwen2-1.5B-instruct"]
DOCS = [
    "A tecnica 369 ativa a lei da atracao e traz abundancia",
    "Portal 12 12 para atrair riqueza saude e felicidade",
    "Como dormir dormindo para mudar sua vida",
    "Portal 99 da sabedoria da riqueza",
]


def _vetor(alvo: int) -> list[float]:
    """Embedding sintético: um pico na posição `alvo`."""
    bruto = [0.0] * 4
    bruto[alvo] = 1.0
    return bruto


@pytest.fixture
def base(tmp_path):
    """Coleção com 4 documentos, 4 dimensões e um índice lexical pronto."""
    spec = ModelSpec(nome="teste", dimensao=4)
    pasta = tmp_path / "db"
    # Criada via abrir_colecao para que o fingerprint fique nos metadados,
    # que é exatamente o que a validação de compatibilidade inspeciona.
    collection = abrir_colecao(pasta, "testes", spec)
    collection.upsert(
        ids=list("abcd"),
        embeddings=[_vetor(i) for i in range(4)],
        documents=DOCS,
        metadatas=[{"video_id": f"v{i}", "title": f"Vídeo {i}"} for i in range(4)],
    )
    indice = IndiceLexical(dict(zip("abcd", DOCS)))
    return collection, spec, indice, pasta


# --------------------------------------------------------------------------- #
# Validação de modelo
# --------------------------------------------------------------------------- #

def test_colecao_nova_grava_o_fingerprint(tmp_path) -> None:
    spec = ModelSpec(nome="teste", dimensao=8)
    abrir_colecao(tmp_path, "nova", spec)
    cliente = chromadb.PersistentClient(path=str(tmp_path))
    meta = cliente.get_collection("nova").metadata
    assert meta["embedding_fingerprint"] == spec.fingerprint
    assert meta["embedding_model"] == "teste"
    assert meta["embedding_dimensions"] == 8


def test_modelo_diferente_eh_rejeitado(base) -> None:
    """O erro que motivou o fingerprint: mesma dimensão, espaço vetorial diferente."""
    collection, _, _, caminho = base
    with pytest.raises(IncompatibilidadeDeModelo, match="outro modelo"):
        abrir_colecao(caminho, "testes", SPEC_A, criar=False)


def test_modelo_diferente_passa_com_permissao(base) -> None:
    collection, _, _, caminho = base
    abrir_colecao(caminho, "testes", SPEC_A, criar=False, permitir_reindexacao=True)


def test_mesmo_modelo_passa(base) -> None:
    """Reabrir a própria base não pode gerar alarme."""
    collection, spec, _, caminho = base
    reaberta = abrir_colecao(caminho, "testes", spec, criar=False)
    assert reaberta.count() == 4


def test_colecao_inexistente_sem_criar_falha(tmp_path) -> None:
    abrir_colecao(tmp_path, "criada", ModelSpec(nome="x", dimensao=4))
    with pytest.raises(FileNotFoundError, match="não existe"):
        abrir_colecao(tmp_path, "outra", ModelSpec(nome="x", dimensao=4), criar=False)


def test_banco_inexistente_falha_com_mensagem(tmp_path) -> None:
    """Regressão: a mensagem de erro referenciava um nome inexistente (NameError)."""
    pasta_vazia = tmp_path / "nao_existe"
    with pytest.raises(FileNotFoundError, match="nao_existe"):
        abrir_colecao(pasta_vazia, "qualquer", ModelSpec(nome="x", dimensao=4), criar=False)


# --------------------------------------------------------------------------- #
# Manifesto
# --------------------------------------------------------------------------- #

def test_manifesto_vazio_quando_ausente(tmp_path) -> None:
    assert ler_manifesto(tmp_path) == {}


def test_manifesto_sobrevive_a_json_corrompido(tmp_path) -> None:
    (tmp_path / ".manifesto.json").write_text("{quebrado", encoding="utf-8")
    assert ler_manifesto(tmp_path) == {}


def test_manifesto_ida_e_volta(tmp_path) -> None:
    salvar_manifesto(tmp_path, {"videos": {"abc": {"chunks": 3}}})
    assert ler_manifesto(tmp_path)["videos"]["abc"]["chunks"] == 3


def test_manifesto_reconstroi_a_partir_da_colecao(base) -> None:
    collection, _, _, _ = base
    manifesto = reconstruir_manifesto(collection)
    assert manifesto["videos"] == {
        "v0": {"chunks": 1},
        "v1": {"chunks": 1},
        "v2": {"chunks": 1},
        "v3": {"chunks": 1},
    }


# --------------------------------------------------------------------------- #
# Busca híbrida
# --------------------------------------------------------------------------- #

def test_busca_vetorial_pura(base) -> None:
    collection, _, indice, _ = base
    resultados = buscar(
        collection, _vetor(1), "portal", hibrido=False, indice_lexical=indice
    )
    assert resultados[0].id == "b"
    assert resultados[0].ranque_vetorial == 0
    assert resultados[0].ranque_lexical is None


def test_busca_lexical_encontra_termo_exato(base) -> None:
    """Com vetor ruim, o BM25 ainda acha 'portal' nos dois vídeos certos."""
    collection, _, indice, _ = base
    resultados = buscar(collection, _vetor(2), "portal", indice_lexical=indice)
    ids = {r.id for r in resultados}
    assert {"b", "d"} <= ids
    assert all(r.ranque_lexical is not None for r in resultados if r.id in {"b", "d"})


def test_rrf_soma_as_duas_modalidades(base) -> None:
    collection, _, indice, _ = base
    resultados = buscar(collection, _vetor(1), "portal 12 12", indice_lexical=indice)
    melhor = resultados[0]
    assert melhor.id == "b"
    assert melhor.ranque_vetorial is not None and melhor.ranque_lexical is not None
    assert melhor.score_rrf > 0


def test_peso_lexical_zero_desliga_o_bm25(base) -> None:
    collection, _, indice, _ = base
    sem = buscar(collection, _vetor(0), "portal", peso_lexical=0.0, indice_lexical=indice)
    assert sem[0].id == "a"  # só vetorial: _vetor(0) aponta para o documento 'a'
    assert all(r.ranque_lexical is None for r in sem)


def test_peso_lexical_alto_puxa_o_termo_exato(base) -> None:
    """Com o vetor apontando para 'a', 'portal' deve puxar 'b' para o topo."""
    collection, _, indice, _ = base
    resultados = buscar(collection, _vetor(0), "portal", peso_lexical=5.0, indice_lexical=indice)
    assert resultados[0].id in {"b", "d"}


def test_filtro_por_video(base) -> None:
    collection, _, indice, _ = base
    resultados = buscar(
        collection, _vetor(0), "portal", where={"video_id": "v3"}, indice_lexical=indice
    )
    assert {r.id for r in resultados} == {"d"}


def test_min_score_descarta_resultados(base) -> None:
    collection, _, indice, _ = base
    todos = buscar(collection, _vetor(0), "portal", indice_lexical=indice)
    alto = todos[0].score_rrf
    assert buscar(collection, _vetor(0), "portal", min_score=alto + 0.001, indice_lexical=indice) == []


def test_resultado_traz_documento_e_metadados(base) -> None:
    collection, _, indice, _ = base
    resultado = buscar(collection, _vetor(0), "369", indice_lexical=indice)[0]
    assert resultado.documento == DOCS[0]
    assert resultado.metadados["video_id"] == "v0"
    assert resultado.distancia is not None and math.isfinite(resultado.distancia)


# --------------------------------------------------------------------------- #
# Diversificação por vídeo
# --------------------------------------------------------------------------- #

def test_diversificar_limita_por_video() -> None:
    """
    Regressão de relevância: o 369 aparece em dezenas de vídeos do canal, e sem
    limite o topo da lista vira vários blocos do mesmo vídeo.
    """
    from chroma_store import ResultadoBusca, diversificar

    def r(video: str, score: float) -> ResultadoBusca:
        return ResultadoBusca(
            id=f"{video}_{score}",
            documento="x",
            metadados={"video_id": video},
            distancia=0.1,
            ranque_vetorial=0,
            ranque_lexical=0,
            score_rrf=score,
        )

    resultados = [r("a", 0.9), r("a", 0.8), r("a", 0.7), r("b", 0.6), r("c", 0.5)]
    misto = diversificar(resultados, max_por_video=1)
    assert [x.metadados["video_id"] for x in misto[:3]] == ["a", "b", "c"]
    # A lista não encurta: os excedentes ficam no fim, na ordem original.
    assert len(misto) == len(resultados)


def test_diversificar_zero_desliga() -> None:
    from chroma_store import ResultadoBusca, diversificar

    lista = [
        ResultadoBusca("1", "x", {"video_id": "a"}, 0.1, 0, 0, 0.9),
        ResultadoBusca("2", "x", {"video_id": "a"}, 0.1, 1, 0, 0.8),
    ]
    assert diversificar(lista, max_por_video=0) == lista


def test_diversificar_agrupa_quando_cabe() -> None:
    from chroma_store import ResultadoBusca, diversificar

    lista = [
        ResultadoBusca("1", "x", {"video_id": "a"}, 0.1, 0, 0, 0.9),
        ResultadoBusca("2", "x", {"video_id": "a"}, 0.1, 1, 0, 0.8),
        ResultadoBusca("3", "x", {"video_id": "b"}, 0.1, 2, 0, 0.7),
    ]
    assert [x.id for x in diversificar(lista, max_por_video=2)] == ["1", "2", "3"]
