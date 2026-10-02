"""
Acesso ao banco vetorial ChromaDB, compartilhado entre a indexação e a busca.

Concentra três responsabilidades que antes eram duplicadas nos scripts:

1. abrir a coleção verificando que ela foi construída com o mesmo modelo;
2. saber quais vídeos já estão indexados, sem carregar todos os metadados;
3. fazer a busca híbrida (vetorial + BM25 local) com fusão RRF.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from embedding_engine import ModelSpec

CHAVE_MODELO = "embedding_model"
CHAVE_FINGERPRINT = "embedding_fingerprint"
CHAVE_DIMENSAO = "embedding_dimensions"
CHAVE_ESPACO = "hnsw:space"
NOME_MANIFESTO = ".manifesto.json"


class IncompatibilidadeDeModelo(RuntimeError):
    """A coleção existente foi criada com outro modelo de embeddings."""


@dataclass
class ResultadoBusca:
    id: str
    documento: str
    metadados: Dict[str, Any]
    distancia: Optional[float]
    ranque_vetorial: Optional[int]
    ranque_lexical: Optional[int]
    score_rrf: float


# --------------------------------------------------------------------------- #
# Abertura da coleção
# --------------------------------------------------------------------------- #

def _metadados_de(spec: ModelSpec) -> Dict[str, Any]:
    return {
        CHAVE_ESPACO: "cosine",
        CHAVE_DIMENSAO: int(spec.dimensao),
        CHAVE_MODELO: str(spec.nome),
        CHAVE_FINGERPRINT: str(spec.fingerprint),
    }


def abrir_colecao(
    chroma_dir: str | os.PathLike[str],
    nome: str,
    spec: ModelSpec,
    *,
    criar: bool = True,
    permitir_reindexacao: bool = False,
) -> Any:
    """
    Abre (e opcionalmente cria) a coleção, exigindo que o modelo bata.

    Sem essa checagem, indexar com `infly/inf-retriever` e buscar com o padrão
    `Alibaba-NLP/gte-Qwen2` devolve resultados sem sentido: as duas famílias
    geram 1536 dimensões, então nada acusa a troca.
    """
    import chromadb

    caminho = Path(chroma_dir)
    if criar:
        caminho.mkdir(parents=True, exist_ok=True)
    elif not caminho.exists():
        raise FileNotFoundError(
            f"Banco ChromaDB não encontrado em '{chroma_dir}'. "
            "Rode gerar_embeddings.py antes de buscar."
        )

    client = chromadb.PersistentClient(path=str(caminho))
    existentes = {col.name for col in client.list_collections()}

    if nome not in existentes:
        if not criar:
            raise FileNotFoundError(
                f"Coleção '{nome}' não existe. Rode gerar_embeddings.py primeiro."
            )
        return client.get_or_create_collection(name=nome, metadata=_metadados_de(spec))

    collection = client.get_collection(name=nome)
    _conferir_modelo(collection, spec, permitir_reindexacao)
    return collection


def _conferir_modelo(collection: Any, spec: ModelSpec, permitir: bool) -> None:
    try:
        atual = collection.metadata or {}
    except Exception:  # pragma: no cover - defensivo
        return

    if not atual:
        return

    modelo_atual = atual.get(CHAVE_MODELO)
    fingerprint_atual = atual.get(CHAVE_FINGERPRINT)
    dimensao_atual = atual.get(CHAVE_DIMENSAO)

    problemas = []
    if dimensao_atual is not None and int(dimensao_atual) != int(spec.dimensao):
        problemas.append(f"dimensão {dimensao_atual} != {spec.dimensao}")
    if fingerprint_atual and fingerprint_atual != spec.fingerprint:
        problemas.append(f"fingerprint {fingerprint_atual} != {spec.fingerprint}")
    if not fingerprint_atual and modelo_atual and modelo_atual != spec.nome:
        problemas.append(f"modelo '{modelo_atual}' != '{spec.nome}'")

    if not problemas:
        return

    mensagem = (
        "A coleção foi indexada com outro modelo de embeddings:\n  - "
        + "\n  - ".join(problemas)
        + f"\n\nA base atual usa: {modelo_atual or '(desconhecido)'}"
        + f"\nVocê pediu:     {spec.nome}\n\nBuscas com modelos diferentes não são"
        " comparáveis e retornam ruído.\n\nPara corrigir, reindexe:\n"
        f"  python gerar_embeddings.py --model {spec.nome} --reindex"
    )
    if permitir:
        print(f"[AVISO] {mensagem}\n[AVISO] Reindexando mesmo assim (--reindex).")
        return
    raise IncompatibilidadeDeModelo(mensagem)


# --------------------------------------------------------------------------- #
# Manifesto: controle de incremental sem varrer toda a coleção
# --------------------------------------------------------------------------- #

def caminho_manifesto(chroma_dir: str | os.PathLike[str]) -> Path:
    return Path(chroma_dir) / NOME_MANIFESTO


def ler_manifesto(chroma_dir: str | os.PathLike[str]) -> Dict[str, Any]:
    """Lê o manifesto. Um .json simples é muito mais barato que `collection.get()`."""
    arquivo = caminho_manifesto(chroma_dir)
    if not arquivo.exists():
        return {}
    try:
        return json.loads(arquivo.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def salvar_manifesto(chroma_dir: str | os.PathLike[str], dados: Dict[str, Any]) -> None:
    """Escrita atômica: uma queda no meio não deixa o manifesto corrompido."""
    arquivo = caminho_manifesto(chroma_dir)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=arquivo.parent, delete=False, suffix=".tmp"
    ) as temporario:
        json.dump(dados, temporario, ensure_ascii=False, indent=2)
        caminho_temp = temporario.name
    os.replace(caminho_temp, arquivo)


def reconstruir_manifesto(collection: Any) -> Dict[str, Any]:
    """
    Reconstrói o manifesto a partir da coleção (usado quando o .json se perde).
    `collection.get()` traz todos os metadados de uma vez, então é caro: só
    rode isto quando o manifesto realmente não existir.
    """
    entradas: Dict[str, Any] = {}
    try:
        resultado = collection.get(include=["metadatas"])
    except Exception:
        return entradas
    for meta in resultado.get("metadatas") or []:
        if not meta:
            continue
        video_id = str(meta.get("video_id") or "")
        if not video_id:
            continue
        registro = entradas.setdefault(video_id, {"chunks": 0})
        registro["chunks"] = int(registro.get("chunks", 0)) + 1
    return {
        "videos": entradas,
        "reconstruido_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------- #
# Busca
# --------------------------------------------------------------------------- #

def _rrf(listas: Sequence[Sequence[str]], k: int = 60) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion: combina rankings sem precisar normalizar escores
    incompatíveis (cosseno vs BM25). É a razão de ser da busca híbrida.
    """
    pontuacoes: Dict[str, float] = {}
    for ranqueados in listas:
        for posicao, identificador in enumerate(ranqueados):
            pontuacoes[identificador] = pontuacoes.get(identificador, 0.0) + 1.0 / (k + posicao + 1)
    return pontuacoes


def _empacotar(
    ids: Sequence[str],
    documentos: Sequence[str],
    metadatas: Sequence[Dict[str, Any]],
    distancias: Optional[Sequence[float]],
) -> List[Dict[str, Any]]:
    return [
        {
            "id": identificador,
            "documento": documento,
            "metadados": dict(meta or {}),
            "distancia": None if distancias is None else float(distancias[pos]),
        }
        for pos, (identificador, documento, meta) in enumerate(
            zip(ids, documentos, metadatas)
        )
    ]


def buscar(
    collection: Any,
    embedding_query: Sequence[float],
    texto_query: str,
    *,
    candidatos: int = 50,
    hibrido: bool = True,
    peso_lexical: float = 1.0,
    min_score: float = 0.0,
    where: Optional[Dict[str, Any]] = None,
    chroma_dir: str | os.PathLike[str] | None = None,
    indice_lexical: Any = None,
) -> List[ResultadoBusca]:
    """
    Busca híbrida: vetorial (cosseno) + BM25 local, fundidos por RRF.

    A parte lexical importa em transcrições como estas: termos como "369",
    "Portal 12 12" ou "Grabovoi" são essenciais, e a similaridade semântica
    sozinha tende a diluí-los entre centenas de blocos do mesmo tema.

    RRF é usado em vez de somar escores porque cosseno e BM25 têm escalas
    incomparáveis; a fusão olha só a posição de cada documento no ranking.
    """
    vetorial = collection.query(
        query_embeddings=[list(embedding_query)],
        n_results=min(candidatos, max(1, collection.count())),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids_v = (vetorial.get("ids") or [[]])[0]
    docs_v = (vetorial.get("documents") or [[]])[0]
    metas_v = (vetorial.get("metadatas") or [[]])[0]
    dists_v = (vetorial.get("distances") or [[]])[0]
    mapa_vetorial = {
        item["id"]: item for item in _empacotar(ids_v, docs_v, metas_v, dists_v)
    }
    ranked_vetorial = {item["id"]: pos for pos, item in enumerate(mapa_vetorial.values())}

    mapa_lexical: Dict[str, Dict[str, Any]] = {}
    if hibrido and peso_lexical > 0 and texto_query.strip():
        try:
            if indice_lexical is None:
                if chroma_dir is None:
                    raise ValueError("Informe chroma_dir para habilitar a busca híbrida.")
                from busca_lexical import carregar_indice

                indice_lexical = carregar_indice(chroma_dir, collection)

            achados = indice_lexical.buscar(texto_query, limite=candidatos)
            # `where` também precisa valer aqui: sem isso, restringindo a busca
            # a um vídeo, o BM25 ainda traria trechos dos outros.
            documentos = _carregar_por_ids(collection, [i for i, _ in achados], where)
            mapa_lexical = {
                identificador: {
                    "id": identificador,
                    "documento": documentos.get(identificador, {}).get("documento", ""),
                    "metadados": documentos.get(identificador, {}).get("metadados", {}),
                    "distancia": None,
                    "score_lexical": score,
                }
                for identificador, score in achados
                if identificador in documentos
            }
        except Exception as erro:
            print(f"[Busca] parte híbrida indisponível, seguindo só com vetorial: {erro}")

    if not mapa_lexical:
        return _montar_resultados(
            {i: 1.0 / (60 + p) for i, p in ranked_vetorial.items()},
            mapa_vetorial, {}, ranked_vetorial, min_score,
        )

    vetorial_rrf = _rrf([list(ranked_vetorial)])
    lexical_rrf = _rrf([list(mapa_lexical)])

    fundido = {
        identificador: vetorial_rrf.get(identificador, 0.0) + peso_lexical * valor
        for identificador, valor in lexical_rrf.items()
    }
    for identificador, valor in vetorial_rrf.items():
        fundido.setdefault(identificador, valor)

    return _montar_resultados(
        fundido, mapa_vetorial, mapa_lexical, ranked_vetorial, min_score
    )


def _carregar_por_ids(
    collection: Any,
    ids: Sequence[str],
    where: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Busca documentos e metadados de uma lista de ids, aplicando `where`."""
    if not ids:
        return {}
    resultado = collection.get(ids=list(ids), include=["documents", "metadatas"], where=where)
    saida: Dict[str, Dict[str, Any]] = {}
    for identificador, documento, meta in zip(
        resultado.get("ids") or [],
        resultado.get("documents") or [],
        resultado.get("metadatas") or [],
    ):
        saida[str(identificador)] = {
            "documento": documento or "",
            "metadados": dict(meta or {}),
        }
    return saida


def diversificar(
    resultados: List[ResultadoBusca], max_por_video: int
) -> List[ResultadoBusca]:
    """
    Limita quantos trechos de cada vídeo entram no resultado.

    Sem isto, uma busca sobre um tema muito coberto pelo canal (o 369 aparece em
    dezenas de vídeos) devolve vários blocos do mesmo vídeo e esconde a variety
    de fontes. Os resultados que sobrarem preenchem as vagas restantes, então a
    lista não encurta.
    """
    if max_por_video <= 0:
        return resultados

    por_video: Dict[str, int] = {}
    escolhidos: List[ResultadoBusca] = []
    excedente: List[ResultadoBusca] = []

    for resultado in resultados:
        video_id = str(resultado.metadados.get("video_id") or "")
        usados = por_video.get(video_id, 0)
        if usados < max_por_video:
            por_video[video_id] = usados + 1
            escolhidos.append(resultado)
        else:
            excedente.append(resultado)

    return escolhidos + excedente


def _montar_resultados(
    pontuacoes: Dict[str, float],
    mapa_vetorial: Dict[str, Dict[str, Any]],
    mapa_lexical: Dict[str, Dict[str, Any]],
    ranked_vetorial: Dict[str, int],
    min_score: float,
) -> List[ResultadoBusca]:
    ranked_lexical = {identificador: pos for pos, identificador in enumerate(mapa_lexical)}
    ordenados = sorted(pontuacoes.items(), key=lambda par: par[1], reverse=True)

    resultados: List[ResultadoBusca] = []
    for identificador, score in ordenados:
        base = mapa_vetorial.get(identificador) or mapa_lexical.get(identificador)
        if base is None:
            continue
        if score < min_score:
            continue
        resultados.append(
            ResultadoBusca(
                id=identificador,
                documento=base["documento"] or "",
                metadados=base["metadados"],
                distancia=base["distancia"],
                ranque_vetorial=ranked_vetorial.get(identificador),
                ranque_lexical=ranked_lexical.get(identificador),
                score_rrf=score,
            )
        )
    return resultados
