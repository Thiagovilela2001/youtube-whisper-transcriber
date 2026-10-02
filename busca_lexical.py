"""
Busca lexical BM25 sobre os documentos da coleção.

Por que existe: o `query_texts` do Chroma não faz BM25 — ele baixa um modelo
ONNX 384d e faz busca vetorial, o que quebra numa coleção de 1536 dimensões. A
API `collection.search()` com `Rrf` existe, mas lança `NotImplementedError` no
Chroma local ("Search is not implemented for Local Chroma"). Como o corpus aqui
tem apenas alguns milhares de blocos, compensa fazer o BM25 em Python.

O índice invertido é serializado em disco e invalidado pela contagem da coleção,
então o custo de construir paga-se uma vez por base.
"""

from __future__ import annotations

import math
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

K1 = 1.5
B = 0.75
NOME_CACHE = ".indice_lexical.pkl"

#: Stopwords em português e inglês. Termos como "369" ou "portal" sobrevivem.
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "pelos", "pelas", "para",
    "pra", "com", "sem", "sob", "sobre", "entre", "ate", "até", "que", "se", "ao",
    "aos", "e", "ou", "mas", "porem", "porém", "porque", "pois", "entao", "então",
    "ja", "já", "mais", "muito", "tambem", "também", "so", "só", "ser", "estar",
    "ter", "fazer", "ir", "vao", "vão", "voce", "você", "eu", "tu", "ele", "ela",
    "isso", "isto", "aquilo", "qual", "quais", "quando", "como", "onde", "qualquer",
    "todos", "todas", "cada", "seu", "sua", "meu", "minha", "nosso", "nossa",
    "dele", "dela", "deles", "delas", "the", "and", "for", "you", "are", "with",
    "this", "that", "from", "have", "not", "but", "was", "will",
}

_TOKEN = re.compile(r"[a-z0-9]+")


def normalizar(texto: str) -> str:
    """Minúsculas e sem acentos: 'portal' casa com 'PORTAL' e 'Pörtal'."""
    decomposto = unicodedata.normalize("NFD", str(texto or "").lower())
    sem_acento = "".join(c for c in decomposto if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", sem_acento)


def tokenizar(texto: str) -> List[str]:
    return _TOKEN.findall(normalizar(texto))


class IndiceLexical:
    """Índice invertido com BM25."""

    def __init__(self, documentos: Dict[str, str]):
        self.ids: List[str] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.tamanhos: List[int] = []
        self._construir(documentos)

    def _construir(self, documentos: Dict[str, str]) -> None:
        frequencias: List[Counter] = []
        for identificador, documento in documentos.items():
            tokens = tokenizar(documento)
            self.ids.append(identificador)
            self.tamanhos.append(len(tokens))
            frequencias.append(Counter(tokens))

        for posicao, contagem in enumerate(frequencias):
            for termo, ocorrencias in contagem.items():
                self.postings[termo].append((posicao, ocorrencias))

        self.total_documentos = len(self.ids)
        self.media_tamanho = (
            sum(self.tamanhos) / self.total_documentos if self.total_documentos else 0.0
        )
        self.postings = dict(self.postings)

    def buscar(self, consulta: str, limite: int = 50) -> List[Tuple[str, float]]:
        """Retorna os `limite` documentos mais relevantes, com o score BM25."""
        termos = [t for t in tokenizar(consulta) if t not in STOPWORDS]
        if not termos or not self.total_documentos:
            return []

        # Termos repetidos na query contam menos que termos únicos.
        pesos_termo: Dict[str, float] = {}
        for termo in Counter(termos).items():
            pesos_termo[termo[0]] = termo[1]

        placar: Dict[int, float] = defaultdict(float)
        for termo, peso in pesos_termo.items():
            ocorrencias = self.postings.get(termo)
            if not ocorrencias:
                continue
            df = len(ocorrencias)
            # IDF de Robertson com piso positivo, para não inverter o sinal.
            idf = math.log(1.0 + (self.total_documentos - df + 0.5) / (df + 0.5))
            for posicao, frequencia in ocorrencias:
                tamanho = self.tamanhos[posicao]
                normalizador = K1 * (
                    1.0 - B + B * (tamanho / self.media_tamanho if self.media_tamanho else 1.0)
                )
                placar[posicao] += peso * idf * (frequencia * (K1 + 1.0)) / (
                    frequencia + normalizador
                )

        ordenados = sorted(placar.items(), key=lambda par: par[1], reverse=True)[:limite]
        return [(self.ids[posicao], score) for posicao, score in ordenados]


def _caminho_cache(chroma_dir: str | Path) -> Path:
    return Path(chroma_dir) / NOME_CACHE


def carregar_indice(
    chroma_dir: str | Path,
    collection: Any,
    usar_cache: bool = True,
) -> IndiceLexical:
    """
    Carrega o índice do disco, revalidando contra a contagem da coleção.
    Sem esse cache, toda busca pagaria a leitura de todos os documentos.
    """
    caminho = _caminho_cache(chroma_dir)
    total = collection.count()

    if usar_cache and caminho.exists():
        try:
            with caminho.open("rb") as arquivo:
                bruto = pickle.load(arquivo)
            if bruto.get("total") == total:
                return bruto["indice"]
        except Exception:
            pass  # cache corrompido: reconstrói

    documentos = _carregar_documentos(collection)
    indice = IndiceLexical(documentos)
    _salvar_cache(caminho, indice, total)
    return indice


def _carregar_documentos(collection: Any) -> Dict[str, str]:
    """Lê os documentos em lotes, para não estourar a memória de uma vez."""
    documentos: Dict[str, str] = {}
    total = collection.count()
    if not total:
        return documentos

    tamanho_lote = 2000
    for inicio in range(0, total, tamanho_lote):
        lote = collection.get(
            limit=min(tamanho_lote, total - inicio), offset=inicio, include=["documents"]
        )
        for identificador, documento in zip(lote.get("ids") or [], lote.get("documents") or []):
            documentos[str(identificador)] = documento or ""
    return documentos


def _salvar_cache(caminho: Path, indice: IndiceLexical, total: int) -> None:
    try:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        with caminho.open("wb") as arquivo:
            pickle.dump({"total": total, "indice": indice}, arquivo, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as erro:  # pragma: no cover - cache é otimização
        print(f"[Lexical] Não foi possível salvar o índice em cache: {erro}")


def invalidar_cache(chroma_dir: str | Path) -> None:
    _caminho_cache(chroma_dir).unlink(missing_ok=True)
