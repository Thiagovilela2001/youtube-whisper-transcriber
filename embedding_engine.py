"""
Motor de embeddings — módulo reaproveitável, sem dependência de CLI ou de RAG.

Conceitos centrais:

* `ModelSpec` descreve o que um modelo exige: dimensão, prefixo de instrução
  para query, prefixo para documento, normalização e `trust_remote_code`.
* `fingerprint` resume essas exigências num hash. Coleções vetoriais gravam esse
  hash nos metadados, o que impede o erro clássico de indexar com um modelo e
  buscar com outro: como quase todos geram 1536 dimensões, a incompatibilidade
  passaria despercebida e a busca devolveria ruído.
* As engines expõem `embed_documents` e `embed_query` como operações distintas.
  Modelos da família e5/gte/bge/stella só funcionam bem quando a query recebe o
  prefixo de instrução e o documento não (ou recebe o seu).

Exemplo:

    engine = create_engine(provider="local")
    engine.embed_documents(["trecho de um vídeo"])
    engine.embed_query("como funciona a técnica 369?")
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "ModelSpec",
    "BaseEmbeddingEngine",
    "LocalEmbeddingEngine",
    "ApiEmbeddingEngine",
    "create_engine",
    "resolver_spec",
    "MODELOS",
    "MODELO_PADRAO_LOCAL",
]

MODELO_PADRAO_LOCAL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
MODELO_PADRAO_API = "text-embedding-3-small"

# Compatibilidade: versões antigas importavam este nome para validar a coleção.
EXPECTED_DIMENSION = 1536


@dataclass(frozen=True)
class ModelSpec:
    """Contrato de uso de um modelo de embeddings."""

    nome: str
    dimensao: int
    # Prefixos de instrução. `None` significa que o texto vai cru.
    prefixo_query: Optional[str] = None
    prefixo_documento: Optional[str] = None
    # Alguns modelos (família e5) perdem qualidade com maiúsculas.
    lowercase: bool = False
    # Normalizar para norma unitária é o padrão para distância cosseno.
    normalizar: bool = True
    # `None` = tentar False e cair para True (comportamento seguro por padrão).
    trust_remote_code: Optional[bool] = None
    max_seq_length: Optional[int] = None
    #: Observação sobre o modelo, útil para logs e para a UI.
    nota: str = ""

    @property
    def fingerprint(self) -> str:
        """
        Hash estável das exigências do modelo. Qualquer mudança que afete a
        espaço vetorial (dimensão, prefixos, normalização) invalida a base.
        """
        relevante = {
            "nome": self.nome,
            "dimensao": self.dimensao,
            "prefixo_query": self.prefixo_query,
            "prefixo_documento": self.prefixo_documento,
            "lowercase": self.lowercase,
            "normalizar": self.normalizar,
        }
        bruto = json.dumps(relevante, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(bruto.encode("utf-8")).hexdigest()[:16]

    def resumo(self) -> str:
        return f"{self.nome} ({self.dimensao}d, fp={self.fingerprint})"


#: Modelos conhecidos, com o contrato correto de uso de cada um.
MODELOS: Dict[str, ModelSpec] = {
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct": ModelSpec(
        nome="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        dimensao=1536,
        prefixo_query=(
            "Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query\nQuery: "
        ),
        prefixo_documento=None,
        trust_remote_code=False,
        nota="Multilíngue, bom em português. Query recebe instrução; documento vai cru.",
    ),
    "infly/inf-retriever-v1-1.5b": ModelSpec(
        nome="infly/inf-retriever-v1-1.5b",
        dimensao=1536,
        prefixo_query="query: ",
        prefixo_documento="passage: ",
        trust_remote_code=True,
        nota="Família e5/Infini. Exige prefixo em query e em documento.",
    ),
    "dunzhang/stella_en_1.5B_v5": ModelSpec(
        nome="dunzhang/stella_en_1.5B_v5",
        dimensao=1536,
        prefixo_query="s2p_query: ",
        prefixo_documento="s2p_document: ",
        trust_remote_code=True,
        max_seq_length=512,
        nota="Stella v5. Repositório exige trust_remote_code.",
    ),
    "BAAI/bge-m3": ModelSpec(
        nome="BAAI/bge-m3",
        dimensao=1024,
        prefixo_query=None,
        prefixo_documento=None,
        trust_remote_code=False,
        nota="Multilíngue denso+esparseo. A partir da v1.5 não usa prefixo.",
    ),
    "intfloat/multilingual-e5-large": ModelSpec(
        nome="intfloat/multilingual-e5-large",
        dimensao=1024,
        prefixo_query="query: ",
        prefixo_documento="passage: ",
        trust_remote_code=False,
        nota="Multilíngue, exige os prefixos da família e5.",
    ),
    # Leve, para validar o pipeline inteiro sem GPU nem download de GB.
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": ModelSpec(
        nome="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dimensao=384,
        trust_remote_code=False,
        nota="Leve e multilíngue. Bom para testes; qualidade menor que os 1536d.",
    ),
    "text-embedding-3-small": ModelSpec(
        nome="text-embedding-3-small",
        dimensao=1536,
        nota="OpenAI. Sem prefixo; `dimensions` é enviado na requisição.",
    ),
    "text-embedding-3-large": ModelSpec(
        nome="text-embedding-3-large",
        dimensao=3072,
        nota="OpenAI. Sem prefixo; `dimensions` é enviado na requisição.",
    ),
}

#: Prefixos de modelo aceitos pelo argumento `--provider`.
PROVIDER_LOCAL = {"local", "huggingface", "hf", "st", "stella", "infini", "gte", "e5", "bge"}
PROVIDER_API = {"api", "openai", "infini-cloud", "cloud", "infini_api"}


def resolver_spec(model: str, *, provider: str = "local") -> ModelSpec:
    """
    Devolve o `ModelSpec` de um modelo.

    Modelos desconhecidos são aceitos com um contrato neutro (sem prefixo), mas
    isso é registrado: sem o contrato correto, prefixos específicos de família
    deixam de ser aplicados e a recuperação piora silenciosamente.
    """
    nome = (model or "").strip()
    if not nome:
        raise ValueError("Informe um modelo de embeddings.")

    if nome in MODELOS:
        return MODELOS[nome]

    registrado = _por_prefixo(nome)
    if registrado is not None:
        return ModelSpec(
            nome=registrado.nome,
            dimensao=registrado.dimensao,
            prefixo_query=registrado.prefixo_query,
            prefixo_documento=registrado.prefixo_documento,
            lowercase=registrado.lowercase,
            normalizar=registrado.normalizar,
            trust_remote_code=registrado.trust_remote_code,
            max_seq_length=registrado.max_seq_length,
            nota=f"Alias de {registrado.nome} (deduzido pelo nome).",
        )

    dimensao = _inferir_dimensao_api(nome) if provider in PROVIDER_API else 1536
    return ModelSpec(
        nome=nome,
        dimensao=dimensao,
        nota="Modelo não registrado: aplicado contrato neutro, sem prefixo de instrução.",
    )


def _por_prefixo(nome: str) -> Optional[ModelSpec]:
    """Casa um modelo desconhecido com um registrado pelo nome da família."""
    alvo = nome.lower()
    familias = {
        "gte-qwen2": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "inf-retriever": "infly/inf-retriever-v1-1.5b",
        "stella": "dunzhang/stella_en_1.5B_v5",
        "bge-m3": "BAAI/bge-m3",
        "multilingual-e5": "intfloat/multilingual-e5-large",
        "paraphrase-multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "text-embedding-3-small": "text-embedding-3-small",
        "text-embedding-3-large": "text-embedding-3-large",
    }
    for marca, nome_registrado in familias.items():
        if marca in alvo:
            return MODELOS[nome_registrado]
    return None


def _inferir_dimensao_api(nome: str) -> int:
    return MODELOS[nome].dimensao if nome in MODELOS else 1536


class BaseEmbeddingEngine:
    """Interface comum das engines."""

    spec: ModelSpec

    def embed_documents(self, texts: Sequence[str], batch_size: int = 16) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError

    # Mantido para compatibilidade com chamadores antigos.
    def get_embeddings(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        return self.embed_documents(texts, batch_size=batch_size)

    @property
    def fingerprint(self) -> str:
        return self.spec.fingerprint

    def _preparar(self, texts: Sequence[str], prefixo: Optional[str]) -> List[str]:
        saida = []
        for texto in texts:
            limpo = str(texto or "").strip()
            if self.spec.lowercase:
                limpo = limpo.lower()
            if prefixo:
                limpo = f"{prefixo}{limpo}"
            saida.append(limpo)
        return saida

    def _validar(self, vetores: List[List[float]], origem: str) -> List[List[float]]:
        if not vetores:
            return vetores
        dimensao = len(vetores[0])
        if dimensao != self.spec.dimensao:
            raise ValueError(
                f"{origem} produziu vetores de {dimensao} dimensões, mas "
                f"'{self.spec.nome}' deveria produzir {self.spec.dimensao}. "
                "Isso costuma significar que outro modelo foi carregado."
            )
        return vetores


class LocalEmbeddingEngine(BaseEmbeddingEngine):
    """Embeddings locais via sentence-transformers."""

    def __init__(self, spec: ModelSpec, device: Optional[str] = None):
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.spec = spec
        self.device = device
        print(f"[EmbeddingEngine] Carregando '{spec.nome}' no dispositivo '{device}'...")

        def carregar(trust: bool) -> Any:
            extra: Dict[str, Any] = {"trust_remote_code": trust}
            if spec.max_seq_length:
                extra["model_kwargs"] = {"max_seq_length": spec.max_seq_length}
            return SentenceTransformer(spec.nome, device=device, **extra)

        if spec.trust_remote_code is False:
            # Modelos Qwen2 não precisam (e não devem) de código remoto.
            self.model = carregar(False)
        elif spec.trust_remote_code is True:
            self.model = carregar(True)
        else:
            try:
                self.model = carregar(False)
            except Exception:
                self.model = carregar(True)

    def _codificar(self, textos: Sequence[str], batch_size: int) -> List[List[float]]:
        vetores = self.model.encode(
            list(textos),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.spec.normalizar,
            convert_to_numpy=True,
        )
        return vetores.tolist()

    def embed_documents(self, texts: Sequence[str], batch_size: int = 16) -> List[List[float]]:
        if not texts:
            return []
        preparados = self._preparar(texts, self.spec.prefixo_documento)
        return self._validar(self._codificar(preparados, batch_size), "O modelo local")

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text], batch_size=1)[0]


class ApiEmbeddingEngine(BaseEmbeddingEngine):
    """Embeddings via API compatível com OpenAI (OpenAI, Infini-AI, vLLM, Ollama)."""

    def __init__(
        self,
        spec: ModelSpec,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        from openai import OpenAI

        self.spec = spec
        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("INFINI_API_KEY")
            or os.getenv("EMBEDDING_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Chave de API não informada! Defina OPENAI_API_KEY, INFINI_API_KEY "
                "ou passe --api-key."
            )

        self.base_url = base_url or os.getenv("EMBEDDING_BASE_URL")
        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        self.client = OpenAI(**kwargs)
        destino = self.base_url or "OpenAI oficial"
        print(f"[EmbeddingEngine] API: modelo '{spec.nome}' em {destino}")

    def _lotes(self, textos: Sequence[str], batch_size: int) -> List[List[float]]:
        todos: List[List[float]] = []
        for inicio in range(0, len(textos), batch_size):
            lote = [
                str(t).replace("\r\n", " ").replace("\n", " ").strip()
                for t in textos[inicio : inicio + batch_size]
            ]
            payload: Dict[str, Any] = {"input": lote, "model": self.spec.nome}
            if self.spec.nome.startswith("text-embedding-3"):
                payload["dimensions"] = self.spec.dimensao
            resposta = self.client.embeddings.create(**payload)
            todos.extend(item.embedding for item in resposta.data)
        return todos

    def embed_documents(self, texts: Sequence[str], batch_size: int = 64) -> List[List[float]]:
        if not texts:
            return []
        preparados = self._preparar(texts, self.spec.prefixo_documento)
        return self._validar(self._lotes(preparados, batch_size), "A API")

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text], batch_size=1)[0]


def create_engine(
    provider: str = "local",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    device: Optional[str] = None,
) -> BaseEmbeddingEngine:
    """
    Cria a engine adequada.

    `provider` decide a classe (local ou API); `model` escolhe dentro dela.
    Modelos desconhecidos são permitidos, com um aviso sobre o contrato neutro.
    """
    escolha = (provider or "local").strip().lower()

    if escolha in PROVIDER_LOCAL:
        nome = model or _modelo_padrao_local(escolha, padrao=MODELO_PADRAO_LOCAL)
    elif escolha in PROVIDER_API:
        nome = model or os.getenv("EMBEDDING_MODEL") or MODELO_PADRAO_API
    else:
        raise ValueError(
            f"Provedor desconhecido: '{provider}'. Use 'local' ou 'api'."
        )

    spec = resolver_spec(nome, provider=escolha)

    if escolha in PROVIDER_LOCAL:
        return LocalEmbeddingEngine(spec, device=device)
    return ApiEmbeddingEngine(spec, api_key=api_key, base_url=base_url)


def _modelo_padrao_local(escolha: str, padrao: str) -> str:
    """
    Antes `--provider stella` caía no mesmo ramo de `--provider local` e
    carregava o modelo padrão, ignorando o provider escolhido.
    """
    if escolha in {"stella"}:
        return "dunzhang/stella_en_1.5B_v5"
    if escolha in {"infini"}:
        return "infly/inf-retriever-v1-1.5b"
    return padrao
