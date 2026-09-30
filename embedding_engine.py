"""
Módulo para geração de embeddings em 1536 dimensões.
Suporta modelos locais (HuggingFace / SentenceTransformers) e APIs compatíveis (Infini-AI, OpenAI).
"""

import os
from typing import List, Optional
import numpy as np


EXPECTED_DIMENSION = 1536


class BaseEmbeddingEngine:
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        raise NotImplementedError


class LocalSentenceTransformerEngine(BaseEmbeddingEngine):
    """
    Gera embeddings localmente usando SentenceTransformers.
    Modelos recomendados com 1536 dimensões nativas:
    - 'Alibaba-NLP/gte-Qwen2-1.5B-instruct' (Multilíngue, excelente em Português)
    - 'infly/inf-retriever-v1-1.5b' (Modelo da INF TECH / Infini-AI baseado em Qwen2, 1536d)
    - 'dunzhang/stella_en_1.5B_v5' (Modelo Stella v5)
    """

    def __init__(self, model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct", device: Optional[str] = None):
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[EmbeddingEngine] Carregando modelo local '{model_name}' no dispositivo '{device}'...")
        self.model_name = model_name
        self.device = device

        # No Transformers v5+, o Qwen2 tem suporte nativo. Usar trust_remote_code=False
        # evita o script legado modeling_qwen.py que causava o erro de 'rope_theta'.
        try:
            self.model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=False,
            )
        except Exception:
            self.model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=True,
            )

    def get_embeddings(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        dim = embeddings.shape[1]
        if dim != EXPECTED_DIMENSION:
            raise ValueError(
                f"Dimensão incorreta gerada pelo modelo: {dim}. "
                f"Esperado exatamente {EXPECTED_DIMENSION} dimensões."
            )

        return embeddings.tolist()


class OpenAICompatibleAPIEngine(BaseEmbeddingEngine):
    """
    Gera embeddings via API compatível com OpenAI (inclui Infini-AI e OpenAI).
    - Para Infini-AI:
        base_url = "https://cloud.infini-ai.com/maas/v1"
        model = "bge-m3" ou o modelo disponível na sua conta Infini-AI
    - Para OpenAI:
        model = "text-embedding-3-small" (gera nativamente 1536 dimensões)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ):
        from openai import OpenAI

        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("INFINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Chave de API não informada! Defina a variável de ambiente OPENAI_API_KEY "
                "ou INFINI_API_KEY, ou passe via parâmetro --api-key."
            )

        self.base_url = base_url or os.getenv("EMBEDDING_BASE_URL")
        self.model = model
        self.dimensions = dimensions

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        self.client = OpenAI(**kwargs)
        print(f"[EmbeddingEngine] Conectado à API (modelo: {self.model}, base_url: {self.base_url or 'OpenAI Oficial'})")

    def get_embeddings(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Limpa quebras de linha excessivas
            cleaned_batch = [t.replace("\r\n", " ").replace("\n", " ").strip() for t in batch]

            kwargs = {
                "input": cleaned_batch,
                "model": self.model,
            }
            # Se for OpenAI text-embedding-3, podemos explicitar dimensions
            if "text-embedding-3" in self.model:
                kwargs["dimensions"] = self.dimensions

            response = self.client.embeddings.create(**kwargs)
            batch_vectors = [item.embedding for item in response.data]

            for v in batch_vectors:
                if len(v) != EXPECTED_DIMENSION:
                    raise ValueError(
                        f"Dimensão incorreta retornada pela API: {len(v)}. "
                        f"Esperado exatamente {EXPECTED_DIMENSION} dimensões."
                    )
            all_embeddings.extend(batch_vectors)

        return all_embeddings


def create_engine(
    provider: str = "local",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    device: Optional[str] = None,
) -> BaseEmbeddingEngine:
    """
    Fábrica para instanciar a engine correta.
    """
    provider = provider.lower().strip()

    if provider in ("local", "huggingface", "stella", "infini"):
        # Se for modelo Stella / Infini local
        default_model = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
        chosen_model = model or default_model
        return LocalSentenceTransformerEngine(model_name=chosen_model, device=device)

    elif provider in ("api", "openai", "infini-cloud"):
        default_model = model or "text-embedding-3-small"
        return OpenAICompatibleAPIEngine(
            api_key=api_key,
            base_url=base_url,
            model=default_model,
            dimensions=EXPECTED_DIMENSION,
        )

    else:
        raise ValueError(
            f"Provedor desconhecido: '{provider}'. Escolha 'local' ou 'api'."
        )
