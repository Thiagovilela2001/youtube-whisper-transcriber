"""
Testes do motor de embeddings.

O foco é o contrato por modelo: prefixos de instrução, fingerprint e a separação
entre `embed_query` e `embed_documents`. São exatamente as coisas que, quando
erradas, degradam a busca sem gerar erro visível.
"""

from __future__ import annotations

import pytest

from embedding_engine import (
    MODELOS,
    BaseEmbeddingEngine,
    ModelSpec,
    create_engine,
    resolver_spec,
)


class EngineFalsa(BaseEmbeddingEngine):
    """Engine que só registra o que recebeu, para inspecionar o contrato."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.documentos: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts, batch_size: int = 16):
        self.documentos = self._preparar(texts, self.spec.prefixo_documento)
        return [[0.0] * self.spec.dimensao for _ in texts]

    def embed_query(self, text: str):
        self.queries = self._preparar([text], self.spec.prefixo_query)
        return [0.0] * self.spec.dimensao


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def test_modelos_registrados_tem_dimensao_positiva() -> None:
    for nome, spec in MODELOS.items():
        assert spec.dimensao > 0, nome
        assert spec.nome == nome


def test_prefixos_de_query_e_documento_sao_distintos() -> None:
    """Se fossem iguais, a query e o documento cairiam no mesmo espaço."""
    for nome, spec in MODELOS.items():
        if spec.prefixo_query is None and spec.prefixo_documento is None:
            continue  # modelos sem instrução (OpenAI, bge-m3) são legítimos
        assert spec.prefixo_query != spec.prefixo_documento, nome


def test_familias_e5_gte_e_stella_tem_prefixo_de_query() -> None:
    """O erro que motivou o registry: indexar sem prefixo degrada a recuperação."""
    for nome in (
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "infly/inf-retriever-v1-1.5b",
        "dunzhang/stella_en_1.5B_v5",
    ):
        assert MODELOS[nome].prefixo_query, nome


def test_fingerprint_distingue_modelos_da_mesma_dimensao() -> None:
    """
    Regressão do problema central: `infly` e `Alibaba` geram 1536 dimensões,
    então uma troca entre eles passava despercebida.
    """
    a = MODELOS["infly/inf-retriever-v1-1.5b"].fingerprint
    b = MODELOS["Alibaba-NLP/gte-Qwen2-1.5B-instruct"].fingerprint
    assert a != b


def test_fingerprint_muda_com_o_contrato() -> None:
    base = ModelSpec(nome="x", dimensao=1536)
    assert base.fingerprint != ModelSpec(nome="x", dimensao=1024).fingerprint
    assert base.fingerprint != ModelSpec(nome="x", dimensao=1536, prefixo_query="query: ").fingerprint


def test_fingerprint_e_estavel() -> None:
    """O mesmo contrato precisa gerar o mesmo hash entre execuções."""
    a = ModelSpec(nome="x", dimensao=1536, prefixo_query="query: ")
    b = ModelSpec(nome="x", dimensao=1536, prefixo_query="query: ")
    assert a.fingerprint == b.fingerprint


def test_fingerprint_ignora_campos_cosméticos() -> None:
    """Nota e max_seq_length não mudam o espaço vetorial."""
    a = ModelSpec(nome="x", dimensao=1536, nota="alpha")
    b = ModelSpec(nome="x", dimensao=1536, nota="beta", max_seq_length=512)
    assert a.fingerprint == b.fingerprint


# --------------------------------------------------------------------------- #
# Resolução de modelos
# --------------------------------------------------------------------------- #

def test_alias_de_familia_recupera_o_contrato() -> None:
    spec = resolver_spec("infly/inf-retriever-v1-5b")
    assert spec.prefixo_query == "query: "
    assert spec.prefixo_documento == "passage: "


def test_modelo_desconhecido_usa_contrato_neutro_com_aviso() -> None:
    spec = resolver_spec("meu/modelo-inedito")
    assert spec.prefixo_query is None
    assert "não registrado" in spec.nota


def test_resolver_spec_exige_modelo() -> None:
    with pytest.raises(ValueError):
        resolver_spec("")


def test_provider_desconhecido_falha() -> None:
    with pytest.raises(ValueError):
        create_engine(provider="banana")


# --------------------------------------------------------------------------- #
# Preparo dos textos
# --------------------------------------------------------------------------- #

def test_query_recebe_prefixo_e_documento_nao() -> None:
    """A assimetria passage/query é intencional nos modelos de instrução."""
    engine = EngineFalsa(MODELOS["infly/inf-retriever-v1-1.5b"])
    engine.embed_query("como usar o 369")
    engine.embed_documents(["um trecho de vídeo"])
    assert engine.queries == ["query: como usar o 369"]
    assert engine.documentos == ["passage: um trecho de vídeo"]


def test_modelo_sem_instrucao_nao_altera_o_texto() -> None:
    engine = EngineFalsa(MODELOS["text-embedding-3-small"])
    engine.embed_query("  pergunta  ")
    assert engine.queries == ["pergunta"]


def test_gte_aplica_instrucao_apenas_na_query() -> None:
    engine = EngineFalsa(MODELOS["Alibaba-NLP/gte-Qwen2-1.5B-instruct"])
    engine.embed_query("portal 12 12")
    engine.embed_documents(["conteúdo"])
    assert engine.queries[0].startswith("Instruct:")
    assert engine.queries[0].endswith("Query: portal 12 12")
    assert engine.documentos == ["conteúdo"]


def test_lowercase_e_aplicado_quando_o_modelo_exige() -> None:
    engine = EngineFalsa(ModelSpec(nome="x", dimensao=8, lowercase=True))
    engine.embed_documents(["Portal Do SOL"])
    assert engine.documentos == ["portal do sol"]


def test_validacao_de_dimensao_rejeita_vetor_errado() -> None:
    engine = EngineFalsa(ModelSpec(nome="x", dimensao=4))
    with pytest.raises(ValueError, match="dimensões"):
        engine._validar([[0.0, 1.0]], "O modelo local")


def test_entradas_vazias_nao_geram_vetores() -> None:
    engine = EngineFalsa(MODELOS["infly/inf-retriever-v1-1.5b"])
    assert engine.embed_documents([]) == []
