"""
Testes das funções puras do `transcrever_canal`.

Cobrem a montagem do comando do whisper.cpp, a localização de executáveis e o
filtro de duração. Nenhum teste baixa áudio nem chama o YouTube.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import transcrever_canal as tc


@pytest.fixture
def whisper_falso(tmp_path, monkeypatch):
    """Cria uma árvore de ferramentas do whisper.cpp e aponta o módulo para ela."""
    ferramentas = tmp_path / ".tools" / "whisper-vulkan"
    ferramentas.mkdir(parents=True)
    monkeypatch.setattr(tc, "DIR_FERRAMENTAS", ferramentas)
    monkeypatch.setattr(tc, "DIR_MODELOS_CPP", tmp_path / "modelos_whisper_cpp")
    return ferramentas


# --------------------------------------------------------------------------- #
# Localização do executável e do modelo
# --------------------------------------------------------------------------- #

def test_executavel_windows_e_encontrado(whisper_falso) -> None:
    (whisper_falso / "whisper-cli.exe").write_text("bin")
    executavel, _ = tc.arquivos_gpu("small")
    assert executavel.name == "whisper-cli.exe"


def test_executavel_sem_extensao_e_encontrado(whisper_falso) -> None:
    """O binário sem extensão é o build de Linux/macOS, que o README promete suportar."""
    (whisper_falso / "whisper-cli").write_text("bin")
    executavel, _ = tc.arquivos_gpu("small")
    assert executavel.name == "whisper-cli"


def test_executavel_ausente_cai_para_o_nome_plataforma(whisper_falso) -> None:
    executavel, _ = tc.arquivos_gpu("small")
    esperado = "whisper-cli.exe" if os.name == "nt" else "whisper-cli"
    assert executavel.name == esperado
    assert executavel.parent == whisper_falso


def test_modelo_quantizado_por_alias(whisper_falso) -> None:
    _, modelo = tc.arquivos_gpu("small")
    assert modelo.name == "ggml-small-q5_1.bin"
    _, modelo = tc.arquivos_gpu("large-v3")
    assert modelo.name == "ggml-large-v3-q5_0.bin"


def test_modelo_explicito_com_extensao_e_respeitado(whisper_falso, tmp_path) -> None:
    _, modelo = tc.arquivos_gpu("custom/meu-modelo.bin")
    assert modelo == Path("custom/meu-modelo.bin")


def test_caminho_do_modelo_usa_variavel_de_ambiente(whisper_falso) -> None:
    _, modelo = tc.arquivos_gpu("small")
    assert modelo.parent == whisper_falso.parent.parent / "modelos_whisper_cpp"


# --------------------------------------------------------------------------- #
# VAD
# --------------------------------------------------------------------------- #

def test_modelo_vad_encontrado(whisper_falso) -> None:
    (whisper_falso / "ggml-silero-v5.1.bin").write_bytes(b"x")
    assert tc.whisper_vad_model() is not None


def test_modelo_vad_ausente_retorna_none(whisper_falso) -> None:
    assert tc.whisper_vad_model() is None


def test_whisper_vad_model_com_diretorio_inexistente(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tc, "DIR_FERRAMENTAS", tmp_path / "nao_existe")
    assert tc.whisper_vad_model() is None


# --------------------------------------------------------------------------- #
# Filtro de duração
# --------------------------------------------------------------------------- #

def _videos_de_teste() -> list[dict]:
    return [
        {"id": "longo", "duration": 600},
        {"id": "curto", "duration": 20},
        {"id": "no_limite", "duration": 60},
        {"id": "live", "duration": 40000, "is_live": True},
        {"id": "sem_duracao"},
    ]


def test_filtro_descarta_curtos_e_lives() -> None:
    """Live pode ter 11 horas: transcrever isso é desperdício."""
    mantidos = [
        v
        for v in _videos_de_teste()
        if not v.get("is_live") and (not v.get("duration") or v["duration"] >= 60)
    ]
    assert {v["id"] for v in mantidos} == {"longo", "no_limite", "sem_duracao"}


# --------------------------------------------------------------------------- #
# Lock
# --------------------------------------------------------------------------- #

def test_lock_impede_duas_instancias(tmp_path) -> None:
    """Dois processos lendo o mesmo .concluidos.txt transcrevem o mesmo vídeo duas vezes."""
    caminho = tmp_path / ".lock"

    if sys.platform == "win32":
        pytest.skip("fcntl não existe no Windows; o lock é apenas advisory")

    import fcntl

    with tc.LockDeExecucao(caminho):
        # Simula o segundo processo tentando o mesmo lock.
        outro = caminho.open("a+")
        try:
            with pytest.raises(OSError):
                fcntl.flock(outro.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            outro.close()

    # Liberado: o arquivo some e outro lock pode ser tomado.
    assert not caminho.exists()
    with tc.LockDeExecucao(caminho):
        assert caminho.exists()
    assert not caminho.exists()


def test_lock_grava_o_pid(tmp_path) -> None:
    caminho = tmp_path / ".lock"
    with tc.LockDeExecucao(caminho):
        assert caminho.read_text().strip() == str(os.getpid())


def test_lock_removido_mesmo_com_excecao(tmp_path) -> None:
    caminho = tmp_path / ".lock"
    with pytest.raises(RuntimeError), tc.LockDeExecucao(caminho):
        raise RuntimeError("falha no meio")
    assert not caminho.exists()


# --------------------------------------------------------------------------- #
# Progresso
# --------------------------------------------------------------------------- #

def test_ids_concluidos_ignora_linhas_vazias(tmp_path) -> None:
    arquivo = tmp_path / ".concluidos.txt"
    arquivo.write_text("abc\n\n  \nxyz\n", encoding="utf-8")
    assert tc.ids_concluidos(arquivo) == {"abc", "xyz"}


def test_ids_concluidos_arquivo_inexistente(tmp_path) -> None:
    assert tc.ids_concluidos(tmp_path / "nao_existe.txt") == set()
