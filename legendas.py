"""
Parser de legendas WebVTT (SRT também), para reaproveitar as legendas que o
YouTube já publica em vez de re-transcrever o áudio.

Transcrever 341 vídeos com Whisper leva horas; baixar as legendas leva minutos.
A diferença é maior ainda quando a legenda é humana (digitada por alguém) ou
automática de qualidade razoável: o texto de origem costuma ser melhor que o
reconhecimento de voz, que troca "saciedade" por "sociedade" e "momento" por
"gerginha".

O formato de saída é o mesmo dos segmentos do Whisper, para que
`salvar_transcricao` funcione sem mudanças.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Dict, List

# 00:00:01.234 --> 00:00:04.567  (aceita os dois separadores de milissegundos)
_TEMPO = re.compile(
    r"(?P<inicio>\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})"
    r"\s*-->\s*"
    r"(?P<fim>\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})"
)
_TAG = re.compile(r"<[^>]+>")
_CURSOR = re.compile(r"</?c[^>]*>|</?v[^>]*>|\{\\[^}]*\}", re.IGNORECASE)
_ORDEM_NUMERICA = re.compile(r"<(\d+)>\s*$")


def tempo_para_segundos(texto: str) -> float:
    """Converte 'HH:MM:SS.mmm' ou 'MM:SS.mmm' em segundos."""
    texto = texto.strip().replace(",", ".")
    partes = texto.split(":")
    try:
        numeros = [float(p) for p in partes]
    except ValueError:
        return 0.0

    if len(numeros) == 3:
        horas, minutos, segundos = numeros
    elif len(numeros) == 2:
        horas, minutos, segundos = 0.0, numeros[0], numeros[1]
    else:
        return 0.0
    return horas * 3600 + minutos * 60 + segundos


def limpar_texto(texto: str) -> str:
    """Remove marcação, cursores e entidades; o YouTube escapa o texto como HTML."""
    sem_curso = _CURSOR.sub("", texto)
    sem_tags = _TAG.sub("", sem_curso)
    limpo = re.sub(r"\s+", " ", html.unescape(sem_tags)).strip()
    # alguma legenda traz a linha de tempo dentro do próprio texto
    return _TEMPO.sub("", limpo).strip()


def _tem_conteudo_util(texto: str) -> bool:
    if not texto:
        return False
    # Legendas repetem a linha anterior quando a fala é contínua. Sem esta
    # checagem, o mesmo trecho entra várias vezes no índice.
    partes = {p for p in texto.split() if p}
    return len(partes) >= 1 and any(c.isalnum() for c in texto)


def parse_vtt(conteudo: str) -> List[Dict[str, Any]]:
    """
    Converte WebVTT/SRT em segmentos no formato do Whisper.

    Devolve uma legenda por *cue*, sem mexer no tamanho. Quem decide o tamanho
    do bloco de recuperação é `rag_chunker`, e duplicar essa regra aqui faria
    os dois módulos brigar. O que este parser faz de específico é remover a
    repetição que o YouTube insere: sem isso, a mesma frase entrava no índice
    duas ou três vezes.
    """
    segmentos: List[Dict[str, Any]] = []
    if not conteudo:
        return segmentos

    linhas = conteudo.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    total = len(linhas)
    anterior = ""

    while i < total:
        linha = linhas[i]

        # Pula cabeçalho, índices e styled blocks.
        if linha.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            i += 1
            continue

        casamento = _TEMPO.search(linha)
        if not casamento:
            i += 1
            continue

        inicio = tempo_para_segundos(casamento.group("inicio"))
        fim = tempo_para_segundos(casamento.group("fim"))

        i += 1
        partes: List[str] = []
        while i < total and linhas[i].strip():
            partes.append(linhas[i])
            i += 1

        if partes:
            # A primeira parte costuma ser o número da legenda, sozinho na linha.
            if len(partes) > 1 and partes[0].strip().isdigit():
                partes = partes[1:]
            else:
                partes[0] = _ORDEM_NUMERICA.sub("", partes[0])
        texto = limpar_texto(" ".join(partes))

        if _tem_conteudo_util(texto):
            limpo = _sem_repeticao(texto, anterior)
            if limpo and _tem_conteudo_util(limpo):
                segmentos.append({"start": inicio, "end": max(fim, inicio), "text": limpo})
                anterior = limpo

    return segmentos


def _sem_repeticao(atual: str, anterior: str) -> str | None:
    """
    Remove da legenda atual o trecho que repete o fim da anterior.

    É o padrão das legendas automáticas do YouTube: uma frase contínua é
    quebrada em legendas que se sobrepõem na emenda.

        legenda 1: "o amor e a forca que move"
        legenda 2: "o amor e a forca que move todas as coisas"

    Sem remover, a mesma frase entra no índice duas vezes e o BM25 pontua aquele
    trecho em dobro. Devolve `None` quando não sobra texto novo, o que evita
    criar um segmento vazio.
    """
    if not anterior:
        return atual or None

    anteriores = anterior.split()
    atuais = atual.split()

    # Maior bloco de palavras comum na emenda: fim da anterior contra começo
    # da atual. O laço vai do maior para o menor e para no primeiro casamento.
    for tamanho in range(min(len(anteriores), len(atuais)), 1, -1):
        if anteriores[-tamanho:] == atuais[:tamanho]:
            restante = atuais[tamanho:]
            return " ".join(restante) if restante else None

    return None if atual == anterior else atual


def ler_legenda(caminho: str | Path) -> List[Dict[str, Any]]:
    """Lê um arquivo de legenda do disco e devolve os segmentos."""
    arquivo = Path(caminho)
    for codificacao in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return parse_vtt(arquivo.read_text(encoding=codificacao))
        except UnicodeDecodeError:
            continue
    return []
