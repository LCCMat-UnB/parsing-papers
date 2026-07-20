"""
Parser numerico unico e testado para valores extraidos de papers.

Antes deste modulo, sanity_checks.py tinha DUAS funcoes de parsing numerico
com heuristicas inconsistentes entre si (_parse_numeric para float,
_parse_int para inteiro), cada uma tratando separador decimal/milhar de um
jeito diferente. Isso causava dois problemas reais:
  1. Um valor como "1.234,56" (formato PT-BR completo) ou "1,234.56" (EN
     completo) podia ser truncado silenciosamente errado por uma das duas
     funcoes.
  2. Como esse parsing tambem alimenta a comparacao de divergencia entre as
     duas extracoes (dual_extraction.py), um erro de parsing podia mascarar
     ou inventar divergencia que nao era real.

CONVENCAO ADOTADA (documentada aqui por ser inerentemente ambigua sem saber o
locale do paper original):
  - Se o numero tem os dois separadores (ex: "1.234,56" ou "1,234.56"), o
    ULTIMO separador que aparece e tratado como decimal, e todos os
    anteriores como separador de milhar. Isso cobre tanto o formato PT-BR
    (milhar=".", decimal=",") quanto o formato EN (milhar=",", decimal=".").
  - Se o numero tem SO UM separador, a decisao e por contagem de digitos
    apos ele:
      - Exatamente 3 digitos apos o separador E existe mais de um grupo de
        3 digitos antes dele (ex: "1.234", "12.345") -> provavelmente
        separador de milhar, sem parte decimal.
      - Qualquer outra contagem de digitos apos o separador (1, 2, ou 4+)
        -> tratado como separador decimal (ex: "63,02" -> 63.02; "0.87" ->
        0.87; "1.5" -> 1.5).
      - Exceção: um separador seguido de exatamente 3 digitos mas com so um
        grupo de digitos antes dele (ex: "1.234") e AMBIGUO -- pode ser
        "mil duzentos e trinta e quatro" (milhar) ou "um vírgula duzentos e
        trinta e quatro" (decimal raro). Neste caso, tratamos como milhar
        (inteiro), por ser o caso mais comum em sample_size/contagens neste
        dominio (tamanhos de amostra na casa dos milhares).
  - "%" no final sempre implica divisao por 100 no resultado, ADEMAIS da
    logica acima (aplicada aos digitos antes do "%").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_NUMBER_RE = re.compile(r"-?\d[\d.,]*\d|-?\d")


@dataclass
class ParsedNumber:
    value: float
    is_percent: bool
    raw_match: str


def _split_last_separator(digits_part: str) -> tuple[str, str, Optional[str]]:
    """
    Dado um bloco tipo "1.234,56" ou "1,234" ou "63,02", retorna
    (parte_inteira_normalizada_sem_separadores, parte_decimal_ou_vazia, separador_decimal_usado).
    """
    # posicoes de '.' e ',' no bloco
    seps = [(i, c) for i, c in enumerate(digits_part) if c in ".,"]
    if not seps:
        return digits_part, "", None

    last_idx, last_sep = seps[-1]
    before = digits_part[:last_idx]
    after = digits_part[last_idx + 1 :]

    other_seps = [c for i, c in seps[:-1]]

    if other_seps:
        # multiplos separadores -- o ultimo e decimal, os anteriores sao milhar
        integer_part = re.sub(r"[.,]", "", before)
        return integer_part, after, last_sep

    # um unico separador -- decide por contagem de digitos depois dele
    if len(after) == 3 and len(before) >= 1:
        # ambiguo: "1.234" -- convencao deste projeto: trata como milhar (inteiro)
        integer_part = before + after
        return integer_part, "", None
    else:
        # trata como separador decimal
        return before, after, last_sep


def parse_number(raw: Optional[str]) -> Optional[ParsedNumber]:
    """
    Extrai um numero de uma string livre, tratando separador decimal/milhar
    de forma consistente (ver convencao no docstring do modulo).

    Exemplos:
        "0.87"      -> 0.87
        "87%"       -> 0.87 (is_percent=True)
        "63,02%"    -> 0.6302 (is_percent=True)
        "1.234"     -> 1234.0 (milhar, sem parte decimal)
        "1,234"     -> 1234.0 (milhar, sem parte decimal)
        "1.234,56"  -> 1234.56
        "1,234.56"  -> 1234.56
        "3.844"     -> 3844.0 (tamanho de amostra, nao "3 virgula 844")
        "AUC=0.91"  -> 0.91
    """
    if raw is None:
        return None
    s = raw.strip()
    is_percent = "%" in s

    match = _NUMBER_RE.search(s)
    if not match:
        return None

    digits_part = match.group(0)
    negative = digits_part.startswith("-")
    if negative:
        digits_part = digits_part[1:]

    integer_part, decimal_part, _sep = _split_last_separator(digits_part)
    integer_part = integer_part or "0"

    try:
        if decimal_part:
            val = float(f"{integer_part}.{decimal_part}")
        else:
            val = float(integer_part)
    except ValueError:
        return None

    if negative:
        val = -val

    if is_percent:
        val = val / 100.0
    elif val > 1.0 and any(k in s.lower() for k in ["auc", "acc", "f1", "recall", "specificity"]):
        # heuristica preexistente: valor tipo "87" sem simbolo % mas
        # claramente uma metrica de proporcao mencionada no contexto textual
        val = val / 100.0

    return ParsedNumber(value=val, is_percent=is_percent, raw_match=match.group(0))


def parse_float(raw: Optional[str]) -> Optional[float]:
    """Atalho: retorna so o float (ou None). Substitui sanity_checks._parse_numeric."""
    parsed = parse_number(raw)
    return parsed.value if parsed is not None else None


def parse_int(raw: Optional[str]) -> Optional[int]:
    """
    Extrai um inteiro de uma string livre (ex: tamanho de amostra).
    Substitui sanity_checks._parse_int -- usa a MESMA logica de separador
    milhar/decimal de parse_number, em vez de uma regex diferente.
    """
    parsed = parse_number(raw)
    if parsed is None:
        return None
    return int(round(parsed.value))


def numbers_close(a: Optional[str], b: Optional[str], relative_tolerance: float = 0.05) -> Optional[bool]:
    """
    Compara dois valores numericos em texto livre, usando o mesmo parser
    (evita divergencia espuria entre extracoes A/B por erro de parsing
    diferente em cada lado). Retorna None se algum dos dois nao for parseavel
    (nesse caso, quem chama deve tratar como divergencia por falta de dado
    comparavel, nao usar este resultado como "sao iguais").
    """
    va, vb = parse_float(a), parse_float(b)
    if va is None or vb is None:
        return None
    denom = max(abs(va), abs(vb), 1e-9)
    return abs(va - vb) / denom <= relative_tolerance
