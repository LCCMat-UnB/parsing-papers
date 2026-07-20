"""
Etapa 5 do pipeline: checagem de sanidade nos valores.

Regras simples baseadas em conhecimento de dominio:
  - AUC deve estar entre 0.5 e 1.0
  - Accuracy, F1, Recall, Specificity, Default_rate sao proporcoes em [0, 1]
    (ou [0, 100] se reportadas em %)
  - Sample_size deve ser inteiro positivo
  - Standard_deviation / Standard_error devem ser >= 0
  - "value" deve ser um dado pontual (numero/rotulo curto), nao um paragrafo
    em prosa/resumo (ver _looks_like_prose)

Valores fora da faixa esperada sao marcados com um flag, mas NAO sao
descartados automaticamente -- ficam sinalizados para revisao humana (etapa 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .numparse import parse_float, parse_int
from .schema import ModelRecord

PROPORTION_METRICS = ["default_rate", "accuracy", "f1_score", "recall", "specificity"]

# Todos os campos EvidenceField/NumericEvidenceField do ModelRecord -- usados
# na checagem de "value parece prosa" (ver _looks_like_prose).
ALL_EVIDENCE_FIELDS = [
    "model_used", "sample_size", "default_rate", "variables_used",
    "auc", "accuracy", "f1_score", "recall", "specificity",
    "standard_deviation", "standard_error",
]

# variables_used e legitimamente uma lista/frase de variaveis (ex: "renda,
# historico de credito, area plantada"), entao fica de fora do limite de
# palavras -- os demais campos devem ser dados pontuais (numero, nome curto).
MAX_WORDS_BY_FIELD = {
    "model_used": 6,
    "sample_size": 5,
    "default_rate": 5,
    "auc": 5,
    "accuracy": 5,
    "f1_score": 5,
    "recall": 5,
    "specificity": 5,
    "standard_deviation": 5,
    "standard_error": 5,
}


def _looks_like_prose(value: str, max_words: int) -> bool:
    """
    Heuristica: se `value` tem mais palavras que o esperado para um dado
    pontual, provavelmente o LLM colou um resumo/paragrafo em prosa em vez do
    numero/rotulo isolado (caso real observado: sample_size.value virou
    "They analyzed a sample of micro, small and medium-sized enterprises...").
    """
    return len(value.split()) > max_words


@dataclass
class SanityFlag:
    field_name: str
    raw_value: str
    parsed_value: Optional[float]
    ok: bool
    message: str


# _parse_numeric/_parse_int antigos foram substituidos por parse_float/parse_int
# de numparse.py -- eram duas implementacoes com heuristicas de separador
# decimal/milhar INCONSISTENTES entre si (uma podia truncar "1.234,56"
# diferente da outra), e essa inconsistencia tambem contaminava a comparacao
# de divergencia entre extracoes A/B em dual_extraction.py. Ver numparse.py
# para a convencao adotada.


def check_record(record: ModelRecord) -> list[SanityFlag]:
    flags: list[SanityFlag] = []

    # "value" parece prosa em vez de dado pontual (numero/nome curto) --
    # sinaliza mesmo que o valor tecnicamente "parseie" para algo (ver caso
    # real: sample_size virou uma frase em ingles cujo unico numero era o
    # limite de credito, nao o tamanho da amostra -- passava despercebido
    # pelo check de "inteiro positivo" com um numero errado).
    for field_name in ALL_EVIDENCE_FIELDS:
        ev = getattr(record, field_name)
        if ev.value is None:
            continue
        max_words = MAX_WORDS_BY_FIELD.get(field_name, 8)
        if _looks_like_prose(ev.value, max_words):
            flags.append(
                SanityFlag(
                    field_name=f"{field_name}.value_shape",
                    raw_value=ev.value,
                    parsed_value=None,
                    ok=False,
                    message=(
                        f"{field_name}.value parece prosa/resumo ({len(ev.value.split())} palavras), "
                        f"nao um dado pontual -- provavel erro de extracao do LLM, revisar manualmente."
                    ),
                )
            )

    # AUC entre 0.5 e 1.0
    if record.auc.value is not None:
        v = parse_float(record.auc.value)
        ok = v is not None and 0.5 <= v <= 1.0
        flags.append(
            SanityFlag(
                field_name="auc",
                raw_value=record.auc.value,
                parsed_value=v,
                ok=ok,
                message="ok" if ok else f"AUC parseado={v} fora do intervalo esperado [0.5, 1.0]",
            )
        )

    # Proporcoes em [0, 1]
    for field_name in PROPORTION_METRICS:
        ev = getattr(record, field_name)
        if ev.value is None:
            continue
        v = parse_float(ev.value)
        ok = v is not None and 0.0 <= v <= 1.0
        flags.append(
            SanityFlag(
                field_name=field_name,
                raw_value=ev.value,
                parsed_value=v,
                ok=ok,
                message="ok" if ok else f"{field_name} parseado={v} fora do intervalo esperado [0, 1]",
            )
        )

    # sample_size inteiro positivo
    if record.sample_size.value is not None:
        v = parse_int(record.sample_size.value)
        ok = v is not None and v > 0
        flags.append(
            SanityFlag(
                field_name="sample_size",
                raw_value=record.sample_size.value,
                parsed_value=float(v) if v is not None else None,
                ok=ok,
                message="ok" if ok else f"sample_size parseado={v} nao e inteiro positivo",
            )
        )

    # desvio padrao / erro padrao >= 0
    for field_name in ["standard_deviation", "standard_error"]:
        ev = getattr(record, field_name)
        if ev.value is None:
            continue
        v = parse_float(ev.value)
        ok = v is not None and v >= 0
        flags.append(
            SanityFlag(
                field_name=field_name,
                raw_value=ev.value,
                parsed_value=v,
                ok=ok,
                message="ok" if ok else f"{field_name} parseado={v} negativo ou nao numerico",
            )
        )

    return flags


def record_has_failures(flags: list[SanityFlag]) -> bool:
    return any(not f.ok for f in flags)
