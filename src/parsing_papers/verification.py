"""
Etapa 3 do pipeline: verificacao automatica das citacoes.

Para cada EvidenceField/NumericEvidenceField extraido, confere se `quote`
realmente existe (aproximadamente) no texto fonte, via fuzzy match
(rapidfuzz.fuzz.partial_ratio) com limiar de similaridade configuravel
(default 90%). Isso captura tanto invencao pura (quote nao existe) quanto
citacoes levemente alteradas pelo LLM (parafraseadas, truncadas etc).

Se a citacao nao bate, o campo e marcado como nao verificado -- o `value` NAO
e descartado (fica disponivel para revisao humana), mas ganha um flag que
impede que ele va para a planilha final sem passar por auditoria.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz

from .schema import ModelRecord, PaperExtraction

DEFAULT_THRESHOLD = 90.0


def _normalize(text: str) -> str:
    """Normaliza espacos/quebras de linha para tornar o fuzzy match mais robusto
    a diferencas de whitespace introduzidas pelo parsing do GROBID."""
    return re.sub(r"\s+", " ", text or "").strip()


@dataclass
class FieldVerification:
    field_name: str
    value: Optional[str]
    quote: Optional[str]
    verified: bool
    similarity: float
    reason: str = ""


@dataclass
class RecordVerificationResult:
    record_index: int
    model_used: Optional[str]
    field_results: list[FieldVerification] = field(default_factory=list)

    @property
    def all_verified(self) -> bool:
        return all(fr.verified for fr in self.field_results if fr.value is not None)

    @property
    def unverified_fields(self) -> list[str]:
        return [fr.field_name for fr in self.field_results if fr.value is not None and not fr.verified]


EVIDENCE_FIELD_NAMES = [
    "model_used",
    "sample_size",
    "default_rate",
    "variables_used",
    "auc",
    "accuracy",
    "f1_score",
    "recall",
    "specificity",
    "standard_deviation",
    "standard_error",
]


def verify_quote(quote: Optional[str], source_text: str, threshold: float = DEFAULT_THRESHOLD) -> tuple[bool, float, str]:
    """Retorna (verificado, similaridade, motivo)."""
    if quote is None or not quote.strip():
        return False, 0.0, "quote vazio"

    norm_quote = _normalize(quote)
    norm_source = _normalize(source_text)

    if len(norm_quote) < 3:
        return False, 0.0, "quote muito curto para verificar com confianca"

    # partial_ratio: encontra a melhor janela do source que se aproxima do quote,
    # tolerante a quote ser um substring exato ou quase-exato.
    score = fuzz.partial_ratio(norm_quote, norm_source)

    verified = score >= threshold
    reason = "ok" if verified else f"similaridade {score:.1f} < limiar {threshold}"
    return verified, score, reason


def verify_record(
    record: ModelRecord, record_index: int, source_text: str, threshold: float = DEFAULT_THRESHOLD
) -> RecordVerificationResult:
    result = RecordVerificationResult(record_index=record_index, model_used=record.model_used.value)

    for field_name in EVIDENCE_FIELD_NAMES:
        ev = getattr(record, field_name)
        if ev.value is None:
            result.field_results.append(
                FieldVerification(field_name=field_name, value=None, quote=None, verified=True, similarity=100.0, reason="valor nulo, nada a verificar")
            )
            continue

        verified, score, reason = verify_quote(ev.quote, source_text, threshold=threshold)
        result.field_results.append(
            FieldVerification(
                field_name=field_name,
                value=ev.value,
                quote=ev.quote,
                verified=verified,
                similarity=score,
                reason=reason,
            )
        )

    return result


def verify_paper_extraction(
    extraction: PaperExtraction, source_text: str, threshold: float = DEFAULT_THRESHOLD
) -> list[RecordVerificationResult]:
    return [
        verify_record(record, idx, source_text, threshold=threshold)
        for idx, record in enumerate(extraction.records)
    ]
