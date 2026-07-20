# src/parsing_papers/arbiter.py
"""
Arbitro de divergencias da dupla extracao (PRISMA item 9, agora com resolucao).

Antes: divergencias A/B viravam flag needs_review e a planilha usava so a
extracao A -- a fila de revisao humana crescia com ruido (8/13 linhas no lote
de referencia). Agora: para cada registro divergente, UMA chamada de arbitro
(prompt minusculo: so os campos divergentes + quotes + o texto fonte ja
reduzido pelas janelas) decide campo a campo "a", "b" ou "neither".

Principios:
- O arbitro NUNCA inventa um terceiro valor -- so escolhe entre A, B ou
  declara empate ("neither" -> campo continua flagado para revisao humana).
- Falha do arbitro (transporte, JSON invalido) NUNCA bloqueia o pipeline:
  retorna None e as divergencias seguem o comportamento antigo (flag).
- A reconciliacao (apply_arbiter_decisions) e pura e nao muta a extracao A --
  opera numa copia profunda.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ValidationError

from .dual_extraction import CATEGORICAL_FIELDS, NUMERIC_FIELDS, TEXT_FIELDS, FieldDivergence, RecordComparison
from .llm_client import LLMExtractor, _try_recover_json
from .schema import PaperExtraction

logger = logging.getLogger(__name__)

_EVIDENCE_FIELDS = set(NUMERIC_FIELDS + TEXT_FIELDS)
_ALL_FIELDS = _EVIDENCE_FIELDS | set(CATEGORICAL_FIELDS)

ARBITER_SYSTEM_PROMPT = """\
Voce e o arbitro de uma dupla extracao de dados de um paper cientifico.
Dois extratores independentes (A e B) produziram valores diferentes para os
mesmos campos de um mesmo modelo. Para CADA campo divergente, decida qual
valor esta correto com base EXCLUSIVA no texto fonte fornecido.

Regras:
- Escolha "a" ou "b" somente se o valor correspondente estiver claramente
  suportado pelo texto fonte (a quote do extrator ajuda a localizar o trecho).
- Se nenhum dos dois valores estiver claramente no texto, escolha "neither".
- NUNCA invente um terceiro valor. NUNCA parafraseie.

Responda APENAS com JSON valido no formato:
{"decisions": [{"field_name": "<campo>", "choice": "a"|"b"|"neither"}, ...]}
"""

ARBITER_USER_TEMPLATE = """\
PAPER_ID: {paper_id}
MODELO: {model_used}

CAMPOS DIVERGENTES:
{fields_block}

TEXTO FONTE:
---
{source_text}
---

Decida cada campo divergente. Responda APENAS com o JSON.
"""


class ArbiterFieldDecision(BaseModel):
    field_name: str
    choice: Literal["a", "b", "neither"]


class ArbiterRecordDecision(BaseModel):
    decisions: list[ArbiterFieldDecision]


def build_arbiter_messages(
    paper_id: str, model_used: str, divergences: list[FieldDivergence], source_text: str
) -> list[dict]:
    lines = []
    for d in divergences:
        lines.append(f'- campo "{d.field_name}": A="{d.value_a}" | B="{d.value_b}"')
    fields_block = "\n".join(lines)
    return [
        {"role": "system", "content": ARBITER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": ARBITER_USER_TEMPLATE.format(
                paper_id=paper_id, model_used=model_used, fields_block=fields_block, source_text=source_text
            ),
        },
    ]


class ArbiterClient:
    """Uma chamada de arbitro por REGISTRO divergente (todos os campos de uma vez)."""

    def __init__(self, extractor: LLMExtractor):
        self.extractor = extractor

    def arbitrate_record(
        self, paper_id: str, model_used: str, divergences: list[FieldDivergence], source_text: str
    ) -> ArbiterRecordDecision | None:
        messages = build_arbiter_messages(paper_id, model_used, divergences, source_text)
        try:
            raw = self.extractor.complete(
                messages, json_schema=ArbiterRecordDecision.model_json_schema(), schema_name="arbiter_decision"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Arbitro falhou para %s/%s (%s) -- divergencias ficam para revisao humana.",
                paper_id, model_used, e,
            )
            return None
        data = _try_recover_json(raw)
        if data is None:
            logger.warning("Arbitro retornou JSON invalido para %s/%s -- mantendo flags.", paper_id, model_used)
            return None
        try:
            return ArbiterRecordDecision.model_validate(data)
        except ValidationError as e:
            logger.warning("Decisao do arbitro fora do schema para %s/%s (%s) -- mantendo flags.", paper_id, model_used, e)
            return None


def apply_arbiter_decisions(
    extraction_a: PaperExtraction,
    extraction_b: PaperExtraction,
    comparisons: list[RecordComparison],
    decisions_by_index_a: dict[int, ArbiterRecordDecision],
) -> tuple[PaperExtraction, list[dict]]:
    """
    Produz a extracao reconciliada: copia profunda de A com os campos decididos
    "b" substituidos pelos valores de B. Retorna (reconciliada, resolucoes),
    onde resolucoes = [{"index_a", "field_name", "choice"}] apenas para
    choice em ("a", "b") -- "neither" deixa o campo divergente (needs_review).
    Campos decididos que nao constam como divergentes sao ignorados (defesa
    contra alucinacao do arbitro sobre a lista de campos).
    """
    reconciled = extraction_a.model_copy(deep=True)
    resolutions: list[dict] = []

    for comp in comparisons:
        if comp.index_a is None or comp.index_b is None:
            continue  # registros nao pareados: sem arbitro, seguem flagados
        decision = decisions_by_index_a.get(comp.index_a)
        if decision is None:
            continue
        divergent_fields = {d.field_name for d in comp.divergences if d.diverges}
        rec_out = reconciled.records[comp.index_a]
        rec_b = extraction_b.records[comp.index_b]
        applied = []
        for fd in decision.decisions:
            if fd.field_name not in divergent_fields or fd.field_name not in _ALL_FIELDS:
                continue
            if fd.choice == "neither":
                continue
            if fd.choice == "b":
                value_b = getattr(rec_b, fd.field_name)
                if fd.field_name in _EVIDENCE_FIELDS:
                    value_b = value_b.model_copy(deep=True)
                setattr(rec_out, fd.field_name, value_b)
            resolutions.append({"index_a": comp.index_a, "field_name": fd.field_name, "choice": fd.choice})
            applied.append(f"{fd.field_name}<-{fd.choice}")
        if applied:
            prior = rec_out.extraction_notes or ""
            rec_out.extraction_notes = (prior + " " if prior else "") + f"[arbitro: {', '.join(applied)}]"

    return reconciled, resolutions
