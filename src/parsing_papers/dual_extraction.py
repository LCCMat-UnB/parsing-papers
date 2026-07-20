"""
Etapa 4 do pipeline: dupla extracao independente (espirito do PRISMA item 9).

O PRISMA pede que dois revisores extraiam dados independentemente e que
divergencias sejam resolvidas. Aqui replicamos isso automaticamente: rodamos
a extracao duas vezes -- por padrao com o MESMO modelo em temperaturas
diferentes (0.0 e 0.4), mas o pipeline aceita dois LLMExtractor distintos
(ex: dois modelos open source diferentes) via `extractor_b`.

Divergencias acima de uma tolerancia (fuzzy ratio para strings, diferenca
relativa para numeros) sao sinalizadas para revisao humana -- elas NAO sao
resolvidas automaticamente (isso seria reintroduzir o mesmo risco de
alucinacao que o pipeline tenta evitar).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .llm_client import LLMExtractor
from .numparse import parse_float
from .schema import ModelRecord, PaperExtraction

NUMERIC_FIELDS = [
    "sample_size", "default_rate", "auc", "accuracy", "f1_score",
    "recall", "specificity", "standard_deviation", "standard_error",
]
TEXT_FIELDS = ["model_used", "variables_used"]
CATEGORICAL_FIELDS = [
    "model_class", "baseline", "type_data_used", "financial",
    "productive", "climatic", "hybrid", "agricultural_context",
]

DEFAULT_NUMERIC_TOLERANCE = 0.05  # 5% de diferenca relativa tolerada
DEFAULT_TEXT_SIMILARITY_THRESHOLD = 85.0

# Abaixo deste score de similaridade de model_used, dois records NAO sao
# considerados o mesmo modelo -- ficam como "extra" de um dos lados em vez de
# pareados a força (o que geraria divergencia espuria em todos os campos).
MIN_MODEL_NAME_MATCH_SCORE = 60.0


@dataclass
class FieldDivergence:
    field_name: str
    value_a: str | None
    value_b: str | None
    diverges: bool
    detail: str = ""


@dataclass
class RecordComparison:
    index_a: int | None  # None se o record so existe em B (nao pareado)
    index_b: int | None  # None se o record so existe em A (nao pareado)
    model_used_a: str | None
    model_used_b: str | None
    match_score: float | None = None  # similaridade do pareamento por nome, None se nao pareado
    divergences: list[FieldDivergence] = field(default_factory=list)

    @property
    def has_divergence(self) -> bool:
        return self.index_a is None or self.index_b is None or any(d.diverges for d in self.divergences)


@dataclass
class DualExtractionResult:
    extraction_a: PaperExtraction
    extraction_b: PaperExtraction
    comparisons: list[RecordComparison]
    record_count_mismatch: bool

    @property
    def needs_human_review(self) -> bool:
        return self.record_count_mismatch or any(c.has_divergence for c in self.comparisons)


def _compare_field(field_name: str, rec_a: ModelRecord, rec_b: ModelRecord, numeric_tol: float, text_threshold: float) -> FieldDivergence:
    if field_name in NUMERIC_FIELDS:
        ev_a, ev_b = getattr(rec_a, field_name), getattr(rec_b, field_name)
        va, vb = ev_a.value, ev_b.value
        if va is None and vb is None:
            return FieldDivergence(field_name, va, vb, diverges=False, detail="ambos nulos")
        if va is None or vb is None:
            return FieldDivergence(field_name, va, vb, diverges=True, detail="um dos extratores nao reportou valor")
        # Usa o mesmo parser numerico de sanity_checks.py (numparse.py) para
        # os dois lados -- antes, uma inconsistencia de parsing entre modulos
        # podia mascarar ou inventar divergencia que nao era real.
        na, nb = parse_float(va), parse_float(vb)
        if na is None or nb is None:
            diverges = va.strip() != vb.strip()
            return FieldDivergence(field_name, va, vb, diverges=diverges, detail="nao numerico, comparado como string")
        denom = max(abs(na), abs(nb), 1e-9)
        rel_diff = abs(na - nb) / denom
        diverges = rel_diff > numeric_tol
        return FieldDivergence(field_name, va, vb, diverges=diverges, detail=f"diff relativa={rel_diff:.3f}")

    if field_name in TEXT_FIELDS:
        ev_a, ev_b = getattr(rec_a, field_name), getattr(rec_b, field_name)
        va, vb = ev_a.value, ev_b.value
        if va is None and vb is None:
            return FieldDivergence(field_name, va, vb, diverges=False, detail="ambos nulos")
        if va is None or vb is None:
            return FieldDivergence(field_name, va, vb, diverges=True, detail="um dos extratores nao reportou valor")
        score = fuzz.token_sort_ratio(va, vb)
        diverges = score < text_threshold
        return FieldDivergence(field_name, va, vb, diverges=diverges, detail=f"similaridade={score:.1f}")

    # categorico (enums simples, strings ou None)
    va = getattr(rec_a, field_name)
    vb = getattr(rec_b, field_name)
    va_norm = va.value if hasattr(va, "value") else va
    vb_norm = vb.value if hasattr(vb, "value") else vb
    diverges = va_norm != vb_norm
    return FieldDivergence(field_name, str(va_norm), str(vb_norm), diverges=diverges, detail="")


def _match_records_by_model_name(
    records_a: list[ModelRecord], records_b: list[ModelRecord]
) -> list[tuple[int | None, int | None, float | None]]:
    """
    Pareia records de A e B pela similaridade de model_used.value, em vez de
    assumir que aparecem na mesma ordem (posicao). Pareamento por posicao
    falha sempre que o LLM lista os modelos em ordem diferente entre as duas
    chamadas (comum quando a extracao B usa temperatura mais alta) -- isso
    gerava divergencia espuria em quase todo record, mascarando as
    divergencias reais no meio do ruido.

    Estrategia: matching guloso -- calcula a similaridade de todos os pares
    (i, j) possiveis, ordena do maior para o menor score, e vai casando
    greedily, pulando indices ja usados. Pares com score abaixo de
    MIN_MODEL_NAME_MATCH_SCORE nao sao pareados (ficam como "extra" de um
    lado so). Nao e o otimo global (isso seria um problema de assignment,
    ex: algoritmo hungaro), mas para o numero tipico de modelos por paper
    (dezenas no maximo) e uma aproximacao suficiente e muito mais simples.

    Retorna lista de tuplas (index_a, index_b, match_score) -- index_a ou
    index_b pode ser None quando o record so existe de um lado.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, rec_a in enumerate(records_a):
        name_a = rec_a.model_used.value or ""
        for j, rec_b in enumerate(records_b):
            name_b = rec_b.model_used.value or ""
            if not name_a or not name_b:
                continue
            score = fuzz.token_sort_ratio(name_a, name_b)
            if score >= MIN_MODEL_NAME_MATCH_SCORE:
                candidates.append((score, i, j))

    candidates.sort(key=lambda t: t[0], reverse=True)

    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j, score))

    result: list[tuple[int | None, int | None, float | None]] = [(i, j, s) for i, j, s in matches]
    for i in range(len(records_a)):
        if i not in used_a:
            result.append((i, None, None))
    for j in range(len(records_b)):
        if j not in used_b:
            result.append((None, j, None))

    # ordena por index_a (com Nones ao final) para manter saida deterministica e legivel
    result.sort(key=lambda t: (t[0] is None, t[0], t[1] is None, t[1]))
    return result


def compare_extractions(
    extraction_a: PaperExtraction,
    extraction_b: PaperExtraction,
    numeric_tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
    text_similarity_threshold: float = DEFAULT_TEXT_SIMILARITY_THRESHOLD,
) -> DualExtractionResult:
    comparisons: list[RecordComparison] = []
    count_mismatch = len(extraction_a.records) != len(extraction_b.records)

    matches = _match_records_by_model_name(extraction_a.records, extraction_b.records)

    all_fields = NUMERIC_FIELDS + TEXT_FIELDS + CATEGORICAL_FIELDS
    for index_a, index_b, match_score in matches:
        if index_a is not None and index_b is not None:
            rec_a = extraction_a.records[index_a]
            rec_b = extraction_b.records[index_b]
            divergences = [
                _compare_field(f, rec_a, rec_b, numeric_tolerance, text_similarity_threshold) for f in all_fields
            ]
            comparisons.append(
                RecordComparison(
                    index_a=index_a,
                    index_b=index_b,
                    model_used_a=rec_a.model_used.value,
                    model_used_b=rec_b.model_used.value,
                    match_score=match_score,
                    divergences=divergences,
                )
            )
        elif index_a is not None:
            rec_a = extraction_a.records[index_a]
            comparisons.append(
                RecordComparison(
                    index_a=index_a, index_b=None,
                    model_used_a=rec_a.model_used.value, model_used_b=None,
                )
            )
        else:
            rec_b = extraction_b.records[index_b]
            comparisons.append(
                RecordComparison(
                    index_a=None, index_b=index_b,
                    model_used_a=None, model_used_b=rec_b.model_used.value,
                )
            )

    return DualExtractionResult(
        extraction_a=extraction_a,
        extraction_b=extraction_b,
        comparisons=comparisons,
        record_count_mismatch=count_mismatch,
    )


def run_dual_extraction(
    paper_id: str,
    source_text: str,
    extractor_a: LLMExtractor,
    extractor_b: LLMExtractor | None = None,
    existing_extraction_a: PaperExtraction | None = None,
    on_extraction_a_done=None,
) -> DualExtractionResult:
    """
    Roda a extracao duas vezes. Se extractor_b nao for fornecido, reusa o
    mesmo modelo de extractor_a mas com temperatura mais alta (0.4) para
    gerar uma segunda amostra independente.

    Suporte a checkpoint intermediario: cada chamada ao LLM custa minutos (ver
    llm_timeout_s, default 900s) e um lote grande de papers pode ser
    interrompido (crash do Ollama, timeout, queda de energia) no meio do
    processamento de um paper. Sem checkpoint intermediario, uma extracao_a
    ja concluida com sucesso seria descartada e a chamada cara ao LLM
    refeita do zero ao reprocessar.

    existing_extraction_a : se fornecido (ex: recuperado de um checkpoint
        parcial em disco), pula a chamada a extractor_a e reusa esse
        resultado -- so a extracao_b (mais barata de refazer, pois so
        acontece depois) e chamada.
    on_extraction_a_done : callback opcional chamado com a PaperExtraction
        assim que extraction_a termina, ANTES de extraction_b comecar --
        usado por pipeline.py para persistir um checkpoint parcial em disco.
    """
    if extractor_b is None:
        extractor_b = LLMExtractor(
            model=extractor_a.model,
            api_base=extractor_a.api_base,
            temperature=max(extractor_a.temperature, 0.4),
            max_tokens=extractor_a.max_tokens,
            request_timeout=extractor_a.request_timeout,
            min_num_ctx=extractor_a.min_num_ctx,
        )

    if existing_extraction_a is not None:
        extraction_a = existing_extraction_a
    else:
        extraction_a = extractor_a.extract(paper_id, source_text)
        if on_extraction_a_done is not None:
            on_extraction_a_done(extraction_a)

    extraction_b = extractor_b.extract(paper_id, source_text)

    return compare_extractions(extraction_a, extraction_b)
