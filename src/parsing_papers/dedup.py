"""
Deduplicacao de modelos repetidos dentro da MESMA extracao de um paper.

Problema real observado: o LLM as vezes lista o mesmo modelo mais de uma vez
no `records` de um unico PaperExtraction -- por exemplo, porque o paper cita
"Random Forest" tanto na secao de metodos quanto na de resultados e o modelo
gera um ModelRecord para cada mencao, ou porque um modelo aparece em duas
tabelas diferentes (ex: tabela de treino e tabela de teste) e o LLM nao
percebe que e a mesma linha logica. Sem deduplicacao, isso infla o numero de
"modelos testados" na planilha final (cada paper deveria contribuir uma linha
por modelo REALMENTE distinto) e distorce qualquer contagem/meta-analise
subsequente.

Este modulo roda DEPOIS da extracao de UM PaperExtraction (nao confundir com
o pareamento A/B entre duas extracoes independentes, que e outro problema,
tratado em dual_extraction.py).

Estrategia: agrupa records cujo `model_used.value` e muito similar (fuzzy
match >= threshold) E cujas metricas de desempenho (auc/accuracy/f1/recall/
specificity) coincidem ou sao muito proximas -- exigir tambem similaridade
de metricas evita fundir dois modelos genuinamente diferentes que por acaso
tem nomes parecidos (ex: "Gradient Boosting" vs "Extreme Gradient Boosting"
com desempenhos bem diferentes). Quando dois records sao considerados
duplicatas, mantém-se o mais "completo" (mais campos preenchidos) e descarta
o outro, registrando um aviso em extraction_warnings.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from .numparse import numbers_close
from .schema import ModelRecord, PaperExtraction

MODEL_NAME_DEDUP_THRESHOLD = 90.0  # mais estrito que o pareamento A/B (60.0): aqui e a mesma extracao
METRIC_TOLERANCE = 0.02  # 2% -- duplicatas reais devem ter valores identicos ou quase (mesmo numero, reformatado)

METRIC_FIELDS_FOR_DEDUP_CHECK = ["auc", "accuracy", "f1_score", "recall", "specificity"]


def _filled_field_count(record: ModelRecord) -> int:
    """Quantos campos EvidenceField/NumericEvidenceField tem valor preenchido -- usado para escolher o record 'mais completo' entre duplicatas."""
    count = 0
    for field_name in type(record).model_fields:
        val = getattr(record, field_name)
        if hasattr(val, "value") and val.value is not None:
            count += 1
        elif val is not None and not hasattr(val, "value"):
            count += 1
    return count


def _metrics_agree_or_absent(rec_a: ModelRecord, rec_b: ModelRecord) -> bool:
    """
    True se, para cada metrica de desempenho, os dois records concordam
    (numericamente proximos) ou pelo menos um dos dois nao reportou aquela
    metrica (None nao conta como divergencia -- so conflito explicito conta).
    """
    any_common_metric = False
    for field_name in METRIC_FIELDS_FOR_DEDUP_CHECK:
        va = getattr(rec_a, field_name).value
        vb = getattr(rec_b, field_name).value
        if va is None or vb is None:
            continue
        any_common_metric = True
        close = numbers_close(va, vb, relative_tolerance=METRIC_TOLERANCE)
        if close is False:
            return False
    # Se nenhuma metrica em comum existe para comparar, nao afirmamos que
    # concordam com confianca -- exige-se ao menos uma metrica em comum
    # coincidindo para considerar duplicata (evita fundir dois records vazios
    # so pelo nome parecido).
    return any_common_metric


def _is_duplicate_pair(rec_a: ModelRecord, rec_b: ModelRecord) -> bool:
    name_a = rec_a.model_used.value or ""
    name_b = rec_b.model_used.value or ""
    if not name_a or not name_b:
        return False
    name_score = fuzz.token_sort_ratio(name_a, name_b)
    if name_score < MODEL_NAME_DEDUP_THRESHOLD:
        return False
    return _metrics_agree_or_absent(rec_a, rec_b)


def deduplicate_records(extraction: PaperExtraction) -> PaperExtraction:
    """
    Retorna uma nova PaperExtraction com records duplicados (mesmo modelo,
    mesmas metricas) colapsados em um so. Nao modifica `extraction` in place.
    """
    records = extraction.records
    n = len(records)
    if n < 2:
        return extraction

    # union-find simples para agrupar duplicatas transitivas
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _is_duplicate_pair(records[i], records[j]):
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    kept_records = []
    warnings = list(extraction.extraction_warnings)

    for indices in groups.values():
        if len(indices) == 1:
            kept_records.append(records[indices[0]])
            continue
        # escolhe o record mais completo (mais campos preenchidos) como representante
        best_idx = max(indices, key=lambda i: _filled_field_count(records[i]))
        kept_records.append(records[best_idx])
        dropped = [i for i in indices if i != best_idx]
        dropped_names = ", ".join(records[i].model_used.value or "?" for i in dropped)
        warnings.append(
            f"Deduplicacao: {len(dropped)} registro(s) de '{records[best_idx].model_used.value}' "
            f"considerados duplicatas do mesmo modelo (nomes: {dropped_names}) e descartados -- "
            f"mantido o registro com mais campos preenchidos. Revisar se sao de fato o mesmo modelo."
        )

    return PaperExtraction(
        paper_id=extraction.paper_id,
        records=kept_records,
        extraction_warnings=warnings,
    )
