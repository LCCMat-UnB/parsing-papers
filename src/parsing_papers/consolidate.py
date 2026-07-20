"""
Consolida os resultados de todos os papers processados na tabela de metadados
final pedida pelo projeto, e exporta para .xlsx e .csv.

Cada linha = um modelo (ModelRecord) de um paper. Colunas extraidas por este
pipeline sao combinadas com placeholders para os metadados extraidos por
outros metodos (doi, year, author, journal, country, latitude, longitude),
para que a planilha final tenha o layout completo pedido no projeto -- essas
colunas ficam vazias aqui e devem ser preenchidas/mescladas pela etapa que
cuida da identificacao bibliografica.

Quando rodado via `pipeline.py registry-run` (integracao com SPE/pontodoi,
ver INTEGRATION.md e registry.py no repo synoptic-paper-engine), essas
colunas SAO preenchidas automaticamente a partir do bloco `metadata` do
registry compartilhado -- ver registry.merge_bibliographic_columns(). Elas
so ficam vazias no modo standalone (`pipeline.py run` direto sobre um
--pdf-dir, sem integracao).

Reconciliacao A/B (espirito PRISMA item 9): alem do flag simples
`dual_extraction_diverges`, a planilha final traz o detalhe de QUAIS campos
divergiram e com QUE modelo da extracao B o registro foi pareado -- sem isso,
o revisor humano teria que abrir o checkpoint JSON bruto para entender o que
exatamente diverge em cada linha sinalizada.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .sanity_checks import check_record, record_has_failures
from .schema import PaperExtraction
from .verification import RecordVerificationResult

BIBLIOGRAPHIC_PLACEHOLDER_COLUMNS = [
    "doi", "Year", "Author", "Journal", "Country", "Latitude", "Longitude",
]

OUTPUT_COLUMNS = BIBLIOGRAPHIC_PLACEHOLDER_COLUMNS + [
    "paper_id",
    "Model_used",
    "Model_class",
    "Baseline",
    "Sample_size",
    "Default_rate",
    "Type_data_used",
    "Financial",
    "Productive",
    "Climatic",
    "Hybrid",
    "Variables_used",
    "Agricultural_context",
    "AUC",
    "Accuracy",
    "F1-score",
    "Recall",
    "Specificity",
    "Standard_deviation",
    "Standard_error",
    # metadados de proveniencia / QA (nao fazem parte da planilha "oficial" mas
    # sao essenciais para auditoria -- podem ser removidos/ocultados na entrega final)
    "citation_verified",
    "unverified_fields",
    "sanity_failed",
    "sanity_messages",
    "dual_extraction_diverges",
    "dual_model_used_b",
    "dual_match_score",
    "dual_divergent_fields",
    "dual_unmatched",
    "arbiter_resolved_fields",
    "needs_review",
    "extraction_notes",
]


def _dual_comparison_summary(comparison: dict | None, resolved_fields: set[str] | None = None) -> dict:
    """
    Extrai da comparacao A/B (dict serializado a partir de RecordComparison,
    ver pipeline.py) os campos que vao para a planilha final. `comparison` e
    None quando nao ha checkpoint de dupla extracao disponivel para o registro
    (ex: consolidacao de dados legados).

    resolved_fields: campos ja decididos pelo arbitro (choice "a" ou "b") --
    sao REMOVIDOS da lista de divergentes e reportados em
    arbiter_resolved_fields. dual_extraction_diverges passa a significar
    "diverge E nao foi resolvido", que e o que merece revisao humana.
    """
    resolved_fields = resolved_fields or set()
    if comparison is None:
        return {
            "dual_extraction_diverges": None,
            "dual_model_used_b": "",
            "dual_match_score": None,
            "dual_divergent_fields": "",
            "dual_unmatched": False,
            "arbiter_resolved_fields": "",
        }

    unmatched = comparison.get("index_b") is None
    remaining_divergent = [
        d["field_name"]
        for d in comparison.get("divergences", [])
        if d.get("diverges") and d["field_name"] not in resolved_fields
    ]
    return {
        "dual_extraction_diverges": bool(remaining_divergent) or unmatched,
        "dual_model_used_b": comparison.get("model_used_b") or "",
        "dual_match_score": comparison.get("match_score"),
        "dual_divergent_fields": ", ".join(remaining_divergent),
        "dual_unmatched": unmatched,
        "arbiter_resolved_fields": ", ".join(sorted(resolved_fields)),
    }


def record_to_row(
    paper_id: str,
    record,
    verification: RecordVerificationResult | None = None,
    dual_comparison: dict | None = None,
    arbiter_resolved: set[str] | None = None,
) -> dict:
    flags = check_record(record)
    sanity_failed = record_has_failures(flags)
    sanity_messages = "; ".join(f"{f.field_name}: {f.message}" for f in flags if not f.ok)

    citation_verified = verification.all_verified if verification else None
    unverified_fields = ", ".join(verification.unverified_fields) if verification else ""

    dual_summary = _dual_comparison_summary(dual_comparison, arbiter_resolved)

    needs_review = bool(
        sanity_failed
        or (verification and not verification.all_verified)
        or dual_summary["dual_extraction_diverges"]
        or dual_summary["dual_unmatched"]
    )

    row = {c: "" for c in BIBLIOGRAPHIC_PLACEHOLDER_COLUMNS}
    row.update(
        {
            "paper_id": paper_id,
            "Model_used": record.model_used.value,
            "Model_class": record.model_class.value if record.model_class else "",
            "Baseline": record.baseline.value,
            "Sample_size": record.sample_size.value,
            "Default_rate": record.default_rate.value,
            "Type_data_used": record.type_data_used.value if record.type_data_used else "",
            "Financial": record.financial.value,
            "Productive": record.productive.value,
            "Climatic": record.climatic.value,
            "Hybrid": record.hybrid.value,
            "Variables_used": record.variables_used.value,
            "Agricultural_context": record.agricultural_context.value if record.agricultural_context else "",
            "AUC": record.auc.value,
            "Accuracy": record.accuracy.value,
            "F1-score": record.f1_score.value,
            "Recall": record.recall.value,
            "Specificity": record.specificity.value,
            "Standard_deviation": record.standard_deviation.value,
            "Standard_error": record.standard_error.value,
            "citation_verified": citation_verified,
            "unverified_fields": unverified_fields,
            "sanity_failed": sanity_failed,
            "sanity_messages": sanity_messages,
            "extraction_notes": record.extraction_notes or "",
            "needs_review": needs_review,
        }
    )
    row.update(dual_summary)
    return row


def build_dataframe(
    extractions: list[PaperExtraction],
    verifications_by_paper: dict[str, list[RecordVerificationResult]] | None = None,
    dual_comparisons_by_paper: dict[str, dict[int, dict]] | None = None,
    arbiter_resolutions_by_paper: dict[str, dict[int, set[str]]] | None = None,
) -> pd.DataFrame:
    """
    dual_comparisons_by_paper: paper_id -> {index_a: comparison_dict}, onde
    comparison_dict e o dict serializado de um RecordComparison (ver
    pipeline.py: result["comparisons"]). Indexado por index_a (posicao do
    registro dentro de extraction_a.records) em vez de posicao na lista de
    comparacoes, porque comparacoes de registros "so em B" nao tem index_a e
    nao devem ser confundidas com as de registros de A.

    arbiter_resolutions_by_paper: paper_id -> {index_a: {field_name, ...}}
    com os campos ja resolvidos pelo arbitro (ver pipeline.py:
    result["arbiter_resolutions"]).
    """
    rows = []
    for extraction in extractions:
        verifications = (verifications_by_paper or {}).get(extraction.paper_id)
        comparisons_by_index_a = (dual_comparisons_by_paper or {}).get(extraction.paper_id, {})
        resolutions_by_index_a = (arbiter_resolutions_by_paper or {}).get(extraction.paper_id, {})
        for idx, record in enumerate(extraction.records):
            verification = verifications[idx] if verifications and idx < len(verifications) else None
            dual_comparison = comparisons_by_index_a.get(idx)
            arbiter_resolved = resolutions_by_index_a.get(idx)
            rows.append(record_to_row(extraction.paper_id, record, verification, dual_comparison, arbiter_resolved))

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return df


def export(df: pd.DataFrame, out_dir: str | Path, basename: str = "metadados_extraidos") -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / f"{basename}.xlsx"
    csv_path = out_dir / f"{basename}.csv"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="metadados")
        ws = writer.sheets["metadados"]
        # auto-width simples
        for i, col in enumerate(df.columns, start=1):
            max_len = max([len(str(col))] + [len(str(v)) for v in df[col].astype(str).tolist()[:500]])
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(max_len + 2, 60)

    return xlsx_path, csv_path
