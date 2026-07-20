"""
Testes de deduplicacao de modelo repetido dentro da MESMA extracao de um paper.

Caso real motivador: o LLM as vezes gera dois ModelRecord para o mesmo modelo
(ex: "Random Forest" citado em metodos e de novo em resultados/tabela),
inflando artificialmente o numero de "modelos testados" na planilha final.
"""

from __future__ import annotations

from parsing_papers.dedup import deduplicate_records
from parsing_papers.schema import (
    EvidenceField,
    ModelRecord,
    NumericEvidenceField,
    PaperExtraction,
    YesNo,
)


def _ev(value):
    return EvidenceField(value=value, quote=value, source_section="results")


def _num(value):
    return NumericEvidenceField(value=value, quote=value, source_section="results")


def _record(model_name, auc="0.90", accuracy=None, extra_notes=None):
    return ModelRecord(
        model_used=_ev(model_name),
        baseline=YesNo.NO,
        sample_size=_num("1000"),
        default_rate=_num(None),
        financial=YesNo.YES,
        productive=YesNo.NO,
        climatic=YesNo.NO,
        hybrid=YesNo.NO,
        variables_used=_ev("renda, historico de credito"),
        auc=_num(auc),
        accuracy=_num(accuracy),
        f1_score=_num(None),
        recall=_num(None),
        specificity=_num(None),
        standard_deviation=_num(None),
        standard_error=_num(None),
        extraction_notes=extra_notes,
    )


def test_exact_duplicate_model_is_collapsed():
    """Mesmo nome, mesma metrica (mencionado 2x) -- deve virar 1 record so."""
    extraction = PaperExtraction(
        paper_id="p1",
        records=[
            _record("Random Forest", auc="0.91"),
            _record("Random Forest", auc="0.91"),
            _record("XGBoost", auc="0.88"),
        ],
    )

    result = deduplicate_records(extraction)

    assert len(result.records) == 2
    names = sorted(r.model_used.value for r in result.records)
    assert names == ["Random Forest", "XGBoost"]
    assert any("Deduplicacao" in w for w in result.extraction_warnings)


def test_more_complete_record_is_kept():
    """Entre duas duplicatas, mantém a que tem mais campos preenchidos."""
    less_complete = _record("Random Forest", auc="0.91", accuracy=None)
    more_complete = _record("Random Forest", auc="0.91", accuracy="0.85", extra_notes="usa 5-fold CV")
    extraction = PaperExtraction(paper_id="p2", records=[less_complete, more_complete])

    result = deduplicate_records(extraction)

    assert len(result.records) == 1
    assert result.records[0].accuracy.value == "0.85"
    assert result.records[0].extraction_notes == "usa 5-fold CV"


def test_same_name_different_metrics_not_merged():
    """
    Nome igual mas metricas claramente diferentes (ex: dois experimentos
    distintos com o mesmo tipo de modelo, ou erro de digitacao no nome que
    na verdade sao modelos diferentes) NAO deve ser deduplicado -- fundir
    esconderia dados legitimamente distintos.
    """
    extraction = PaperExtraction(
        paper_id="p3",
        records=[
            _record("Random Forest", auc="0.91"),
            _record("Random Forest", auc="0.60"),  # muito diferente -- provavelmente outro experimento
        ],
    )

    result = deduplicate_records(extraction)

    assert len(result.records) == 2


def test_different_models_not_merged():
    """Modelos genuinamente diferentes (nome e metricas diferentes) permanecem separados."""
    extraction = PaperExtraction(
        paper_id="p4",
        records=[
            _record("Random Forest", auc="0.91"),
            _record("XGBoost", auc="0.88"),
            _record("Logistic Regression", auc="0.75"),
        ],
    )

    result = deduplicate_records(extraction)

    assert len(result.records) == 3


def test_single_record_untouched():
    extraction = PaperExtraction(paper_id="p5", records=[_record("Random Forest")])
    result = deduplicate_records(extraction)
    assert len(result.records) == 1
    assert result.extraction_warnings == []


def test_empty_records_untouched():
    extraction = PaperExtraction(paper_id="p6", records=[])
    result = deduplicate_records(extraction)
    assert result.records == []
