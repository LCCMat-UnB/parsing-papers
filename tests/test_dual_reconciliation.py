"""
Testes de reconciliacao A/B visivel na planilha final (tarefa: colunas
dual_model_used_b / dual_match_score / dual_divergent_fields / dual_unmatched).

Simula o formato de `comparisons` tal como pipeline.py serializa a partir de
RecordComparison/FieldDivergence (dataclasses -> dict via asdict/campo a
campo), para garantir que build_dataframe consegue reconstruir as colunas de
QA sem depender do checkpoint JSON bruto.
"""

from __future__ import annotations

from parsing_papers.consolidate import build_dataframe
from parsing_papers.schema import (
    EvidenceField,
    ModelRecord,
    NumericEvidenceField,
    PaperExtraction,
    YesNo,
)


def _ev(value):
    return EvidenceField(value=value, quote=value)


def _num(value):
    return NumericEvidenceField(value=value, quote=value)


def _record(model_name, auc="0.90"):
    return ModelRecord(
        model_used=_ev(model_name),
        baseline=YesNo.NO,
        sample_size=_num("1000"),
        default_rate=_num(None),
        financial=YesNo.YES, productive=YesNo.NO, climatic=YesNo.NO, hybrid=YesNo.NO,
        variables_used=_ev("renda"),
        auc=_num(auc),
        accuracy=_num(None), f1_score=_num(None), recall=_num(None), specificity=_num(None),
        standard_deviation=_num(None), standard_error=_num(None),
    )


def test_matched_record_with_divergence_shows_field_details():
    paper = PaperExtraction(paper_id="p1", records=[_record("Random Forest", auc="0.91")])

    comparisons = {
        "p1": {
            0: {
                "index_a": 0,
                "index_b": 0,
                "model_used_a": "Random Forest",
                "model_used_b": "Random Forest",
                "match_score": 100.0,
                "has_divergence": True,
                "divergences": [
                    {"field_name": "auc", "value_a": "0.91", "value_b": "0.60", "diverges": True, "detail": "diff relativa=0.34"},
                    {"field_name": "sample_size", "value_a": "1000", "value_b": "1000", "diverges": False, "detail": "diff relativa=0.00"},
                ],
            }
        }
    }

    df = build_dataframe([paper], dual_comparisons_by_paper=comparisons)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["dual_extraction_diverges"] == True
    assert row["dual_model_used_b"] == "Random Forest"
    assert row["dual_match_score"] == 100.0
    assert row["dual_divergent_fields"] == "auc"
    assert row["dual_unmatched"] == False
    assert row["needs_review"] == True


def test_unmatched_record_flagged_for_review():
    """Registro de A sem par em B (index_b=None) deve ser marcado dual_unmatched=True e needs_review=True, mesmo sem 'divergences'."""
    paper = PaperExtraction(paper_id="p2", records=[_record("Neural Network")])

    comparisons = {
        "p2": {
            0: {
                "index_a": 0,
                "index_b": None,
                "model_used_a": "Neural Network",
                "model_used_b": None,
                "match_score": None,
                "has_divergence": True,
                "divergences": [],
            }
        }
    }

    df = build_dataframe([paper], dual_comparisons_by_paper=comparisons)

    row = df.iloc[0]
    assert row["dual_unmatched"] == True
    assert row["dual_model_used_b"] == ""
    assert row["needs_review"] == True


def test_no_divergence_does_not_force_review():
    paper = PaperExtraction(paper_id="p3", records=[_record("XGBoost", auc="0.88")])

    comparisons = {
        "p3": {
            0: {
                "index_a": 0,
                "index_b": 0,
                "model_used_a": "XGBoost",
                "model_used_b": "XGBoost",
                "match_score": 100.0,
                "has_divergence": False,
                "divergences": [
                    {"field_name": "auc", "value_a": "0.88", "value_b": "0.88", "diverges": False, "detail": ""},
                ],
            }
        }
    }

    df = build_dataframe([paper], dual_comparisons_by_paper=comparisons)

    row = df.iloc[0]
    assert row["dual_extraction_diverges"] == False
    assert row["dual_divergent_fields"] == ""
    assert row["needs_review"] == False


def test_missing_comparison_data_does_not_crash():
    """Se nao houver dado de dupla extracao para um paper (ex: consolidacao de dado legado), a planilha ainda deve ser gerada."""
    paper = PaperExtraction(paper_id="p4", records=[_record("SVM")])

    df = build_dataframe([paper], dual_comparisons_by_paper={})

    row = df.iloc[0]
    assert row["dual_extraction_diverges"] is None
    assert row["dual_unmatched"] == False
