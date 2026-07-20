# tests/test_consolidate_reconciled.py
from parsing_papers.consolidate import build_dataframe
from parsing_papers.schema import (
    EvidenceField, ModelRecord, NumericEvidenceField, PaperExtraction, YesNo,
)


def _ev(value):
    return EvidenceField(value=value, quote=value, source_section="results")


def _num(value):
    return NumericEvidenceField(value=value, quote=value, source_section="results")


def _record(auc):
    return ModelRecord(
        model_used=_ev("XGBoost"), baseline=YesNo.NO, sample_size=_num("1000"),
        default_rate=_num(None), financial=YesNo.YES, productive=YesNo.NO,
        climatic=YesNo.NO, hybrid=YesNo.NO, variables_used=_ev("renda"),
        auc=_num(auc), accuracy=_num(None), f1_score=_num(None), recall=_num(None),
        specificity=_num(None), standard_deviation=_num(None), standard_error=_num(None),
    )


def _comparison(divergent_fields):
    return {
        "index_a": 0,
        "index_b": 0,
        "model_used_a": "XGBoost",
        "model_used_b": "XGBoost",
        "match_score": 100.0,
        "has_divergence": bool(divergent_fields),
        "divergences": [
            {"field_name": f, "value_a": "0.91", "value_b": "0.87", "diverges": True, "detail": ""}
            for f in divergent_fields
        ],
    }


def test_resolved_divergence_clears_flag_and_lists_resolution():
    extraction = PaperExtraction(paper_id="p", records=[_record("0.87")])  # reconciliada (valor de B)
    df = build_dataframe(
        [extraction],
        dual_comparisons_by_paper={"p": {0: _comparison(["auc"])}},
        arbiter_resolutions_by_paper={"p": {0: {"auc"}}},
    )
    row = df.iloc[0]
    assert row["AUC"] == "0.87"
    assert row["dual_extraction_diverges"] is False or row["dual_extraction_diverges"] == False  # noqa: E712
    assert row["dual_divergent_fields"] == ""
    assert row["arbiter_resolved_fields"] == "auc"
    assert row["needs_review"] == False  # noqa: E712


def test_neither_choice_stays_divergent():
    extraction = PaperExtraction(paper_id="p", records=[_record("0.91")])
    df = build_dataframe(
        [extraction],
        dual_comparisons_by_paper={"p": {0: _comparison(["auc"])}},
        arbiter_resolutions_by_paper={"p": {0: set()}},
    )
    row = df.iloc[0]
    assert row["dual_extraction_diverges"] == True  # noqa: E712
    assert row["needs_review"] == True  # noqa: E712


def test_no_arbiter_data_keeps_old_behavior():
    extraction = PaperExtraction(paper_id="p", records=[_record("0.91")])
    df = build_dataframe(
        [extraction],
        dual_comparisons_by_paper={"p": {0: _comparison(["auc"])}},
    )
    row = df.iloc[0]
    assert row["dual_extraction_diverges"] == True  # noqa: E712
    assert row["arbiter_resolved_fields"] == ""
