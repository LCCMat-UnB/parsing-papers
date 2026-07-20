# tests/test_arbiter.py
from parsing_papers.arbiter import (
    ArbiterClient,
    ArbiterRecordDecision,
    apply_arbiter_decisions,
)
from parsing_papers.dual_extraction import compare_extractions
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


def _record(model_name, auc="0.90", baseline=YesNo.NO):
    return ModelRecord(
        model_used=_ev(model_name),
        baseline=baseline,
        sample_size=_num("1000"),
        default_rate=_num(None),
        financial=YesNo.YES,
        productive=YesNo.NO,
        climatic=YesNo.NO,
        hybrid=YesNo.NO,
        variables_used=_ev("renda"),
        auc=_num(auc),
        accuracy=_num(None),
        f1_score=_num(None),
        recall=_num(None),
        specificity=_num(None),
        standard_deviation=_num(None),
        standard_error=_num(None),
    )


def _dual():
    a = PaperExtraction(paper_id="p", records=[_record("XGBoost", auc="0.91", baseline=YesNo.NO)])
    b = PaperExtraction(paper_id="p", records=[_record("XGBoost", auc="0.87", baseline=YesNo.YES)])
    # tolerancia mais estrita que o default (5%): 0.91 vs 0.87 tem diff
    # relativa ~4.4% e NAO divergiria; o teste precisa de auc divergente.
    return a, b, compare_extractions(a, b, numeric_tolerance=0.01)


def test_apply_choice_b_replaces_evidence_and_categorical():
    a, b, result = _dual()
    decision = ArbiterRecordDecision.model_validate(
        {"decisions": [{"field_name": "auc", "choice": "b"}, {"field_name": "baseline", "choice": "b"}]}
    )
    reconciled, resolutions = apply_arbiter_decisions(a, b, result.comparisons, {0: decision})
    assert reconciled.records[0].auc.value == "0.87"
    assert reconciled.records[0].baseline == YesNo.YES
    assert ("arbitro" in (reconciled.records[0].extraction_notes or ""))
    assert {r["field_name"] for r in resolutions} == {"auc", "baseline"}
    # extracao A original NAO pode ter sido mutada
    assert a.records[0].auc.value == "0.91"


def test_apply_choice_a_and_neither():
    a, b, result = _dual()
    decision = ArbiterRecordDecision.model_validate(
        {"decisions": [{"field_name": "auc", "choice": "a"}, {"field_name": "baseline", "choice": "neither"}]}
    )
    reconciled, resolutions = apply_arbiter_decisions(a, b, result.comparisons, {0: decision})
    assert reconciled.records[0].auc.value == "0.91"
    assert reconciled.records[0].baseline == YesNo.NO
    # "neither" NAO resolve -- fica de fora das resolucoes (continua needs_review)
    assert [r["field_name"] for r in resolutions] == ["auc"]


class _StubExtractor:
    def __init__(self, raw):
        self.raw = raw

    def complete(self, messages, **kwargs):
        return self.raw


def test_arbitrate_record_parses_json():
    client = ArbiterClient(_StubExtractor('{"decisions": [{"field_name": "auc", "choice": "b"}]}'))
    _, _, result = _dual()
    divergent = [d for d in result.comparisons[0].divergences if d.diverges]
    out = client.arbitrate_record("p", "XGBoost", divergent, "TEXTO FONTE")
    assert out is not None and out.decisions[0].choice == "b"


def test_arbitrate_record_returns_none_on_garbage():
    client = ArbiterClient(_StubExtractor("isso nao e JSON"))
    _, _, result = _dual()
    divergent = [d for d in result.comparisons[0].divergences if d.diverges]
    assert client.arbitrate_record("p", "XGBoost", divergent, "TEXTO") is None
