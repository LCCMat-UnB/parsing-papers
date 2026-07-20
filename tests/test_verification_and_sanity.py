import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parsing_papers.verification import verify_quote
from parsing_papers.sanity_checks import check_record, record_has_failures
from parsing_papers.numparse import parse_float, parse_int
from parsing_papers.schema import ModelRecord, EvidenceField, NumericEvidenceField, YesNo

SOURCE = """
The Random Forest model achieved an AUC of 0.87 on the test set, using a
sample of 1,200 farmers from the southern region. The logistic regression
baseline achieved an AUC of 0.79.
"""


def test_verify_quote_exact_match():
    verified, score, reason = verify_quote("AUC of 0.87 on the test set", SOURCE)
    assert verified
    assert score >= 90


def test_verify_quote_hallucinated():
    verified, score, reason = verify_quote("the model reached 99.9% accuracy on unseen countries", SOURCE)
    assert not verified


def test_verify_quote_empty():
    verified, score, reason = verify_quote(None, SOURCE)
    assert not verified


def _record(auc=None, accuracy=None, sample_size=None, default_rate=None, sd=None):
    return ModelRecord(
        model_used=EvidenceField(value="RF", quote="Random Forest model"),
        baseline=YesNo.NO,
        sample_size=NumericEvidenceField(value=sample_size, quote="x"),
        default_rate=NumericEvidenceField(value=default_rate, quote="x"),
        financial=YesNo.YES, productive=YesNo.NO, climatic=YesNo.NO, hybrid=YesNo.NO,
        variables_used=EvidenceField(value="income", quote="income"),
        auc=NumericEvidenceField(value=auc, quote="x"),
        accuracy=NumericEvidenceField(value=accuracy, quote="x"),
        f1_score=NumericEvidenceField(value=None, quote=None),
        recall=NumericEvidenceField(value=None, quote=None),
        specificity=NumericEvidenceField(value=None, quote=None),
        standard_deviation=NumericEvidenceField(value=sd, quote="x"),
        standard_error=NumericEvidenceField(value=None, quote=None),
    )


def test_sanity_auc_in_range():
    rec = _record(auc="0.87")
    flags = check_record(rec)
    auc_flag = next(f for f in flags if f.field_name == "auc")
    assert auc_flag.ok


def test_sanity_auc_out_of_range():
    rec = _record(auc="1.5")  # AUC impossivel
    flags = check_record(rec)
    auc_flag = next(f for f in flags if f.field_name == "auc")
    assert not auc_flag.ok
    assert record_has_failures(flags)


def test_sanity_auc_percent_like_value_below_one_is_suspicious_but_parsed():
    rec = _record(auc="0.3")  # abaixo de 0.5 -- pior que aleatorio, deve falhar
    flags = check_record(rec)
    auc_flag = next(f for f in flags if f.field_name == "auc")
    assert not auc_flag.ok


def test_sanity_sample_size_positive_int():
    rec = _record(sample_size="1,200")
    flags = check_record(rec)
    ss_flag = next(f for f in flags if f.field_name == "sample_size")
    assert ss_flag.ok
    assert ss_flag.parsed_value == 1200


def test_sanity_default_rate_percent():
    rec = _record(default_rate="12%")
    flags = check_record(rec)
    dr_flag = next(f for f in flags if f.field_name == "default_rate")
    assert dr_flag.ok
    assert abs(dr_flag.parsed_value - 0.12) < 1e-6


def test_parse_numeric_and_int_helpers():
    # sanity_checks agora usa numparse.parse_float/parse_int (ver test_numparse.py
    # para cobertura completa) -- mantido aqui so como smoke test de integracao.
    assert parse_float("0.87") == 0.87
    assert parse_float("87%") == 0.87
    assert parse_int("1,200") == 1200
    assert parse_int("1200 farmers") == 1200


def test_sanity_flags_prose_value_as_wrong_shape():
    """Caso real observado: sample_size.value veio como um paragrafo em ingles
    parafraseado ('They analyzed a sample of...') em vez do numero pontual.
    O quote estava correto (citacao literal em PT), mas o value era inutil
    para a planilha -- e um sanity check baseado so em 'e um inteiro positivo'
    nao pegava isso, porque o parser encontrava um numero (errado) escondido
    na frase (ex: 'R$ 25 mil' em vez do tamanho real da amostra)."""
    rec = _record()
    rec.sample_size.value = (
        "They analyzed a sample of micro, small and medium-sized enterprises "
        "(MPMEs) that had up to R$ 25 thousand credit limit with an automobile "
        "rental company operating throughout Brazil."
    )
    flags = check_record(rec)
    shape_flag = next((f for f in flags if f.field_name == "sample_size.value_shape"), None)
    assert shape_flag is not None
    assert not shape_flag.ok
    assert record_has_failures(flags)


def test_sanity_does_not_flag_short_values():
    rec = _record(auc="0.87", sample_size="3844")
    flags = check_record(rec)
    shape_flags = [f for f in flags if f.field_name.endswith(".value_shape")]
    assert shape_flags == []
