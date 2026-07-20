import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parsing_papers.numparse import parse_float, parse_int, parse_number, numbers_close


def test_simple_decimal():
    assert parse_float("0.87") == 0.87


def test_percent():
    assert parse_float("87%") == 0.87


def test_percent_with_comma_decimal():
    assert abs(parse_float("63,02%") - 0.6302) < 1e-9


def test_thousands_dot():
    assert parse_float("1.234") == 1234.0


def test_thousands_comma():
    assert parse_float("1,234") == 1234.0


def test_full_ptbr_format():
    assert parse_float("1.234,56") == 1234.56


def test_full_en_format():
    assert parse_float("1,234.56") == 1234.56


def test_sample_size_thousands_not_decimal():
    """3.844 e um tamanho de amostra (3844 empresas), nao 3.844 (tres virgula
    oito quatro quatro) -- caso real do paper sobre credit scoring."""
    assert parse_float("3.844") == 3844.0


def test_metric_context_percent_heuristic():
    assert parse_float("AUC=0.91") == 0.91


def test_none_returns_none():
    assert parse_float(None) is None


def test_no_number_returns_none():
    assert parse_float("no data available") is None


def test_parse_int_thousands():
    assert parse_int("1,200") == 1200
    assert parse_int("3.844") == 3844


def test_parse_int_with_trailing_text():
    assert parse_int("1200 farmers") == 1200


def test_negative_number():
    assert parse_float("-0.5") == -0.5


def test_numbers_close_within_tolerance():
    assert numbers_close("0.87", "0.88") is True


def test_numbers_close_outside_tolerance():
    assert numbers_close("0.87", "0.99") is False


def test_numbers_close_none_when_unparseable():
    assert numbers_close(None, "0.87") is None
    assert numbers_close("n/a", "0.87") is None


def test_consistency_between_parse_float_and_parse_int():
    """Regressao do bug original: as duas funcoes devem concordar sobre o
    mesmo numero em vez de usar heuristicas diferentes."""
    raw = "1.234,56"
    assert parse_int(raw) == round(parse_float(raw))
