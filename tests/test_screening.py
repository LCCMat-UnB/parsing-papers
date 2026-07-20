"""
Testes da etapa 0 (triagem PRISMA de elegibilidade), cobrindo:
  - schema (ScreeningResult, decisao, vocabulario de codigos de criterio)
  - reparo de saida malformada do LLM (screening.py: _repair_screening_data)
  - integracao no pipeline (run_screening_for_paper, com LLM mockado;
    should_proceed_to_extraction como gate para a extracao de metricas)
  - planilha de triagem (build_screening_dataframe)
"""

from __future__ import annotations

import json

import pytest

from parsing_papers.pipeline import (
    build_screening_dataframe,
    export_screening_outputs,
    run_screening_for_paper,
)
from parsing_papers.screening import (
    ScreeningClient,
    ScreeningError,
    _repair_screening_data,
    should_proceed_to_extraction,
)
from parsing_papers.screening_schema import (
    ALL_CRITERIA_CODES,
    CriterionApplication,
    ScreeningDecision,
    ScreeningResult,
)


def test_all_documented_codes_present_in_vocabulary():
    """Vocabulario deve conter exatamente os codigos do documento de criterios (8 I, 11 E, 7 M, 6 ME = 32)."""
    assert len(ALL_CRITERIA_CODES) == 8 + 11 + 7 + 6
    assert "I1" in ALL_CRITERIA_CODES and "I8" in ALL_CRITERIA_CODES
    assert "E1" in ALL_CRITERIA_CODES and "E11" in ALL_CRITERIA_CODES
    assert "M1" in ALL_CRITERIA_CODES and "M7" in ALL_CRITERIA_CODES
    assert "ME1" in ALL_CRITERIA_CODES and "ME6" in ALL_CRITERIA_CODES


def test_minimal_valid_screening_result():
    result = ScreeningResult(
        paper_id="p1",
        decision=ScreeningDecision.INCLUDE_REVIEW_AND_METAANALYSIS,
        inclusion_criteria_met=[CriterionApplication(code="I4", quote="credit risk default prediction")],
        exclusion_criteria_met=[],
        metaanalysis_inclusion_criteria_met=[CriterionApplication(code="M2", quote="AUC of 0.87")],
        metaanalysis_exclusion_criteria_met=[],
        justification="Aplica modelo de ML a credit scoring com AUC reportado.",
    )
    assert result.decision == ScreeningDecision.INCLUDE_REVIEW_AND_METAANALYSIS
    assert result.inclusion_criteria_met[0].code == "I4"


def test_repair_coerces_string_list_to_criterion_objects():
    """LLM as vezes retorna so os codigos como lista de strings em vez de {code, quote}."""
    raw = {
        "paper_id": "p2",
        "decision": "include_review_only",
        "inclusion_criteria_met": ["I4", "I5"],
        "exclusion_criteria_met": [],
        "metaanalysis_inclusion_criteria_met": [],
        "metaanalysis_exclusion_criteria_met": ["ME1"],
        "justification": "ok",
    }
    fixed = _repair_screening_data(raw)
    result = ScreeningResult.model_validate(fixed)
    assert [c.code for c in result.inclusion_criteria_met] == ["I4", "I5"]
    assert result.inclusion_criteria_met[0].quote is None


def test_repair_drops_unknown_criterion_codes():
    raw = {
        "paper_id": "p3",
        "decision": "exclude",
        "inclusion_criteria_met": [],
        "exclusion_criteria_met": [{"code": "E99", "quote": "invalido"}, {"code": "E4", "quote": "sem relacao com credito"}],
        "metaanalysis_inclusion_criteria_met": [],
        "metaanalysis_exclusion_criteria_met": [],
        "justification": "fora do escopo tematico",
    }
    fixed = _repair_screening_data(raw)
    codes = [c["code"] for c in fixed["exclusion_criteria_met"]]
    assert "E99" not in codes
    assert "E4" in codes


def test_repair_normalizes_decision_aliases():
    raw = {
        "paper_id": "p4",
        "decision": "included",
        "inclusion_criteria_met": [],
        "exclusion_criteria_met": [],
        "metaanalysis_inclusion_criteria_met": [],
        "metaanalysis_exclusion_criteria_met": [],
        "justification": "ok",
    }
    fixed = _repair_screening_data(raw)
    assert fixed["decision"] == "include_review_and_metaanalysis"


def test_repair_defaults_unknown_decision_to_exclude_with_warning():
    """Decisao nao reconhecida e nao mapeavel deve virar 'exclude' por seguranca (nunca incluir silenciosamente algo incerto)."""
    raw = {
        "paper_id": "p5",
        "decision": "talvez",
        "inclusion_criteria_met": [],
        "exclusion_criteria_met": [],
        "metaanalysis_inclusion_criteria_met": [],
        "metaanalysis_exclusion_criteria_met": [],
        "justification": "ambiguo",
    }
    fixed = _repair_screening_data(raw)
    assert fixed["decision"] == "exclude"
    assert any("nao reconhecida" in w for w in fixed["screening_warnings"])


def test_should_proceed_to_extraction_gate():
    include_both = ScreeningResult(paper_id="p6", decision=ScreeningDecision.INCLUDE_REVIEW_AND_METAANALYSIS, justification="ok")
    include_review_only = ScreeningResult(paper_id="p7", decision=ScreeningDecision.INCLUDE_REVIEW_ONLY, justification="ok")
    excluded = ScreeningResult(paper_id="p8", decision=ScreeningDecision.EXCLUDE, justification="fora do escopo")

    assert should_proceed_to_extraction(include_both) is True
    assert should_proceed_to_extraction(include_review_only) is True
    assert should_proceed_to_extraction(excluded) is False


class _FakeParsedPaper:
    def __init__(self, title, abstract):
        self.title = title
        self.abstract = abstract

    def methods_and_results_text(self):
        return f"TITLE: {self.title}\nFULL TEXT FALLBACK"


class _StubScreeningClient:
    def __init__(self, result: ScreeningResult):
        self._result = result
        self.call_count = 0
        self.last_source_text = None

    def screen(self, paper_id, source_text):
        self.call_count += 1
        self.last_source_text = source_text
        return self._result


def test_run_screening_for_paper_uses_abstract_when_available(tmp_path):
    parsed = _FakeParsedPaper(title="A Study of Credit Risk", abstract="We apply XGBoost to predict default with AUC 0.9.")
    stub_result = ScreeningResult(paper_id="p9", decision=ScreeningDecision.INCLUDE_REVIEW_AND_METAANALYSIS, justification="ok")
    client = _StubScreeningClient(stub_result)

    result = run_screening_for_paper("p9", parsed, client, tmp_path / "screening")

    assert client.call_count == 1
    assert "ABSTRACT" in client.last_source_text
    assert "FULL TEXT FALLBACK" not in client.last_source_text
    assert result.decision == ScreeningDecision.INCLUDE_REVIEW_AND_METAANALYSIS
    assert (tmp_path / "screening" / "p9.json").exists()


def test_run_screening_for_paper_falls_back_to_full_text_when_no_abstract(tmp_path):
    parsed = _FakeParsedPaper(title="Some Paper", abstract="")
    stub_result = ScreeningResult(paper_id="p10", decision=ScreeningDecision.EXCLUDE, justification="fora do escopo")
    client = _StubScreeningClient(stub_result)

    run_screening_for_paper("p10", parsed, client, tmp_path / "screening")

    assert "FULL TEXT FALLBACK" in client.last_source_text


def test_run_screening_for_paper_skips_if_checkpoint_exists(tmp_path):
    parsed = _FakeParsedPaper(title="X", abstract="Y")
    stub_result = ScreeningResult(paper_id="p11", decision=ScreeningDecision.INCLUDE_REVIEW_ONLY, justification="ok")
    client = _StubScreeningClient(stub_result)
    screening_dir = tmp_path / "screening"

    run_screening_for_paper("p11", parsed, client, screening_dir)
    assert client.call_count == 1

    # segunda chamada sem force=True deve reusar o checkpoint, nao chamar o LLM de novo
    result2 = run_screening_for_paper("p11", parsed, client, screening_dir)
    assert client.call_count == 1
    assert result2.decision == ScreeningDecision.INCLUDE_REVIEW_ONLY


def test_run_screening_for_paper_returns_none_on_screening_error(tmp_path):
    class _FailingClient:
        def screen(self, paper_id, source_text):
            raise ScreeningError("LLM indisponivel")

    parsed = _FakeParsedPaper(title="X", abstract="Y")
    result = run_screening_for_paper("p12", parsed, _FailingClient(), tmp_path / "screening")
    assert result is None


def test_build_screening_dataframe_includes_all_categories():
    screening_results = [
        {
            "paper_id": "p13",
            "decision": "include_review_and_metaanalysis",
            "inclusion_criteria_met": [{"code": "I4", "quote": "credit risk"}],
            "exclusion_criteria_met": [],
            "metaanalysis_inclusion_criteria_met": [{"code": "M2", "quote": "AUC 0.9"}],
            "metaanalysis_exclusion_criteria_met": [],
            "justification": "atende todos os criterios",
            "screening_warnings": [],
        },
        {
            "paper_id": "p14",
            "decision": "exclude",
            "inclusion_criteria_met": [],
            "exclusion_criteria_met": [{"code": "E4", "quote": "market risk only"}],
            "metaanalysis_inclusion_criteria_met": [],
            "metaanalysis_exclusion_criteria_met": [],
            "justification": "fora do escopo tematico",
            "screening_warnings": ["abstract truncado"],
        },
    ]

    df = build_screening_dataframe(screening_results)

    assert len(df) == 2
    assert df.iloc[0]["decision"] == "include_review_and_metaanalysis"
    assert df.iloc[0]["inclusion_criteria_met"] == "I4"
    assert df.iloc[0]["metaanalysis_inclusion_criteria_met"] == "M2"
    assert df.iloc[1]["decision"] == "exclude"
    assert df.iloc[1]["exclusion_criteria_met"] == "E4"
    assert df.iloc[1]["screening_warnings"] == "abstract truncado"


def test_export_screening_outputs_creates_files(tmp_path):
    screening_results = [
        {
            "paper_id": "p15",
            "decision": "include_review_only",
            "inclusion_criteria_met": [],
            "exclusion_criteria_met": [],
            "metaanalysis_inclusion_criteria_met": [],
            "metaanalysis_exclusion_criteria_met": [{"code": "ME1", "quote": None}],
            "justification": "sem valor numerico de metrica",
            "screening_warnings": [],
        },
    ]
    xlsx_path, csv_path = export_screening_outputs(screening_results, tmp_path)
    assert xlsx_path.exists()
    assert csv_path.exists()
    assert xlsx_path.name == "triagem_prisma.xlsx"
