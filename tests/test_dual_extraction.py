"""
Testes do pareamento A/B por nome de modelo (etapa 4 / PRISMA item 9).

Cobrem o bug que o pareamento por posicao tinha: se a extracao B lista os
mesmos modelos em ordem diferente da extracao A, o pareamento por indice
gerava divergencia espuria em quase todos os campos (comparava
Random Forest de A com XGBoost de B, por exemplo). O pareamento por nome
(fuzzy match em model_used) deve resolver isso.
"""

from __future__ import annotations

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


def _record(model_name, auc="0.90", sample_size="1000"):
    return ModelRecord(
        model_used=_ev(model_name),
        baseline=YesNo.NO,
        sample_size=_num(sample_size),
        default_rate=_num(None),
        financial=YesNo.YES,
        productive=YesNo.NO,
        climatic=YesNo.NO,
        hybrid=YesNo.NO,
        variables_used=_ev("renda, historico de credito"),
        auc=_num(auc),
        accuracy=_num(None),
        f1_score=_num(None),
        recall=_num(None),
        specificity=_num(None),
        standard_deviation=_num(None),
        standard_error=_num(None),
    )


def test_same_models_different_order_are_paired_correctly():
    """
    A lista [Random Forest, XGBoost, Logistic Regression]; B lista os mesmos
    3 modelos em ordem diferente com valores identicos. Pareamento por nome
    deve casar cada um com seu correspondente e NAO reportar divergencia --
    pareamento por posicao reportaria divergencia em quase todo campo aqui.
    """
    extraction_a = PaperExtraction(
        paper_id="p1",
        records=[
            _record("Random Forest", auc="0.91"),
            _record("XGBoost", auc="0.88"),
            _record("Logistic Regression", auc="0.75"),
        ],
    )
    extraction_b = PaperExtraction(
        paper_id="p1",
        records=[
            _record("Logistic Regression", auc="0.75"),
            _record("Random Forest", auc="0.91"),
            _record("XGBoost", auc="0.88"),
        ],
    )

    result = compare_extractions(extraction_a, extraction_b)

    assert result.record_count_mismatch is False
    assert len(result.comparisons) == 3
    for comp in result.comparisons:
        assert comp.index_a is not None
        assert comp.index_b is not None
        assert comp.model_used_a == comp.model_used_b
        assert comp.has_divergence is False


def test_model_only_in_a_is_unmatched_not_forced():
    """
    A tem um modelo extra que B nao reportou (ex: B parou cedo). Esse record
    deve ficar como nao pareado (index_b=None) em vez de ser forcado contra um
    modelo diferente de B, o que geraria divergencia espuria em todos os campos.
    """
    extraction_a = PaperExtraction(
        paper_id="p2",
        records=[_record("Random Forest"), _record("Neural Network")],
    )
    extraction_b = PaperExtraction(
        paper_id="p2",
        records=[_record("Random Forest")],
    )

    result = compare_extractions(extraction_a, extraction_b)

    assert result.record_count_mismatch is True
    matched = [c for c in result.comparisons if c.index_a is not None and c.index_b is not None]
    unmatched_a = [c for c in result.comparisons if c.index_a is not None and c.index_b is None]
    assert len(matched) == 1
    assert matched[0].model_used_a == "Random Forest"
    assert len(unmatched_a) == 1
    assert unmatched_a[0].model_used_a == "Neural Network"
    assert unmatched_a[0].has_divergence is True  # nao pareado conta como divergencia (precisa revisao)


def test_real_numeric_divergence_still_detected_after_pairing():
    """Depois de parear corretamente por nome, uma divergencia numerica real ainda deve ser sinalizada."""
    extraction_a = PaperExtraction(paper_id="p3", records=[_record("XGBoost", auc="0.95")])
    extraction_b = PaperExtraction(paper_id="p3", records=[_record("XGBoost", auc="0.60")])

    result = compare_extractions(extraction_a, extraction_b)

    assert len(result.comparisons) == 1
    comp = result.comparisons[0]
    assert comp.index_a is not None and comp.index_b is not None
    assert comp.has_divergence is True
    auc_div = next(d for d in comp.divergences if d.field_name == "auc")
    assert auc_div.diverges is True


def test_similar_but_distinct_model_names_are_paired_by_fuzzy_match():
    """
    Pequenas variacoes de grafia (ex: 'XGBoost' vs 'XG Boost') ainda devem
    parear como o mesmo registro (score de pareamento acima do minimo) mesmo
    que o campo model_used em si fique sinalizado como divergente no relatorio
    de campo -- pareamento e comparacao de campo sao decisoes independentes:
    o pareamento so precisa achar "e o mesmo modelo", a comparacao de campo
    ainda deve expor a diferenca textual para revisao humana.
    """
    extraction_a = PaperExtraction(paper_id="p4", records=[_record("XGBoost", auc="0.90")])
    extraction_b = PaperExtraction(paper_id="p4", records=[_record("XG Boost", auc="0.90")])

    result = compare_extractions(extraction_a, extraction_b)

    assert len(result.comparisons) == 1
    comp = result.comparisons[0]
    # o ponto central do teste: foram pareados como o MESMO registro (nao
    # ficaram como dois registros nao pareados, um so em A e outro so em B)
    assert comp.index_a is not None and comp.index_b is not None
    assert comp.match_score is not None and comp.match_score >= 60.0
    # todos os campos NAO textuais (numericos/categoricos) devem bater --
    # so o proprio campo model_used e esperado divergir (grafia diferente),
    # o que corretamente marca has_divergence=True para revisao humana.
    non_name_divergences = [d for d in comp.divergences if d.field_name != "model_used" and d.diverges]
    assert non_name_divergences == []
    model_used_div = next(d for d in comp.divergences if d.field_name == "model_used")
    assert model_used_div.diverges is True


def test_extractor_b_inherits_min_num_ctx():
    """Bug: o extractor_b interno era construido sem min_num_ctx e caia no
    default 16000 da classe mesmo quando A pedia mais -- B podia ter o prompt
    truncado em papers grandes. As duas chamadas devem pedir o mesmo num_ctx."""
    from unittest.mock import patch

    from parsing_papers.llm_client import LLMExtractor
    from parsing_papers.dual_extraction import run_dual_extraction
    from parsing_papers.schema import PaperExtraction

    seen_num_ctx = []

    def fake_completion(**kwargs):
        seen_num_ctx.append(kwargs.get("num_ctx"))
        return {"choices": [{"message": {"content": '{"paper_id": "p", "records": []}'}}]}

    extractor_a = LLMExtractor(
        model="ollama_chat/fake", api_base="http://localhost:1",
        temperature=0.1, max_tokens=100, request_timeout=10, min_num_ctx=12345,
    )
    with patch("litellm.completion", side_effect=fake_completion):
        run_dual_extraction("p", "texto curto", extractor_a)

    assert seen_num_ctx == [12345, 12345]
