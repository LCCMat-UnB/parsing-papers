# tests/test_pipeline_windows.py
import json
from pathlib import Path

from parsing_papers.pipeline import process_one_pdf
from parsing_papers.grobid_client import GrobidSection, ParsedPaper
from parsing_papers.schema import (
    EvidenceField, ModelRecord, NumericEvidenceField, PaperExtraction, YesNo,
)


def _ev(value):
    return EvidenceField(value=value, quote=value, source_section="results")


def _num(value):
    return NumericEvidenceField(value=value, quote=value, source_section="results")


def _record():
    return ModelRecord(
        model_used=_ev("XGBoost"), baseline=YesNo.NO, sample_size=_num("1000"),
        default_rate=_num(None), financial=YesNo.YES, productive=YesNo.NO,
        climatic=YesNo.NO, hybrid=YesNo.NO, variables_used=_ev("renda"),
        auc=_num("0.91"), accuracy=_num(None), f1_score=_num(None), recall=_num(None),
        specificity=_num(None), standard_deviation=_num(None), standard_error=_num(None),
    )


class _StubGrobid:
    def parse_pdf(self, pdf_path, tei_cache_dir=None):
        return ParsedPaper(
            paper_id=Path(pdf_path).stem, title="T", abstract="A",
            sections=[
                GrobidSection(header="Intro", text="lorem ipsum " * 200),
                GrobidSection(header="Results", text="XGBoost AUC 0.91 accuracy 0.87"),
            ],
            tables=[],
        )


class _StubExtractor:
    """Simula LLMExtractor: registra os textos recebidos e roteia respostas."""

    model = "stub"
    api_base = None
    temperature = 0.1
    max_tokens = 8000
    request_timeout = 10
    min_num_ctx = 100

    def __init__(self, results):
        self.results = list(results)  # PaperExtraction por chamada, em ordem
        self.seen_texts = []

    def extract(self, paper_id, source_text):
        self.seen_texts.append(source_text)
        return self.results.pop(0)


def _empty():
    return PaperExtraction(paper_id="x", records=[])


def _filled():
    return PaperExtraction(paper_id="x", records=[_record()])


def test_escalation_on_empty_extraction(tmp_path):
    (tmp_path / "paper1.pdf").touch()
    extractor_a = _StubExtractor([_empty(), _filled(), _filled()])  # A1 vazia, A2 ok, B
    extractor_b = _StubExtractor([_filled()])

    result = process_one_pdf(
        tmp_path / "paper1.pdf",
        grobid=_StubGrobid(),
        extractor_a=extractor_a,
        extractor_b=extractor_b,
        tei_cache_dir=tmp_path / "tei",
        checkpoint_dir=tmp_path / "ckpt",
        citation_threshold=90.0,
        fallback_budgets=[50, 100000, None],
    )

    assert result is not None
    assert result["fallback_level"] == 1
    assert len(extractor_a.seen_texts) == 2
    assert len(extractor_a.seen_texts[0]) < len(extractor_a.seen_texts[1])
    ckpt = json.loads((tmp_path / "ckpt" / "paper1.json").read_text(encoding="utf-8"))
    assert "extraction_reconciled" in ckpt
    assert ckpt["arbiter_resolutions"] == []
    assert ckpt["estimated_prompt_tokens"] > 0
    assert "timing" in ckpt


def test_no_escalation_when_first_window_has_records(tmp_path):
    (tmp_path / "paper2.pdf").touch()
    extractor_a = _StubExtractor([_filled(), _filled()])

    result = process_one_pdf(
        tmp_path / "paper2.pdf",
        grobid=_StubGrobid(),
        extractor_a=extractor_a,
        extractor_b=_StubExtractor([_filled()]),
        tei_cache_dir=tmp_path / "tei",
        checkpoint_dir=tmp_path / "ckpt",
        citation_threshold=90.0,
        fallback_budgets=[100000, None],
    )

    assert result["fallback_level"] == 0
    assert len(extractor_a.seen_texts) == 1


def test_default_budgets_keep_old_fulltext_behavior(tmp_path):
    (tmp_path / "paper3.pdf").touch()
    extractor_a = _StubExtractor([_filled(), _filled()])

    result = process_one_pdf(
        tmp_path / "paper3.pdf",
        grobid=_StubGrobid(),
        extractor_a=extractor_a,
        extractor_b=_StubExtractor([_filled()]),
        tei_cache_dir=tmp_path / "tei",
        checkpoint_dir=tmp_path / "ckpt",
        citation_threshold=90.0,
    )

    assert result["fallback_level"] == 0
    # texto enviado = full-text (intro longa incluida)
    assert "lorem ipsum" in extractor_a.seen_texts[0]
