"""
Testes do checkpoint intermediario por paper (pipeline.process_one_pdf).

Simula uma interrupcao entre a extracao_a e a extracao_b: escreve manualmente
um checkpoint parcial (como process_one_pdf faria apos extraction_a) e chama
process_one_pdf de novo, confirmando que ele NAO tenta re-extrair A (o que
"provaria" a economia via um extractor_a fake que falha se chamado 2x) e
completa o checkpoint final normalmente.
"""

from __future__ import annotations

import json

from parsing_papers.pipeline import (
    _load_partial_checkpoint,
    _partial_checkpoint_path,
    _save_partial_checkpoint,
    process_one_pdf,
)
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


class _FailIfCalledExtractor:
    """Extrator fake que falha o teste se extract() for chamado -- representa a chamada de extracao_a que NAO deveria ser refeita apos retomar de checkpoint parcial."""

    model = "fake/model-a"
    api_base = None
    temperature = 0.1
    max_tokens = 8000
    request_timeout = 900

    def extract(self, paper_id, source_text):
        raise AssertionError("extractor_a.extract() nao deveria ser chamado de novo -- deveria reusar o checkpoint parcial")


class _StubExtractorB:
    """Extrator fake para a extracao_b -- sempre chamado, retorna uma extracao simples e deterministica."""

    model = "fake/model-b"
    api_base = None
    temperature = 0.4
    max_tokens = 8000
    request_timeout = 900

    def __init__(self, records):
        self._records = records
        self.call_count = 0

    def extract(self, paper_id, source_text):
        self.call_count += 1
        return PaperExtraction(paper_id=paper_id, records=self._records)


def test_partial_checkpoint_save_and_load_roundtrip(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    extraction_a = PaperExtraction(paper_id="p1", records=[_record("Random Forest")])

    _save_partial_checkpoint(checkpoint_dir, "p1", "texto fonte de teste", extraction_a)

    assert _partial_checkpoint_path(checkpoint_dir, "p1").exists()

    loaded = _load_partial_checkpoint(checkpoint_dir, "p1")
    assert loaded is not None
    assert loaded["source_text"] == "texto fonte de teste"
    reloaded_extraction = PaperExtraction.model_validate(loaded["extraction_a"])
    assert reloaded_extraction.records[0].model_used.value == "Random Forest"


def test_process_one_pdf_resumes_from_partial_checkpoint(tmp_path, monkeypatch):
    """
    Simula: extracao_a ja rodou numa execucao anterior (checkpoint parcial em
    disco), o processo foi interrompido antes de extracao_b. Reprocessar deve
    reusar a extracao_a salva (extractor_a.extract NUNCA e chamado de novo) e
    so chamar extractor_b, produzindo um checkpoint final completo.
    """
    checkpoint_dir = tmp_path / "checkpoints"
    tei_cache_dir = tmp_path / "tei_cache"
    pdf_path = tmp_path / "paper_x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    extraction_a = PaperExtraction(paper_id="paper_x", records=[_record("Random Forest", auc="0.91")])
    _save_partial_checkpoint(checkpoint_dir, "paper_x", "texto fonte simulado do GROBID", extraction_a)

    # monkeypatch run_dual_extraction's dependency on LLMExtractor() default-construction
    # for extractor_b: pipeline.process_one_pdf itself doesn't construct extractor_b,
    # run_dual_extraction does when extractor_b=None -- but since existing_extraction_a
    # is provided, run_dual_extraction still builds a default extractor_b from extractor_a's
    # config unless we pass one. We patch LLMExtractor used inside dual_extraction to avoid
    # hitting a real network call.
    import parsing_papers.dual_extraction as dual_extraction_module

    stub_b = _StubExtractorB([_record("Random Forest", auc="0.91")])

    def _fake_llmextractor_ctor(*args, **kwargs):
        return stub_b

    monkeypatch.setattr(dual_extraction_module, "LLMExtractor", _fake_llmextractor_ctor)

    grobid_stub = object()  # nao deve ser usado -- fonte ja veio do checkpoint parcial

    result = process_one_pdf(
        pdf_path,
        grobid=grobid_stub,
        extractor_a=_FailIfCalledExtractor(),
        tei_cache_dir=tei_cache_dir,
        checkpoint_dir=checkpoint_dir,
        citation_threshold=0.0,  # nao importa para este teste
        force=False,
    )

    assert result is not None
    assert stub_b.call_count == 1
    assert result["extraction_a"]["records"][0]["model_used"]["value"] == "Random Forest"

    # checkpoint final foi criado e o parcial foi removido
    final_ckpt = checkpoint_dir / "paper_x.json"
    partial_ckpt = _partial_checkpoint_path(checkpoint_dir, "paper_x")
    assert final_ckpt.exists()
    assert not partial_ckpt.exists()


def test_force_clears_stale_partial_checkpoint(tmp_path, monkeypatch):
    """--force deve descartar um checkpoint parcial antigo, nao reusa-lo."""
    checkpoint_dir = tmp_path / "checkpoints"
    tei_cache_dir = tmp_path / "tei_cache"
    pdf_path = tmp_path / "paper_y.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    stale_extraction_a = PaperExtraction(paper_id="paper_y", records=[_record("Old Stale Model")])
    _save_partial_checkpoint(checkpoint_dir, "paper_y", "texto antigo", stale_extraction_a)

    class _FakeParsedPaper:
        empty_table_labels: list = []

        def methods_and_results_text(self):
            return "texto novo do grobid"

    class _FakeGrobid:
        def parse_pdf(self, pdf_path, tei_cache_dir):
            return _FakeParsedPaper()

    class _FreshExtractorA:
        model = "fake/model-a"
        api_base = None
        temperature = 0.1
        max_tokens = 8000
        request_timeout = 900

        def __init__(self):
            self.call_count = 0

        def extract(self, paper_id, source_text):
            self.call_count += 1
            assert source_text == "texto novo do grobid"
            return PaperExtraction(paper_id=paper_id, records=[_record("Fresh Model")])

    import parsing_papers.dual_extraction as dual_extraction_module

    stub_b = _StubExtractorB([_record("Fresh Model")])
    monkeypatch.setattr(dual_extraction_module, "LLMExtractor", lambda *a, **k: stub_b)

    fresh_a = _FreshExtractorA()

    result = process_one_pdf(
        pdf_path,
        grobid=_FakeGrobid(),
        extractor_a=fresh_a,
        tei_cache_dir=tei_cache_dir,
        checkpoint_dir=checkpoint_dir,
        citation_threshold=0.0,
        force=True,
    )

    assert fresh_a.call_count == 1  # extracao_a foi refeita, nao reusou o parcial antigo
    assert result["extraction_a"]["records"][0]["model_used"]["value"] == "Fresh Model"
