# tests/test_blocks.py
from parsing_papers.blocks import render_blocks, segment_paper
from parsing_papers.grobid_client import GrobidSection, GrobidTable, ParsedPaper


def _paper():
    return ParsedPaper(
        paper_id="p1",
        title="Titulo",
        abstract="Resumo do estudo.",
        sections=[
            GrobidSection(header="1. Introduction", text="Texto da introducao."),
            GrobidSection(header="4. Results", text="AUC foi de 0.91 para o XGBoost."),
        ],
        tables=[GrobidTable(label="Table 2", caption="Performance", text="| Model | AUC |\n|---|---|\n| XGB | 0.91 |")],
        empty_table_labels=["Table 5 (#5 de 5 tabelas no PDF)"],
    )


def test_segment_paper_kinds_and_order():
    blocks = segment_paper(_paper())
    kinds = [b.kind for b in blocks]
    assert kinds == ["fixed", "section", "section", "table", "warning"]


def test_render_roundtrip_matches_methods_and_results_text():
    paper = _paper()
    assert render_blocks(segment_paper(paper)) == paper.methods_and_results_text()


def test_render_roundtrip_without_empty_tables():
    paper = _paper()
    paper.empty_table_labels = []
    assert render_blocks(segment_paper(paper)) == paper.methods_and_results_text()
