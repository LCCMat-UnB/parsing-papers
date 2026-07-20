# tests/test_selection.py
from parsing_papers.blocks import Block
from parsing_papers.selection import estimate_tokens, score_block, select_blocks


def _fixed():
    return Block(kind="fixed", label="title_abstract", rendered="TITLE: T\nABSTRACT: A")


def test_block_with_metrics_outranks_filler():
    metric = Block(kind="section", label="Results", rendered="\n=== SECTION: Results ===\nThe XGBoost achieved AUC 0.91 and accuracy 0.87 on validation.")
    filler = Block(kind="section", label="Intro", rendered="\n=== SECTION: Intro ===\nCredit is important for agriculture in many countries worldwide.")
    assert score_block(metric) > score_block(filler)


def test_fixed_and_warning_always_included():
    blocks = [_fixed(), Block(kind="section", label="S", rendered="\n=== SECTION: S ===\nfiller"), Block(kind="warning", label="w", rendered="\n=== AVISO ===\nx")]
    selected = select_blocks(blocks, budget_tokens=1)  # orcamento minusculo
    kinds = {b.kind for b in selected}
    assert "fixed" in kinds and "warning" in kinds


def test_budget_respected_and_document_order_preserved():
    big_results = Block(kind="section", label="Results", rendered="\n=== SECTION: Results ===\n" + "AUC 0.91 XGBoost " * 30)
    intro = Block(kind="section", label="Intro", rendered="\n=== SECTION: Intro ===\n" + "lorem ipsum dolor " * 30)
    blocks = [_fixed(), intro, big_results]
    budget = estimate_tokens(blocks[0].rendered) + estimate_tokens(big_results.rendered) + 1
    selected = select_blocks(blocks, budget)
    # results (score maior) entra; intro (score baixo) fica de fora; ordem = do documento
    assert [b.label for b in selected] == ["title_abstract", "Results"]


def test_none_budget_returns_all():
    blocks = [_fixed(), Block(kind="section", label="S", rendered="x")]
    assert select_blocks(blocks, None) == blocks


def test_estimate_tokens_convention():
    assert estimate_tokens("x" * 300) == 100
