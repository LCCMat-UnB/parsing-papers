# src/parsing_papers/selection.py
"""
Selecao heuristica dos blocos relevantes para extracao de metricas.

Motivacao: enviar o paper inteiro (10-32K tokens) deixa o modelo pequeno
lento E impreciso (casos reais de records=[] nos dois maiores prompts). A
selecao por regex de metricas/modelos e deterministica, testavel e barata;
a escada de fallback (pipeline.py) garante recall -- se a selecao nao
render registros, o orcamento dobra ate chegar ao full-text.

Licao do passado (ver grobid_client.py:63-76): uma versao anterior FILTRAVA
secoes por palavra-chave de cabecalho e descartava o resto -- perdia dados
silenciosamente. Aqui nada e descartado: blocos fora da janela voltam no
proximo nivel da escada, e o ultimo nivel e sempre o texto completo.
"""

from __future__ import annotations

import re

from .blocks import Block, render_blocks

METRIC_RE = re.compile(
    r"(auc|roc[ -]?auc|accuracy|acuracia|precision|precisao|recall|sensitiv\w*|"
    r"specificity|especificidade|f1[ -]?score|rmse|\bmae\b|\bmse\b|\br2\b|r²|"
    r"brier|calibra\w*|gini|kolmogorov|\bks\b|log[ -]?loss|matthews|\bmcc\b|balanced accuracy)",
    re.IGNORECASE,
)
MODEL_RE = re.compile(
    r"(regress\w+|random[ ]?forest|floresta|xgboost|lightgbm|catboost|gradient[ ]?boost|"
    r"\bgbm\b|\bsvm\b|support vector|neural|deep learning|lstm|cnn|rnn|transformer|"
    r"naive bayes|bayesian|knn|k-nearest|decision tree|arvore de decis|discriminant|"
    r"ensemble|stacking|voting|bagging|survival|\bcox\b|elastic net|lasso|ridge)",
    re.IGNORECASE,
)
ANCHOR_RE = re.compile(
    r"(performance|desempenho|validation|valida\w+|discrimination|discrimina\w+|"
    r"comparison|compara\w+|evaluation|avalia\w+|baseline)",
    re.IGNORECASE,
)

ALWAYS_INCLUDE_KINDS = ("fixed", "warning")


def estimate_tokens(text: str) -> int:
    """Mesma convencao de llm_client.py (chars // 3) -- margem para PT/EN acentuado."""
    return len(text) // 3


def score_block(block: Block) -> float:
    if block.kind in ALWAYS_INCLUDE_KINDS:
        return float("inf")
    text = block.rendered
    score = (
        3.0 * len(METRIC_RE.findall(text))
        + 2.0 * len(MODEL_RE.findall(text))
        + 1.0 * len(ANCHOR_RE.findall(text))
    )
    if block.kind == "table":
        # tabelas densas em numeros sao candidatas fortes a tabela de resultados
        density = len(re.findall(r"\d", text)) / max(len(text), 1)
        score *= 1.0 + min(density * 10, 2.0)
    return score


def select_blocks(blocks: list[Block], budget_tokens: int | None) -> list[Block]:
    """
    Preenche o orcamento com os blocos de maior score e retorna na ordem
    original do documento (o LLM le melhor texto contiguo; a ordem tambem
    preserva a invariante de que a janela e subsequencia verbatim do full-text).
    budget_tokens=None -> todos os blocos (nivel final do fallback).
    """
    if budget_tokens is None:
        return list(blocks)

    selected = [b for b in blocks if b.kind in ALWAYS_INCLUDE_KINDS]
    used = estimate_tokens(render_blocks(selected))

    candidates = [b for b in blocks if b.kind not in ALWAYS_INCLUDE_KINDS]
    for block in sorted(candidates, key=score_block, reverse=True):
        cost = estimate_tokens(block.rendered)
        if used + cost > budget_tokens:
            continue
        selected.append(block)
        used += cost

    order = {id(b): i for i, b in enumerate(blocks)}
    selected.sort(key=lambda b: order[id(b)])
    return selected
