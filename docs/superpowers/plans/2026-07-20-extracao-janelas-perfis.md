# Extração em Janelas Candidatas + Perfis de Deployment — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduzir prompts de extração de 10–32K para ≤6K tokens via janelas candidatas com fallback, adicionar árbitro automático de divergências A/B, e introduzir perfis de deployment `local` (GPU ≤12GB, Ollama) e `cluster` (>18GB, vLLM) com paralelismo configurável.

**Architecture:** Novos módulos `blocks.py` (segmentação do TEI em blocos), `selection.py` (score heurístico + preenchimento de orçamento), `arbiter.py` (decisão de divergências por LLM com prompt mínimo) e `profiles.py` (perfis JSON). `pipeline.py` integra escada de fallback + árbitro + observabilidade; `consolidate.py` passa a usar a extração reconciliada. Spec: `docs/superpowers/specs/2026-07-20-extracao-janelas-perfis-design.md`.

**Tech Stack:** Python ≥3.10, pydantic v2, LiteLLM, rapidfuzz, click, pytest. Zero dependências novas (exceto declarar `typer`, já importado).

## Global Constraints

- Python ≥3.10 (`pyproject.toml` requires-python). Perfis em **JSON** (não TOML — tomllib só existe em 3.11+).
- Zero novas dependências além de declarar `typer>=0.12.0` (já importado em `cli.py`, hoje não declarado).
- Comentários/docstrings em português sem acentos, seguindo o estilo do codebase (ver `grobid_client.py`, `dual_extraction.py`).
- Checkpoints antigos (sem `extraction_reconciled`/`arbiter_resolutions`/`fallback_level`) DEVEM continuar consolidando — toda leitura de campo novo usa `.get(...)` com fallback.
- Os 69 testes existentes devem continuar verdes ao final de cada task (ajustes em assertions de lista de colunas são permitidos apenas na Task 7, onde a planilha ganha uma coluna).
- Estimativa de tokens: `len(texto) // 3` (mesma convenção de `llm_client.py:93`).
- Testes determinísticos, sem GROBID/Ollama/GPU — stubs e mocks como nos testes atuais (ver `tests/test_dual_extraction.py` para o padrão de fixtures `ModelRecord`).
- Desvio deliberado da spec: blocos NÃO carregam offsets no `source_text` original; em vez disso cada `Block` carrega seu texto já renderizado (`rendered`), e a verificação de citações roda sobre a concatenação dos blocos selecionados (que é exatamente o texto que o LLM viu). Funcionalmente equivalente, mais simples.
- Desvio deliberado da spec: em vez de um fixture TEI sintético grande em arquivo, os testes de orçamento constroem um `ParsedPaper` sintético em código (`tests/test_pipeline_windows.py`) — mesmo efeito, menos arquivo para manter.
- Desvio deliberado da spec: o gatilho de escalada "seletor com poucos blocos" foi descartado — apenas extracao A vazia escala (nao existe oraculo para "incompleto mas nao-vazio"); o recall continua limitado pelo full-text no ultimo nivel.
- Desvio deliberado da spec: o prompt do arbitro NAO inclui as quotes de A/B (FieldDivergence carrega so valores); a janela de texto completa ja vai no prompt, entao o impacto e baixo.
- Commits ao final de cada task, mensagens em português estilo conventional commits (ex: `feat: adiciona segmentacao em blocos`).

---

### Task 1: Perfis de configuração (`profiles.py` + JSONs)

**Files:**
- Create: `src/parsing_papers/profiles.py`
- Create: `config/profiles/local.json`
- Create: `config/profiles/cluster.json`
- Modify: `pyproject.toml:6-19` (adicionar typer)
- Test: `tests/test_profiles.py`

**Interfaces:**
- Consumes: nada.
- Produces: `Profile` (dataclass frozen, campos abaixo), `load_profile(name, profiles_dir) -> Profile`, `DEFAULT_PROFILES_DIR = Path("config/profiles")`, `DEFAULT_PROFILE = "local"`. Tasks 6, 8 e 10 consomem `Profile`.

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_profiles.py
from pathlib import Path

import pytest

from parsing_papers.profiles import DEFAULT_PROFILES_DIR, Profile, load_profile


def test_load_local_profile_defaults():
    p = load_profile("local")
    assert p.name == "local"
    assert p.model.startswith("ollama_chat/")
    assert p.window_token_budget == 4000
    assert p.max_concurrency == 1
    assert p.fallback_budgets[-1] is None  # ultimo nivel = full-text


def test_load_cluster_profile():
    p = load_profile("cluster")
    assert p.model.startswith("openai/")
    assert p.max_concurrency > 1
    assert p.window_token_budget == 6000


def test_missing_profile_raises_with_available_list():
    with pytest.raises(FileNotFoundError, match="nao-encontrado"):
        load_profile("nao-encontrado")


def test_unknown_field_raises(tmp_path):
    (tmp_path / "ruim.json").write_text('{"model": "m", "campo_inventado": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="campo_inventado"):
        load_profile("ruim", profiles_dir=tmp_path)
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `python -m pytest tests/test_profiles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'parsing_papers.profiles'`

- [ ] **Step 3: Implementar `profiles.py` e os JSONs**

```python
# src/parsing_papers/profiles.py
"""
Perfis de deployment: "local" (GPU <=12GB, Ollama) e "cluster" (>18GB, vLLM).

Hoje toda a configuracao e passada por flags de CLI com defaults hardcoded;
trocar de ambiente exigia lembrar um conjunto coerente de flags. O perfil
empacota esse conjunto num JSON versionado em config/profiles/<nome>.json;
flags individuais da CLI continuam existindo e tem precedencia sobre o perfil
(ver pipeline._resolve_profile).

JSON (nao TOML) porque tomllib so existe no Python 3.11+ e o projeto suporta
3.10 -- JSON nao adiciona nenhuma dependencia.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PROFILES_DIR = Path("config/profiles")
DEFAULT_PROFILE = "local"


@dataclass(frozen=True)
class Profile:
    name: str
    model: str  # identificador LiteLLM (ollama_chat/... local; openai/... para vLLM)
    api_base: str
    temperature_a: float = 0.1
    temperature_b: float = 0.4
    max_tokens: int = 8000
    min_num_ctx: int = 8000
    request_timeout_s: int = 900
    window_token_budget: int = 4000  # orcamento de tokens da janela de extracao
    max_concurrency: int = 1  # PDFs processados em paralelo (threads)
    # escada de fallback: orcamentos progressivos; None = full-text (comportamento antigo)
    fallback_budgets: list[int | None] = field(default_factory=lambda: [4000, 12000, None])
    arbiter_max_tokens: int = 1500
    arbiter_min_num_ctx: int = 8000


def load_profile(name: str = DEFAULT_PROFILE, profiles_dir: Path = DEFAULT_PROFILES_DIR) -> Profile:
    profiles_dir = Path(profiles_dir)
    path = profiles_dir / f"{name}.json"
    if not path.exists():
        available = sorted(p.stem for p in profiles_dir.glob("*.json")) if profiles_dir.exists() else []
        raise FileNotFoundError(
            f"Perfil '{name}' nao encontrado ({path}). Disponiveis: {available or 'nenhum'}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    known = set(Profile.__dataclass_fields__) - {"name"}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Campos desconhecidos no perfil '{name}': {sorted(unknown)}")
    return Profile(name=name, **data)
```

```json
// config/profiles/local.json
{
  "model": "ollama_chat/qwen2.5:14b-instruct",
  "api_base": "http://localhost:11434",
  "temperature_a": 0.1,
  "temperature_b": 0.4,
  "max_tokens": 8000,
  "min_num_ctx": 8000,
  "request_timeout_s": 900,
  "window_token_budget": 4000,
  "max_concurrency": 1,
  "fallback_budgets": [4000, 12000, null],
  "arbiter_max_tokens": 1500,
  "arbiter_min_num_ctx": 8000
}
```

```json
// config/profiles/cluster.json
{
  "model": "openai/Qwen/Qwen2.5-32B-Instruct-AWQ",
  "api_base": "http://localhost:8000/v1",
  "temperature_a": 0.1,
  "temperature_b": 0.4,
  "max_tokens": 8000,
  "min_num_ctx": 8192,
  "request_timeout_s": 600,
  "window_token_budget": 6000,
  "max_concurrency": 12,
  "fallback_budgets": [6000, 16000, null],
  "arbiter_max_tokens": 1500,
  "arbiter_min_num_ctx": 8192
}
```

Em `pyproject.toml`, adicionar `"typer>=0.12.0",` à lista `dependencies` (após `"rich>=13.7.0",`).

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `python -m pytest tests/test_profiles.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/parsing_papers/profiles.py config/profiles/ tests/test_profiles.py pyproject.toml
git commit -m "feat: perfis de deployment local/cluster em JSON + declara typer"
```

---

### Task 2: Segmentação em blocos (`blocks.py`)

**Files:**
- Create: `src/parsing_papers/blocks.py`
- Test: `tests/test_blocks.py`

**Interfaces:**
- Consumes: `ParsedPaper`, `GrobidSection`, `GrobidTable` de `grobid_client.py`.
- Produces: `Block` (dataclass: `kind: str`, `label: str`, `rendered: str`), `segment_paper(parsed: ParsedPaper) -> list[Block]`, `render_blocks(blocks: Iterable[Block]) -> str`. Invariante: `render_blocks(segment_paper(p)) == p.methods_and_results_text()` — a Task 3 depende disso para não alterar o texto que o LLM vê.

- [ ] **Step 1: Escrever o teste que falha**

```python
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
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `python -m pytest tests/test_blocks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'parsing_papers.blocks'`

- [ ] **Step 3: Implementar `blocks.py`**

```python
# src/parsing_papers/blocks.py
"""
Segmentacao do ParsedPaper em blocos tipados para a selecao de janelas.

Cada bloco carrega seu texto JA RENDERIZADO exatamente como apareceria no
prompt (mesmos cabecalhos "=== SECTION ===" / "=== TABLE ===" de
ParsedPaper.methods_and_results_text), de modo que:
  1. a janela selecionada e uma subsequencia verbatim do full-text -- quotes
     extraidas pelo LLM continuam verificaveis por fuzzy match;
  2. a invariante render_blocks(segment_paper(p)) == p.methods_and_results_text()
     garante que o fallback full-text e byte-identico ao comportamento antigo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .grobid_client import ParsedPaper


@dataclass
class Block:
    kind: str  # "fixed" (titulo+abstract) | "section" | "table" | "warning"
    label: str  # header da secao / "label - caption" da tabela / descricao
    rendered: str  # texto exato como vai para o prompt


def segment_paper(parsed: ParsedPaper) -> list[Block]:
    blocks = [
        Block(
            kind="fixed",
            label="title_abstract",
            rendered=f"TITLE: {parsed.title}\nABSTRACT: {parsed.abstract}",
        )
    ]
    for sec in parsed.sections:
        blocks.append(
            Block(kind="section", label=sec.header, rendered=f"\n=== SECTION: {sec.header} ===\n{sec.text}")
        )
    for tbl in parsed.tables:
        blocks.append(
            Block(
                kind="table",
                label=f"{tbl.label} - {tbl.caption}",
                rendered=f"\n=== TABLE: {tbl.label} - {tbl.caption} ===\n{tbl.text}",
            )
        )
    if parsed.empty_table_labels:
        labels_str = "; ".join(parsed.empty_table_labels)
        blocks.append(
            Block(
                kind="warning",
                label="aviso_parsing",
                rendered=(
                    "\n=== AVISO DE PARSING ===\n"
                    f"As seguintes tabelas foram identificadas no PDF mas o parser NAO conseguiu "
                    f"extrair o conteudo numerico delas (vieram vazias): {labels_str}. "
                    "Os valores dessas tabelas podem ainda aparecer em texto corrido nas secoes "
                    "de resultados/discussao/comparacao acima -- procure ativamente por eles la "
                    "antes de considerar um dado como nao reportado."
                ),
            )
        )
    return blocks


def render_blocks(blocks: Iterable[Block]) -> str:
    return "\n".join(b.rendered for b in blocks)
```

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `python -m pytest tests/test_blocks.py -v`
Expected: 3 PASS (se o roundtrip falhar por whitespace, comparar contra `methods_and_results_text` em `grobid_client.py:77-96` e ajustar o `rendered`)

- [ ] **Step 5: Commit**

```bash
git add src/parsing_papers/blocks.py tests/test_blocks.py
git commit -m "feat: segmentacao do paper em blocos renderizados"
```

---

### Task 3: Seletor heurístico de janelas (`selection.py`)

**Files:**
- Create: `src/parsing_papers/selection.py`
- Test: `tests/test_selection.py`

**Interfaces:**
- Consumes: `Block`, `render_blocks` de `blocks.py`.
- Produces: `estimate_tokens(text: str) -> int`, `score_block(block: Block) -> float`, `select_blocks(blocks: list[Block], budget_tokens: int | None) -> list[Block]` (retorna na ORDEM ORIGINAL do documento; `budget_tokens=None` retorna todos). Task 6 consome.

- [ ] **Step 1: Escrever o teste que falha**

```python
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
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `python -m pytest tests/test_selection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'parsing_papers.selection'`

- [ ] **Step 3: Implementar `selection.py`**

```python
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
```

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `python -m pytest tests/test_selection.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/parsing_papers/selection.py tests/test_selection.py
git commit -m "feat: seletor heuristico de janelas com orcamento de tokens"
```

---

### Task 4: `llm_client` — `complete()` público + fallback de schema restrito

**Files:**
- Modify: `src/parsing_papers/llm_client.py:85-129`
- Test: `tests/test_llm_client_fallback.py`

**Interfaces:**
- Consumes: nada novo.
- Produces: `LLMExtractor.complete(messages: list[dict]) -> str` ( público, delega ao `_call` com retry — o árbitro da Task 5 usa); `_is_schema_unsupported_error(exc: Exception) -> bool`. Comportamento alterado: o fallback json_schema→json_object SÓ ocorre quando o erro indica falta de suporte a schema; timeout/transporte propagam para o retry do tenacity (hoje são mascarados).

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_llm_client_fallback.py
from unittest.mock import patch

import pytest

from parsing_papers.llm_client import LLMExtractor, _is_schema_unsupported_error


def test_schema_unsupported_detection():
    assert _is_schema_unsupported_error(Exception("Unsupported parameter: response_format json_schema"))
    assert _is_schema_unsupported_error(Exception("Invalid JSON schema provided"))
    assert not _is_schema_unsupported_error(Exception("Request timed out after 900s"))
    assert not _is_schema_unsupported_error(Exception("Connection refused"))


def _extractor():
    return LLMExtractor(model="ollama_chat/fake", api_base="http://localhost:1", min_num_ctx=100)


def test_timeout_is_not_masked_by_json_object_fallback():
    """Erro de transporte NAO pode cair no fallback json_object (que esconderia
    a causa real e queimaria uma chamada extra); deve propagar para o retry."""
    ext = _extractor()
    with patch("litellm.completion", side_effect=TimeoutError("Request timed out")):
        with pytest.raises(TimeoutError):
            ext.complete([{"role": "user", "content": "oi"}])


def test_schema_error_falls_back_to_json_object():
    ext = _extractor()
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs.get("response_format"))
        if kwargs["response_format"].get("type") == "json_schema":
            raise Exception("Unsupported parameter: response_format json_schema")
        return {"choices": [{"message": {"content": '{"paper_id": "p", "records": []}'}}]}

    with patch("litellm.completion", side_effect=fake_completion):
        out = ext.complete([{"role": "user", "content": "oi"}])
    assert '"records"' in out
    assert [c["type"] for c in calls] == ["json_schema", "json_object"]
```

Nota para o implementador: `complete` ainda não existe e `_call` tem `@retry` do tenacity — o `TimeoutError` vai ser retentado 3× antes de propagar; com o mock instantâneo isso é rápido e o `pytest.raises` continua válido.

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `python -m pytest tests/test_llm_client_fallback.py -v`
Expected: FAIL — `ImportError: cannot import name '_is_schema_unsupported_error'`

- [ ] **Step 3: Implementar as mudanças em `llm_client.py`**

Adicionar após a classe `ExtractionError` (linha 39-40):

```python
# Padroes de erro que indicam que o provedor nao suporta response_format
# json_schema -- somente NESSES casos vale tentar o fallback json_object.
# Qualquer outro erro (timeout, conexao, 5xx) deve propagar para o retry do
# tenacity; antes, o catch-all mascarava ate timeout como "schema nao
# suportado" e queimava uma chamada extra por tentativa.
_SCHEMA_UNSUPPORTED_HINTS = (
    "response_format", "json_schema", "json schema", "schema",
    "structured output", "unsupported parameter", "unknown field",
)


def _is_schema_unsupported_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(hint in msg for hint in _SCHEMA_UNSUPPORTED_HINTS)
```

Substituir o bloco try/except de `_call` (linhas 113-127) por:

```python
        # Tenta forcar JSON schema estrito; so cai no fallback json_object
        # quando o erro indica falta de suporte a schema (ver
        # _is_schema_unsupported_error) -- demais erros propagam para o retry.
        try:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "paper_extraction",
                    "schema": PaperExtraction.model_json_schema(),
                    "strict": True,
                },
            }
            resp = litellm.completion(**kwargs)
        except Exception as e:  # noqa: BLE001
            if not _is_schema_unsupported_error(e):
                raise
            logger.warning("json_schema estrito nao suportado (%s); tentando response_format=json_object", e)
            kwargs["response_format"] = {"type": "json_object"}
            resp = litellm.completion(**kwargs)

        return resp["choices"][0]["message"]["content"]
```

Adicionar método público na classe (após `_call`, antes de `extract`):

```python
    def complete(self, messages: list[dict]) -> str:
        """Chamada generica (com retry) fora do fluxo de extracao -- usada pelo
        arbitro (arbiter.py) com seus proprios prompts e schema de resposta."""
        return self._call(messages)
```

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `python -m pytest tests/test_llm_client_fallback.py tests/test_grobid_client.py -v`
Expected: novos PASS; suite existente verde (`python -m pytest -q` → 73 passed)

- [ ] **Step 5: Commit**

```bash
git add src/parsing_papers/llm_client.py tests/test_llm_client_fallback.py
git commit -m "fix: fallback json_object so em erro de schema + LLMExtractor.complete publico"
```

---

### Task 5: Árbitro de divergências (`arbiter.py`)

**Files:**
- Create: `src/parsing_papers/arbiter.py`
- Test: `tests/test_arbiter.py`

**Interfaces:**
- Consumes: `LLMExtractor.complete` (Task 4), `_try_recover_json` de `llm_client.py`, `FieldDivergence`/`RecordComparison`/`NUMERIC_FIELDS`/`TEXT_FIELDS`/`CATEGORICAL_FIELDS` de `dual_extraction.py`, `PaperExtraction` de `schema.py`.
- Produces: `ArbiterFieldDecision`/`ArbiterRecordDecision` (pydantic), `build_arbiter_messages(paper_id, model_used, divergences, source_text) -> list[dict]`, `ArbiterClient(extractor).arbitrate_record(paper_id, model_used, divergences, source_text) -> ArbiterRecordDecision | None`, `apply_arbiter_decisions(extraction_a, extraction_b, comparisons, decisions_by_index_a) -> tuple[PaperExtraction, list[dict]]`. Task 6 chama `arbitrate_record` por registro divergente e `apply_arbiter_decisions` para produzir a extração reconciliada; Task 7 consome a lista de resoluções (`{"index_a", "field_name", "choice"}`).

- [ ] **Step 1: Escrever o teste que falha**

```python
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
    return a, b, compare_extractions(a, b)


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

    def complete(self, messages):
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
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `python -m pytest tests/test_arbiter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'parsing_papers.arbiter'`

- [ ] **Step 3: Implementar `arbiter.py`**

```python
# src/parsing_papers/arbiter.py
"""
Arbitro de divergencias da dupla extracao (PRISMA item 9, agora com resolucao).

Antes: divergencias A/B viravam flag needs_review e a planilha usava so a
extracao A -- a fila de revisao humana crescia com ruido (8/13 linhas no lote
de referencia). Agora: para cada registro divergente, UMA chamada de arbitro
(prompt minusculo: so os campos divergentes + quotes + o texto fonte ja
reduzido pelas janelas) decide campo a campo "a", "b" ou "neither".

Principios:
- O arbitro NUNCA inventa um terceiro valor -- so escolhe entre A, B ou
  declara empate ("neither" -> campo continua flagado para revisao humana).
- Falha do arbitro (transporte, JSON invalido) NUNCA bloqueia o pipeline:
  retorna None e as divergencias seguem o comportamento antigo (flag).
- A reconciliacao (apply_arbiter_decisions) e pura e nao muta a extracao A --
  opera numa copia profunda.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ValidationError

from .dual_extraction import CATEGORICAL_FIELDS, NUMERIC_FIELDS, TEXT_FIELDS, FieldDivergence, RecordComparison
from .llm_client import LLMExtractor, _try_recover_json
from .schema import PaperExtraction

logger = logging.getLogger(__name__)

_EVIDENCE_FIELDS = set(NUMERIC_FIELDS + TEXT_FIELDS)
_ALL_FIELDS = _EVIDENCE_FIELDS | set(CATEGORICAL_FIELDS)

ARBITER_SYSTEM_PROMPT = """\
Voce e o arbitro de uma dupla extracao de dados de um paper cientifico.
Dois extratores independentes (A e B) produziram valores diferentes para os
mesmos campos de um mesmo modelo. Para CADA campo divergente, decida qual
valor esta correto com base EXCLUSIVA no texto fonte fornecido.

Regras:
- Escolha "a" ou "b" somente se o valor correspondente estiver claramente
  suportado pelo texto fonte (a quote do extrator ajuda a localizar o trecho).
- Se nenhum dos dois valores estiver claramente no texto, escolha "neither".
- NUNCA invente um terceiro valor. NUNCA parafraseie.

Responda APENAS com JSON valido no formato:
{"decisions": [{"field_name": "<campo>", "choice": "a"|"b"|"neither"}, ...]}
"""

ARBITER_USER_TEMPLATE = """\
PAPER_ID: {paper_id}
MODELO: {model_used}

CAMPOS DIVERGENTES:
{fields_block}

TEXTO FONTE:
---
{source_text}
---

Decida cada campo divergente. Responda APENAS com o JSON.
"""


class ArbiterFieldDecision(BaseModel):
    field_name: str
    choice: Literal["a", "b", "neither"]


class ArbiterRecordDecision(BaseModel):
    decisions: list[ArbiterFieldDecision]


def build_arbiter_messages(
    paper_id: str, model_used: str, divergences: list[FieldDivergence], source_text: str
) -> list[dict]:
    lines = []
    for d in divergences:
        lines.append(f'- campo "{d.field_name}": A="{d.value_a}" | B="{d.value_b}"')
    fields_block = "\n".join(lines)
    return [
        {"role": "system", "content": ARBITER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": ARBITER_USER_TEMPLATE.format(
                paper_id=paper_id, model_used=model_used, fields_block=fields_block, source_text=source_text
            ),
        },
    ]


class ArbiterClient:
    """Uma chamada de arbitro por REGISTRO divergente (todos os campos de uma vez)."""

    def __init__(self, extractor: LLMExtractor):
        self.extractor = extractor

    def arbitrate_record(
        self, paper_id: str, model_used: str, divergences: list[FieldDivergence], source_text: str
    ) -> ArbiterRecordDecision | None:
        messages = build_arbiter_messages(paper_id, model_used, divergences, source_text)
        try:
            raw = self.extractor.complete(messages)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Arbitro falhou para %s/%s (%s) -- divergencias ficam para revisao humana.",
                paper_id, model_used, e,
            )
            return None
        data = _try_recover_json(raw)
        if data is None:
            logger.warning("Arbitro retornou JSON invalido para %s/%s -- mantendo flags.", paper_id, model_used)
            return None
        try:
            return ArbiterRecordDecision.model_validate(data)
        except ValidationError as e:
            logger.warning("Decisao do arbitro fora do schema para %s/%s (%s) -- mantendo flags.", paper_id, model_used, e)
            return None


def apply_arbiter_decisions(
    extraction_a: PaperExtraction,
    extraction_b: PaperExtraction,
    comparisons: list[RecordComparison],
    decisions_by_index_a: dict[int, ArbiterRecordDecision],
) -> tuple[PaperExtraction, list[dict]]:
    """
    Produz a extracao reconciliada: copia profunda de A com os campos decididos
    "b" substituidos pelos valores de B. Retorna (reconciliada, resolucoes),
    onde resolucoes = [{"index_a", "field_name", "choice"}] apenas para
    choice em ("a", "b") -- "neither" deixa o campo divergente (needs_review).
    Campos decididos que nao constam como divergentes sao ignorados (defesa
    contra alucinacao do arbitro sobre a lista de campos).
    """
    reconciled = extraction_a.model_copy(deep=True)
    resolutions: list[dict] = []

    for comp in comparisons:
        if comp.index_a is None or comp.index_b is None:
            continue  # registros nao pareados: sem arbitro, seguem flagados
        decision = decisions_by_index_a.get(comp.index_a)
        if decision is None:
            continue
        divergent_fields = {d.field_name for d in comp.divergences if d.diverges}
        rec_out = reconciled.records[comp.index_a]
        rec_b = extraction_b.records[comp.index_b]
        applied = []
        for fd in decision.decisions:
            if fd.field_name not in divergent_fields or fd.field_name not in _ALL_FIELDS:
                continue
            if fd.choice == "neither":
                continue
            if fd.choice == "b":
                value_b = getattr(rec_b, fd.field_name)
                if fd.field_name in _EVIDENCE_FIELDS:
                    value_b = value_b.model_copy(deep=True)
                setattr(rec_out, fd.field_name, value_b)
            resolutions.append({"index_a": comp.index_a, "field_name": fd.field_name, "choice": fd.choice})
            applied.append(f"{fd.field_name}<-{fd.choice}")
        if applied:
            prior = rec_out.extraction_notes or ""
            rec_out.extraction_notes = (prior + " " if prior else "") + f"[arbitro: {', '.join(applied)}]"

    return reconciled, resolutions
```

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `python -m pytest tests/test_arbiter.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/parsing_papers/arbiter.py tests/test_arbiter.py
git commit -m "feat: arbitro automatico de divergencias da dupla extracao"
```

---

### Task 6: Integração no pipeline — escada de fallback + árbitro + observabilidade + perfil na CLI

**Files:**
- Modify: `src/parsing_papers/pipeline.py` (`process_one_pdf`:147-275, `process_pdf_directory`:378-441, comandos `run`:449-483 e `registry-run`:486-547)
- Modify: `src/parsing_papers/dual_extraction.py:260-267` (B herda `min_num_ctx`)
- Modify: `src/parsing_papers/cli.py:138-160,189-191,244-246` (chamadas de `process_pdf_directory` com perfil)
- Test: `tests/test_pipeline_windows.py`
- Test: `tests/test_dual_extraction.py` (adicionar teste do `min_num_ctx` de B)

**Interfaces:**
- Consumes: `segment_paper`/`render_blocks` (Task 2), `select_blocks`/`estimate_tokens` (Task 3), `ArbiterClient`/`apply_arbiter_decisions` (Task 5), `Profile`/`load_profile` (Task 1).
- Produces:
  - `process_one_pdf(..., extractor_b=None, arbiter_client=None, fallback_budgets=None)` — novos kwargs opcionais no final; comportamento antigo preservado quando `fallback_budgets=None` (equivale a `[None]`, full-text direto).
  - `process_pdf_directory(pdf_dir, out_dir, grobid_url, citation_threshold, grobid_wait_s, skip_screening, force, profile)` — **assinatura nova** (perfil substitui os 5 knobs de LLM).
  - `_resolve_profile(profile_name, model, api_base, temperature, llm_timeout_s, min_num_ctx) -> Profile` (overrides da CLI).
  - Checkpoint por paper ganha: `extraction_reconciled`, `arbiter_resolutions`, `fallback_level`, `estimated_prompt_tokens`, `timing` (Task 7 consome os dois primeiros; Task 9 os demais).

- [ ] **Step 1: Escrever os testes que falham**

```python
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
```

Adicionar em `tests/test_dual_extraction.py` (ao final):

```python
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
```

- [ ] **Step 2: Rodar os testes e ver falhar**

Run: `python -m pytest tests/test_pipeline_windows.py tests/test_dual_extraction.py -v`
Expected: FAIL — `process_one_pdf() got an unexpected keyword argument 'fallback_budgets'` e `assert seen_num_ctx == [12345, 12345]`

- [ ] **Step 3a: Corrigir `dual_extraction.py` (B herda `min_num_ctx`)**

Em `run_dual_extraction` (`dual_extraction.py:261-267`), adicionar a linha `min_num_ctx` ao construtor de `extractor_b`:

```python
    if extractor_b is None:
        extractor_b = LLMExtractor(
            model=extractor_a.model,
            api_base=extractor_a.api_base,
            temperature=max(extractor_a.temperature, 0.4),
            max_tokens=extractor_a.max_tokens,
            request_timeout=extractor_a.request_timeout,
            min_num_ctx=extractor_a.min_num_ctx,
        )
```

- [ ] **Step 3b: Reescrever `process_one_pdf` em `pipeline.py`**

Novos imports no topo de `pipeline.py`:

```python
import time

from .arbiter import ArbiterClient, apply_arbiter_decisions
from .blocks import segment_paper
from .selection import estimate_tokens, render_blocks, select_blocks
from .profiles import Profile, load_profile
```

Substituir `process_one_pdf` inteiro (linhas 147-275) por:

```python
def process_one_pdf(
    pdf_path: Path,
    grobid: GrobidClient,
    extractor_a: LLMExtractor,
    tei_cache_dir: Path,
    checkpoint_dir: Path,
    citation_threshold: float,
    force: bool = False,
    screening_client: ScreeningClient | None = None,
    screening_dir: Path | None = None,
    extractor_b: LLMExtractor | None = None,
    arbiter_client: ArbiterClient | None = None,
    fallback_budgets: list[int | None] | None = None,
) -> dict | None:
    paper_id = pdf_path.stem
    ckpt_path = _checkpoint_path(checkpoint_dir, paper_id)

    if ckpt_path.exists() and not force:
        logger.info("Checkpoint encontrado para %s, pulando (use --force para reprocessar).", paper_id)
        return json.loads(ckpt_path.read_text(encoding="utf-8"))

    if force:
        _clear_partial_checkpoint(checkpoint_dir, paper_id)

    logger.info("Processando %s ...", paper_id)
    timing: dict[str, float] = {}
    t_start = time.monotonic()

    # Tenta reaproveitar uma extracao_a de um checkpoint intermediario (de uma
    # execucao anterior interrompida antes de completar este paper).
    partial = _load_partial_checkpoint(checkpoint_dir, paper_id)
    extraction_a = None
    source_text = None
    fallback_level = 0
    if partial is not None:
        try:
            extraction_a = PaperExtraction.model_validate(partial["extraction_a"])
            source_text = partial["source_text"]
            fallback_level = partial.get("fallback_level", 0)
            logger.info(
                "Retomando %s a partir de checkpoint intermediario -- reusando extracao_a (%d registros), pulando GROBID e a extracao_a.",
                paper_id, len(extraction_a.records),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Checkpoint intermediario de %s invalido (%s) -- reprocessando do zero.", paper_id, e)
            extraction_a = None
            source_text = None

    parsed = None
    try:
        if extraction_a is None:
            # Etapa 1: GROBID
            t0 = time.monotonic()
            parsed = grobid.parse_pdf(pdf_path, tei_cache_dir=tei_cache_dir)
            timing["grobid_s"] = round(time.monotonic() - t0, 2)

            # Etapa 0: triagem PRISMA (inalterada -- roda antes da extracao cara)
            if screening_client is not None and screening_dir is not None:
                screening_result = run_screening_for_paper(paper_id, parsed, screening_client, screening_dir, force=force)
                if screening_result is not None and not should_proceed_to_extraction(screening_result):
                    logger.info(
                        "%s excluido na triagem PRISMA (decisao=%s) -- pulando extracao de metricas.",
                        paper_id, screening_result.decision.value,
                    )
                    return None

            # Etapa 2a: extracao A sobre JANELAS com escada de fallback.
            # budget=None (ultimo nivel) = full-text = comportamento antigo;
            # so escala quando a extracao vem vazia -- nunca pior que antes.
            budgets = fallback_budgets or [None]
            blocks = segment_paper(parsed)
            t0 = time.monotonic()
            for level, budget in enumerate(budgets):
                source_text = render_blocks(select_blocks(blocks, budget))
                extraction_a = extractor_a.extract(paper_id, source_text)
                fallback_level = level
                if extraction_a.records or level == len(budgets) - 1:
                    break
                logger.info(
                    "%s: extracao A vazia com janela de %s tokens -- escalando orcamento (nivel %d).",
                    paper_id, budget, level + 1,
                )
            timing["extraction_a_s"] = round(time.monotonic() - t0, 2)

            if not source_text.strip():
                logger.warning("Texto vazio apos parsing GROBID para %s -- pulando.", paper_id)
                return None

            # Checkpoint intermediario com o TEXTO FINAL usado (apos escalada)
            _save_partial_checkpoint(checkpoint_dir, paper_id, source_text, extraction_a)

        # Etapas 2b e 4: extracao B sobre o MESMO contexto final de A
        t0 = time.monotonic()
        dual_result = run_dual_extraction(
            paper_id,
            source_text,
            extractor_a,
            extractor_b=extractor_b,
            existing_extraction_a=extraction_a,
        )
        timing["extraction_b_s"] = round(time.monotonic() - t0, 2)
    except ExtractionError as e:
        logger.error("Falha na extracao de %s: %s", paper_id, e)
        return None

    empty_table_labels = parsed.empty_table_labels if parsed is not None else []

    # Etapa 4b: arbitro resolve divergencias campo a campo (uma chamada por
    # registro divergente; prompts minusculos). Falha do arbitro -> comportamento
    # antigo (flags), nunca bloqueia.
    resolutions: list[dict] = []
    reconciled = dual_result.extraction_a
    if arbiter_client is not None and dual_result.needs_human_review:
        t0 = time.monotonic()
        decisions = {}
        for comp in dual_result.comparisons:
            if comp.index_a is None or comp.index_b is None:
                continue
            divergent = [d for d in comp.divergences if d.diverges]
            if not divergent:
                continue
            dec = arbiter_client.arbitrate_record(paper_id, comp.model_used_a or "", divergent, source_text)
            if dec is not None:
                decisions[comp.index_a] = dec
        if decisions:
            reconciled, resolutions = apply_arbiter_decisions(
                dual_result.extraction_a, dual_result.extraction_b, dual_result.comparisons, decisions
            )
        timing["arbiter_s"] = round(time.monotonic() - t0, 2)

    # Etapa 3: verificacao de citacoes sobre a extracao RECONCILIADA, contra o
    # texto que o LLM efetivamente viu (janela) -- quotes validas por construcao.
    verifications = verify_paper_extraction(reconciled, source_text, threshold=citation_threshold)
    timing["total_s"] = round(time.monotonic() - t_start, 2)

    result = {
        "paper_id": paper_id,
        "source_text_len": len(source_text),
        "estimated_prompt_tokens": estimate_tokens(source_text),
        "fallback_level": fallback_level,
        "timing": timing,
        "empty_table_labels": empty_table_labels,
        "extraction_a": dual_result.extraction_a.model_dump(),
        "extraction_b": dual_result.extraction_b.model_dump(),
        "extraction_reconciled": reconciled.model_dump(),
        "arbiter_resolutions": resolutions,
        "record_count_mismatch": dual_result.record_count_mismatch,
        "comparisons": [
            {
                "index_a": c.index_a,
                "index_b": c.index_b,
                "model_used_a": c.model_used_a,
                "model_used_b": c.model_used_b,
                "match_score": c.match_score,
                "has_divergence": c.has_divergence,
                "divergences": [asdict(d) for d in c.divergences],
            }
            for c in dual_result.comparisons
        ],
        "verifications": [
            {
                "record_index": v.record_index,
                "model_used": v.model_used,
                "all_verified": v.all_verified,
                "unverified_fields": v.unverified_fields,
                "field_results": [asdict(fr) for fr in v.field_results],
            }
            for v in verifications
        ],
    }

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Checkpoint salvo: %s", ckpt_path)
    _clear_partial_checkpoint(checkpoint_dir, paper_id)
    return result
```

Nota: o aviso de `empty_table_labels` que existia antes da extração (linhas 211-217 do original) deve ser mantido — inserir este bloco logo após o ramo da triagem, ainda dentro de `if extraction_a is None:`:

```python
            if parsed.empty_table_labels:
                logger.warning(
                    "%s: %d tabela(s) identificadas pelo GROBID vieram sem conteudo extraido (%s). "
                    "O bloco de aviso na janela instrui o LLM a buscar esses valores em texto corrido.",
                    paper_id, len(parsed.empty_table_labels), "; ".join(parsed.empty_table_labels),
                )
```

- [ ] **Step 3c: `process_pdf_directory` com perfil + paralelismo posterior (Task 8 usa `profile.max_concurrency`)**

Substituir `process_pdf_directory` (linhas 378-441) por:

```python
def process_pdf_directory(
    pdf_dir: Path,
    out_dir: Path,
    grobid_url: str,
    citation_threshold: float,
    grobid_wait_s: int,
    skip_screening: bool,
    force: bool,
    profile: Profile,
) -> list[dict]:
    """Corpo de `run`, extraido para ser reusado por `registry-run` (que monta
    seu proprio --pdf-dir/--out-dir a partir do registry compartilhado antes
    de chamar isto). Retorna a lista de checkpoints processados nesta chamada
    (nao a lista completa historica -- ver load_checkpoints para isso)."""
    checkpoint_dir = out_dir / "checkpoints"
    tei_cache_dir = out_dir / "tei_cache"
    screening_dir = out_dir / "screening"

    grobid = GrobidClient(base_url=grobid_url)
    logger.info("Verificando GROBID em %s (aguardando ate %ss) ...", grobid_url, grobid_wait_s)
    grobid.wait_until_ready(max_wait_s=grobid_wait_s)
    logger.info("GROBID ok.")

    extractor_a = LLMExtractor(
        model=profile.model,
        api_base=profile.api_base,
        temperature=profile.temperature_a,
        max_tokens=profile.max_tokens,
        request_timeout=profile.request_timeout_s,
        min_num_ctx=profile.min_num_ctx,
    )
    extractor_b = LLMExtractor(
        model=profile.model,
        api_base=profile.api_base,
        temperature=profile.temperature_b,
        max_tokens=profile.max_tokens,
        request_timeout=profile.request_timeout_s,
        min_num_ctx=profile.min_num_ctx,
    )
    # Arbitro: determinístico (temp 0) e com saida curta -- o prompt ja e
    # pequeno por carregar so os campos divergentes + a janela.
    arbiter_client = ArbiterClient(
        LLMExtractor(
            model=profile.model,
            api_base=profile.api_base,
            temperature=0.0,
            max_tokens=profile.arbiter_max_tokens,
            request_timeout=profile.request_timeout_s,
            min_num_ctx=profile.arbiter_min_num_ctx,
        )
    )

    screening_client = None
    if not skip_screening:
        screening_client = ScreeningClient(model=profile.model, api_base=profile.api_base, request_timeout=profile.request_timeout_s)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("Nenhum PDF encontrado em %s", pdf_dir)
        return []

    def _process(pdf_path: Path) -> dict:
        try:
            result = process_one_pdf(
                pdf_path,
                grobid=grobid,
                extractor_a=extractor_a,
                tei_cache_dir=tei_cache_dir,
                checkpoint_dir=checkpoint_dir,
                citation_threshold=citation_threshold,
                force=force,
                screening_client=screening_client,
                screening_dir=screening_dir if not skip_screening else None,
                extractor_b=extractor_b,
                arbiter_client=arbiter_client,
                fallback_budgets=profile.fallback_budgets,
            )
            return {"paper_id": pdf_path.stem, "result": result}
        except Exception:  # noqa: BLE001
            logger.exception("Erro nao tratado processando %s -- pulando para o proximo.", pdf_path.name)
            return {"paper_id": pdf_path.stem, "result": None, "error": True}

    processed = []
    for pdf_path in tqdm(pdfs, desc="Processando papers"):
        processed.append(_process(pdf_path))

    return processed
```

(O loop sequencial acima é substituído por `_process_all` na Task 8 — mantido sequencial aqui para isolar a revisão.)

- [ ] **Step 3d: CLI `run`/`registry-run` com `--profile` + overrides**

Adicionar em `pipeline.py` (antes de `cli()`):

```python
def _resolve_profile(
    profile_name: str,
    model: str | None,
    api_base: str | None,
    temperature: float | None,
    llm_timeout_s: int | None,
    min_num_ctx: int | None,
) -> Profile:
    """Perfil do JSON + overrides individuais da CLI (flags tem precedencia)."""
    from dataclasses import replace

    profile = load_profile(profile_name)
    overrides = {}
    if model is not None:
        overrides["model"] = model
    if api_base is not None:
        overrides["api_base"] = api_base
    if temperature is not None:
        overrides["temperature_a"] = temperature
    if llm_timeout_s is not None:
        overrides["request_timeout_s"] = llm_timeout_s
    if min_num_ctx is not None:
        overrides["min_num_ctx"] = min_num_ctx
    resolved = replace(profile, **overrides) if overrides else profile
    logger.info("Perfil '%s': model=%s api_base=%s budget=%s concorrencia=%d",
                resolved.name, resolved.model, resolved.api_base, resolved.window_token_budget, resolved.max_concurrency)
    return resolved
```

No comando `run`: adicionar a opção `--profile` e trocar os defaults das flags de LLM para `None` (override):

```python
@cli.command()
@click.option("--pdf-dir", required=True, type=click.Path(exists=True, file_okay=False), help="Diretorio com os PDFs a processar.")
@click.option("--out-dir", required=True, type=click.Path(file_okay=False), help="Diretorio de saida (checkpoints + planilhas).")
@click.option("--profile", "profile_name", default="local", show_default=True, help="Perfil de deployment em config/profiles (local | cluster).")
@click.option("--grobid-url", default="http://localhost:8070", show_default=True)
@click.option("--model", default=None, help="Override do modelo do perfil (identificador LiteLLM).")
@click.option("--api-base", default=None, help="Override do endpoint do perfil.")
@click.option("--temperature", default=None, type=float, help="Override da temperatura da extracao A.")
@click.option("--citation-threshold", default=90.0, show_default=True, help="Limiar (%) de similaridade fuzzy para aceitar uma citacao.")
@click.option("--grobid-wait-s", default=300, show_default=True, help="Tempo maximo (s) para aguardar o GROBID ficar pronto.")
@click.option("--llm-timeout-s", default=None, type=int, help="Override do timeout (s) por chamada ao LLM.")
@click.option("--min-num-ctx", default=None, type=int, help="Override do piso de contexto (num_ctx) do perfil.")
@click.option("--skip-screening", is_flag=True, help="Pula a etapa 0 (triagem PRISMA).")
@click.option("--force", is_flag=True, help="Reprocessa mesmo que ja exista checkpoint.")
def run(pdf_dir, out_dir, profile_name, grobid_url, model, api_base, temperature, citation_threshold, grobid_wait_s, llm_timeout_s, min_num_ctx, skip_screening, force):
    """Roda o pipeline completo (etapas 0-6) sobre todos os PDFs de --pdf-dir."""
    pdf_dir = Path(pdf_dir)
    out_dir = Path(out_dir)
    screening_dir = out_dir / "screening"

    profile = _resolve_profile(profile_name, model, api_base, temperature, llm_timeout_s, min_num_ctx)

    process_pdf_directory(
        pdf_dir, out_dir, grobid_url, citation_threshold,
        grobid_wait_s, skip_screening, force, profile,
    )
    # ... resto inalterado (load_checkpoints, build_final_outputs, screening outputs)
```

Aplicar o mesmo padrão em `registry-run` (adicionar `--profile`, defaults `None` nas mesmas 5 flags, chamar `_resolve_profile`, e passar `profile` no lugar dos knobs na chamada de `process_pdf_directory`).

Em `cli.py`, `_perguntar_parametros_llm` passa a montar o perfil (substituir a função inteira):

```python
def _perguntar_parametros_llm() -> dict:
    """Monta o perfil de execucao a partir do perfil local + override de modelo
    perguntado ao usuario. A pergunta de perfil (local/cluster) chega na Task 10;
    aqui o menu continua funcionando como antes, agora via Profile."""
    from dataclasses import replace

    from .profiles import load_profile

    modelo = ui.perguntar("Modelo (Ollama)", doctor.MODELO_RECOMENDADO)
    skip_screening = ui.confirmar(
        "Pular a triagem PRISMA e processar todos os PDFs direto? "
        "(responda 'não' se ainda não filtrou manualmente os PDFs elegíveis)",
        padrao=False,
    )
    force = ui.confirmar("Reprocessar mesmo papers que já têm checkpoint salvo?", padrao=False)
    profile = load_profile("local")
    if modelo != doctor.MODELO_RECOMENDADO:
        profile = replace(profile, model=f"ollama_chat/{modelo}" if not modelo.startswith("ollama_chat/") else modelo)
    return {
        "grobid_url": doctor.DEFAULT_GROBID_URL,
        "citation_threshold": 90.0,
        "grobid_wait_s": 300,
        "skip_screening": skip_screening,
        "force": force,
        "profile": profile,
    }
```

E as duas chamadas de `process_pdf_directory` (cli.py:189-191 e 244-246) viram:

```python
    process_pdf_directory(pdf_dir, out_dir, params["grobid_url"], params["citation_threshold"],
                          params["grobid_wait_s"], params["skip_screening"], params["force"], params["profile"])
```

```python
    process_pdf_directory(staging_dir, out_dir, params["grobid_url"], params["citation_threshold"],
                          params["grobid_wait_s"], params["skip_screening"], params["force"], params["profile"])
```

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `python -m pytest -q`
Expected: suite inteira verde (69 antigos + novos). Se algum teste antigo chamar `process_pdf_directory` com a assinatura antiga, atualizar a chamada para o novo formato (sem mudar a lógica do teste).

- [ ] **Step 5: Commit**

```bash
git add src/parsing_papers/pipeline.py src/parsing_papers/dual_extraction.py src/parsing_papers/cli.py tests/test_pipeline_windows.py tests/test_dual_extraction.py
git commit -m "feat: janelas com escada de fallback + arbitro + observabilidade no pipeline"
```

---

### Task 7: Consolidação reconciliada + coluna `arbiter_resolved_fields`

**Files:**
- Modify: `src/parsing_papers/consolidate.py` (`OUTPUT_COLUMNS`:40-74, `_dual_comparison_summary`:77-103, `record_to_row`:106-160, `build_dataframe`:163-186)
- Modify: `src/parsing_papers/pipeline.py` (`build_final_outputs`:340-375)
- Test: `tests/test_consolidate_reconciled.py`

**Interfaces:**
- Consumes: checkpoint com `extraction_reconciled` e `arbiter_resolutions` (Task 6).
- Produces: `build_dataframe(extractions, verifications_by_paper, dual_comparisons_by_paper, arbiter_resolutions_by_paper=None)` — novo param opcional `dict[paper_id, dict[index_a, set[field_name]]]`; coluna nova `arbiter_resolved_fields` na planilha; `dual_extraction_diverges` passa a refletir apenas divergências NÃO resolvidas.

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_consolidate_reconciled.py
from parsing_papers.consolidate import build_dataframe
from parsing_papers.schema import (
    EvidenceField, ModelRecord, NumericEvidenceField, PaperExtraction, YesNo,
)


def _ev(value):
    return EvidenceField(value=value, quote=value, source_section="results")


def _num(value):
    return NumericEvidenceField(value=value, quote=value, source_section="results")


def _record(auc):
    return ModelRecord(
        model_used=_ev("XGBoost"), baseline=YesNo.NO, sample_size=_num("1000"),
        default_rate=_num(None), financial=YesNo.YES, productive=YesNo.NO,
        climatic=YesNo.NO, hybrid=YesNo.NO, variables_used=_ev("renda"),
        auc=_num(auc), accuracy=_num(None), f1_score=_num(None), recall=_num(None),
        specificity=_num(None), standard_deviation=_num(None), standard_error=_num(None),
    )


def _comparison(divergent_fields):
    return {
        "index_a": 0,
        "index_b": 0,
        "model_used_a": "XGBoost",
        "model_used_b": "XGBoost",
        "match_score": 100.0,
        "has_divergence": bool(divergent_fields),
        "divergences": [
            {"field_name": f, "value_a": "0.91", "value_b": "0.87", "diverges": True, "detail": ""}
            for f in divergent_fields
        ],
    }


def test_resolved_divergence_clears_flag_and_lists_resolution():
    extraction = PaperExtraction(paper_id="p", records=[_record("0.87")])  # reconciliada (valor de B)
    df = build_dataframe(
        [extraction],
        dual_comparisons_by_paper={"p": {0: _comparison(["auc"])}},
        arbiter_resolutions_by_paper={"p": {0: {"auc"}}},
    )
    row = df.iloc[0]
    assert row["AUC"] == "0.87"
    assert row["dual_extraction_diverges"] is False or row["dual_extraction_diverges"] == False  # noqa: E712
    assert row["dual_divergent_fields"] == ""
    assert row["arbiter_resolved_fields"] == "auc"
    assert row["needs_review"] == False  # noqa: E712


def test_neither_choice_stays_divergent():
    extraction = PaperExtraction(paper_id="p", records=[_record("0.91")])
    df = build_dataframe(
        [extraction],
        dual_comparisons_by_paper={"p": {0: _comparison(["auc"])}},
        arbiter_resolutions_by_paper={"p": {0: set()}},
    )
    row = df.iloc[0]
    assert row["dual_extraction_diverges"] == True  # noqa: E712
    assert row["needs_review"] == True  # noqa: E712


def test_no_arbiter_data_keeps_old_behavior():
    extraction = PaperExtraction(paper_id="p", records=[_record("0.91")])
    df = build_dataframe(
        [extraction],
        dual_comparisons_by_paper={"p": {0: _comparison(["auc"])}},
    )
    row = df.iloc[0]
    assert row["dual_extraction_diverges"] == True  # noqa: E712
    assert row["arbiter_resolved_fields"] == ""
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `python -m pytest tests/test_consolidate_reconciled.py -v`
Expected: FAIL — `build_dataframe() got an unexpected keyword argument 'arbiter_resolutions_by_paper'`

- [ ] **Step 3: Implementar as mudanças**

Em `consolidate.py`:

1. Em `OUTPUT_COLUMNS`, inserir `"arbiter_resolved_fields",` logo após `"dual_unmatched",`.

2. Substituir `_dual_comparison_summary` por:

```python
def _dual_comparison_summary(comparison: dict | None, resolved_fields: set[str] | None = None) -> dict:
    """
    Extrai da comparacao A/B (dict serializado a partir de RecordComparison,
    ver pipeline.py) os campos que vao para a planilha final. `comparison` e
    None quando nao ha checkpoint de dupla extracao disponivel para o registro
    (ex: consolidacao de dados legados).

    resolved_fields: campos ja decididos pelo arbitro (choice "a" ou "b") --
    sao REMOVIDOS da lista de divergentes e reportados em
    arbiter_resolved_fields. dual_extraction_diverges passa a significar
    "diverge E nao foi resolvido", que e o que merece revisao humana.
    """
    resolved_fields = resolved_fields or set()
    if comparison is None:
        return {
            "dual_extraction_diverges": None,
            "dual_model_used_b": "",
            "dual_match_score": None,
            "dual_divergent_fields": "",
            "dual_unmatched": False,
            "arbiter_resolved_fields": "",
        }

    unmatched = comparison.get("index_b") is None
    remaining_divergent = [
        d["field_name"]
        for d in comparison.get("divergences", [])
        if d.get("diverges") and d["field_name"] not in resolved_fields
    ]
    return {
        "dual_extraction_diverges": bool(remaining_divergent) or unmatched,
        "dual_model_used_b": comparison.get("model_used_b") or "",
        "dual_match_score": comparison.get("match_score"),
        "dual_divergent_fields": ", ".join(remaining_divergent),
        "dual_unmatched": unmatched,
        "arbiter_resolved_fields": ", ".join(sorted(resolved_fields)),
    }
```

3. `record_to_row` ganha o param e repassa (mudar assinatura e a linha do `dual_summary`):

```python
def record_to_row(
    paper_id: str,
    record,
    verification: RecordVerificationResult | None = None,
    dual_comparison: dict | None = None,
    arbiter_resolved: set[str] | None = None,
) -> dict:
```

```python
    dual_summary = _dual_comparison_summary(dual_comparison, arbiter_resolved)
```

4. `build_dataframe` ganha o quarto param e resolve por registro:

```python
def build_dataframe(
    extractions: list[PaperExtraction],
    verifications_by_paper: dict[str, list[RecordVerificationResult]] | None = None,
    dual_comparisons_by_paper: dict[str, dict[int, dict]] | None = None,
    arbiter_resolutions_by_paper: dict[str, dict[int, set[str]]] | None = None,
) -> pd.DataFrame:
    """
    dual_comparisons_by_paper: paper_id -> {index_a: comparison_dict}, onde
    comparison_dict e o dict serializado de um RecordComparison (ver
    pipeline.py: result["comparisons"]). Indexado por index_a (posicao do
    registro dentro de extraction_a.records) em vez de posicao na lista de
    comparacoes, porque comparacoes de registros "so em B" nao tem index_a e
    nao devem ser confundidas com as de registros de A.

    arbiter_resolutions_by_paper: paper_id -> {index_a: {field_name, ...}}
    com os campos ja resolvidos pelo arbitro (ver pipeline.py:
    result["arbiter_resolutions"]).
    """
    rows = []
    for extraction in extractions:
        verifications = (verifications_by_paper or {}).get(extraction.paper_id)
        comparisons_by_index_a = (dual_comparisons_by_paper or {}).get(extraction.paper_id, {})
        resolutions_by_index_a = (arbiter_resolutions_by_paper or {}).get(extraction.paper_id, {})
        for idx, record in enumerate(extraction.records):
            verification = verifications[idx] if verifications and idx < len(verifications) else None
            dual_comparison = comparisons_by_index_a.get(idx)
            arbiter_resolved = resolutions_by_index_a.get(idx)
            rows.append(record_to_row(extraction.paper_id, record, verification, dual_comparison, arbiter_resolved))

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return df
```

Em `pipeline.py`, `build_final_outputs`: usar a extração reconciliada e montar o mapa de resoluções (substituir a primeira linha da função e adicionar o mapa antes de `build_dataframe`):

```python
def build_final_outputs(checkpoints: list[dict], out_dir: Path) -> tuple[Path, Path, Path, Path]:
    # Extracao reconciliada (A + decisoes do arbitro) e a fonte da planilha;
    # checkpoints antigos (sem extraction_reconciled) caem no fallback para A.
    extractions = [
        PaperExtraction.model_validate(c.get("extraction_reconciled") or c["extraction_a"])
        for c in checkpoints
    ]
```

```python
        arbiter_resolutions_by_paper[paper_id] = {}
        for r in c.get("arbiter_resolutions", []):
            arbiter_resolutions_by_paper[paper_id].setdefault(r["index_a"], set()).add(r["field_name"])
```

(declarar `arbiter_resolutions_by_paper = {}` junto de `verifications_by_paper = {}` no topo do loop, e passar como 4º argumento em `build_dataframe(extractions, verifications_by_paper, dual_comparisons_by_paper, arbiter_resolutions_by_paper)`.)

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `python -m pytest -q`
Expected: verde. Se `tests/test_consolidate_and_audit.py` (ou outro) assertar a lista exata de colunas, atualizar a expectativa incluindo `"arbiter_resolved_fields"` — única mudança permitida em teste antigo.

- [ ] **Step 5: Commit**

```bash
git add src/parsing_papers/consolidate.py src/parsing_papers/pipeline.py tests/test_consolidate_reconciled.py tests/test_consolidate_and_audit.py
git commit -m "feat: planilha final usa extracao reconciliada pelo arbitro"
```

---

### Task 8: Paralelismo por perfil (`_process_all` + ThreadPoolExecutor)

**Files:**
- Modify: `src/parsing_papers/pipeline.py` (imports; loop final de `process_pdf_directory`)
- Test: `tests/test_process_all.py`

**Interfaces:**
- Consumes: `profile.max_concurrency` (Task 1), closure `_process` (Task 6).
- Produces: `_process_all(pdfs, process_fn, max_concurrency: int) -> list` — Task 6 já chama `_process(pdf_path)`; esta task só troca o loop. `process_one_pdf` permanece inalterada (checkpoints são 1 arquivo por PDF → escrita paralela segura).

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_process_all.py
import threading
import time

from parsing_papers.pipeline import _process_all


def test_process_all_sequential():
    out = _process_all([1, 2], lambda x: x + 1, max_concurrency=1)
    assert out == [2, 3]


def test_process_all_runs_concurrently():
    active = 0
    max_active = 0
    lock = threading.Lock()

    def work(x):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return x * 2

    out = _process_all(list(range(6)), work, max_concurrency=3)
    assert sorted(out) == [0, 2, 4, 6, 8, 10]
    assert max_active > 1  # prova que houve sobreposicao real de execucao
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `python -m pytest tests/test_process_all.py -v`
Expected: FAIL — `ImportError: cannot import name '_process_all'`

- [ ] **Step 3: Implementar**

Em `pipeline.py`, adicionar o import no topo:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
```

Adicionar a função antes de `process_pdf_directory`:

```python
def _process_all(pdfs: list[Path], process_fn, max_concurrency: int) -> list:
    """
    Roda process_fn sobre os PDFs. max_concurrency>1 usa pool de threads:
    as chamadas LLM/GROBID sao HTTP bloqueante (I/O-bound), entao threads
    bastam -- o GIL nao e gargalo e o batching do lado do servidor (vLLM)
    resolve a GPU. O resultado sai em ordem de conclusao; nao importa, pois
    load_checkpoints rele os arquivos do disco em ordem alfabetica.
    """
    if max_concurrency <= 1:
        return [process_fn(p) for p in tqdm(pdfs, desc="Processando papers")]

    results = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = [pool.submit(process_fn, p) for p in pdfs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processando papers"):
            results.append(fut.result())
    return results
```

Em `process_pdf_directory`, substituir o loop final:

```python
    processed = []
    for pdf_path in tqdm(pdfs, desc="Processando papers"):
        processed.append(_process(pdf_path))

    return processed
```

por:

```python
    return _process_all(pdfs, _process, profile.max_concurrency)
```

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `python -m pytest -q`
Expected: verde

- [ ] **Step 5: Commit**

```bash
git add src/parsing_papers/pipeline.py tests/test_process_all.py
git commit -m "feat: processamento paralelo de PDFs conforme max_concurrency do perfil"
```

---

### Task 9: Comando `compare` — benchmark sem gabarito

**Files:**
- Modify: `src/parsing_papers/pipeline.py` (novas funções ao final, antes de `if __name__`)
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: `load_checkpoints` (pipeline.py), campos de checkpoint `estimated_prompt_tokens`/`fallback_level`/`arbiter_resolutions`/`timing` (Task 6) — todos lidos com `.get` e fallback para checkpoints antigos (`source_text_len // 3`).
- Produces: `build_compare_report(run_a_dir: Path, run_b_dir: Path) -> pd.DataFrame`; comando CLI `parsing-papers compare`/`python -m parsing_papers.pipeline compare --run-a DIR --run-b DIR`.

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_compare.py
import json

from parsing_papers.pipeline import build_compare_report


def _ckpt(paper_id, records, tokens, fallback, resolutions, total_s):
    return {
        "paper_id": paper_id,
        "estimated_prompt_tokens": tokens,
        "fallback_level": fallback,
        "extraction_a": {"paper_id": paper_id, "records": records},
        "comparisons": [
            {"index_a": 0, "index_b": 0, "divergences": [{"field_name": "auc", "diverges": True}]}
        ] if records else [],
        "arbiter_resolutions": resolutions,
        "timing": {"total_s": total_s},
    }


def _write_run(base, ckpts):
    d = base / "checkpoints"
    d.mkdir(parents=True)
    for c in ckpts:
        (d / f"{c['paper_id']}.json").write_text(json.dumps(c), encoding="utf-8")


def test_build_compare_report(tmp_path):
    rec = {"model_used": {"value": "XGBoost", "quote": None, "source_section": None}}
    _write_run(tmp_path / "old", [_ckpt("p1", [rec], 30000, 0, [], 900.0)])
    _write_run(tmp_path / "new", [_ckpt("p1", [rec], 5000, 1, [{"index_a": 0, "field_name": "auc", "choice": "b"}], 120.0)])

    df = build_compare_report(tmp_path / "old", tmp_path / "new")

    row = df[df["paper_id"] == "p1"].iloc[0]
    assert row["prompt_tokens_a"] == 30000
    assert row["prompt_tokens_b"] == 5000
    assert row["fallback_level_b"] == 1
    assert row["divergent_fields_a"] == 1
    assert row["arbiter_resolved_b"] == 1
    assert row["total_s_a"] == 900.0


def test_compare_tolerates_old_checkpoints(tmp_path):
    # checkpoint legado: so paper_id + extraction_a + source_text_len
    legacy = {"paper_id": "p9", "source_text_len": 90000, "extraction_a": {"paper_id": "p9", "records": []}}
    _write_run(tmp_path / "old", [legacy])
    _write_run(tmp_path / "new", [legacy])

    df = build_compare_report(tmp_path / "old", tmp_path / "new")
    assert df.iloc[0]["prompt_tokens_a"] == 30000  # source_text_len // 3
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `python -m pytest tests/test_compare.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_compare_report'`

- [ ] **Step 3: Implementar**

Em `pipeline.py`, antes de `if __name__ == "__main__":`:

```python
def _summarize_checkpoint(c: dict) -> dict:
    """Metricas de um checkpoint para o compare. Le campos novos com .get e
    cai no fallback para checkpoints legados (pre-janelas/arbitro)."""
    reconciled = c.get("extraction_reconciled") or c.get("extraction_a") or {}
    return {
        "records": len(reconciled.get("records", [])),
        "prompt_tokens": c.get("estimated_prompt_tokens") or (c.get("source_text_len", 0) // 3),
        "fallback_level": c.get("fallback_level", 0),
        "divergent_fields": sum(
            1 for comp in c.get("comparisons", []) for d in comp.get("divergences", []) if d.get("diverges")
        ),
        "unmatched": sum(1 for comp in c.get("comparisons", []) if comp.get("index_b") is None),
        "arbiter_resolved": len(c.get("arbiter_resolutions", [])),
        "total_s": (c.get("timing") or {}).get("total_s"),
    }


def build_compare_report(run_a_dir: Path, run_b_dir: Path) -> pd.DataFrame:
    """Compara os checkpoints de duas rodadas (ex: baseline full-text vs
    janelas+arbitro). Linhas = papers (uniao); colunas com sufixo _a/_b."""
    run_a = {c["paper_id"]: _summarize_checkpoint(c) for c in load_checkpoints(Path(run_a_dir) / "checkpoints")}
    run_b = {c["paper_id"]: _summarize_checkpoint(c) for c in load_checkpoints(Path(run_b_dir) / "checkpoints")}
    keys = ("records", "prompt_tokens", "fallback_level", "divergent_fields", "unmatched", "arbiter_resolved", "total_s")
    rows = []
    for paper_id in sorted(set(run_a) | set(run_b)):
        row = {"paper_id": paper_id}
        for suffix, data in (("_a", run_a.get(paper_id)), ("_b", run_b.get(paper_id))):
            for key in keys:
                row[f"{key}{suffix}"] = data[key] if data else None
        rows.append(row)
    return pd.DataFrame(rows)


@cli.command()
@click.option("--run-a", required=True, type=click.Path(exists=True, file_okay=False), help="Out-dir da rodada baseline (ex: full-text).")
@click.option("--run-b", required=True, type=click.Path(exists=True, file_okay=False), help="Out-dir da rodada nova (ex: janelas).")
@click.option("--csv", "csv_path", default=None, type=click.Path(file_okay=False), help="Opcional: salva o relatorio em CSV.")
def compare(run_a, run_b, csv_path):
    """Compara duas rodadas lado a lado: tokens, registros, divergencias, arbitro, tempo.

    Uso tipico (benchmark sem gabarito -- ver spec):
      python -m parsing_papers.pipeline compare --run-a data/extracted --run-b data/extracted_novo
    """
    df = build_compare_report(Path(run_a), Path(run_b))
    if df.empty:
        logger.warning("Nenhum checkpoint encontrado nas duas pastas.")
        return
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df.to_string(index=False))
    numeric = df.drop(columns=["paper_id"]).apply(pd.to_numeric, errors="coerce")
    print("\nMedias:")
    print(numeric.mean().round(2).to_string())
    if csv_path:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("Relatorio salvo em %s", csv_path)
```

- [ ] **Step 4: Rodar os testes e ver passar**

Run: `python -m pytest tests/test_compare.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/parsing_papers/pipeline.py tests/test_compare.py
git commit -m "feat: comando compare para benchmark de rodadas (tokens/tempo/divergencias)"
```

---

### Task 10: vLLM no compose + doctor por perfil + menu com perfil + README

**Files:**
- Modify: `docker-compose.yml` (novo serviço `vllm` + volume)
- Modify: `src/parsing_papers/doctor.py` (funções vLLM + `diagnosticar_cluster`)
- Modify: `src/parsing_papers/cli.py` (`doctor_cmd`:44-51, `_perguntar_parametros_llm`:138-160)
- Modify: `README.md` (seções de modelos/VRAM e uso da CLI)
- Test: `tests/test_doctor_vllm.py`

**Interfaces:**
- Consumes: `Profile`/`load_profile` (Task 1); padrão `Checagem`/`DiagnosticoResultado` de `doctor.py`.
- Produces: `doctor._vllm_respondendo(api_base) -> tuple[Checagem, list[str]]`, `doctor._modelo_servido_vllm(modelos, modelo_desejado) -> Checagem`, `doctor.diagnosticar_cluster(api_base, modelo_desejado) -> DiagnosticoResultado`, `doctor.CONTAINER_VLLM = "parsing_papers_vllm"`.

- [ ] **Step 1: Escrever o teste que falha**

```python
# tests/test_doctor_vllm.py
from unittest.mock import patch

from parsing_papers import doctor


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_vllm_respondendo_lists_models():
    with patch("requests.get", return_value=_Resp({"data": [{"id": "Qwen/Qwen2.5-32B-Instruct-AWQ"}]})):
        check, modelos = doctor._vllm_respondendo("http://gpu:8000/v1")
    assert check.ok
    assert modelos == ["Qwen/Qwen2.5-32B-Instruct-AWQ"]


def test_modelo_servido_strips_litellm_prefix():
    check = doctor._modelo_servido_vllm(
        ["Qwen/Qwen2.5-32B-Instruct-AWQ"], "openai/Qwen/Qwen2.5-32B-Instruct-AWQ"
    )
    assert check.ok


def test_modelo_servido_missing():
    check = doctor._modelo_servido_vllm(["outro/modelo"], "openai/Qwen/Qwen2.5-32B-Instruct-AWQ")
    assert not check.ok
    assert "outro/modelo" in check.detalhe
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `python -m pytest tests/test_doctor_vllm.py -v`
Expected: FAIL — `AttributeError: module 'parsing_papers.doctor' has no attribute '_vllm_respondendo'`

- [ ] **Step 3a: Serviço vLLM no `docker-compose.yml`**

Adicionar após o serviço `ollama` (e o volume ao final):

```yaml
  vllm:
    image: vllm/vllm-openai:latest
    container_name: parsing_papers_vllm
    # Perfil "cluster": so sobe com `docker compose --profile cluster up -d vllm`.
    # Modelo default: Qwen2.5-32B-Instruct-AWQ (~19,5GB de pesos) -- requer
    # GPU >= 24GB. Para GPUs de 18-20GB, troque --model para
    # Qwen/Qwen2.5-14B-Instruct-AWQ e ajuste "model" em config/profiles/cluster.json.
    profiles: ["cluster"]
    ports:
      - "8000:8000"
    volumes:
      - vllm_cache:/root/.cache/huggingface
    command: >
      --model Qwen/Qwen2.5-32B-Instruct-AWQ
      --max-model-len 16384
      --gpu-memory-utilization 0.92
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

```yaml
volumes:
  ollama_data:
  vllm_cache:
```

- [ ] **Step 3b: `doctor.py` — checagens vLLM**

Adicionar ao final de `doctor.py`:

```python
CONTAINER_VLLM = "parsing_papers_vllm"


def _vllm_respondendo(api_base: str) -> tuple[Checagem, list[str]]:
    """Retorna (checagem, lista_de_modelos_servidos). api_base no formato http://host:8000/v1."""
    import requests

    try:
        r = requests.get(f"{api_base.rstrip('/')}/models", timeout=10)
        r.raise_for_status()
        modelos = [m["id"] for m in r.json().get("data", [])]
        return Checagem("vLLM respondendo", True, f"OK em {api_base}."), modelos
    except requests.RequestException as e:
        return (
            Checagem(
                "vLLM respondendo", False,
                f"sem resposta em {api_base}: {e}",
                "docker compose --profile cluster up -d vllm  (a primeira carga baixa ~20GB de modelo; "
                f"acompanhe com `docker logs {CONTAINER_VLLM}`)",
            ),
            [],
        )


def _modelo_servido_vllm(modelos: list[str], modelo_desejado: str) -> Checagem:
    # cluster.json usa o prefixo LiteLLM "openai/"; o id servido pelo vLLM nao o tem
    model_id = modelo_desejado.split("/", 1)[1] if modelo_desejado.startswith("openai/") else modelo_desejado
    if any(m == model_id or model_id in m for m in modelos):
        return Checagem("Modelo LLM servido", True, f"'{model_id}' disponivel.")
    return Checagem(
        "Modelo LLM servido", False,
        f"'{model_id}' nao esta entre os servidos: {', '.join(modelos) or 'nenhum'}.",
        "Confira o --model do servico vllm no docker-compose.yml e o campo 'model' de config/profiles/cluster.json.",
    )


def diagnosticar_cluster(api_base: str, modelo_desejado: str) -> DiagnosticoResultado:
    """Diagnostico do perfil cluster: o servidor vLLM pode ser REMOTO (cluster),
    entao nao checa Docker/Ollama locais -- so o endpoint e o modelo servido."""
    resultado = DiagnosticoResultado()
    vllm_check, modelos = _vllm_respondendo(api_base)
    resultado.checagens.append(vllm_check)
    if vllm_check.ok:
        resultado.checagens.append(_modelo_servido_vllm(modelos, modelo_desejado))
    return resultado
```

- [ ] **Step 3c: `cli.py` — `doctor --profile` e pergunta de perfil no menu**

Substituir `doctor_cmd` (linhas 44-51) por:

```python
@app.command(name="doctor")
def doctor_cmd(
    grobid_url: str = typer.Option(doctor.DEFAULT_GROBID_URL, "--grobid-url"),
    ollama_url: str = typer.Option(doctor.DEFAULT_OLLAMA_URL, "--ollama-url"),
    modelo: str = typer.Option(doctor.MODELO_RECOMENDADO, "--model"),
    profile_name: str = typer.Option("local", "--profile", help="Perfil a diagnosticar (local | cluster)."),
):
    """Verifica se o ambiente do perfil esta pronto (Docker/GROBID/Ollama no local; endpoint vLLM no cluster)."""
    if profile_name == "cluster":
        from .profiles import load_profile

        p = load_profile("cluster")
        ui.secao("Verificando perfil cluster (vLLM)")
        with ui.console.status("[primaria]checando endpoint vLLM...[/]", spinner="dots"):
            resultado = doctor.diagnosticar_cluster(p.api_base, p.model)
        ui.tabela_diagnostico(resultado.checagens)
        if not resultado.tudo_ok:
            ui.erro("Perfil cluster nao esta pronto -- resolva os itens marcados acima.")
    else:
        _rodar_diagnostico(grobid_url, ollama_url, modelo)
```

Em `_perguntar_parametros_llm` (reescrita na Task 6), trocar a linha do modelo por pergunta de perfil:

```python
def _perguntar_parametros_llm() -> dict:
    """Monta o perfil de execucao: pergunta local/cluster e, no local, permite
    trocar o modelo Ollama. Enter aceita os defaults."""
    from dataclasses import replace

    from .profiles import load_profile

    perfil_nome = ui.perguntar("Perfil de execução (local/cluster)", "local")
    try:
        profile = load_profile(perfil_nome)
    except (FileNotFoundError, ValueError) as e:
        ui.erro(f"{e} -- usando perfil 'local'.")
        profile = load_profile("local")

    if profile.name == "local":
        modelo = ui.perguntar("Modelo (Ollama)", doctor.MODELO_RECOMENDADO)
        if modelo != doctor.MODELO_RECOMENDADO:
            profile = replace(profile, model=f"ollama_chat/{modelo}" if not modelo.startswith("ollama_chat/") else modelo)

    skip_screening = ui.confirmar(
        "Pular a triagem PRISMA e processar todos os PDFs direto? "
        "(responda 'não' se ainda não filtrou manualmente os PDFs elegíveis)",
        padrao=False,
    )
    force = ui.confirmar("Reprocessar mesmo papers que já têm checkpoint salvo?", padrao=False)
    return {
        "grobid_url": doctor.DEFAULT_GROBID_URL,
        "citation_threshold": 90.0,
        "grobid_wait_s": 300,
        "skip_screening": skip_screening,
        "force": force,
        "profile": profile,
    }
```

- [ ] **Step 3d: README**

Atualizar três pontos do `README.md` (ler as seções antes de editar; não alterar o restante):

1. Na seção de modelos/VRAM (tabela que hoje recomenda `qwen2.5:14b-instruct` para 12GB), adicionar nota: *"Desde a versão com janelas candidatas, o prompt de extração caiu de 10–32K para ~4–6K tokens; o gargalo de `num_ctx` em GPUs de 12GB foi eliminado. Para cluster (>18GB), use o perfil `cluster`: `Qwen2.5-32B-Instruct-AWQ` em vLLM (GPU ≥24GB; para 18–20GB, `Qwen2.5-14B-Instruct-AWQ`)."*
2. Na seção de uso da CLI, substituir o bloco de flags de modelo por: *"`--profile local|cluster` seleciona o conjunto de parâmetros de `config/profiles/<nome>.json` (modelo, endpoint, orçamento de janela, concorrência). Flags `--model`, `--api-base`, `--temperature`, `--llm-timeout-s`, `--min-num-ctx` agora são overrides individuais do perfil."* e adicionar exemplo: `python -m parsing_papers.pipeline run --pdf-dir data/pdfs --out-dir data/extracted --profile cluster`.
3. Adicionar seção curta "Comparando rodadas (benchmark)": *"`python -m parsing_papers.pipeline compare --run-a <out_dir_baseline> --run-b <out_dir_novo>` imprime tokens/prompt, registros, divergências, resoluções do árbitro e tempo por paper, lado a lado."* e mencionar `docker compose --profile cluster up -d vllm` na seção de serviços Docker.

- [ ] **Step 4: Rodar a suite completa**

Run: `python -m pytest -q`
Expected: verde (69 antigos + ~20 novos)

- [ ] **Step 5: Smoke test manual (não automatizado — requer GPU/serviços)**

Checklist para o usuário executar (registrar evidência no PR/commit):
1. `docker compose up -d grobid ollama` + `parsing-papers doctor` → tudo ok.
2. Rodada nova: `python -m parsing_papers.pipeline run --pdf-dir data/pdfs --out-dir data/extracted_janelas --profile local --force`
3. Benchmark: `python -m parsing_papers.pipeline compare --run-a data/extracted --run-b data/extracted_janelas` → verificar queda de `prompt_tokens` (10–32K → ≤6K nos papers sem fallback), os 2 papers que davam 0 registros agora com ≥1 (via janela ou fallback), e `arbiter_resolved` > 0.
4. Cluster (quando disponível): `docker compose --profile cluster up -d vllm`, `parsing-papers doctor --profile cluster`, rodada com `--profile cluster`.

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml src/parsing_papers/doctor.py src/parsing_papers/cli.py README.md tests/test_doctor_vllm.py
git commit -m "feat: perfil cluster com vLLM, doctor por perfil e menu com selecao de perfil"
```

---

## Notas de execução

- **Ordem das tasks:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10. Tasks 2/3 são independentes entre si; 4/5 são independentes entre si; 6 depende de 1–5; 7 depende de 6; 8/9 dependem de 6; 10 depende de 1.
- **Validação contínua:** `python -m pytest -q` verde ao final de CADA task. Os 69 testes atuais não podem quebrar (exceção: assertions de lista de colunas na Task 7).
- **Critérios de sucesso (spec §5)** medidos no smoke test da Task 10 via `compare`.
