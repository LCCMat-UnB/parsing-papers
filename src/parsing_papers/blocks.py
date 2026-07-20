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
