"""
Etapa 1 do pipeline: parsing estruturado deterministico via GROBID.

GROBID converte o PDF em TEI XML com secoes rotuladas (abstract, body dividido
em divs com headers, tabelas). Isso e baseado em regras/ML classico, nao gera
alucinacao, e permite que a gente extraia so metodos+resultados+tabelas antes
de mandar pro LLM -- cortando ruido e volume de tokens.

Requer um servico GROBID rodando (ver docker-compose.yml -- `docker compose up -d grobid`).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from lxml import etree

logger = logging.getLogger(__name__)

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# Headers de secao que indicam metodos/resultados (case-insensitive, PT/EN).
METHODS_KEYWORDS = [
    "method", "materials", "data", "model", "metodolog", "método", "dados", "modelo",
]
RESULTS_KEYWORDS = [
    "result", "discussion", "resultado", "discussão", "findings", "evaluation",
]


@dataclass
class GrobidSection:
    header: str
    text: str


@dataclass
class GrobidTable:
    label: str
    caption: str
    text: str  # texto bruto extraido da tabela (linhas/celulas concatenadas)


@dataclass
class ParsedPaper:
    paper_id: str
    title: str
    abstract: str
    sections: list[GrobidSection] = field(default_factory=list)
    tables: list[GrobidTable] = field(default_factory=list)
    # Rotulos de tabelas que o GROBID identificou (figure type="table") mas cujo
    # conteudo veio vazio -- limitacao conhecida do parser de tabelas do GROBID
    # em PDFs com layout complexo (celulas mescladas, fontes pequenas, tabela
    # cruzando pagina). Os dados dessas tabelas podem ainda existir em texto
    # corrido nas secoes -- ver prompts.py regras 7/8.
    empty_table_labels: list[str] = field(default_factory=list)
    raw_tei_path: Path | None = None

    def methods_and_results_text(self) -> str:
        """
        Concatena abstract + TODAS as secoes do corpo + tabelas.

        Antes este metodo filtrava secoes por palavra-chave (METHODS_KEYWORDS/
        RESULTS_KEYWORDS) para reduzir tokens. Isso se mostrou fragil: papers
        com cabecalhos fora do vocabulario previsto (ex: em portugues, ou
        titulos criativos como "Comparison with individual models") tinham
        secoes inteiras descartadas antes mesmo de chegar ao LLM -- em um caso
        real, 7 de 11 secoes foram cortadas, incluindo as que descreviam os
        metodos. Preferimos gastar mais tokens a perder dados silenciosamente
        na etapa deterministica; o LLM e instruido a focar em metodos/
        resultados/tabelas mesmo recebendo o texto completo.
        """
        parts = [f"TITLE: {self.title}", f"ABSTRACT: {self.abstract}"]

        for sec in self.sections:
            parts.append(f"\n=== SECTION: {sec.header} ===\n{sec.text}")

        for tbl in self.tables:
            parts.append(f"\n=== TABLE: {tbl.label} - {tbl.caption} ===\n{tbl.text}")

        if self.empty_table_labels:
            labels_str = "; ".join(self.empty_table_labels)
            parts.append(
                "\n=== AVISO DE PARSING ===\n"
                f"As seguintes tabelas foram identificadas no PDF mas o parser NAO conseguiu "
                f"extrair o conteudo numerico delas (vieram vazias): {labels_str}. "
                "Os valores dessas tabelas podem ainda aparecer em texto corrido nas secoes "
                "de resultados/discussao/comparacao acima -- procure ativamente por eles la "
                "antes de considerar um dado como nao reportado."
            )

        return "\n".join(parts)

    def full_text(self) -> str:
        parts = [f"TITLE: {self.title}", f"ABSTRACT: {self.abstract}"]
        for sec in self.sections:
            parts.append(f"\n=== SECTION: {sec.header} ===\n{sec.text}")
        for tbl in self.tables:
            parts.append(f"\n=== TABLE: {tbl.label} - {tbl.caption} ===\n{tbl.text}")
        return "\n".join(parts)


class GrobidClient:
    def __init__(self, base_url: str = "http://localhost:8070", timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_alive(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/isalive", timeout=10)
            return r.status_code == 200 and r.text.strip().lower() == "true"
        except requests.RequestException:
            return False

    def wait_until_ready(self, max_wait_s: int = 300, poll_s: int = 5) -> None:
        """
        Aguarda o endpoint /api/isalive responder. O GROBID carrega varios
        modelos CRF na inicializacao e pode legitimamente demorar 1-3 minutos
        (mais em disco lento, ex: WSL2 acessando /mnt/c/...) -- nao e um erro,
        e um crash real do container so aparece como container reiniciando em
        `docker ps` (STATUS "Restarting"), nao como demora aqui.
        """
        start = time.time()
        attempt = 0
        while time.time() - start < max_wait_s:
            if self.is_alive():
                return
            attempt += 1
            elapsed = int(time.time() - start)
            logger.info("Aguardando GROBID ficar pronto... (%ss decorridos, tentativa %d)", elapsed, attempt)
            time.sleep(poll_s)
        raise RuntimeError(
            f"GROBID nao respondeu em {max_wait_s}s em {self.base_url}. "
            "Verifique com `docker ps` se o container esta 'Up' (nao 'Restarting') "
            "e com `docker logs parsing_papers_grobid` se ha erro real, "
            "ou aumente --grobid-wait-s se a maquina/disco for lento."
        )

    def process_pdf(self, pdf_path: str | Path) -> str:
        """Envia o PDF para o endpoint processFulltextDocument e retorna o TEI XML bruto."""
        pdf_path = Path(pdf_path)
        with open(pdf_path, "rb") as f:
            files = {"input": (pdf_path.name, f, "application/pdf")}
            data = {
                "consolidateHeader": "1",
                "consolidateCitations": "0",
                "includeRawCitations": "0",
                "teiCoordinates": "0",
            }
            resp = requests.post(
                f"{self.base_url}/api/processFulltextDocument",
                files=files,
                data=data,
                timeout=self.timeout,
            )
        resp.raise_for_status()
        return resp.text

    def parse_pdf(self, pdf_path: str | Path, tei_cache_dir: str | Path | None = None) -> ParsedPaper:
        pdf_path = Path(pdf_path)
        paper_id = pdf_path.stem

        tei_xml = self.process_pdf(pdf_path)

        raw_tei_path = None
        if tei_cache_dir is not None:
            tei_cache_dir = Path(tei_cache_dir)
            tei_cache_dir.mkdir(parents=True, exist_ok=True)
            raw_tei_path = tei_cache_dir / f"{paper_id}.tei.xml"
            raw_tei_path.write_text(tei_xml, encoding="utf-8")

        parsed = self._parse_tei(tei_xml, paper_id=paper_id)
        parsed.raw_tei_path = raw_tei_path
        return parsed

    def parse_tei_file(self, tei_path: str | Path) -> ParsedPaper:
        tei_path = Path(tei_path)
        paper_id = tei_path.stem.replace(".tei", "")
        tei_xml = tei_path.read_text(encoding="utf-8")
        parsed = self._parse_tei(tei_xml, paper_id=paper_id)
        parsed.raw_tei_path = tei_path
        return parsed

    @staticmethod
    def _text_of(el) -> str:
        if el is None:
            return ""
        return " ".join("".join(el.itertext()).split())

    @staticmethod
    def _table_to_markdown(table_el) -> tuple[str, int]:
        """
        Converte um <tei:table> em uma tabela Markdown (linhas/colunas
        preservadas com '|'), em vez de achatar tudo numa unica string.

        Achatar celulas com itertext() (como o resto do parser faz para texto
        corrido) destroi a correspondencia linha/coluna: uma tabela como
            Regressao logistica | 30,36% | 63,02%
            Random forest       | 47,00% | 66,81%
        vira "Regressao logistica30,36%63,02%Random forest47,00%66,81%...",
        o que torna impossivel para o LLM (ou um humano) saber com confianca
        qual numero pertence a qual modelo/metrica. Isso e a causa raiz de
        varios casos de "records: []" mesmo com a tabela certa presente no
        texto: o LLM prefere nao reportar a arriscar uma correspondencia errada.

        Retorna (markdown, n_data_rows) -- n_data_rows exclui a linha de
        cabecalho, usado para decidir se a tabela tem conteudo util (ver
        MIN_TABLE_DATA_ROWS).
        """
        rows_text: list[list[str]] = []
        for row in table_el.findall("tei:row", TEI_NS):
            cells = row.findall("tei:cell", TEI_NS)
            row_text = [" ".join("".join(c.itertext()).split()) for c in cells]
            if any(row_text):
                rows_text.append(row_text)

        if not rows_text:
            return "", 0

        max_cols = max(len(r) for r in rows_text)
        lines = []
        for i, row in enumerate(rows_text):
            padded = row + [""] * (max_cols - len(row))
            lines.append("| " + " | ".join(padded) + " |")
            if i == 0:
                lines.append("|" + "|".join(["---"] * max_cols) + "|")
        n_data_rows = max(0, len(rows_text) - 1)
        return "\n".join(lines), n_data_rows

    def _parse_tei(self, tei_xml: str, paper_id: str) -> ParsedPaper:
        root = etree.fromstring(tei_xml.encode("utf-8"))

        title_el = root.find(".//tei:titleStmt/tei:title", TEI_NS)
        title = self._text_of(title_el)

        abstract_el = root.find(".//tei:abstract", TEI_NS)
        abstract = self._text_of(abstract_el)

        sections: list[GrobidSection] = []
        body = root.find(".//tei:text/tei:body", TEI_NS)
        if body is not None:
            for div in body.findall(".//tei:div", TEI_NS):
                head_el = div.find("tei:head", TEI_NS)
                header = self._text_of(head_el) if head_el is not None else ""
                # texto do div excluindo o head (para nao duplicar)
                clone_parts = []
                for child in div:
                    tag = etree.QName(child).localname
                    if tag == "head":
                        continue
                    clone_parts.append(self._text_of(child))
                text = " ".join(p for p in clone_parts if p)
                if text:
                    sections.append(GrobidSection(header=header or "(sem titulo)", text=text))

        # limiar minimo de LINHAS DE DADO (excluindo cabecalho) para considerar
        # que a tabela realmente tem conteudo util -- tabelas com 0 linhas de
        # dado (ex: so um fragmento de cabecalho como "Models" ou "Importance
        # ranking") sao efetivamente vazias: o GROBID capturou o rotulo, nao
        # os dados. Usamos contagem de linhas, nao tamanho de string, porque a
        # marcacao Markdown adiciona overhead fixo que distorce um limiar por
        # caracteres.
        MIN_TABLE_DATA_ROWS = 1

        tables: list[GrobidTable] = []
        empty_table_labels: list[str] = []
        figures = root.findall(".//tei:text//tei:figure[@type='table']", TEI_NS)
        for position, fig in enumerate(figures, start=1):
            label_el = fig.find("tei:label", TEI_NS)
            head_el = fig.find("tei:head", TEI_NS)
            table_el = fig.find("tei:table", TEI_NS)
            label = self._text_of(label_el) or "table"
            caption = self._text_of(head_el)
            # tenta primeiro preservar a estrutura linha/coluna em Markdown;
            # cai para texto achatado so se a tabela nao tiver row/cell (raro).
            text, n_data_rows = ("", 0)
            if table_el is not None:
                text, n_data_rows = self._table_to_markdown(table_el)
            if not text:
                text = self._text_of(table_el) if table_el is not None else self._text_of(fig)
                # sem estrutura row/cell para contar, usa heuristica de tamanho
                n_data_rows = 1 if len(text) >= 25 else 0
            if text and n_data_rows >= MIN_TABLE_DATA_ROWS:
                tables.append(GrobidTable(label=label, caption=caption, text=text))
            else:
                # GROBID identificou a tabela (rotulo/legenda) mas nao conseguiu
                # extrair o conteudo -- registra para nao desaparecer em silencio.
                # Inclui a posicao (#N de M) porque varias tabelas sem <head>/
                # <label> caem no mesmo fallback generico "table" e ficariam
                # indistinguiveis umas das outras no log e no aviso ao LLM.
                base_identifier = caption or f"table label={label!r}"
                identifier = f"{base_identifier} (#{position} de {len(figures)} tabelas no PDF)"
                empty_table_labels.append(identifier)
                logger.warning(
                    "GROBID nao extraiu conteudo da tabela '%s' em %s (parser de tabela falhou "
                    "nesse layout de PDF -- os dados podem ainda existir em texto corrido).",
                    identifier, paper_id,
                )

        return ParsedPaper(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            sections=sections,
            tables=tables,
            empty_table_labels=empty_table_labels,
        )
