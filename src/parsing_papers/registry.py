"""
Leitura/escrita do registro compartilhado (registry.jsonl) usado para o
handoff entre SPE (busca + PRISMA), pontodoi (download de PDF) e este
pipeline (extracao de metricas). Ver INTEGRATION.md e registry.schema.json
no repo synoptic-paper-engine para o contrato completo.

Autocontido (so stdlib + o schema.py/consolidate.py ja existentes neste
pacote): a mesma logica de registry e reimplementada, nao importada, em
cada um dos tres projetos -- "contrato de dados, nao codigo compartilhado"
(INTEGRATION.md secao 7). Mantem os tres repositorios desacoplados.

Fluxo consumido por este modulo (ver pipeline.py, comando `registry-run`):

  1. Ler registry.jsonl, filtrar fulltext_status=done AND
     extraction_status IN (pending, failed).
  2. Copiar/expor os PDFs pendentes num diretorio local (--pdf-dir do
     pipeline existente), usando o `record_id` como nome de arquivo --
     que tambem vira o `paper_id` (em vez do nome original do PDF, que
     pode ter acentos/caracteres problematicos).
  3. Apos rodar o pipeline normal (screening + extracao), atualizar
     extraction_status de volta no registry, e fazer o merge das colunas
     placeholder bibliograficas (doi/Year/Author/Journal/Country/
     Latitude/Longitude) a partir do bloco `metadata` de cada registro --
     resolvendo o TODO documentado em consolidate.py.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

STATUS_VALIDOS_EXTRACTION = {"pending", "done", "failed", "not_applicable"}


def normalize_doi(doi: str | None) -> str:
    """Mesma normalizacao usada em record_identity.py (SPE) e registry.py (pontodoi)."""
    if not doi:
        return ""
    doi = str(doi).strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip()


def normalize_title(title: str | None) -> str:
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(title).lower())


def build_record_id(doi: str | None = None, title: str | None = None) -> str | None:
    d = normalize_doi(doi)
    if d:
        return f"doi:{d}"
    t = normalize_title(title)
    if t:
        return f"title:{t}"
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_registry(registry_path: Path) -> dict[str, dict]:
    """Carrega registry.jsonl inteiro em um dict {record_id: registro}."""
    records: dict[str, dict] = {}
    if not registry_path.exists():
        return records
    with open(registry_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"! registry: linha {line_num} malformada em {registry_path}: {e}")
                continue
            rid = rec.get("record_id")
            if rid:
                records[rid] = rec
    return records


def save_registry(registry_path: Path, records: dict[str, dict]) -> None:
    """Reescreve o registry inteiro de forma atomica (tmp + Path.replace)."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = registry_path.with_suffix(registry_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rid in sorted(records.keys()):
            f.write(json.dumps(records[rid], ensure_ascii=False))
            f.write("\n")
    tmp_path.replace(registry_path)


def pending_for_extraction(registry_path: Path) -> list[dict]:
    """Registros com fulltext_status=done e extraction_status em
    (pending, failed) -- o que este pipeline deve processar a seguir."""
    records = load_registry(registry_path)
    return [
        r
        for r in records.values()
        if r.get("fulltext_status") == "done"
        and r.get("extraction_status") in ("pending", "failed")
    ]


def stage_pdfs(pending: list[dict], workspace_root: Path, staging_dir: Path) -> dict[str, str]:
    """Copia os PDFs pendentes para staging_dir, nomeados <record_id_sanitizado>.pdf,
    para servirem de --pdf-dir ao pipeline existente (que usa pdf_path.stem como
    paper_id -- ver pipeline.py:process_one_pdf).

    Retorna um dict {paper_id (=stem do arquivo copiado): record_id original},
    necessario porque record_id contem ':' e outros caracteres que nao viram
    nome de arquivo com seguranca em todos os SOs.

    fulltext.pdf_path no registro e relativo a workspace_root (a raiz do
    workspace compartilhado onde vive o proprio registry.jsonl) -- ver
    INTEGRATION.md secao 2.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    paper_id_to_record_id: dict[str, str] = {}

    for rec in pending:
        fulltext = rec.get("fulltext") or {}
        pdf_rel = fulltext.get("pdf_path")
        if not pdf_rel:
            print(f"! registry: {rec['record_id']} tem fulltext_status=done mas sem pdf_path -- pulando.")
            continue

        pdf_src = Path(pdf_rel)
        if not pdf_src.is_absolute():
            pdf_src = workspace_root / pdf_rel
        if not pdf_src.exists():
            print(f"! registry: PDF nao encontrado para {rec['record_id']}: {pdf_src} -- pulando.")
            continue

        paper_id = _sanitize_paper_id(rec["record_id"])
        pdf_dst = staging_dir / f"{paper_id}.pdf"
        if not pdf_dst.exists():
            shutil.copyfile(pdf_src, pdf_dst)
        paper_id_to_record_id[paper_id] = rec["record_id"]

    return paper_id_to_record_id


def _sanitize_paper_id(record_id: str) -> str:
    """record_id tipo 'doi:10.1234/example.2024' -> nome de arquivo seguro."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", record_id)


def mark_extraction_status(
    registry_path: Path,
    updates: list[dict],
) -> None:
    """Atualiza extraction_status (e opcionalmente o bloco extraction) em lote.

    `updates` e uma lista de dicts com chaves: record_id, status, e
    opcionalmente checkpoint_path / n_models_extracted / needs_review.
    """
    if not updates:
        return
    records = load_registry(registry_path)
    now = _now_iso()

    for upd in updates:
        rid = upd["record_id"]
        rec = records.get(rid)
        if rec is None:
            continue
        status = upd["status"]
        if status not in STATUS_VALIDOS_EXTRACTION:
            continue
        rec["extraction_status"] = status
        if status == "done":
            rec["extraction"] = {
                "checkpoint_path": upd.get("checkpoint_path"),
                "n_models_extracted": upd.get("n_models_extracted"),
                "needs_review": upd.get("needs_review"),
                "extracted_at": now,
            }
        rec["updated_at"] = now
        records[rid] = rec

    save_registry(registry_path, records)


def merge_bibliographic_columns(df: pd.DataFrame, registry_path: Path, paper_id_to_record_id: dict[str, str]) -> pd.DataFrame:
    """Preenche as colunas placeholder bibliograficas (doi/Year/Author/Journal/
    Country/Latitude/Longitude) do dataframe final a partir do bloco
    `metadata` de cada registro no registry -- em vez de deixa-las vazias,
    como consolidate.py faz quando rodado sem integracao (ver comentario em
    BIBLIOGRAPHIC_PLACEHOLDER_COLUMNS em consolidate.py).

    Nao modifica df in-place; retorna uma copia.
    """
    if df.empty:
        return df

    records = load_registry(registry_path)
    df = df.copy()

    for idx, row in df.iterrows():
        paper_id = row["paper_id"]
        record_id = paper_id_to_record_id.get(paper_id)
        if not record_id:
            continue
        rec = records.get(record_id)
        if not rec:
            continue
        meta = rec.get("metadata") or {}
        df.at[idx, "doi"] = rec.get("doi") or ""
        df.at[idx, "Year"] = meta.get("year") or ""
        df.at[idx, "Author"] = meta.get("author") or ""
        df.at[idx, "Journal"] = meta.get("journal") or ""
        df.at[idx, "Country"] = meta.get("country") or ""
        df.at[idx, "Latitude"] = meta.get("latitude") or ""
        df.at[idx, "Longitude"] = meta.get("longitude") or ""

    return df
