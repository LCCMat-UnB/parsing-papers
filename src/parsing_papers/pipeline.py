"""
Orquestrador do pipeline completo (etapas 0-6), com CLI e checkpointing.

Fluxo por PDF:
  0. Triagem PRISMA de elegibilidade (screening.py) -- decide se o paper
     entra na revisao sistematica, na metanalise, ou e excluido. Papers
     excluidos NAO prosseguem para as etapas de extracao (2+), economizando
     as chamadas de LLM mais caras (dupla extracao com texto completo).
  1. GROBID parse (cacheado em disco -- se o .tei.xml ja existir, reusa)
  2. Extracao dupla via LLM (extracao A + extracao B)
  3. Verificacao de citacoes (fuzzy match) em cada extracao
  4. Comparacao das duas extracoes (divergencias)
  5. Checagem de sanidade nos valores
  6. Persistencia incremental em JSON (checkpoint por paper) -- permite retomar
     um lote grande sem reprocessar papers ja feitos.

Checkpoint INTERMEDIARIO (dentro de um mesmo paper): alem do checkpoint final
por paper, a extracao_a e persistida em disco assim que termina, ANTES da
extracao_b comecar. Cada chamada ao LLM custa minutos (timeout default de
900s) -- sem isso, uma interrupcao (crash do Ollama, timeout, queda de
energia) durante a extracao_b jogaria fora a extracao_a ja concluida com
sucesso, obrigando a repetir as DUAS chamadas ao reprocessar. Com o
checkpoint parcial, reprocessar reusa a extracao_a salva e so refaz a
extracao_b.

Ao final, roda a amostragem de auditoria e exporta xlsx/csv, alem da
planilha de triagem PRISMA (todos os papers, incluidos ou nao).

Uso:
    python -m parsing_papers.pipeline run --pdf-dir data/pdfs --out-dir data/extracted \\
        --model ollama_chat/qwen2.5:14b-instruct --grobid-url http://localhost:8070
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

from .audit_sampling import AuditSampleConfig, select_audit_sample
from .consolidate import build_dataframe, export
from .dual_extraction import compare_extractions, run_dual_extraction
from .grobid_client import GrobidClient
from .llm_client import ExtractionError, LLMExtractor
from .schema import PaperExtraction
from .screening import ScreeningClient, ScreeningError, should_proceed_to_extraction
from .screening_schema import ScreeningResult
from .verification import verify_paper_extraction
from . import registry as reg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _checkpoint_path(checkpoint_dir: Path, paper_id: str) -> Path:
    return checkpoint_dir / f"{paper_id}.json"


def _partial_checkpoint_path(checkpoint_dir: Path, paper_id: str) -> Path:
    """
    Checkpoint intermediario: guarda so a extracao_a (mais o texto fonte, para
    nao precisar re-rodar o GROBID) assim que ela termina. Removido apos o
    checkpoint final do paper ser gravado com sucesso.
    """
    return checkpoint_dir / f"{paper_id}.partial.json"


def _save_partial_checkpoint(checkpoint_dir: Path, paper_id: str, source_text: str, extraction_a: PaperExtraction) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _partial_checkpoint_path(checkpoint_dir, paper_id)
    payload = {
        "paper_id": paper_id,
        "source_text": source_text,
        "extraction_a": extraction_a.model_dump(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Checkpoint intermediario salvo (extracao_a concluida): %s", path)


def _load_partial_checkpoint(checkpoint_dir: Path, paper_id: str) -> dict | None:
    path = _partial_checkpoint_path(checkpoint_dir, paper_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Checkpoint intermediario de %s corrompido/ilegivel (%s) -- ignorando e reprocessando do zero.", paper_id, e)
        return None


def _clear_partial_checkpoint(checkpoint_dir: Path, paper_id: str) -> None:
    path = _partial_checkpoint_path(checkpoint_dir, paper_id)
    if path.exists():
        path.unlink()


def _screening_checkpoint_path(screening_dir: Path, paper_id: str) -> Path:
    return screening_dir / f"{paper_id}.json"


def run_screening_for_paper(
    paper_id: str,
    parsed,
    screening_client: ScreeningClient,
    screening_dir: Path,
    force: bool = False,
) -> ScreeningResult | None:
    """
    Etapa 0: roda a triagem PRISMA para um paper ja parseado pelo GROBID.

    Usa titulo+abstract como material de triagem por padrao (mais proximo do
    que um revisor humano avalia na etapa de triagem real, e mais barato que
    mandar o texto completo) -- cai para o texto completo (metodos+resultados)
    se o abstract vier vazio (comum quando o GROBID nao consegue segmentar o
    abstract de alguns PDFs).

    Retorna None se o checkpoint de triagem ja existir e force=False.
    """
    ckpt_path = _screening_checkpoint_path(screening_dir, paper_id)
    if ckpt_path.exists() and not force:
        data = json.loads(ckpt_path.read_text(encoding="utf-8"))
        return ScreeningResult.model_validate(data)

    if parsed.abstract and parsed.abstract.strip():
        screening_text = f"TITLE: {parsed.title}\nABSTRACT: {parsed.abstract}"
    else:
        logger.info("%s: abstract vazio/ausente, usando texto completo para a triagem.", paper_id)
        screening_text = parsed.methods_and_results_text()

    try:
        result = screening_client.screen(paper_id, screening_text)
    except ScreeningError as e:
        logger.error("Falha na triagem de %s: %s -- paper NAO sera excluido automaticamente, marcar para revisao manual.", paper_id, e)
        return None

    screening_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text(json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Triagem de %s: decisao=%s", paper_id, result.decision.value)
    return result


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
) -> dict | None:
    paper_id = pdf_path.stem
    ckpt_path = _checkpoint_path(checkpoint_dir, paper_id)

    if ckpt_path.exists() and not force:
        logger.info("Checkpoint encontrado para %s, pulando (use --force para reprocessar).", paper_id)
        return json.loads(ckpt_path.read_text(encoding="utf-8"))

    if force:
        # --force reprocessa do zero -- descarta tambem qualquer checkpoint
        # parcial obsoleto (ex: de uma extracao_a com um prompt/modelo antigo).
        _clear_partial_checkpoint(checkpoint_dir, paper_id)

    logger.info("Processando %s ...", paper_id)

    # Tenta reaproveitar uma extracao_a de um checkpoint intermediario (de uma
    # execucao anterior interrompida antes de completar este paper).
    partial = _load_partial_checkpoint(checkpoint_dir, paper_id)
    existing_extraction_a = None
    source_text = None
    if partial is not None:
        try:
            existing_extraction_a = PaperExtraction.model_validate(partial["extraction_a"])
            source_text = partial["source_text"]
            logger.info(
                "Retomando %s a partir de checkpoint intermediario -- reusando extracao_a (%d registros), pulando GROBID e a chamada de extracao_a.",
                paper_id, len(existing_extraction_a.records),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Checkpoint intermediario de %s invalido (%s) -- reprocessando do zero.", paper_id, e)
            existing_extraction_a = None
            source_text = None

    parsed = None
    if source_text is None:
        # Etapa 1: GROBID
        parsed = grobid.parse_pdf(pdf_path, tei_cache_dir=tei_cache_dir)

        # Etapa 0: triagem PRISMA -- roda ANTES da extracao cara. Paper
        # excluido na triagem nao prossegue (economiza a dupla extracao com
        # texto completo, que e a chamada mais cara do pipeline).
        if screening_client is not None and screening_dir is not None:
            screening_result = run_screening_for_paper(paper_id, parsed, screening_client, screening_dir, force=force)
            if screening_result is not None and not should_proceed_to_extraction(screening_result):
                logger.info(
                    "%s excluido na triagem PRISMA (decisao=%s) -- pulando extracao de metricas.",
                    paper_id, screening_result.decision.value,
                )
                return None

        source_text = parsed.methods_and_results_text()
        if not source_text.strip():
            logger.warning("Texto vazio apos parsing GROBID para %s -- pulando.", paper_id)
            return None
        if parsed.empty_table_labels:
            logger.warning(
                "%s: %d tabela(s) identificadas pelo GROBID vieram sem conteudo extraido (%s). "
                "O LLM foi instruido a buscar esses valores em texto corrido, mas vale conferir "
                "manualmente se os numeros criticos ficaram de fora.",
                paper_id, len(parsed.empty_table_labels), "; ".join(parsed.empty_table_labels),
            )

    empty_table_labels = parsed.empty_table_labels if parsed is not None else []

    def _on_extraction_a_done(extraction_a: PaperExtraction) -> None:
        _save_partial_checkpoint(checkpoint_dir, paper_id, source_text, extraction_a)

    # Etapas 2 e 4: dupla extracao
    try:
        dual_result = run_dual_extraction(
            paper_id,
            source_text,
            extractor_a,
            existing_extraction_a=existing_extraction_a,
            on_extraction_a_done=_on_extraction_a_done,
        )
    except ExtractionError as e:
        logger.error("Falha na extracao de %s: %s", paper_id, e)
        return None

    # Etapa 3: verificacao de citacoes (roda sobre a extracao A, que e a "principal")
    verifications = verify_paper_extraction(dual_result.extraction_a, source_text, threshold=citation_threshold)

    result = {
        "paper_id": paper_id,
        "source_text_len": len(source_text),
        "empty_table_labels": empty_table_labels,
        "extraction_a": dual_result.extraction_a.model_dump(),
        "extraction_b": dual_result.extraction_b.model_dump(),
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


def load_checkpoints(checkpoint_dir: Path) -> list[dict]:
    results = []
    for f in sorted(checkpoint_dir.glob("*.json")):
        if f.name.endswith(".partial.json"):
            continue  # checkpoints intermediarios nao sao resultados finais
        results.append(json.loads(f.read_text(encoding="utf-8")))
    return results


def load_screening_results(screening_dir: Path) -> list[dict]:
    if not screening_dir.exists():
        return []
    results = []
    for f in sorted(screening_dir.glob("*.json")):
        results.append(json.loads(f.read_text(encoding="utf-8")))
    return results


def build_screening_dataframe(screening_results: list[dict]) -> pd.DataFrame:
    """
    Planilha de triagem PRISMA (secao 6 do documento de criterios): uma linha
    por paper triado, incluidos ou nao, com a decisao, os codigos de criterio
    aplicados em cada categoria, e a justificativa -- rastreavel ate os
    codigos formais (I1-I8, E1-E11, M1-M7, ME1-ME6).
    """
    rows = []
    for r in screening_results:
        inclusion_codes = ", ".join(c["code"] for c in r.get("inclusion_criteria_met", []))
        exclusion_codes = ", ".join(c["code"] for c in r.get("exclusion_criteria_met", []))
        meta_inclusion_codes = ", ".join(c["code"] for c in r.get("metaanalysis_inclusion_criteria_met", []))
        meta_exclusion_codes = ", ".join(c["code"] for c in r.get("metaanalysis_exclusion_criteria_met", []))
        rows.append(
            {
                "paper_id": r["paper_id"],
                "decision": r["decision"],
                "inclusion_criteria_met": inclusion_codes,
                "exclusion_criteria_met": exclusion_codes,
                "metaanalysis_inclusion_criteria_met": meta_inclusion_codes,
                "metaanalysis_exclusion_criteria_met": meta_exclusion_codes,
                "justification": r.get("justification", ""),
                "screening_warnings": "; ".join(r.get("screening_warnings", [])),
            }
        )
    columns = [
        "paper_id", "decision",
        "inclusion_criteria_met", "exclusion_criteria_met",
        "metaanalysis_inclusion_criteria_met", "metaanalysis_exclusion_criteria_met",
        "justification", "screening_warnings",
    ]
    return pd.DataFrame(rows, columns=columns)


def export_screening_outputs(screening_results: list[dict], out_dir: Path) -> tuple[Path, Path]:
    df = build_screening_dataframe(screening_results)
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / "triagem_prisma.xlsx"
    csv_path = out_dir / "triagem_prisma.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)
    return xlsx_path, csv_path


def build_final_outputs(checkpoints: list[dict], out_dir: Path) -> tuple[Path, Path, Path, Path]:
    extractions = [PaperExtraction.model_validate(c["extraction_a"]) for c in checkpoints]

    verifications_by_paper = {}
    dual_comparisons_by_paper = {}
    for c in checkpoints:
        paper_id = c["paper_id"]
        from .verification import FieldVerification, RecordVerificationResult

        v_list = []
        for v in c["verifications"]:
            frs = [FieldVerification(**fr) for fr in v["field_results"]]
            v_list.append(RecordVerificationResult(record_index=v["record_index"], model_used=v["model_used"], field_results=frs))
        verifications_by_paper[paper_id] = v_list

        # Indexado por index_a (posicao do registro em extraction_a.records),
        # nao por posicao na lista de comparacoes -- comparacoes de registros
        # "so em B" (index_a=None) nao correspondem a nenhuma linha de A e
        # ficam de fora deste dict (elas nao aparecem na planilha hoje, pois
        # a planilha e construida a partir de extraction_a; ficam registradas
        # no checkpoint JSON bruto para quem quiser auditar manualmente).
        dual_comparisons_by_paper[paper_id] = {
            comp["index_a"]: comp for comp in c["comparisons"] if comp["index_a"] is not None
        }

    df = build_dataframe(extractions, verifications_by_paper, dual_comparisons_by_paper)

    xlsx_path, csv_path = export(df, out_dir, basename="metadados_extraidos")

    audit_df = select_audit_sample(df, AuditSampleConfig())
    audit_xlsx = out_dir / "auditoria_amostra.xlsx"
    audit_csv = out_dir / "auditoria_amostra.csv"
    audit_df.to_excel(audit_xlsx, index=False)
    audit_df.to_csv(audit_csv, index=False, encoding="utf-8-sig")

    return xlsx_path, csv_path, audit_xlsx, audit_csv


def process_pdf_directory(
    pdf_dir: Path,
    out_dir: Path,
    grobid_url: str,
    model: str,
    api_base: str,
    temperature: float,
    citation_threshold: float,
    grobid_wait_s: int,
    llm_timeout_s: int,
    min_num_ctx: int,
    skip_screening: bool,
    force: bool,
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
        model=model,
        api_base=api_base,
        temperature=temperature,
        request_timeout=llm_timeout_s,
        min_num_ctx=min_num_ctx,
    )

    screening_client = None
    if not skip_screening:
        screening_client = ScreeningClient(model=model, api_base=api_base, request_timeout=llm_timeout_s)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("Nenhum PDF encontrado em %s", pdf_dir)
        return []

    processed = []
    for pdf_path in tqdm(pdfs, desc="Processando papers"):
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
            )
            processed.append({"paper_id": pdf_path.stem, "result": result})
        except Exception:  # noqa: BLE001
            logger.exception("Erro nao tratado processando %s -- pulando para o proximo.", pdf_path.name)
            processed.append({"paper_id": pdf_path.stem, "result": None, "error": True})

    return processed


@click.group()
def cli():
    """Pipeline de extracao de metadados de papers (meta-analise credito agricola)."""


@cli.command()
@click.option("--pdf-dir", required=True, type=click.Path(exists=True, file_okay=False), help="Diretorio com os PDFs a processar.")
@click.option("--out-dir", required=True, type=click.Path(file_okay=False), help="Diretorio de saida (checkpoints + planilhas).")
@click.option("--grobid-url", default="http://localhost:8070", show_default=True)
@click.option("--model", default="ollama_chat/qwen2.5:14b-instruct", show_default=True, help="Identificador LiteLLM do modelo (etapa 2).")
@click.option("--api-base", default="http://localhost:11434", show_default=True, help="Endpoint do servidor do modelo (Ollama).")
@click.option("--temperature", default=0.1, show_default=True)
@click.option("--citation-threshold", default=90.0, show_default=True, help="Limiar (%) de similaridade fuzzy para aceitar uma citacao.")
@click.option("--grobid-wait-s", default=300, show_default=True, help="Tempo maximo (s) para aguardar o GROBID ficar pronto (o carregamento inicial dos modelos CRF pode demorar).")
@click.option("--llm-timeout-s", default=900, show_default=True, help="Timeout (s) por chamada ao LLM. Papers geram prompts grandes (milhares de tokens); em GPU com pouca VRAM ou modo hibrido GPU+CPU o prefill pode demorar bastante -- aumente se ver timeouts.")
@click.option("--min-num-ctx", default=16000, show_default=True, help="Piso minimo de contexto (tokens) pedido ao Ollama via num_ctx. O Ollama trunca o prompt em silencio se num_ctx for pequeno demais para o texto enviado.")
@click.option("--skip-screening", is_flag=True, help="Pula a etapa 0 (triagem PRISMA) e manda todos os PDFs direto para a extracao -- use se a triagem ja foi feita manualmente e so os PDFs elegiveis estao em --pdf-dir.")
@click.option("--force", is_flag=True, help="Reprocessa mesmo que ja exista checkpoint.")
def run(pdf_dir, out_dir, grobid_url, model, api_base, temperature, citation_threshold, grobid_wait_s, llm_timeout_s, min_num_ctx, skip_screening, force):
    """Roda o pipeline completo (etapas 0-6) sobre todos os PDFs de --pdf-dir."""
    pdf_dir = Path(pdf_dir)
    out_dir = Path(out_dir)
    screening_dir = out_dir / "screening"

    process_pdf_directory(
        pdf_dir, out_dir, grobid_url, model, api_base, temperature, citation_threshold,
        grobid_wait_s, llm_timeout_s, min_num_ctx, skip_screening, force,
    )

    checkpoint_dir = out_dir / "checkpoints"
    checkpoints = load_checkpoints(checkpoint_dir)
    xlsx_path, csv_path, audit_xlsx, audit_csv = build_final_outputs(checkpoints, out_dir)
    logger.info("Concluido. Planilha final: %s / %s", xlsx_path, csv_path)
    logger.info("Amostra de auditoria: %s / %s", audit_xlsx, audit_csv)

    if not skip_screening:
        screening_results = load_screening_results(screening_dir)
        if screening_results:
            screening_xlsx, screening_csv = export_screening_outputs(screening_results, out_dir)
            logger.info("Planilha de triagem PRISMA: %s / %s", screening_xlsx, screening_csv)


@cli.command(name="registry-run")
@click.option("--registry", required=True, type=click.Path(exists=True, dir_okay=False), help="Caminho do registry.jsonl compartilhado (do SPE/pontodoi).")
@click.option("--out-dir", required=True, type=click.Path(file_okay=False), help="Diretorio de saida (checkpoints + planilhas).")
@click.option("--grobid-url", default="http://localhost:8070", show_default=True, help="Endpoint do GROBID.")
@click.option("--model", default="ollama_chat/qwen2.5:14b-instruct", show_default=True, help="Identificador LiteLLM do modelo (etapa 2).")
@click.option("--api-base", default="http://localhost:11434", show_default=True, help="Endpoint do servidor do modelo (Ollama).")
@click.option("--temperature", default=0.1, show_default=True)
@click.option("--citation-threshold", default=90.0, show_default=True, help="Limiar (%) de similaridade fuzzy para aceitar uma citacao.")
@click.option("--grobid-wait-s", default=300, show_default=True)
@click.option("--llm-timeout-s", default=900, show_default=True)
@click.option("--min-num-ctx", default=16000, show_default=True)
@click.option("--skip-screening", is_flag=True)
@click.option("--force", is_flag=True, help="Reprocessa mesmo que ja exista checkpoint.")
def registry_run(registry, out_dir, grobid_url, model, api_base, temperature, citation_threshold, grobid_wait_s, llm_timeout_s, min_num_ctx, skip_screening, force):
    """Roda o pipeline sobre os PDFs pendentes no registry compartilhado
    (fulltext_status=done AND extraction_status in pending/failed), em vez de
    um --pdf-dir fixo. Ao final, atualiza extraction_status de volta no
    registry e faz o merge das colunas bibliograficas placeholder
    (doi/Year/Author/Journal/Country/Latitude/Longitude) a partir do bloco
    `metadata` de cada registro -- ver INTEGRATION.md no repo synoptic-paper-engine.
    """
    registry_path = Path(registry)
    out_dir = Path(out_dir)
    workspace_root = registry_path.parent  # raiz do workspace compartilhado (ver INTEGRATION.md secao 2)
    staging_dir = out_dir / "registry_pdfs"
    screening_dir = out_dir / "screening"
    checkpoint_dir = out_dir / "checkpoints"

    pending = reg.pending_for_extraction(registry_path)
    if not pending:
        logger.info("Nenhum paper pendente no registry (fulltext_status=done + extraction_status pendente/falho).")
        return

    logger.info("%d paper(s) pendentes de extracao no registry.", len(pending))
    paper_id_to_record_id = reg.stage_pdfs(pending, workspace_root, staging_dir)
    if not paper_id_to_record_id:
        logger.warning("Nenhum PDF pode ser localizado a partir dos registros pendentes -- abortando.")
        return

    process_pdf_directory(
        staging_dir, out_dir, grobid_url, model, api_base, temperature, citation_threshold,
        grobid_wait_s, llm_timeout_s, min_num_ctx, skip_screening, force,
    )

    checkpoints = load_checkpoints(checkpoint_dir)
    xlsx_path, csv_path, audit_xlsx, audit_csv = build_final_outputs(checkpoints, out_dir)

    # Merge das colunas bibliograficas placeholder a partir do registry, e
    # reexportacao da planilha final -- so para os papers desta rodada
    # (paper_id_to_record_id), sem tocar em checkpoints de rodadas anteriores
    # que ja tenham sido processados sem essa informacao.
    df = pd.read_csv(csv_path)
    df = reg.merge_bibliographic_columns(df, registry_path, paper_id_to_record_id)
    xlsx_path, csv_path = export(df, out_dir, basename="metadados_extraidos")
    logger.info("Concluido. Planilha final (com metadados bibliograficos mesclados): %s / %s", xlsx_path, csv_path)
    logger.info("Amostra de auditoria: %s / %s", audit_xlsx, audit_csv)

    if not skip_screening:
        screening_results = load_screening_results(screening_dir)
        if screening_results:
            screening_xlsx, screening_csv = export_screening_outputs(screening_results, out_dir)
            logger.info("Planilha de triagem PRISMA: %s / %s", screening_xlsx, screening_csv)

    # Atualiza extraction_status de volta no registry: done para quem tem
    # checkpoint final, not_applicable para quem foi excluido na triagem
    # (screening_result existe mas nao prosseguiu -- ver should_proceed_to_extraction),
    # failed para quem nao tem nem checkpoint nem screening (erro nao tratado).
    checkpoint_paper_ids = {c["paper_id"] for c in checkpoints}
    screening_results = load_screening_results(screening_dir) if not skip_screening else []
    screened_paper_ids = {r["paper_id"] for r in screening_results}

    # needs_review por paper_id: OR de todos os ModelRecords daquele paper na planilha final.
    needs_review_by_paper = df.groupby("paper_id")["needs_review"].any().to_dict() if not df.empty else {}
    n_models_by_paper = df.groupby("paper_id").size().to_dict() if not df.empty else {}

    updates = []
    for paper_id, record_id in paper_id_to_record_id.items():
        if paper_id in checkpoint_paper_ids:
            updates.append({
                "record_id": record_id,
                "status": "done",
                "checkpoint_path": str(checkpoint_dir / f"{paper_id}.json"),
                "n_models_extracted": int(n_models_by_paper.get(paper_id, 0)),
                "needs_review": bool(needs_review_by_paper.get(paper_id, False)),
            })
        elif paper_id in screened_paper_ids:
            # Triado mas excluido (decision=exclude) -- nao e uma falha, e uma
            # decisao valida da etapa 0. Ver should_proceed_to_extraction.
            updates.append({"record_id": record_id, "status": "not_applicable"})
        else:
            updates.append({"record_id": record_id, "status": "failed"})

    reg.mark_extraction_status(registry_path, updates)
    n_done = sum(1 for u in updates if u["status"] == "done")
    n_na = sum(1 for u in updates if u["status"] == "not_applicable")
    n_failed = sum(1 for u in updates if u["status"] == "failed")
    logger.info("Registry atualizado: %d done, %d not_applicable (excluidos na triagem), %d failed.", n_done, n_na, n_failed)


@cli.command()
@click.option("--out-dir", required=True, type=click.Path(file_okay=False), help="Diretorio com os checkpoints ja gerados.")
def consolidate(out_dir):
    """So reconsolida a planilha final a partir dos checkpoints existentes (sem reprocessar PDFs)."""
    out_dir = Path(out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoints = load_checkpoints(checkpoint_dir)
    if not checkpoints:
        logger.warning("Nenhum checkpoint encontrado em %s", checkpoint_dir)
        return
    xlsx_path, csv_path, audit_xlsx, audit_csv = build_final_outputs(checkpoints, out_dir)
    logger.info("Planilha final: %s / %s", xlsx_path, csv_path)
    logger.info("Amostra de auditoria: %s / %s", audit_xlsx, audit_csv)

    screening_dir = out_dir / "screening"
    screening_results = load_screening_results(screening_dir)
    if screening_results:
        screening_xlsx, screening_csv = export_screening_outputs(screening_results, out_dir)
        logger.info("Planilha de triagem PRISMA: %s / %s", screening_xlsx, screening_csv)


if __name__ == "__main__":
    cli()
