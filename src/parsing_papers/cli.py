"""parsing_papers/cli.py — menu interativo (Typer), para quem não tem
intimidade com a linha de comando de flags do pipeline.py.

Dois jeitos de usar:
  * Sem argumentos  -> menu interativo colorido (`parsing-papers`).
  * `python -m parsing_papers.pipeline <comando> --flags...` continua
    funcionando exatamente como antes — este módulo não substitui aquele,
    só oferece um caminho de entrada mais amigável por cima das mesmas
    funções (process_pdf_directory, registry_run helpers, build_final_outputs
    etc. em pipeline.py e registry.py).

Esta é a única camada que "conhece" o usuário (perguntas, cores). A lógica
de processamento vive em pipeline.py/registry.py/doctor.py.
"""

from __future__ import annotations

from pathlib import Path

import typer

from . import doctor, ui
from .pipeline import (
    build_final_outputs,
    export_screening_outputs,
    load_checkpoints,
    load_screening_results,
    process_pdf_directory,
)
from . import registry as reg
from .consolidate import export

app = typer.Typer(
    name="parsing-papers",
    help="Extração de métricas de papers científicos (GROBID + LLM com verificação).",
    rich_markup_mode="rich",
    no_args_is_help=False,
    add_completion=False,
)


# ============================================================ subcomandos ===

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


@app.callback(invoke_without_command=True)
def principal(ctx: typer.Context):
    """Sem subcomando: abre o menu interativo."""
    if ctx.invoked_subcommand is None:
        menu()


# ============================================================ menu loop =====

OPCOES_MENU = [
    "Verificar ambiente (Docker / GROBID / Ollama / modelo)",
    "Rodar sobre uma pasta de PDFs",
    "Rodar via registry compartilhado (integração com SPE/pontodoi)",
    "Reconsolidar planilha (sem reprocessar PDFs)",
    "Ajuda",
]


def menu():
    ui.top()
    ui.banner()

    while True:
        ui.console.print()
        ui.secao("Menu")
        for i, rotulo in enumerate(OPCOES_MENU, 1):
            ui.console.print(f"  [aviso]{i}[/]  {rotulo}")
        ui.console.print("  [aviso]0[/]  Sair")

        escolha = ui.escolher_opcao("Escolha uma opção", 0, len(OPCOES_MENU))
        if escolha == 0:
            ui.console.print("[suave]até a próxima![/]")
            return

        try:
            _despachar(escolha)
        except KeyboardInterrupt:
            ui.aviso("interrompido — voltando ao menu.")
        except Exception as e:  # noqa: BLE001 — menu não pode morrer por um erro
            ui.erro(f"algo deu errado: {e}")


def _despachar(escolha: int):
    if escolha == 1:
        _fluxo_diagnostico_interativo()
    elif escolha == 2:
        _fluxo_rodar_pdf_dir()
    elif escolha == 3:
        _fluxo_rodar_registry()
    elif escolha == 4:
        _fluxo_reconsolidar()
    elif escolha == 5:
        _ajuda()


# ============================================================ fluxos ========

def _rodar_diagnostico(grobid_url: str, ollama_url: str, modelo: str) -> bool:
    """Roda o diagnóstico e imprime o resultado. Retorna True se está tudo
    pronto para rodar o pipeline (ver DiagnosticoResultado.pronto_para_rodar)."""
    ui.secao("Verificando ambiente")
    with ui.console.status("[primaria]checando Docker, GROBID, Ollama...[/]", spinner="dots"):
        resultado = doctor.diagnosticar(grobid_url, ollama_url, modelo)
    ui.tabela_diagnostico(resultado.checagens)

    if resultado.tudo_ok:
        ui.console.print()
        ui.ok("Tudo pronto — pode rodar o pipeline.")
    elif resultado.pronto_para_rodar:
        ui.console.print()
        ui.aviso("Serviços no ar, mas o modelo desejado ainda não foi baixado — veja acima.")
    else:
        ui.console.print()
        ui.erro("Ambiente ainda não está pronto — resolva os itens marcados com ✗ acima.")

    return resultado.pronto_para_rodar


def _fluxo_diagnostico_interativo():
    _rodar_diagnostico(doctor.DEFAULT_GROBID_URL, doctor.DEFAULT_OLLAMA_URL, doctor.MODELO_RECOMENDADO)
    ui.console.print()
    input("Pressione Enter para voltar ao menu...")


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


def _fluxo_rodar_pdf_dir():
    ui.secao("Rodar sobre uma pasta de PDFs")

    if not _rodar_diagnostico(doctor.DEFAULT_GROBID_URL, doctor.DEFAULT_OLLAMA_URL, doctor.MODELO_RECOMENDADO):
        ui.dica("Resolva os itens acima antes de continuar (ou tente mesmo assim, por sua conta).")
        if not ui.confirmar("Continuar mesmo assim?", padrao=False):
            return

    pdf_dir_str = ui.perguntar("Pasta com os PDFs", "data/pdfs")
    pdf_dir = Path(pdf_dir_str).expanduser()
    if not pdf_dir.exists():
        ui.erro(f"pasta não encontrada: {pdf_dir}")
        return
    n_pdfs = len(list(pdf_dir.glob("*.pdf")))
    if n_pdfs == 0:
        ui.erro(f"nenhum PDF encontrado em {pdf_dir}")
        return
    ui.info(f"{n_pdfs} PDF(s) encontrado(s) em {pdf_dir}")

    out_dir_str = ui.perguntar("Pasta de saída (planilhas + checkpoints)", "data/extracted")
    out_dir = Path(out_dir_str).expanduser()

    params = _perguntar_parametros_llm()

    ui.console.print()
    ui.secao("Processando")
    process_pdf_directory(pdf_dir, out_dir, params["grobid_url"], params["citation_threshold"],
                          params["grobid_wait_s"], params["skip_screening"], params["force"], params["profile"])

    checkpoint_dir = out_dir / "checkpoints"
    screening_dir = out_dir / "screening"
    checkpoints = load_checkpoints(checkpoint_dir)
    if not checkpoints:
        ui.aviso("Nenhum checkpoint gerado (todos os papers podem ter sido excluídos na triagem, ou houve erro).")
        return

    xlsx_path, csv_path, audit_xlsx, audit_csv = build_final_outputs(checkpoints, out_dir)
    _imprimir_resumo(out_dir, checkpoints, screening_dir, params["skip_screening"], xlsx_path, audit_xlsx)

    input("\nPressione Enter para voltar ao menu...")


def _fluxo_rodar_registry():
    ui.secao("Rodar via registry compartilhado")
    ui.dica("Requer um registry.jsonl já exportado pelo SPE e com PDFs baixados pelo pontodoi.")

    if not _rodar_diagnostico(doctor.DEFAULT_GROBID_URL, doctor.DEFAULT_OLLAMA_URL, doctor.MODELO_RECOMENDADO):
        ui.dica("Resolva os itens acima antes de continuar (ou tente mesmo assim, por sua conta).")
        if not ui.confirmar("Continuar mesmo assim?", padrao=False):
            return

    registry_str = ui.perguntar("Caminho do registry.jsonl")
    if not registry_str:
        ui.erro("nenhum caminho informado.")
        return
    registry_path = Path(registry_str.strip().strip('"')).expanduser()
    if not registry_path.exists():
        ui.erro(f"registry não encontrado: {registry_path}")
        return

    pending = reg.pending_for_extraction(registry_path)
    if not pending:
        ui.aviso("Nada pendente no registry (fulltext_status=done + extraction_status pendente/falho).")
        return
    ui.info(f"{len(pending)} paper(s) pendente(s) de extração no registry.")

    out_dir_str = ui.perguntar("Pasta de saída (planilhas + checkpoints)", "data/extracted")
    out_dir = Path(out_dir_str).expanduser()
    workspace_root = registry_path.parent
    staging_dir = out_dir / "registry_pdfs"

    params = _perguntar_parametros_llm()

    paper_id_to_record_id = reg.stage_pdfs(pending, workspace_root, staging_dir)
    if not paper_id_to_record_id:
        ui.erro("Nenhum PDF pôde ser localizado a partir dos registros pendentes.")
        return

    ui.console.print()
    ui.secao("Processando")
    process_pdf_directory(staging_dir, out_dir, params["grobid_url"], params["citation_threshold"],
                          params["grobid_wait_s"], params["skip_screening"], params["force"], params["profile"])

    checkpoint_dir = out_dir / "checkpoints"
    screening_dir = out_dir / "screening"
    checkpoints = load_checkpoints(checkpoint_dir)
    if not checkpoints:
        ui.aviso("Nenhum checkpoint gerado nesta rodada.")
        return

    xlsx_path, csv_path, audit_xlsx, audit_csv = build_final_outputs(checkpoints, out_dir)

    import pandas as pd
    df = pd.read_csv(csv_path)
    df = reg.merge_bibliographic_columns(df, registry_path, paper_id_to_record_id)
    xlsx_path, csv_path = export(df, out_dir, basename="metadados_extraidos")

    _imprimir_resumo(out_dir, checkpoints, screening_dir, params["skip_screening"], xlsx_path, audit_xlsx)

    # Atualiza extraction_status de volta no registry — mesma lógica de
    # pipeline.py:registry_run, reaproveitada aqui para o menu.
    checkpoint_paper_ids = {c["paper_id"] for c in checkpoints}
    screening_results = load_screening_results(screening_dir) if not params["skip_screening"] else []
    screened_paper_ids = {r["paper_id"] for r in screening_results}
    needs_review_by_paper = df.groupby("paper_id")["needs_review"].any().to_dict() if not df.empty else {}
    n_models_by_paper = df.groupby("paper_id").size().to_dict() if not df.empty else {}

    updates = []
    for paper_id, record_id in paper_id_to_record_id.items():
        if paper_id in checkpoint_paper_ids:
            updates.append({
                "record_id": record_id, "status": "done",
                "checkpoint_path": str(checkpoint_dir / f"{paper_id}.json"),
                "n_models_extracted": int(n_models_by_paper.get(paper_id, 0)),
                "needs_review": bool(needs_review_by_paper.get(paper_id, False)),
            })
        elif paper_id in screened_paper_ids:
            updates.append({"record_id": record_id, "status": "not_applicable"})
        else:
            updates.append({"record_id": record_id, "status": "failed"})
    reg.mark_extraction_status(registry_path, updates)
    ui.ok(f"Registry atualizado: {sum(1 for u in updates if u['status'] == 'done')} paper(s) marcados extraction_status=done.")

    input("\nPressione Enter para voltar ao menu...")


def _fluxo_reconsolidar():
    ui.secao("Reconsolidar planilha")
    ui.dica("Reconstrói a planilha final a partir de checkpoints já existentes, sem reprocessar PDFs.")
    out_dir_str = ui.perguntar("Pasta de saída (onde estão os checkpoints)", "data/extracted")
    out_dir = Path(out_dir_str).expanduser()
    checkpoint_dir = out_dir / "checkpoints"
    checkpoints = load_checkpoints(checkpoint_dir)
    if not checkpoints:
        ui.erro(f"nenhum checkpoint encontrado em {checkpoint_dir}")
        return
    xlsx_path, csv_path, audit_xlsx, audit_csv = build_final_outputs(checkpoints, out_dir)
    screening_dir = out_dir / "screening"
    _imprimir_resumo(out_dir, checkpoints, screening_dir, skip_screening=False, xlsx_path=xlsx_path, audit_xlsx=audit_xlsx)
    input("\nPressione Enter para voltar ao menu...")


def _imprimir_resumo(out_dir: Path, checkpoints: list[dict], screening_dir: Path, skip_screening: bool, xlsx_path, audit_xlsx):
    n_excluidos = 0
    if not skip_screening:
        screening_results = load_screening_results(screening_dir)
        if screening_results:
            export_screening_outputs(screening_results, out_dir)
            processados_ids = {c["paper_id"] for c in checkpoints}
            n_excluidos = sum(1 for r in screening_results if r["paper_id"] not in processados_ids)

    import pandas as pd
    try:
        df = pd.read_csv(out_dir / "metadados_extraidos.csv")
        n_needs_review = int(df["needs_review"].astype(bool).sum()) if "needs_review" in df.columns else 0
    except (FileNotFoundError, pd.errors.EmptyDataError):
        n_needs_review = 0

    ui.resumo_rodada(
        n_processados=len(checkpoints),
        n_excluidos_triagem=n_excluidos,
        n_needs_review=n_needs_review,
        n_erro=0,
        xlsx_path=str(xlsx_path),
        audit_path=str(audit_xlsx),
    )


def _ajuda():
    ui.secao("Ajuda")
    ui.console.print("[primaria]parsing-papers[/] extrai métricas de desempenho de papers científicos")
    ui.console.print("(modelos de ML, AUC, Accuracy etc.) para uma meta-análise, com verificação")
    ui.console.print("automática de citações e dupla extração para reduzir alucinação do LLM.\n")
    ui.secao("Fluxo típico")
    ui.console.print("  1) Verificar ambiente — confirma que Docker/GROBID/Ollama/modelo estão prontos.")
    ui.console.print("  2) Colocar os PDFs em uma pasta (ex: data/pdfs/) — ou usar a opção 3 se estiver")
    ui.console.print("     integrado com o SPE/pontodoi (registry compartilhado).")
    ui.console.print("  3) Rodar — a triagem PRISMA roda primeiro (decide inclusão/exclusão), e só os")
    ui.console.print("     papers elegíveis passam para a extração de métricas.")
    ui.console.print("  4) Conferir a planilha final e a amostra de auditoria (~18% dos registros,")
    ui.console.print("     priorizando os marcados 'needs_review').\n")
    ui.secao("O que significa 'needs_review'")
    ui.console.print("  Um registro é marcado assim quando alguma verificação automática pede uma")
    ui.console.print("  checagem humana: citação da fonte não confirmada por fuzzy match, valor fora")
    ui.console.print("  da faixa esperada (ex: AUC > 1), ou divergência entre as duas extrações (o")
    ui.console.print("  pipeline extrai cada paper duas vezes e compara — espírito PRISMA item 9).")
    ui.console.print("  Recomendação: não aceitar esses registros na meta-análise sem revisão manual.\n")
    ui.secao("Pela linha de comando (uso avançado)")
    ui.console.print("  [suave]python -m parsing_papers.pipeline run --pdf-dir data/pdfs --out-dir data/extracted[/]")
    ui.console.print("  [suave]python -m parsing_papers.pipeline registry-run --registry .../registry.jsonl --out-dir data/extracted[/]")
    ui.console.print("  [suave]python -m parsing_papers.pipeline consolidate --out-dir data/extracted[/]")
    ui.console.print("  [suave]parsing-papers doctor[/]  — roda só o diagnóstico, fora do menu.")


if __name__ == "__main__":
    app()
