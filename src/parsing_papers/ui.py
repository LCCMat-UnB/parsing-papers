"""parsing_papers/ui.py — camada de apresentação (cores, banner, tabelas, prompts).

Tudo que é "como aparece na tela" mora aqui, para o resto do código não
precisar saber de Rich diretamente. Mesmo padrão/paleta usado no pontodoi
(papers/ui.py) para manter a mesma linguagem visual entre os CLIs da
pipeline (SPE + pontodoi + parsing-papers) — um usuário que já usou um dos
três reconhece as cores e o estilo dos outros dois.

Paleta:
  primaria  -> ciano  (títulos, destaques de ação)
  ok        -> verde  (sucesso, checagem passou)
  aviso     -> amarelo (atenção, não bloqueante)
  erro      -> vermelho (falha, checagem bloqueante)
  info      -> azul   (metadados, dicas)
  suave     -> cinza  (texto secundário)
"""

from __future__ import annotations

from typing import Iterable

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

TEMA = Theme(
    {
        "primaria": "bold cyan",
        "ok": "bold green",
        "aviso": "bold yellow",
        "erro": "bold red",
        "info": "blue",
        "suave": "grey62",
    }
)

console = Console(theme=TEMA)

BANNER = r"""
 ██████╗  █████╗ ██████╗ ███████╗██╗███╗   ██╗ ██████╗
 ██╔══██╗██╔══██╗██╔══██╗██╔════╝██║████╗  ██║██╔════╝
 ██████╔╝███████║██████╔╝███████╗██║██╔██╗ ██║██║  ███╗
 ██╔═══╝ ██╔══██║██╔══██╗╚════██║██║██║╚██╗██║██║   ██║
 ██║     ██║  ██║██║  ██║███████║██║██║ ╚████║╚██████╔╝
 ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝
"""


def banner() -> None:
    console.print(Text(BANNER.strip("\n"), style="primaria"), highlight=False)
    console.print(
        Text("extração de métricas de papers científicos · GROBID + LLM com verificação", style="suave"),
        highlight=False,
    )


def top() -> None:
    console.rule("", style="cyan")


def secao(texto: str) -> None:
    console.rule(f"[primaria]{texto}[/]", style="cyan")


def ok(msg: str) -> None:
    console.print(f"[ok]✓[/] {msg}")


def aviso(msg: str) -> None:
    console.print(f"[aviso]![/] {msg}")


def erro(msg: str) -> None:
    console.print(f"[erro]✗[/] {msg}")


def info(msg: str) -> None:
    console.print(f"[info]i[/] {msg}")


def dica(msg: str) -> None:
    console.print(f"  [suave]{msg}[/]")


# ---------------------------------------------------------------- prompts ---

def perguntar(msg: str, padrao: str | None = None) -> str:
    return Prompt.ask(
        f"[primaria]{msg}[/]", default=padrao or "", show_default=bool(padrao), console=console,
    ).strip()


def confirmar(msg: str, padrao: bool = True) -> bool:
    return Confirm.ask(f"[primaria]{msg}[/]", default=padrao, console=console)


def escolher_opcao(msg: str, minimo: int, maximo: int) -> int:
    while True:
        resp = Prompt.ask(f"[primaria]{msg}[/]", default="", show_default=False, console=console).strip()
        if resp.isdigit() and minimo <= int(resp) <= maximo:
            return int(resp)
        erro(f"digite um número entre {minimo} e {maximo}.")


# ------------------------------------------------------------- diagnostico ---

def tabela_diagnostico(checagens: Iterable) -> None:
    """Renderiza a lista de Checagem (doctor.py) como uma tabela de status,
    com o comando de correção logo abaixo de cada linha que falhou — para o
    usuário não precisar caçar em outro lugar o que rodar."""
    tab = Table(border_style="cyan", header_style="primaria", show_lines=False)
    tab.add_column("", width=2)
    tab.add_column("Checagem", style="white")
    tab.add_column("Detalhe", style="suave")

    for c in checagens:
        icone = "[ok]✓[/]" if c.ok else "[erro]✗[/]"
        tab.add_row(icone, c.nome, c.detalhe)
    console.print(tab)

    faltando = [c for c in checagens if not c.ok and c.correcao]
    if faltando:
        console.print()
        secao("Como resolver")
        for c in faltando:
            console.print(f"  [aviso]{c.nome}[/]:")
            for linha in c.correcao.splitlines():
                console.print(f"    [suave]{linha}[/]")


# ------------------------------------------------------------------ resumo ---

def resumo_rodada(
    n_processados: int,
    n_excluidos_triagem: int,
    n_needs_review: int,
    n_erro: int,
    xlsx_path: str,
    audit_path: str,
) -> None:
    """Resumo em linguagem simples do que aconteceu numa rodada do pipeline —
    pensado para quem não conhece os termos técnicos das colunas de QA
    (needs_review, dual_extraction_diverges etc.) saber o que fazer a seguir
    sem precisar ler o README."""
    secao("Resumo da rodada")
    console.print(f"[ok]✓[/] {n_processados} paper(s) tiveram métricas extraídas com sucesso.")
    if n_excluidos_triagem:
        console.print(
            f"[info]i[/] {n_excluidos_triagem} paper(s) foram excluídos na triagem PRISMA "
            "(não atendiam aos critérios de inclusão) — não geraram métricas, mas ficaram "
            "registrados na planilha de triagem para auditoria."
        )
    if n_needs_review:
        console.print(
            f"[aviso]![/] {n_needs_review} registro(s) foram marcados com [aviso]needs_review=True[/] — "
            "alguma verificação automática (citação não confirmada, valor fora da faixa esperada, "
            "ou divergência entre as duas extrações) pede uma checagem humana antes de usar o dado "
            "na meta-análise. Eles aparecem priorizados na planilha de auditoria."
        )
    if n_erro:
        console.print(f"[erro]✗[/] {n_erro} paper(s) tiveram erro durante o processamento — veja os logs acima.")

    console.print()
    console.print(f"  [suave]Planilha final:[/]     {xlsx_path}")
    console.print(f"  [suave]Amostra p/ revisão:[/] {audit_path}")
