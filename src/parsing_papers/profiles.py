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

def _find_project_root(start_dir: Path) -> Path | None:
    """Sobe a partir de `start_dir` ate achar uma pasta com config/profiles ou
    pyproject.toml -- caso de quem clonou o repo (dev, ou `pip install -e .`).
    Retorna None se nao achar: caso de instalacao nao-editavel (`pip install
    git+...`), onde so existe o pacote Python em site-packages, sem repo por
    perto -- o config/profiles fica a cargo de `parsing-papers init` no cwd
    do usuario (ver init_cmd.py e cli.py:_perguntar_parametros_llm)."""
    for parent in [start_dir, *start_dir.parents]:
        if (parent / "config" / "profiles").is_dir() or (parent / "pyproject.toml").is_file():
            return parent
    return None


# Resolvido a partir da localizacao deste arquivo (nao do cwd de quem chama),
# assim `parsing-papers` funciona rodando de qualquer diretorio quando o repo
# foi clonado. Antes era Path("config/profiles"), relativo ao cwd -- quebrava
# se voce rodasse o comando fora da raiz do projeto.
#
# Numa instalacao via `pip install git+...` (nao-editavel) nao ha repo por
# perto do pacote instalado em site-packages -- nesse caso caimos de volta
# para Path("config/profiles") relativo ao CWD, que e onde `parsing-papers
# init` os cria (ver init_cmd.py). load_profile() abaixo tambem detecta esse
# caso para dar uma mensagem de erro acionavel, nao so "nao encontrado".
_project_root = _find_project_root(Path(__file__).resolve().parent)
DEFAULT_PROFILES_DIR = (_project_root / "config" / "profiles") if _project_root else Path("config/profiles")
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
        dica = (
            " -- rode `parsing-papers init` para criar config/profiles/ no diretorio atual "
            "(comum em instalacoes via `pip install git+...`)."
            if not available else ""
        )
        raise FileNotFoundError(
            f"Perfil '{name}' nao encontrado ({path}). Disponiveis: {available or 'nenhum'}{dica}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    known = set(Profile.__dataclass_fields__) - {"name"}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Campos desconhecidos no perfil '{name}': {sorted(unknown)}")
    return Profile(name=name, **data)
