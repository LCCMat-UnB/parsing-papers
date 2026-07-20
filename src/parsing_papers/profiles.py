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
