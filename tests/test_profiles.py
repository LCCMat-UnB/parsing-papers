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
