from pathlib import Path

from parsing_papers.init_cmd import arquivos_faltando, inicializar


def test_inicializar_cria_todos_arquivos_em_diretorio_vazio(tmp_path: Path):
    assert arquivos_faltando(tmp_path) == [
        "docker-compose.yml",
        "config/profiles/local.json",
        "config/profiles/cluster.json",
    ]

    escritos, pulados = inicializar(tmp_path)

    assert set(escritos) == {"docker-compose.yml", "config/profiles/local.json", "config/profiles/cluster.json"}
    assert pulados == []
    assert (tmp_path / "docker-compose.yml").exists()
    assert (tmp_path / "config" / "profiles" / "local.json").exists()
    assert (tmp_path / "config" / "profiles" / "cluster.json").exists()
    assert arquivos_faltando(tmp_path) == []


def test_inicializar_nao_sobrescreve_sem_force(tmp_path: Path):
    inicializar(tmp_path)
    (tmp_path / "docker-compose.yml").write_text("customizado pelo usuario", encoding="utf-8")

    escritos, pulados = inicializar(tmp_path)

    assert "docker-compose.yml" in pulados
    assert "docker-compose.yml" not in escritos
    assert (tmp_path / "docker-compose.yml").read_text(encoding="utf-8") == "customizado pelo usuario"


def test_inicializar_sobrescreve_com_force(tmp_path: Path):
    inicializar(tmp_path)
    (tmp_path / "docker-compose.yml").write_text("customizado pelo usuario", encoding="utf-8")

    escritos, pulados = inicializar(tmp_path, force=True)

    assert "docker-compose.yml" in escritos
    assert pulados == []
    assert (tmp_path / "docker-compose.yml").read_text(encoding="utf-8") != "customizado pelo usuario"


def test_conteudo_gerado_bate_com_perfis_carregaveis(tmp_path: Path):
    """O local.json/cluster.json gerados por init devem ser validos para
    profiles.load_profile -- pega drift entre resources/ e o schema de Profile."""
    from parsing_papers.profiles import load_profile

    inicializar(tmp_path)
    profile = load_profile("local", profiles_dir=tmp_path / "config" / "profiles")
    assert profile.model.startswith("ollama_chat/")

    profile_cluster = load_profile("cluster", profiles_dir=tmp_path / "config" / "profiles")
    assert profile_cluster.model.startswith("openai/")
