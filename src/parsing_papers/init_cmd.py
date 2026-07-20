"""
`parsing-papers init` -- materializa no diretorio atual os arquivos que
`pip install git+https://...` NAO traz consigo.

O pip (inclusive via git+) so instala o pacote Python (src/parsing_papers/);
o resto do repositorio clonado (docker-compose.yml, config/profiles/*.json)
e descartado apos o build. Sem este comando, quem instala via pip nao tem
como montar os containers Docker (GROBID/Ollama/vLLM) nem escolher um perfil
de deployment -- ambos vivem em arquivos fora do pacote Python.

Os arquivos-fonte destes templates vivem em resources/ (dentro do pacote,
acessivel via importlib.resources em qualquer forma de instalacao) e sao
copiados 1:1 -- nao ha geracao dinamica de conteudo aqui.
"""

from __future__ import annotations

import importlib.resources as resources
from pathlib import Path

# (destino relativo ao cwd do usuario, origem dentro de resources/)
_ARQUIVOS = [
    ("docker-compose.yml", "docker-compose.yml"),
    ("config/profiles/local.json", "config_profiles/local.json"),
    ("config/profiles/cluster.json", "config_profiles/cluster.json"),
]


def arquivos_faltando(destino: Path) -> list[str]:
    """Lista (paths relativos) dos arquivos de `_ARQUIVOS` que ainda nao
    existem em `destino`. Usado pelo doctor/menu para sugerir `init` sem
    forcar o usuario a rodar o comando as cegas."""
    return [rel for rel, _ in _ARQUIVOS if not (destino / rel).exists()]


def inicializar(destino: Path, force: bool = False) -> tuple[list[str], list[str]]:
    """Copia docker-compose.yml e config/profiles/*.json para `destino`.

    Retorna (escritos, pulados). Nao sobrescreve arquivos existentes a menos
    que force=True -- evita apagar um docker-compose.yml que o usuario ja
    customizou (ex: trocou o modelo do vLLM, ajustou memoria do GROBID).
    """
    escritos: list[str] = []
    pulados: list[str] = []
    pkg = resources.files("parsing_papers.resources")

    for rel_destino, rel_origem in _ARQUIVOS:
        alvo = destino / rel_destino
        if alvo.exists() and not force:
            pulados.append(rel_destino)
            continue
        alvo.parent.mkdir(parents=True, exist_ok=True)
        conteudo = (pkg / rel_origem).read_bytes()
        alvo.write_bytes(conteudo)
        escritos.append(rel_destino)

    return escritos, pulados
