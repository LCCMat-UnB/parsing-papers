"""
Diagnostico de pre-requisitos do pipeline (Docker, GROBID, Ollama, modelo).

Publico-alvo: quem esta configurando o parsing-papers pela primeira vez e
nao tem intimidade com Docker/GROBID/Ollama. Cada checagem roda em ordem de
dependencia (Docker antes de container, container antes de servico
respondendo, servico antes de modelo baixado) e para no primeiro problema
que bloqueia os seguintes -- reportar "GROBID nao responde" E "Ollama nao
responde" ao mesmo tempo quando na verdade o Docker nem esta rodando so
adiciona ruido.

Deliberadamente so DIAGNOSTICA -- nunca executa `docker compose up`, `ollama
pull`, etc. por conta propria. Subir containers ou baixar alguns GB de
modelo sao acoes com efeito colateral real (uso de disco/rede/GPU) que o
usuario deve escolher rodar, nao algo que um comando de "verificar" faz
silenciosamente.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field

from .grobid_client import GrobidClient

DEFAULT_GROBID_URL = "http://localhost:8070"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Containers esperados pelo docker-compose.yml deste projeto -- nomeados
# explicitamente (container_name:) para que este modulo os identifique sem
# ambiguidade mesmo se o usuario tiver outros containers Docker no ar.
CONTAINER_GROBID = "parsing_papers_grobid"
CONTAINER_OLLAMA = "parsing_papers_ollama"

# Modelo recomendado no README para GPUs de 12GB (ex: RTX 3060) -- usado so
# como sugestao no diagnostico, nao como exigencia.
MODELO_RECOMENDADO = "qwen2.5:14b-instruct"


@dataclass
class Checagem:
    nome: str
    ok: bool
    detalhe: str = ""
    correcao: str = ""  # comando/orientacao para resolver, se ok=False


@dataclass
class DiagnosticoResultado:
    checagens: list[Checagem] = field(default_factory=list)

    @property
    def tudo_ok(self) -> bool:
        return all(c.ok for c in self.checagens)

    @property
    def pronto_para_rodar(self) -> bool:
        """True se as checagens bloqueantes (Docker, containers, GROBID,
        Ollama) passaram -- a checagem de modelo baixado fica de fora daqui
        porque o pipeline ainda funciona sem ela (só falha depois, ao tentar
        chamar um modelo que não existe; preferimos deixar isso visível como
        aviso, não travar o menu por causa disso)."""
        bloqueantes = {"Docker", "Container GROBID", "Container Ollama", "GROBID respondendo", "Ollama respondendo"}
        return all(c.ok for c in self.checagens if c.nome in bloqueantes)


def _docker_disponivel() -> Checagem:
    if shutil.which("docker") is None:
        return Checagem(
            "Docker", False,
            "comando 'docker' não encontrado no PATH.",
            "Instale o Docker Desktop (https://docs.docker.com/desktop/) e certifique-se "
            "de que o WSL2 integration está habilitada, se estiver no Windows/WSL.",
        )
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=10, text=True)
        if r.returncode != 0:
            return Checagem(
                "Docker", False,
                "Docker instalado, mas o daemon não está respondendo (`docker info` falhou).",
                "Abra o Docker Desktop e aguarde ele inicializar completamente.",
            )
    except (subprocess.TimeoutExpired, OSError) as e:
        return Checagem("Docker", False, f"erro ao checar Docker: {e}", "Abra o Docker Desktop e tente de novo.")
    return Checagem("Docker", True, "instalado e respondendo.")


def _container_no_ar(nome_container: str, rotulo: str) -> Checagem:
    try:
        r = subprocess.run(
            ["docker", "ps", "--filter", f"name={nome_container}", "--format", "{{.Status}}"],
            capture_output=True, timeout=10, text=True,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return Checagem(rotulo, False, f"erro ao consultar containers: {e}", "docker compose up -d")

    status = r.stdout.strip()
    if not status:
        return Checagem(
            rotulo, False, "container não está rodando.",
            "docker compose up -d   (rode a partir da raiz do repo parsing-papers)",
        )
    if status.lower().startswith("restarting"):
        return Checagem(
            rotulo, False, f"container está reiniciando em loop ({status}) — provável erro de configuração.",
            f"docker logs {nome_container}   (veja o erro real antes de tentar de novo)",
        )
    return Checagem(rotulo, True, f"rodando ({status}).")


def _grobid_respondendo(grobid_url: str) -> Checagem:
    client = GrobidClient(base_url=grobid_url)
    if client.is_alive():
        return Checagem("GROBID respondendo", True, f"OK em {grobid_url}.")
    return Checagem(
        "GROBID respondendo", False,
        f"sem resposta em {grobid_url}/api/isalive.",
        "O GROBID pode levar 1-3 minutos para carregar os modelos após o container subir "
        "— espere um pouco e rode o diagnóstico de novo. Se persistir, confira "
        f"`docker logs {CONTAINER_GROBID}`.",
    )


def _ollama_respondendo(ollama_url: str) -> tuple[Checagem, list[str]]:
    """Retorna (checagem, lista_de_modelos_baixados)."""
    import requests

    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=10)
        r.raise_for_status()
        modelos = [m["name"] for m in r.json().get("models", [])]
        return Checagem("Ollama respondendo", True, f"OK em {ollama_url}."), modelos
    except requests.RequestException as e:
        return (
            Checagem(
                "Ollama respondendo", False,
                f"sem resposta em {ollama_url}: {e}",
                f"Confira `docker logs {CONTAINER_OLLAMA}` e se a porta 11434 está livre.",
            ),
            [],
        )


def _modelo_disponivel(modelos_baixados: list[str], modelo_desejado: str) -> Checagem:
    if not modelos_baixados:
        return Checagem(
            "Modelo LLM baixado", False,
            "nenhum modelo encontrado no Ollama.",
            f"docker exec -it {CONTAINER_OLLAMA} ollama pull {modelo_desejado}   "
            "(~9GB em Q4, cabe em GPUs de 12GB; veja o README para alternativas menores/maiores)",
        )
    # match exato ou por prefixo (usuario pode ter so "qwen2.5:14b-instruct"
    # sem o sufixo de quantizacao que o Ollama as vezes anexa no nome listado)
    if any(modelo_desejado in m or m in modelo_desejado for m in modelos_baixados):
        return Checagem("Modelo LLM baixado", True, f"'{modelo_desejado}' disponível.")
    return Checagem(
        "Modelo LLM baixado", False,
        f"'{modelo_desejado}' não está entre os baixados: {', '.join(modelos_baixados)}.",
        f"docker exec -it {CONTAINER_OLLAMA} ollama pull {modelo_desejado}\n"
        f"   (ou rode o pipeline apontando --model para um dos já baixados acima)",
    )


def diagnosticar(
    grobid_url: str = DEFAULT_GROBID_URL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    modelo_desejado: str = MODELO_RECOMENDADO,
) -> DiagnosticoResultado:
    """Roda todas as checagens em ordem, parando checagens dependentes assim
    que uma bloqueante falha (ex: sem Docker, não faz sentido checar
    containers)."""
    resultado = DiagnosticoResultado()

    docker_ok = _docker_disponivel()
    resultado.checagens.append(docker_ok)
    if not docker_ok.ok:
        return resultado

    grobid_container = _container_no_ar(CONTAINER_GROBID, "Container GROBID")
    ollama_container = _container_no_ar(CONTAINER_OLLAMA, "Container Ollama")
    resultado.checagens.append(grobid_container)
    resultado.checagens.append(ollama_container)

    if grobid_container.ok:
        resultado.checagens.append(_grobid_respondendo(grobid_url))
    if ollama_container.ok:
        ollama_check, modelos = _ollama_respondendo(ollama_url)
        resultado.checagens.append(ollama_check)
        if ollama_check.ok:
            resultado.checagens.append(_modelo_disponivel(modelos, modelo_desejado))

    return resultado
