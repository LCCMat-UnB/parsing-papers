from unittest.mock import patch

from typer.testing import CliRunner

from parsing_papers import cli, doctor


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_vllm_respondendo_lists_models():
    with patch("requests.get", return_value=_Resp({"data": [{"id": "Qwen/Qwen2.5-32B-Instruct-AWQ"}]})):
        check, modelos = doctor._vllm_respondendo("http://gpu:8000/v1")
    assert check.ok
    assert modelos == ["Qwen/Qwen2.5-32B-Instruct-AWQ"]


def test_modelo_servido_strips_litellm_prefix():
    check = doctor._modelo_servido_vllm(
        ["Qwen/Qwen2.5-32B-Instruct-AWQ"], "openai/Qwen/Qwen2.5-32B-Instruct-AWQ"
    )
    assert check.ok


def test_modelo_servido_missing():
    check = doctor._modelo_servido_vllm(["outro/modelo"], "openai/Qwen/Qwen2.5-32B-Instruct-AWQ")
    assert not check.ok
    assert "outro/modelo" in check.detalhe


def test_doctor_profile_inexistente_sai_com_erro():
    """--profile com typo deve falhar (exit != 0) com a mensagem do load_profile,
    em vez de diagnosticar silenciosamente o perfil local."""
    result = CliRunner().invoke(cli.app, ["doctor", "--profile", "nao-existe"])
    assert result.exit_code != 0
    assert "nao-existe" in result.output


def test_doctor_profile_cluster_usa_api_base_e_modelo_do_perfil(monkeypatch):
    """--profile cluster despacha para diagnosticar_cluster com o api_base/model
    de config/profiles/cluster.json (sem rede: diagnosticar_cluster e stubado)."""
    from parsing_papers.profiles import load_profile

    chamadas = []

    def _stub_diagnosticar_cluster(api_base, modelo):
        chamadas.append((api_base, modelo))
        return doctor.DiagnosticoResultado(
            checagens=[doctor.Checagem("vLLM respondendo", True, "OK (stub).")]
        )

    monkeypatch.setattr(doctor, "diagnosticar_cluster", _stub_diagnosticar_cluster)

    result = CliRunner().invoke(cli.app, ["doctor", "--profile", "cluster"])

    assert result.exit_code == 0
    esperado = load_profile("cluster")
    assert chamadas == [(esperado.api_base, esperado.model)]
