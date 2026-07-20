from unittest.mock import patch

from parsing_papers import doctor


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
