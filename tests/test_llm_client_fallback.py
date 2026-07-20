from unittest.mock import patch

import pytest

from parsing_papers.llm_client import LLMExtractor, _is_schema_unsupported_error


def test_schema_unsupported_detection():
    assert _is_schema_unsupported_error(Exception("Unsupported parameter: response_format json_schema"))
    assert _is_schema_unsupported_error(Exception("Invalid JSON schema provided"))
    assert not _is_schema_unsupported_error(Exception("Request timed out after 900s"))
    assert not _is_schema_unsupported_error(Exception("Connection refused"))


def _extractor():
    return LLMExtractor(model="ollama_chat/fake", api_base="http://localhost:1", min_num_ctx=100)


def test_timeout_is_not_masked_by_json_object_fallback():
    """Erro de transporte NAO pode cair no fallback json_object (que esconderia
    a causa real e queimaria uma chamada extra); deve propagar para o retry."""
    ext = _extractor()
    with patch("litellm.completion", side_effect=TimeoutError("Request timed out")):
        with pytest.raises(TimeoutError):
            ext.complete([{"role": "user", "content": "oi"}])


def test_schema_error_falls_back_to_json_object():
    ext = _extractor()
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs.get("response_format"))
        if kwargs["response_format"].get("type") == "json_schema":
            raise Exception("Unsupported parameter: response_format json_schema")
        return {"choices": [{"message": {"content": '{"paper_id": "p", "records": []}'}}]}

    with patch("litellm.completion", side_effect=fake_completion):
        out = ext.complete([{"role": "user", "content": "oi"}])
    assert '"records"' in out
    assert [c["type"] for c in calls] == ["json_schema", "json_object"]
