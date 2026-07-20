"""
Etapa 2 do pipeline: extracao com schema forcado via LLM.

Usa LiteLLM como camada de abstracao: por padrao aponta para um modelo local
via Ollama (ex: "ollama_chat/llama3.1:70b" ou "ollama_chat/qwen2.5:32b"), mas
o mesmo codigo funciona com qualquer provedor suportado pelo LiteLLM (basta
trocar o `model` no config, ex: "groq/llama-3.1-70b-versatile",
"together_ai/...", "openai/gpt-4o") sem mudar uma linha deste modulo.

IMPORTANTE -- use o prefixo "ollama_chat/", NAO "ollama/": o provider
"ollama/" do LiteLLM usa o endpoint legado /api/generate, que achata
system+user prompt numa unica string de texto (perdendo a separacao formal
entre instrucao e conteudo). O provider "ollama_chat/" usa /api/chat, a API
atual do Ollama, com melhor aderencia a instrucoes -- e a propria doc do
LiteLLM recomenda isso explicitamente ("We recommend using ollama_chat for
better responses").

Structured output e forcado via `response_format` (json_schema), com fallback
para "modo JSON solto + validacao Pydantic" quando o provedor/modelo local nao
suportar json_schema estrito (comum em alguns modelos via Ollama).
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from .dedup import deduplicate_records
from .prompts import build_messages
from .repair import repair_paper_extraction
from .schema import PaperExtraction

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    pass


class LLMExtractor:
    """
    Wrapper fino sobre litellm.completion.

    Parameters
    ----------
    model : str
        Identificador LiteLLM do modelo, ex: "ollama_chat/qwen2.5:14b-instruct",
        "ollama_chat/llama3.1:70b", "groq/llama-3.1-70b-versatile". Para Ollama
        local, use sempre o prefixo "ollama_chat/" (nao "ollama/") -- ver nota
        no docstring do modulo.
    api_base : str | None
        Endpoint do servidor (para Ollama local, default "http://localhost:11434").
    temperature : float
        Temperatura de amostragem. Usada para diversificar a dupla extracao (etapa 4).
    """

    def __init__(
        self,
        model: str = "ollama_chat/qwen2.5:14b-instruct",
        api_base: str | None = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 8000,
        request_timeout: int = 600,
        min_num_ctx: int = 16000,
    ):
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        # Piso minimo de contexto (em tokens) a pedir ao Ollama. O Ollama, por
        # padrao, roda com num_ctx=2048 INDEPENDENTE do que o modelo suporta
        # -- ele trunca o prompt em silencio para caber nesse limite, sem erro
        # nem aviso. Um paper cientifico facilmente gera prompts de 6000-10000+
        # tokens (texto completo + schema JSON); sem fixar num_ctx, o LLM
        # simplesmente nunca ve a maior parte do texto (caso real observado:
        # prompt de ~7400 tokens truncado para 2050 pelo Ollama, fazendo o
        # modelo "nao ver" a tabela com os dados e retornar records=[] mesmo
        # dizendo ter tido sucesso).
        self.min_num_ctx = min_num_ctx

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
    def _call(self, messages: list[dict]) -> str:
        import litellm

        # estimativa grosseira de tokens do prompt (~4 chars/token em ingles/
        # portugues); usada so para calcular um num_ctx seguro, nao para
        # cobranca/limite real.
        prompt_chars = sum(len(m["content"]) for m in messages)
        estimated_prompt_tokens = prompt_chars // 3  # margem generosa (PT/EN acentuado usa mais bytes/token)
        num_ctx = max(self.min_num_ctx, estimated_prompt_tokens + self.max_tokens + 1024)

        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.request_timeout,
        )
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.model.startswith("ollama/") or self.model.startswith("ollama_chat/"):
            # litellm repassa kwargs extras nao reconhecidos como "options" do
            # Ollama; num_ctx e a chave que controla o tamanho da janela de
            # contexto carregada para essa chamada.
            kwargs["num_ctx"] = num_ctx
            logger.debug("Ollama num_ctx=%d (prompt estimado em ~%d tokens)", num_ctx, estimated_prompt_tokens)

        # Tenta forcar JSON schema estrito; se o provedor nao suportar,
        # litellm ignora ou levanta -- cai no fallback abaixo.
        try:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "paper_extraction",
                    "schema": PaperExtraction.model_json_schema(),
                    "strict": True,
                },
            }
            resp = litellm.completion(**kwargs)
        except Exception as e:  # noqa: BLE001
            logger.warning("json_schema estrito falhou (%s); tentando response_format=json_object", e)
            kwargs["response_format"] = {"type": "json_object"}
            resp = litellm.completion(**kwargs)

        return resp["choices"][0]["message"]["content"]

    def extract(self, paper_id: str, source_text: str) -> PaperExtraction:
        messages = build_messages(paper_id=paper_id, source_text=source_text)
        raw = self._call(messages)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            # tenta recuperar JSON embutido em texto (alguns modelos locais
            # insistem em cercar o JSON com ```json ... ``` ou comentarios)
            data = _try_recover_json(raw)
            if data is None:
                raise ExtractionError(f"LLM nao retornou JSON valido para {paper_id}: {e}\nRAW: {raw[:2000]}")

        data.setdefault("paper_id", paper_id)

        try:
            extraction = PaperExtraction.model_validate(data)
        except ValidationError as e_first:
            # Modelos locais via Ollama frequentemente nao seguem o schema a
            # risca mesmo com response_format=json_schema "strict" (aliases de
            # campo, listas em vez de objetos EvidenceField, enums fora do
            # vocabulario). Em vez de descartar a extracao inteira -- jogando
            # fora uma chamada de LLM cara -- tenta reparar os desvios mais
            # comuns (repair.py) e revalida antes de desistir.
            logger.warning(
                "Validacao inicial falhou para %s (%d erros); tentando auto-reparo.",
                paper_id, len(e_first.errors()),
            )
            try:
                repaired = repair_paper_extraction(data)
                extraction = PaperExtraction.model_validate(repaired)
            except ValidationError as e_second:
                raise ExtractionError(
                    f"JSON nao bate com o schema para {paper_id} mesmo apos auto-reparo: {e_second}\n"
                    f"Erros originais (pre-reparo): {e_first}"
                )

        # Etapa de dedup: o LLM as vezes lista o mesmo modelo mais de uma vez
        # (ex: mencionado em metodos e de novo em resultados/tabela). Colapsa
        # antes de devolver, para que nem a comparacao A/B nem a planilha
        # final vejam "modelos" duplicados como se fossem distintos.
        return deduplicate_records(extraction)


def _try_recover_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
