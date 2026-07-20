"""
Etapa 0 do pipeline: triagem PRISMA de elegibilidade, ANTES da extracao de
metricas (etapa 2 em diante).

Aplica os criterios formais definidos em screening_schema.py (baseados no
documento "Criterios de Inclusao e Exclusao para Revisao Sistematica e
Metanalise") a cada paper, via LLM com citacao obrigatoria por criterio
aplicado -- mesmo padrao anti-alucinacao usado na extracao de metricas
(llm_client.py). So papers com decisao "include_review_and_metaanalysis" ou
"include_review_only" seguem para a extracao; papers "exclude" sao
registrados na planilha de triagem mas nao geram ModelRecord.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from .screening_prompts import build_screening_messages
from .screening_schema import ALL_CRITERIA_CODES, CriterionApplication, ScreeningDecision, ScreeningResult

logger = logging.getLogger(__name__)


class ScreeningError(Exception):
    pass


def _coerce_criterion_list(raw) -> list:
    """Tolera o LLM retornar uma lista de strings (so codigos) em vez de objetos {code, quote}."""
    if not raw:
        return []
    coerced = []
    for item in raw:
        if isinstance(item, dict):
            coerced.append(item)
        elif isinstance(item, str):
            coerced.append({"code": item, "quote": None, "note": None})
    return coerced


def _drop_unknown_codes(raw_list: list) -> list:
    """Remove entradas com codigo fora do vocabulario controlado (nao inventa correcao -- so descarta)."""
    kept = []
    for item in raw_list:
        code = item.get("code") if isinstance(item, dict) else None
        if code in ALL_CRITERIA_CODES:
            kept.append(item)
        else:
            logger.warning("Codigo de criterio desconhecido descartado na triagem: %r", code)
    return kept


def _repair_screening_data(raw_data: dict) -> dict:
    fixed = dict(raw_data)
    for field_name in [
        "inclusion_criteria_met",
        "exclusion_criteria_met",
        "metaanalysis_inclusion_criteria_met",
        "metaanalysis_exclusion_criteria_met",
    ]:
        coerced = _coerce_criterion_list(fixed.get(field_name))
        fixed[field_name] = _drop_unknown_codes(coerced)

    decision = fixed.get("decision")
    valid_decisions = {d.value for d in ScreeningDecision}
    if decision not in valid_decisions:
        # normaliza sinonimos comuns antes de desistir
        alias_map = {
            "include": "include_review_and_metaanalysis",
            "included": "include_review_and_metaanalysis",
            "review_only": "include_review_only",
            "include_review": "include_review_only",
            "excluded": "exclude",
            "excluir": "exclude",
            "incluir": "include_review_and_metaanalysis",
        }
        normalized = alias_map.get(str(decision).strip().lower()) if decision is not None else None
        fixed["decision"] = normalized or "exclude"
        if normalized is None:
            fixed.setdefault("screening_warnings", [])
            fixed["screening_warnings"].append(
                f"decisao original '{decision}' nao reconhecida -- default para 'exclude' por seguranca, revisar manualmente."
            )

    fixed.setdefault("justification", "")
    fixed.setdefault("screening_warnings", [])
    return fixed


class ScreeningClient:
    """
    Wrapper fino sobre litellm.completion para a etapa de triagem. Deliberadamente
    separado de LLMExtractor (llm_client.py) porque o schema/prompt sao bem
    menores -- reusar o mesmo cliente evitaria duplicar a logica de num_ctx/
    retry, mas o schema de resposta e outro; mantido simples e explicito aqui.
    """

    def __init__(
        self,
        model: str = "ollama_chat/qwen2.5:14b-instruct",
        api_base: str | None = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        request_timeout: int = 300,
        min_num_ctx: int = 8000,
    ):
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.min_num_ctx = min_num_ctx

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30))
    def _call(self, messages: list[dict]) -> str:
        import litellm

        prompt_chars = sum(len(m["content"]) for m in messages)
        estimated_prompt_tokens = prompt_chars // 3
        num_ctx = max(self.min_num_ctx, estimated_prompt_tokens + self.max_tokens + 512)

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
            kwargs["num_ctx"] = num_ctx

        try:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "screening_result",
                    "schema": ScreeningResult.model_json_schema(),
                    "strict": True,
                },
            }
            resp = litellm.completion(**kwargs)
        except Exception as e:  # noqa: BLE001
            logger.warning("json_schema estrito falhou na triagem (%s); tentando response_format=json_object", e)
            kwargs["response_format"] = {"type": "json_object"}
            resp = litellm.completion(**kwargs)

        return resp["choices"][0]["message"]["content"]

    def screen(self, paper_id: str, source_text: str) -> ScreeningResult:
        messages = build_screening_messages(paper_id=paper_id, source_text=source_text)
        raw = self._call(messages)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            data = _try_recover_json(raw)
            if data is None:
                raise ScreeningError(f"LLM nao retornou JSON valido na triagem de {paper_id}: {e}\nRAW: {raw[:2000]}")

        data.setdefault("paper_id", paper_id)

        try:
            return ScreeningResult.model_validate(data)
        except ValidationError as e_first:
            logger.warning("Validacao inicial da triagem falhou para %s (%d erros); tentando auto-reparo.", paper_id, len(e_first.errors()))
            try:
                repaired = _repair_screening_data(data)
                return ScreeningResult.model_validate(repaired)
            except ValidationError as e_second:
                raise ScreeningError(
                    f"JSON de triagem nao bate com o schema para {paper_id} mesmo apos auto-reparo: {e_second}\n"
                    f"Erros originais (pre-reparo): {e_first}"
                )


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


def should_proceed_to_extraction(result: ScreeningResult) -> bool:
    """Papers excluidos na triagem nao devem prosseguir para a extracao de metricas (etapas 2+)."""
    return result.decision != ScreeningDecision.EXCLUDE
