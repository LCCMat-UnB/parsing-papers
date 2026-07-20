"""
Normalizacao/reparo de JSON antes da validacao Pydantic (schema.py).

Modelos locais via Ollama, mesmo com response_format=json_schema "strict",
frequentemente nao seguem o schema a risca: usam nomes de campo alternativos
(model_name em vez de model_used), retornam listas onde o schema pede um
objeto EvidenceField, retornam varios valores num campo enum de valor unico
(ex: type_data_used=['financial','productive'] em vez de 'hybrid'), ou usam
sinonimos fora do enum (ex: agricultural_context='no').

Em vez de descartar a extracao inteira quando isso acontece (jogando fora
uma chamada de LLM cara), este modulo tenta reparar os desvios mais comuns e
sinaliza no proprio registro (via extraction_notes) o que foi ajustado
automaticamente -- para o revisor humano poder conferir.

Isto NAO inventa valores: so reorganiza/renomeia a estrutura de dados que o
LLM ja retornou, preservando o conteudo.
"""

from __future__ import annotations

import logging

from .schema import AgriculturalContext, ModelClass, TypeDataUsed, YesNo

logger = logging.getLogger(__name__)

VALID_MODEL_CLASS = {e.value for e in ModelClass}
VALID_TYPE_DATA = {e.value for e in TypeDataUsed}
VALID_AGRI_CONTEXT = {e.value for e in AgriculturalContext}
VALID_YES_NO = {e.value for e in YesNo}

# Aliases de nomes de campo que modelos locais usam com frequencia em vez do
# nome oficial do schema.
FIELD_ALIASES: dict[str, list[str]] = {
    "model_used": ["model_name", "model", "modelo", "model_used_name"],
    "sample_size": ["sample_size_n", "n_sample", "amostra"],
    "default_rate": ["default_rate_pct", "inadimplencia", "taxa_inadimplencia"],
    "variables_used": ["variables", "features", "variaveis", "variaveis_utilizadas"],
    "auc": ["roc_auc", "auc_score"],
    "accuracy": ["acc", "acuracia"],
    "f1_score": ["f1", "f1score"],
    "recall": ["sensitivity", "sensibilidade"],
    "specificity": ["especificidade"],
    "standard_deviation": ["std", "sd", "desvio_padrao"],
    "standard_error": ["se", "erro_padrao"],
}

EVIDENCE_FIELDS = [
    "model_used", "sample_size", "default_rate", "variables_used",
    "auc", "accuracy", "f1_score", "recall", "specificity",
    "standard_deviation", "standard_error",
]

YES_NO_ALIASES = {
    "yes": ["yes", "y", "true", "sim", "1"],
    "no": ["no", "n", "false", "nao", "não", "0"],
}

AGRICULTURAL_CONTEXT_ALIASES = {
    "small": ["small", "pequeno", "pequeno produtor", "smallholder"],
    "medium": ["medium", "medio", "médio", "medio produtor"],
    "large": ["large", "grande", "grande produtor"],
    "mixed": ["mixed", "misto", "varios", "diverso"],
    "not_reported": ["not_reported", "no", "none", "nao informado", "não informado", "n/a", "na", ""],
}

TYPE_DATA_ALIASES = {
    "financial": "financial",
    "productive": "productive",
    "climatic": "climatic",
    "hybrid": "hybrid",
}


def _coerce_evidence_field(value) -> dict | None:
    """
    Garante que um campo evidence vire {'value': ..., 'quote': ..., 'source_section': ...}.

    Aceita:
      - ja no formato certo (dict com 'value')
      - string solta -> vira {'value': str, 'quote': None}
      - lista de strings -> junta com '; ' -> {'value': "a; b; c", 'quote': None}
      - numero solto -> stringifica
    """
    if value is None:
        return None
    if isinstance(value, dict):
        if "value" in value:
            # normaliza subcampos ausentes
            value.setdefault("quote", None)
            value.setdefault("source_section", None)
            return value
        # dict sem 'value' -- tenta achar algo utilizavel
        return {"value": str(value), "quote": None, "source_section": None}
    if isinstance(value, list):
        joined = ", ".join(str(v) for v in value if v is not None)
        return {"value": joined or None, "quote": None, "source_section": None}
    # string ou numero solto
    return {"value": str(value), "quote": None, "source_section": None}


def _find_alias(record: dict, canonical: str) -> object:
    if canonical in record:
        return record[canonical]
    for alias in FIELD_ALIASES.get(canonical, []):
        if alias in record:
            return record[alias]
    return None


def _normalize_yes_no(value) -> str:
    if value is None:
        return "not_reported"
    if isinstance(value, bool):
        return "yes" if value else "no"
    v = str(value).strip().lower()
    if v in VALID_YES_NO:
        return v
    for canonical, aliases in YES_NO_ALIASES.items():
        if v in aliases:
            return canonical
    return "not_reported"


def _normalize_agricultural_context(value) -> str | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    for canonical, aliases in AGRICULTURAL_CONTEXT_ALIASES.items():
        if v in aliases:
            return canonical
    return "not_reported"


def _normalize_model_class(value) -> str | None:
    if value is None:
        return None
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if v in VALID_MODEL_CLASS:
        return v
    # tentativas comuns de match parcial
    aliases = {
        "logistic_regression": ["logistic", "logit", "regressao_logistica", "regressão_logística"],
        "random_forest": ["rf", "randomforest"],
        "gradient_boosting": ["xgboost", "lightgbm", "catboost", "gbm", "boosting"],
        "svm": ["support_vector_machine", "support_vector_machines"],
        "neural_network": ["ann", "mlp", "rede_neural"],
        "deep_learning": ["cnn", "rnn", "lstm", "deep_neural_network"],
        "decision_tree": ["arvore_de_decisao", "cart"],
        "knn": ["k_nearest_neighbors", "k_nn"],
        "naive_bayes": ["nb"],
    }
    for canonical, alts in aliases.items():
        if v in alts:
            return canonical
    return "other"


def _normalize_type_data_used(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        types = {str(v).strip().lower() for v in value if v}
        types = {TYPE_DATA_ALIASES.get(t, t) for t in types}
        types &= {"financial", "productive", "climatic"}
        if len(types) >= 2:
            return "hybrid"
        if len(types) == 1:
            return next(iter(types))
        return None
    v = str(value).strip().lower()
    return TYPE_DATA_ALIASES.get(v, v if v in TYPE_DATA_ALIASES.values() else None)


def repair_record(raw_record: dict) -> dict:
    """Repara um dict de ModelRecord bruto (saida do LLM) antes da validacao Pydantic."""
    notes: list[str] = []
    fixed: dict = dict(raw_record)  # copia rasa

    # 1. Campos evidence: renomear aliases + coagir formato
    for canonical in EVIDENCE_FIELDS:
        raw_value = _find_alias(fixed, canonical)
        coerced = _coerce_evidence_field(raw_value)
        if coerced is None:
            coerced = {"value": None, "quote": None, "source_section": None}
        if canonical not in fixed or fixed.get(canonical) != coerced:
            if raw_value is not None and canonical not in raw_record:
                notes.append(f"campo '{canonical}' recuperado de alias")
            if isinstance(raw_value, list):
                notes.append(f"campo '{canonical}' era lista, convertido para string unica")
        fixed[canonical] = coerced

    # remove aliases residuais para nao poluir (Pydantic ignoraria mesmo, mas por clareza)
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            fixed.pop(alias, None)

    # 2. yes/no fields
    for field_name in ["financial", "productive", "climatic", "hybrid"]:
        original = fixed.get(field_name)
        normalized = _normalize_yes_no(original)
        if str(original).strip().lower() if original is not None else None:
            pass
        if normalized != original:
            notes.append(f"campo '{field_name}'='{original}' normalizado para '{normalized}'")
        fixed[field_name] = normalized

    if "baseline" in fixed:
        normalized = _normalize_yes_no(fixed["baseline"])
        if normalized != fixed["baseline"]:
            notes.append(f"campo 'baseline'='{fixed['baseline']}' normalizado para '{normalized}'")
        fixed["baseline"] = normalized
    else:
        fixed["baseline"] = "not_reported"

    # 3. agricultural_context
    if "agricultural_context" in fixed:
        original = fixed["agricultural_context"]
        normalized = _normalize_agricultural_context(original)
        if normalized != original:
            notes.append(f"campo 'agricultural_context'='{original}' normalizado para '{normalized}'")
        fixed["agricultural_context"] = normalized

    # 4. type_data_used (pode vir como lista -> hybrid)
    if "type_data_used" in fixed:
        original = fixed["type_data_used"]
        normalized = _normalize_type_data_used(original)
        if normalized != original:
            notes.append(f"campo 'type_data_used'={original!r} normalizado para {normalized!r}")
        fixed["type_data_used"] = normalized

    # 5. model_class: normaliza para o enum valido; usa 'other' se nao reconhecer
    if "model_class" in fixed and fixed["model_class"] is not None:
        original = fixed["model_class"]
        normalized = _normalize_model_class(original)
        if normalized != original:
            notes.append(f"campo 'model_class'={original!r} normalizado para {normalized!r}")
        fixed["model_class"] = normalized

    # 6. agricultural_context / type_data_used: garante que caiu em valor valido do enum
    if fixed.get("agricultural_context") not in VALID_AGRI_CONTEXT and fixed.get("agricultural_context") is not None:
        notes.append(f"agricultural_context invalido descartado: {fixed['agricultural_context']!r}")
        fixed["agricultural_context"] = "not_reported"
    if fixed.get("type_data_used") not in VALID_TYPE_DATA and fixed.get("type_data_used") is not None:
        notes.append(f"type_data_used invalido descartado: {fixed['type_data_used']!r}")
        fixed["type_data_used"] = None

    if notes:
        existing_notes = fixed.get("extraction_notes") or ""
        combined = "; ".join(notes)
        fixed["extraction_notes"] = f"{existing_notes} [auto-reparo: {combined}]".strip()

    return fixed


def repair_paper_extraction(raw_data: dict) -> dict:
    """Repara o dict completo de PaperExtraction (nivel paper) antes da validacao."""
    fixed = dict(raw_data)
    records = fixed.get("records") or []
    fixed["records"] = [repair_record(r) if isinstance(r, dict) else r for r in records]
    return fixed
