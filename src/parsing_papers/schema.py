"""
Schema de extracao de metadados de papers sobre modelos de credito agricola.

Cada campo numerico/categorico exigido pela planilha vem acompanhado de um
"EvidenceField": o valor extraido + o trecho EXATO do texto fonte de onde o
LLM tirou aquele valor. Isso permite verificacao automatica via fuzzy match
(etapa 3 do pipeline) antes de aceitar o dado como valido.

Metadados como doi, year, author, journal, country, latitude, longitude sao
extraidos por outros metodos (fora deste pipeline) e NAO fazem parte deste
schema.

IMPORTANTE: um paper pode reportar varios modelos (ex: testou 10 modelos).
Cada modelo = um ModelRecord = uma linha na planilha final. A extracao
retorna uma lista de ModelRecord por paper (PaperExtraction.records).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums controlados (vocabulario fechado ajuda o LLM e facilita analise depois)
# ---------------------------------------------------------------------------

class ModelClass(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    DISCRIMINANT_ANALYSIS = "discriminant_analysis"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"  # XGBoost, LightGBM, CatBoost, GBM
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    DEEP_LEARNING = "deep_learning"  # CNN, RNN, LSTM, transformers
    NAIVE_BAYES = "naive_bayes"
    KNN = "knn"
    ENSEMBLE_OTHER = "ensemble_other"  # stacking/voting/bagging genericos
    BAYESIAN_MODEL = "bayesian_model"
    SURVIVAL_MODEL = "survival_model"
    OTHER_STATISTICAL = "other_statistical"
    OTHER = "other"


class TypeDataUsed(str, Enum):
    FINANCIAL = "financial"
    PRODUCTIVE = "productive"
    CLIMATIC = "climatic"
    HYBRID = "hybrid"


class AgriculturalContext(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    NOT_REPORTED = "not_reported"
    MIXED = "mixed"


class YesNo(str, Enum):
    YES = "yes"
    NO = "no"
    NOT_REPORTED = "not_reported"


# ---------------------------------------------------------------------------
# Campo com evidencia obrigatoria
# ---------------------------------------------------------------------------

class EvidenceField(BaseModel):
    """
    Valor extraido + citacao textual exata (verbatim) de onde veio.

    O campo `quote` DEVE ser uma copia literal de uma frase/trecho do texto
    fonte fornecido ao LLM -- nao um resumo, nao uma paráfrase. É contra essa
    string que rodaremos o fuzzy match na etapa de verificacao.
    """

    value: Optional[str] = Field(
        default=None,
        description=(
            "Valor extraido, como string (mesmo para numeros, para preservar "
            "formatacao original, ex: '0.87' ou '87%'). Use null se o dado "
            "nao for reportado no texto."
        ),
    )
    quote: Optional[str] = Field(
        default=None,
        description=(
            "Trecho EXATO (copiado literalmente, sem parafrasear) do texto "
            "fonte de onde o valor foi extraido. Null se value for null."
        ),
    )
    source_section: Optional[str] = Field(
        default=None,
        description="Secao do paper de onde veio o trecho (ex: 'results', 'methods', 'table 2').",
    )


class NumericEvidenceField(EvidenceField):
    """Igual a EvidenceField, mas para valores que devem ser numericos apos parse."""

    pass


# ---------------------------------------------------------------------------
# Registro por modelo (= 1 linha da planilha final)
# ---------------------------------------------------------------------------

class ModelRecord(BaseModel):
    # --- Características metodológicas ---
    model_used: EvidenceField = Field(
        description="Nome especifico do modelo (ex: 'Random Forest', 'XGBoost', 'Logistic Regression')."
    )
    model_class: Optional[ModelClass] = Field(
        default=None, description="Classe/familia do modelo, do vocabulario controlado."
    )
    baseline: YesNo = Field(
        description=(
            "Se este modelo foi usado como baseline/controle no estudo. "
            "Regressao Logistica deve ser marcada 'yes' quando usada como baseline "
            "conforme convencao do projeto; caso contrario 'no'."
        )
    )

    # --- Características do dataset ---
    sample_size: NumericEvidenceField = Field(description="Tamanho da amostra usada para este modelo.")
    default_rate: NumericEvidenceField = Field(
        description="Taxa de inadimplencia/default (proporcao 0-1 ou %, conforme reportado)."
    )
    type_data_used: Optional[TypeDataUsed] = Field(
        default=None, description="Tipo de dados: financial, productive, climatic ou hybrid."
    )
    financial: YesNo = Field(description="Usa dados financeiros (renda, historico de credito, score bancario etc)?")
    productive: YesNo = Field(description="Usa dados produtivos (area plantada, produtividade etc)?")
    climatic: YesNo = Field(description="Usa dados climaticos (NDVI, clima, chuva etc)?")
    hybrid: YesNo = Field(description="Combina múltiplos tipos de dados (financeiro+produtivo+climatico)?")
    variables_used: EvidenceField = Field(
        description=(
            "Lista das variaveis/features usadas pelo modelo, como texto "
            "(ex: 'renda, historico de credito, area plantada, NDVI')."
        )
    )
    agricultural_context: Optional[AgriculturalContext] = Field(
        default=None, description="Porte do produtor: small, medium, large, mixed ou not_reported."
    )

    # --- Métricas de desempenho ---
    auc: NumericEvidenceField = Field(description="AUC / ROC-AUC do modelo, se reportado (0.5-1.0).")
    accuracy: NumericEvidenceField = Field(description="Acuracia do modelo, se reportada (0-1 ou %).")
    f1_score: NumericEvidenceField = Field(description="F1-score do modelo, se reportado (0-1).")
    recall: NumericEvidenceField = Field(description="Recall/sensibilidade do modelo, se reportado (0-1).")
    specificity: NumericEvidenceField = Field(description="Especificidade do modelo, se reportada (0-1).")
    standard_deviation: NumericEvidenceField = Field(
        description="Desvio padrao associado a metrica principal, se reportado."
    )
    standard_error: NumericEvidenceField = Field(
        description="Erro padrao associado a metrica principal, se reportado."
    )

    extraction_notes: Optional[str] = Field(
        default=None,
        description="Observacoes livres do extrator sobre ambiguidades, unidades, ou dados conflitantes.",
    )


class PaperExtraction(BaseModel):
    """Resultado completo da extracao de um paper: 0..N modelos reportados."""

    paper_id: str = Field(description="Identificador do paper (ex: nome do arquivo PDF sem extensao).")
    records: list[ModelRecord] = Field(
        default_factory=list,
        description="Um ModelRecord por modelo testado/reportado no paper. Um paper com 10 modelos gera 10 records.",
    )
    extraction_warnings: list[str] = Field(
        default_factory=list,
        description="Avisos gerais sobre a extracao (ex: 'tabela de resultados truncada pelo GROBID').",
    )


# ---------------------------------------------------------------------------
# JSON Schema para uso em structured output / function calling
# ---------------------------------------------------------------------------

def paper_extraction_json_schema() -> dict:
    """Retorna o JSON schema (dict) usado para forcar structured output no LLM."""
    return PaperExtraction.model_json_schema()


FIELD_NAMES_FOR_SPREADSHEET = [
    "Model_used",
    "Model_class",
    "Baseline",
    "Sample_size",
    "Default_rate",
    "Type_data_used",
    "Financial",
    "Productive",
    "Climatic",
    "Hybrid",
    "Variables_used",
    "Agricultural_context",
    "AUC",
    "Accuracy",
    "F1-score",
    "Recall",
    "Specificity",
    "Standard_deviation",
    "Standard_error",
]
