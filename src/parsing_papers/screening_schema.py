"""
Schema da etapa 0 do pipeline: triagem PRISMA de elegibilidade (inclusao/
exclusao) de cada paper na revisao sistematica e, adicionalmente, na
metanalise.

Baseado no documento "Criterios de Inclusao e Exclusao para Revisao
Sistematica e Metanalise" (aplicacao a modelos de IA/ML/DL em risco de
credito). Os codigos de criterio (I1-I8, E1-E11, M1-M7, ME1-ME6) sao os
mesmos definidos no documento -- mantidos aqui como vocabulario controlado
para que a planilha de triagem final seja diretamente rastreavel ate o
criterio formal que motivou cada decisao.

Segue o mesmo padrao anti-alucinacao do schema.py de extracao de metricas:
todo criterio marcado como aplicavel deve vir acompanhado de uma citacao
(`quote`) literal do texto fonte (titulo/abstract/corpo) que evidencia por
que aquele criterio se aplica -- sujeita a verificacao automatica por fuzzy
match, igual as citacoes da etapa de extracao (etapa 3, verification.py).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ScreeningDecision(str, Enum):
    """
    As 3 categorias de decisao de triagem, conforme secao 6 do documento de
    criterios ("Categorias recomendadas para a planilha de triagem").
    """

    INCLUDE_REVIEW_AND_METAANALYSIS = "include_review_and_metaanalysis"
    INCLUDE_REVIEW_ONLY = "include_review_only"
    EXCLUDE = "exclude"


# ---------------------------------------------------------------------------
# Vocabulario controlado dos codigos de criterio (secoes 2, 3, 4 e 5 do
# documento). Usado para validar que o LLM so cita codigos que realmente
# existem na proposta metodologica -- um codigo inventado (ex: "I9") deve ser
# rejeitado/reparado, nao aceito silenciosamente.
# ---------------------------------------------------------------------------

INCLUSION_CRITERIA_REVIEW = {
    "I1": "Artigos publicados entre 2000 e 2025.",
    "I2": "Artigos redigidos em ingles, portugues ou espanhol.",
    "I3": "Apenas artigos cientificos.",
    "I4": "Foco em risco de credito, inadimplencia, default, credit scoring, probabilidade de default ou previsao de default.",
    "I5": "Uso de modelos quantitativos, estatisticos, econometricos, de machine learning, deep learning ou inteligencia artificial.",
    "I6": "Apresenta pelo menos uma metrica de desempenho do modelo.",
    "I7": "Dados empiricos ou aplicacao real/simulada de modelos de classificacao ou previsao de risco de credito.",
    "I8": "Informacoes minimas disponiveis: autor, ano, titulo, modelo utilizado, base/dados analisados e metrica de desempenho.",
}

EXCLUSION_CRITERIA_REVIEW = {
    "E1": "Publicado antes de 2000 ou apos 2025.",
    "E2": "Idioma diferente de ingles, portugues ou espanhol.",
    "E3": "Revisao de literatura, revisao sistematica, bibliometria, metanalise, editorial, capitulo, tese, dissertacao, relatorio ou documento tecnico.",
    "E4": "Nao trata diretamente de risco de credito.",
    "E5": "Trata de outros riscos sem relacao direta com credito (mercado, operacional, climatico, liquidez, seguro).",
    "E6": "Trata de credito sem analise de risco, inadimplencia, default ou credit scoring.",
    "E7": "Puramente teorico, conceitual ou normativo, sem aplicacao de modelo.",
    "E8": "Menciona IA/ML/DL apenas perifericamente, sem aplicacao ou avaliacao de modelo.",
    "E9": "Sem metricas de desempenho reportadas.",
    "E10": "Registro duplicado.",
    "E11": "Sem informacoes minimas para triagem ou extracao dos dados.",
}

METAANALYSIS_INCLUSION_CRITERIA = {
    "M1": "Reporta claramente o modelo avaliado (ex: Random Forest, XGBoost, SVM, ANN, LSTM, regressao logistica).",
    "M2": "Apresenta pelo menos uma metrica comparavel: AUC, Accuracy, F1-score, Recall ou Specificity.",
    "M3": "A metrica esta associada a um modelo especifico.",
    "M4": "Informa valor numerico da metrica (ex: AUC = 0,87).",
    "M5": "Preferencialmente apresenta medida de dispersao/incerteza (standard deviation, standard error, IC ou validacao cruzada).",
    "M6": "Informa minimamente a estrategia de validacao (treino/teste, validacao cruzada ou validacao externa).",
    "M7": "Quando ha multiplos modelos no mesmo estudo, os resultados sao extraiveis separadamente por modelo.",
}

METAANALYSIS_EXCLUSION_CRITERIA = {
    "ME1": "Incluido na revisao, mas sem valor numerico de metrica.",
    "ME2": "Metrica reportada sem associacao clara ao modelo.",
    "ME3": "Resultado apenas em grafico, sem valor numerico recuperavel.",
    "ME4": "Sem informacao minima sobre validacao do modelo.",
    "ME5": "Metrica nao comparavel aos demais estudos.",
    "ME6": "Sem medida de dispersao quando a metanalise exigir erro padrao, desvio padrao ou intervalo de confianca.",
}

ALL_CRITERIA_CODES = (
    set(INCLUSION_CRITERIA_REVIEW)
    | set(EXCLUSION_CRITERIA_REVIEW)
    | set(METAANALYSIS_INCLUSION_CRITERIA)
    | set(METAANALYSIS_EXCLUSION_CRITERIA)
)


class CriterionApplication(BaseModel):
    """
    Um criterio (codigo do vocabulario acima) que o LLM identificou como
    aplicavel a este paper, com a evidencia textual que justifica a
    aplicacao -- mesmo espirito do EvidenceField do schema.py de extracao.
    """

    code: str = Field(description="Codigo do criterio, ex: 'I4', 'E9', 'M2', 'ME1'. Deve ser um dos codigos definidos na proposta metodologica.")
    quote: Optional[str] = Field(
        default=None,
        description=(
            "Trecho EXATO (copiado literalmente do texto fonte -- titulo, abstract ou corpo) "
            "que evidencia por que este criterio se aplica. Null apenas quando o criterio for "
            "de natureza formal/estrutural nao textual (ex: E10 duplicidade, E11 dados ausentes)."
        ),
    )
    note: Optional[str] = Field(default=None, description="Observacao curta opcional sobre a aplicacao do criterio.")


class ScreeningResult(BaseModel):
    """Resultado da triagem PRISMA de um paper (etapa 0 do pipeline)."""

    paper_id: str = Field(description="Identificador do paper (mesmo usado na extracao de metricas).")

    decision: ScreeningDecision = Field(
        description=(
            "Decisao de triagem: incluir na revisao sistematica E na metanalise; "
            "incluir apenas na revisao sistematica; ou excluir integralmente."
        )
    )

    inclusion_criteria_met: list[CriterionApplication] = Field(
        default_factory=list,
        description="Criterios de INCLUSAO (I1-I8) identificados como atendidos, com evidencia.",
    )
    exclusion_criteria_met: list[CriterionApplication] = Field(
        default_factory=list,
        description="Criterios de EXCLUSAO (E1-E11) identificados como aplicaveis, com evidencia. Presenca de qualquer um justifica decision=exclude.",
    )
    metaanalysis_inclusion_criteria_met: list[CriterionApplication] = Field(
        default_factory=list,
        description="Criterios adicionais de inclusao na metanalise (M1-M7) atendidos, com evidencia. So relevante se o paper nao foi excluido da revisao.",
    )
    metaanalysis_exclusion_criteria_met: list[CriterionApplication] = Field(
        default_factory=list,
        description="Criterios especificos de exclusao da metanalise (ME1-ME6) aplicaveis -- paper pode permanecer na revisao mesmo com esses.",
    )

    justification: str = Field(
        description="Resumo em 1-3 frases da razao da decisao, sintetizando os criterios aplicados."
    )
    screening_warnings: list[str] = Field(
        default_factory=list,
        description="Avisos sobre ambiguidades ou informacao insuficiente para decidir com confianca (ex: abstract truncado).",
    )


def screening_result_json_schema() -> dict:
    return ScreeningResult.model_json_schema()
