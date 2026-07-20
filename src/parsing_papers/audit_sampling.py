"""
Etapa 6 do pipeline: auditoria humana amostral.

Seleciona aleatoriamente 15-20% dos registros extraidos (linhas = modelos,
nao papers) para revisao manual. Prioriza incluir na amostra:
  - todos os registros com flags (citacao nao verificada OU sanidade falhou
    OU divergencia entre extracoes duplas) -- pois sao os que mais precisam
    de olho humano;
  - complementa ate atingir o percentual alvo com registros "limpos"
    sorteados aleatoriamente, para calibrar a taxa de erro tambem no que
    passou automaticamente.

Gera uma planilha de auditoria com os campos extraidos + evidencias + todos
os flags, e uma coluna vazia "human_verdict" para o revisor preencher.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd


@dataclass
class AuditSampleConfig:
    fraction: float = 0.18  # 18%, dentro da faixa 15-20% pedida
    min_sample: int = 5
    seed: int = 42
    always_include_flagged: bool = True


def select_audit_sample(df: pd.DataFrame, config: AuditSampleConfig = AuditSampleConfig()) -> pd.DataFrame:
    """
    df deve ter uma coluna booleana 'needs_review' (True se citacao nao
    verificada, sanidade falhou, ou divergencia entre extracoes).
    Retorna subconjunto de df selecionado para auditoria, com coluna
    'audit_reason' e coluna vazia 'human_verdict'.
    """
    rng = random.Random(config.seed)
    n_total = len(df)
    target_n = max(config.min_sample, int(round(n_total * config.fraction)))

    flagged = df[df.get("needs_review", False) == True].copy()  # noqa: E712
    flagged["audit_reason"] = "auto-flagged"

    clean = df[df.get("needs_review", False) != True].copy()  # noqa: E712

    remaining_slots = max(0, target_n - len(flagged))
    if remaining_slots > 0 and len(clean) > 0:
        sampled_idx = rng.sample(list(clean.index), k=min(remaining_slots, len(clean)))
        clean_sample = clean.loc[sampled_idx].copy()
        clean_sample["audit_reason"] = "random_calibration_sample"
    else:
        clean_sample = clean.iloc[0:0].copy()
        if "audit_reason" not in clean_sample.columns:
            clean_sample["audit_reason"] = pd.Series(dtype=str)

    audit_df = pd.concat([flagged, clean_sample], ignore_index=False)
    audit_df["human_verdict"] = ""  # a preencher: "correct" / "incorrect" / "partially_correct"
    audit_df["human_notes"] = ""

    return audit_df
