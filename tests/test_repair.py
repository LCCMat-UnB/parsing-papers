import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parsing_papers.repair import repair_paper_extraction
from parsing_papers.schema import PaperExtraction


def test_repairs_real_ollama_malformed_output():
    """Caso real observado: qwen2.5:32b-instruct via Ollama retornou uma
    estrutura com aliases de campo, listas em vez de EvidenceField, e enums
    fora do vocabulario -- isso derrubava a extracao inteira antes do reparo."""
    raw = {
        "paper_id": "paper_x",
        "records": [
            {
                "model_name": "Logistic Regression",
                "type_data_used": ["financial", "productive"],
                "variables_used": ["Limite de Credito", "Receita Bruta", "CNAE"],
                "agricultural_context": "no",
                "sample_size": "1000",
                "default_rate": "10%",
                "auc": "0.70",
                "accuracy": "0.6302",
                "f1_score": None,
                "recall": None,
                "specificity": None,
                "standard_deviation": None,
                "standard_error": None,
            }
        ],
    }
    repaired = repair_paper_extraction(raw)
    paper = PaperExtraction.model_validate(repaired)

    assert len(paper.records) == 1
    rec = paper.records[0]
    assert rec.model_used.value == "Logistic Regression"
    assert rec.type_data_used.value == "hybrid"
    assert "Limite de Credito" in rec.variables_used.value
    assert rec.agricultural_context.value == "not_reported"
    assert "auto-reparo" in rec.extraction_notes


def test_repair_preserves_already_valid_record():
    raw = {
        "paper_id": "paper_y",
        "records": [
            {
                "model_used": {"value": "XGBoost", "quote": "we used XGBoost"},
                "baseline": "no",
                "sample_size": {"value": "500", "quote": "n=500"},
                "default_rate": {"value": None, "quote": None},
                "financial": "yes",
                "productive": "no",
                "climatic": "no",
                "hybrid": "no",
                "variables_used": {"value": "income", "quote": "income"},
                "auc": {"value": "0.9", "quote": "AUC=0.9"},
                "accuracy": {"value": None, "quote": None},
                "f1_score": {"value": None, "quote": None},
                "recall": {"value": None, "quote": None},
                "specificity": {"value": None, "quote": None},
                "standard_deviation": {"value": None, "quote": None},
                "standard_error": {"value": None, "quote": None},
            }
        ],
    }
    repaired = repair_paper_extraction(raw)
    paper = PaperExtraction.model_validate(repaired)
    assert paper.records[0].model_used.value == "XGBoost"
    assert paper.records[0].financial.value == "yes"


def test_repair_normalizes_model_class_aliases():
    raw = {
        "paper_id": "paper_z",
        "records": [
            {
                "model_used": {"value": "XGBoost", "quote": "XGBoost"},
                "model_class": "XGBoost",
                "baseline": "no",
                "sample_size": {"value": None, "quote": None},
                "default_rate": {"value": None, "quote": None},
                "financial": "yes", "productive": "no", "climatic": "no", "hybrid": "no",
                "variables_used": {"value": None, "quote": None},
                "auc": {"value": None, "quote": None},
                "accuracy": {"value": None, "quote": None},
                "f1_score": {"value": None, "quote": None},
                "recall": {"value": None, "quote": None},
                "specificity": {"value": None, "quote": None},
                "standard_deviation": {"value": None, "quote": None},
                "standard_error": {"value": None, "quote": None},
            }
        ],
    }
    repaired = repair_paper_extraction(raw)
    paper = PaperExtraction.model_validate(repaired)
    assert paper.records[0].model_class.value == "gradient_boosting"
