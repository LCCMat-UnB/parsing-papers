import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parsing_papers.schema import PaperExtraction, ModelRecord, EvidenceField, NumericEvidenceField, YesNo


def test_minimal_valid_record():
    rec = ModelRecord(
        model_used=EvidenceField(value="Random Forest", quote="we used a Random Forest classifier"),
        baseline=YesNo.NO,
        sample_size=NumericEvidenceField(value="1200", quote="a sample of 1200 farmers"),
        default_rate=NumericEvidenceField(value=None, quote=None),
        financial=YesNo.YES,
        productive=YesNo.YES,
        climatic=YesNo.NO,
        hybrid=YesNo.YES,
        variables_used=EvidenceField(value="income, credit history", quote="income and credit history were used"),
        auc=NumericEvidenceField(value="0.87", quote="AUC of 0.87"),
        accuracy=NumericEvidenceField(value=None, quote=None),
        f1_score=NumericEvidenceField(value=None, quote=None),
        recall=NumericEvidenceField(value=None, quote=None),
        specificity=NumericEvidenceField(value=None, quote=None),
        standard_deviation=NumericEvidenceField(value=None, quote=None),
        standard_error=NumericEvidenceField(value=None, quote=None),
    )
    paper = PaperExtraction(paper_id="paper_x", records=[rec])
    assert len(paper.records) == 1
    assert paper.records[0].auc.value == "0.87"


def test_json_schema_generation():
    schema = PaperExtraction.model_json_schema()
    assert "records" in schema["properties"]
