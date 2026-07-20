import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parsing_papers.schema import PaperExtraction, ModelRecord, EvidenceField, NumericEvidenceField, YesNo
from parsing_papers.verification import verify_paper_extraction
from parsing_papers.consolidate import build_dataframe, export
from parsing_papers.audit_sampling import select_audit_sample, AuditSampleConfig

SOURCE = """
The Random Forest model achieved an AUC of 0.87 on the test set, using a
sample of 1200 farmers with income and credit history as predictors.
The logistic regression baseline achieved an AUC of 0.79.
"""


def _paper():
    rf = ModelRecord(
        model_used=EvidenceField(value="Random Forest", quote="Random Forest model achieved an AUC of 0.87"),
        baseline=YesNo.NO,
        sample_size=NumericEvidenceField(value="1200", quote="sample of 1200 farmers"),
        default_rate=NumericEvidenceField(value=None, quote=None),
        financial=YesNo.YES, productive=YesNo.NO, climatic=YesNo.NO, hybrid=YesNo.NO,
        variables_used=EvidenceField(value="income", quote="income and credit history as predictors"),
        auc=NumericEvidenceField(value="0.87", quote="AUC of 0.87"),
        accuracy=NumericEvidenceField(value=None, quote=None),
        f1_score=NumericEvidenceField(value=None, quote=None),
        recall=NumericEvidenceField(value=None, quote=None),
        specificity=NumericEvidenceField(value=None, quote=None),
        standard_deviation=NumericEvidenceField(value=None, quote=None),
        standard_error=NumericEvidenceField(value=None, quote=None),
    )
    lr = ModelRecord(
        model_used=EvidenceField(value="Logistic Regression", quote="THIS QUOTE DOES NOT EXIST IN SOURCE AT ALL"),
        baseline=YesNo.YES,
        sample_size=NumericEvidenceField(value="1200", quote="sample of 1200 farmers"),
        default_rate=NumericEvidenceField(value=None, quote=None),
        financial=YesNo.YES, productive=YesNo.NO, climatic=YesNo.NO, hybrid=YesNo.NO,
        variables_used=EvidenceField(value="income", quote="income"),
        auc=NumericEvidenceField(value="0.79", quote="AUC of 0.79"),
        accuracy=NumericEvidenceField(value=None, quote=None),
        f1_score=NumericEvidenceField(value=None, quote=None),
        recall=NumericEvidenceField(value=None, quote=None),
        specificity=NumericEvidenceField(value=None, quote=None),
        standard_deviation=NumericEvidenceField(value=None, quote=None),
        standard_error=NumericEvidenceField(value=None, quote=None),
    )
    return PaperExtraction(paper_id="paper_1", records=[rf, lr])


def test_full_consolidation_flags_hallucinated_citation(tmp_path):
    paper = _paper()
    verifications = verify_paper_extraction(paper, SOURCE, threshold=90.0)

    # RF quote real deve verificar; LR quote inventado nao deve
    assert verifications[0].all_verified
    assert not verifications[1].all_verified
    assert "model_used" in verifications[1].unverified_fields

    df = build_dataframe([paper], verifications_by_paper={"paper_1": verifications})
    assert len(df) == 2
    assert df.iloc[0]["needs_review"] == False
    assert df.iloc[1]["needs_review"] == True  # citacao alucinada deve acender o flag

    xlsx_path, csv_path = export(df, tmp_path)
    assert xlsx_path.exists()
    assert csv_path.exists()


def test_audit_sample_includes_flagged_records():
    paper = _paper()
    verifications = verify_paper_extraction(paper, SOURCE, threshold=90.0)
    df = build_dataframe([paper], verifications_by_paper={"paper_1": verifications})

    audit_df = select_audit_sample(df, AuditSampleConfig(fraction=0.5, min_sample=1))
    assert (audit_df["Model_used"] == "Logistic Regression").any()
    assert "human_verdict" in audit_df.columns
