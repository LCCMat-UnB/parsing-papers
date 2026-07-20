import json

from parsing_papers.pipeline import build_compare_report


def _ckpt(paper_id, records, tokens, fallback, resolutions, total_s):
    return {
        "paper_id": paper_id,
        "estimated_prompt_tokens": tokens,
        "fallback_level": fallback,
        "extraction_a": {"paper_id": paper_id, "records": records},
        "comparisons": [
            {"index_a": 0, "index_b": 0, "divergences": [{"field_name": "auc", "diverges": True}]}
        ] if records else [],
        "arbiter_resolutions": resolutions,
        "timing": {"total_s": total_s},
    }


def _write_run(base, ckpts):
    d = base / "checkpoints"
    d.mkdir(parents=True)
    for c in ckpts:
        (d / f"{c['paper_id']}.json").write_text(json.dumps(c), encoding="utf-8")


def test_build_compare_report(tmp_path):
    rec = {"model_used": {"value": "XGBoost", "quote": None, "source_section": None}}
    _write_run(tmp_path / "old", [_ckpt("p1", [rec], 30000, 0, [], 900.0)])
    _write_run(tmp_path / "new", [_ckpt("p1", [rec], 5000, 1, [{"index_a": 0, "field_name": "auc", "choice": "b"}], 120.0)])

    df = build_compare_report(tmp_path / "old", tmp_path / "new")

    row = df[df["paper_id"] == "p1"].iloc[0]
    assert row["prompt_tokens_a"] == 30000
    assert row["prompt_tokens_b"] == 5000
    assert row["fallback_level_b"] == 1
    assert row["divergent_fields_a"] == 1
    assert row["arbiter_resolved_b"] == 1
    assert row["total_s_a"] == 900.0


def test_compare_tolerates_old_checkpoints(tmp_path):
    # checkpoint legado: so paper_id + extraction_a + source_text_len
    legacy = {"paper_id": "p9", "source_text_len": 90000, "extraction_a": {"paper_id": "p9", "records": []}}
    _write_run(tmp_path / "old", [legacy])
    _write_run(tmp_path / "new", [legacy])

    df = build_compare_report(tmp_path / "old", tmp_path / "new")
    assert df.iloc[0]["prompt_tokens_a"] == 30000  # source_text_len // 3
