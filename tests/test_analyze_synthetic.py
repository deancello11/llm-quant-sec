"""Runs the whole analysis on fake data where the true effect is known: NF4 raises ASR by ~15 points, INT8 does nothing."""
import zlib

import numpy as np
import pytest

from src import analyze
from src.common import append_jsonl, load_config

PRECS = {"fp16": 0.10, "int8": 0.10, "nf4": 0.25, "gptq4": 0.18, "awq4": 0.16}


@pytest.fixture
def fake(tmp_path, monkeypatch):
    cfg = load_config()
    out, jud, res = tmp_path / "o", tmp_path / "j", tmp_path / "r"
    for d in (out, jud, res):
        d.mkdir()
    rng = np.random.default_rng(0)
    for m in ["qwen25-7b", "llama31-8b"]:
        for p, rate in PRECS.items():
            for ds, n_b, variants in [("harmbench", 120, ["direct", "tmpl_a"]), ("xstest", 60, ["direct"])]:
                harmful = ds == "harmbench"
                for b in range(n_b):
                    latent = rng.random()   # per-behavior difficulty, shared across precisions via seed below
                    for v in variants:
                        pid = f"{ds}:{b}::{v}"
                        # same latent for every precision keeps outcomes paired
                        u = np.random.default_rng(zlib.crc32(f"{m}{ds}{b}{v}".encode())).random()
                        unsafe = bool(harmful and u < rate)
                        refuse = bool((harmful and not unsafe) or (not harmful and u < 0.05))
                        append_jsonl(out / f"{m}__{p}__{ds}.jsonl", {"model": m, "precision": p, "dataset": ds, "prompt_id": pid,
                            "behavior_id": f"{ds}:{b}", "variant": v, "category": "", "prompt": "x", "response": "y",
                            "n_tokens": 10, "truncated": False})
                        append_jsonl(jud / f"{m}__{p}__{ds}.jsonl", {"model": m, "precision": p, "dataset": ds, "prompt_id": pid,
                            "judge_unsafe": unsafe, "refusal": refuse})
    monkeypatch.setattr(analyze, "OUTPUTS_DIR", out)
    monkeypatch.setattr(analyze, "JUDGMENTS_DIR", jud)
    monkeypatch.setattr(analyze, "RESULTS_DIR", res)
    cfg["stats"]["n_boot"] = 500
    return cfg, res


def test_pipeline_recovers_known_effects(fake, monkeypatch):
    cfg, res = fake
    monkeypatch.setattr(analyze, "load_config", lambda: cfg)
    analyze.main()
    pw = __import__("pandas").read_csv(res / "pairwise.csv")
    nf4 = pw[(pw.family == "vs_fp16") & (pw.condition == "nf4")]
    int8 = pw[(pw.family == "vs_fp16") & (pw.condition == "int8")]
    assert (nf4.diff_pts > 8).all() and nf4.significant_holm.all()
    assert (int8.diff_pts.abs() < 1e-9).all() and not int8.significant_holm.any()
    assert "H2" in (res / "hypotheses.md").read_text()
    for f in ["asr_by_precision.png", "asr_vs_overrefusal.png", "method_comparison.png", "summary.csv"]:
        assert (res / f).exists(), f
