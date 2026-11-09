"""Robustness checks on the headline comparison (4-bit/INT8 vs FP16, per model).

    python -m src.robustness      # writes results/robustness.csv and results/robustness.md

Each variant answers "does the conclusion survive a different way of measuring?". All use the same paired
cluster bootstrap as src.analyze (whole behaviors are resampled together, the same resamples for every condition).

  base            judge says unsafe, all harmful prompts (the headline measure)
  strict          judge says unsafe AND the response is not a keyword refusal
  not_truncated   only prompts whose FP16 and quantized responses both finished before max_new_tokens
  direct / template   the same measure on plain requests only, and on jailbreak-template prompts only
  judge_scaled    base difference x (PPV + NPV - 1) from the pooled hand labels: a rough "if the judge's errors are
                  not precision-dependent, the true difference is about this size" rescaling, not a new measurement

Where it fits: this runs after the main analysis (generate -> judge -> analyze -> robustness). The headline result
is "quantized model X is this many points more or less jailbreakable than FP16". A skeptic's first question is
"is that just an accident of how you measured it?" So this file re-measures the same comparison five or six
different ways and shows whether the sign (up or down) and the "error bar excludes zero" verdict stay the same.
If they do, the finding is sturdy. If one variant flips the answer, that is worth knowing too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analyze import PREC_ORDER, load_all
from .common import RESULTS_DIR, load_config
from .stats import boot_indices, percentile_ci

# Pooled independent hand labels (see validation/agreement_*.json and REPORT.md): judge precision and NPV.
# Two label passes exist for one sample, so the rescaling is shown for both ends of the range.
JUDGE_SCALE = {"pass1_pool": (0.72, 0.97), "pass2_pool": (0.81, 0.97)}


def paired_diff(df: pd.DataFrame, model: str, base: str, cond: str, flag: str, row_filter, n_boot: int, seed: int):
    """Percentage-point difference (cond - base) on the same prompts, with a cluster-bootstrap CI."""
    sub = df[(df.model == model) & df.harmful]
    # row_filter is either a function (keep some rows), the special word "not_truncated", or None (keep everything).
    if callable(row_filter):
        sub = sub[row_filter(sub)]
    # Line the two conditions up by prompt_id, like laying two copies of the same exam side by side, so we only
    # compare a prompt with itself and drop any prompt that is missing from one side.
    a = sub[sub.precision == base].set_index("prompt_id")
    b = sub[sub.precision == cond].set_index("prompt_id")
    ids = a.index.intersection(b.index)
    a, b = a.loc[ids], b.loc[ids]
    keep = np.ones(len(ids), dtype=bool)
    # "not_truncated" needs both versions, so it is applied after pairing: a prompt is dropped if EITHER reply was
    # cut off. That way a difference cannot come from one model simply running out of room.
    if row_filter == "not_truncated":
        keep = (~a.truncated.astype(bool)).to_numpy() & (~b.truncated.astype(bool)).to_numpy()
        a, b = a[keep], b[keep]
    if len(a) == 0:
        return None
    # Collapse to one row per behavior (successes under each condition, and how many prompts), so the bootstrap
    # can resample whole behaviors, with a behavior and its template copies staying together.
    d = pd.DataFrame({"cluster": a.behavior_id.to_numpy(), "x": a[flag].to_numpy(float), "y": b[flag].to_numpy(float)})
    g = d.groupby("cluster").agg(sx=("x", "sum"), sy=("y", "sum"), n=("x", "count"))
    # Same seed and same number of clusters as the main analysis gives the same resamples, so results line up.
    idx = boot_indices(len(g), n_boot, seed)
    sx, sy, n = g.sx.to_numpy(), g.sy.to_numpy(), g.n.to_numpy()
    est = 100 * (sy.sum() - sx.sum()) / n.sum()
    # Each resample recomputes the difference using the SAME drawn behaviors for both conditions. That is what
    # "paired" means here, and it makes the interval much tighter than treating the two sides as strangers.
    boot = 100 * (sy[idx].sum(axis=1) - sx[idx].sum(axis=1)) / np.maximum(n[idx].sum(axis=1), 1)
    lo, hi = percentile_ci(boot)
    return {"diff_pts": est, "ci_lo": lo, "ci_hi": hi, "n_prompts": int(n.sum()), "base_asr": 100 * sx.sum() / n.sum()}


def main():
    """Run every variant for every model and precision, then write the CSV and a compact markdown table."""
    cfg = load_config()
    df = load_all(cfg)
    df["unsafe"] = df.judge_unsafe.astype(float)
    nb, seed = cfg["stats"]["n_boot"], cfg["seed"]
    # name -> (which 0/1 column to average, which rows to keep). Each entry is one "way of measuring".
    variants = {
        "base": ("unsafe", None),
        "strict": ("strict_flag", None),
        "not_truncated": ("unsafe", "not_truncated"),
        "direct": ("unsafe", lambda s: s.variant == "direct"),
        "template": ("unsafe", lambda s: s.variant.str.startswith("tmpl_")),
    }
    rows = []
    for model in [m for m in cfg["models"] if m in set(df.model)]:
        for cond in PREC_ORDER[1:]:
            for name, (flag, filt) in variants.items():
                if filt == "not_truncated":
                    r = paired_diff(df, model, "fp16", cond, flag, "not_truncated", nb, seed)
                else:
                    r = paired_diff(df, model, "fp16", cond, flag, filt, nb, seed)
                if r:
                    rows.append({"model": model, "condition": cond, "variant": name, **r,
                                 "excludes_zero": bool(r["ci_lo"] > 0 or r["ci_hi"] < 0)})
    out = pd.DataFrame(rows)
    base = out[out.variant == "base"].set_index(["model", "condition"])
    for label, (ppv, npv) in JUDGE_SCALE.items():
        # The judge sometimes cries wolf (precision below 1) and sometimes misses things (NPV below 1). If those
        # mistakes are the same at every precision, they shrink real differences by roughly (PPV + NPV - 1).
        # Multiplying the base result by that factor is a back-of-envelope guess, not a new measurement.
        k = ppv + npv - 1
        for _, r in base.reset_index().iterrows():
            rows.append({"model": r.model, "condition": r.condition, "variant": f"judge_scaled_{label}",
                         "diff_pts": r.diff_pts * k, "ci_lo": r.ci_lo * k, "ci_hi": r.ci_hi * k,
                         "n_prompts": r.n_prompts, "base_asr": float("nan"),
                         "excludes_zero": bool(r.ci_lo * k > 0 or r.ci_hi * k < 0)})
    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(RESULTS_DIR / "robustness.csv", index=False)

    # one readable table: does the sign and significance of each headline effect survive each variant?
    # (The judge_scaled rows stay in the CSV only; this table shows the five directly measured variants.)
    key = out[out.variant == "base"][["model", "condition"]]
    lines = ["# Robustness of the headline comparison (difference vs FP16, percentage points, 95% cluster-bootstrap CI)", "",
             "`*` = CI excludes 0. Same prompts in both conditions. See src/robustness.py for the definition of each column.", ""]
    names = ["base", "strict", "not_truncated", "direct", "template"]
    lines.append("| model | condition | " + " | ".join(names) + " |")
    lines.append("|---|---|" + "---|" * len(names))
    for _, k in key.iterrows():
        cells = []
        for v in names:
            r = out[(out.model == k.model) & (out.condition == k.condition) & (out.variant == v)]
            if r.empty:
                cells.append("n/a")
                continue
            r = r.iloc[0]
            cells.append(f"{r.diff_pts:+.1f}{'*' if r.excludes_zero else ''} [{r.ci_lo:+.1f},{r.ci_hi:+.1f}]")
        lines.append(f"| {k.model} | {k.condition} | " + " | ".join(cells) + " |")
    (RESULTS_DIR / "robustness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nWrote results/robustness.csv and results/robustness.md")


if __name__ == "__main__":
    main()
