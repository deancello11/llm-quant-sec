"""Metrics, confidence intervals, paired tests and charts.

    python -m src.analyze                  # reads outputs/, judgments/, results/perf.jsonl -> results/*.csv, results/*.png, results/hypotheses.md

All rates are in percent. CIs are 95% cluster-bootstrap intervals that resample whole behaviors (a behavior and its
jailbreak-template copies stay together). McNemar's exact test is run on the paired prompts and Holm-corrected,
but it treats prompts as independent, so the bootstrap CI is the primary evidence.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .common import JUDGMENTS_DIR, OUTPUTS_DIR, RESULTS_DIR, load_config, read_jsonl
from .stats import boot_indices, holm, mcnemar_exact, mcnemar_mde, pareto_frontier, percentile_ci, ratio

PREC_ORDER = ["fp16", "int8", "nf4", "gptq4", "awq4"]
FOUR_BIT = ["nf4", "gptq4", "awq4"]
# Okabe-Ito colors (colorblind-safe). Precision is also shown by legend and marker, never by color alone.
COLORS = {"fp16": "#555555", "int8": "#0072B2", "nf4": "#D55E00", "gptq4": "#009E73", "awq4": "#CC79A7"}
MARKERS = ["o", "s", "^", "D"]


# ---------------------------------------------------------------- loading
def load_all(cfg: dict) -> pd.DataFrame:
    outs, judg = [], []
    for f in sorted(OUTPUTS_DIR.glob("*.jsonl")):
        if not f.name.startswith("_"):
            outs += read_jsonl(f)
    for f in sorted(JUDGMENTS_DIR.glob("*.jsonl")):
        if not f.name.startswith("_"):
            judg += read_jsonl(f)
    if not outs or not judg:
        raise SystemExit("No outputs or judgments found. Run src.generate and src.judge first.")
    keys = ["model", "precision", "dataset", "prompt_id"]
    j = pd.DataFrame(judg)[keys + ["judge_unsafe", "refusal"]]
    df = pd.DataFrame(outs).merge(j, on=keys, how="inner")
    df["harmful"] = df.dataset.map({d: s["harmful"] for d, s in cfg["datasets"].items()})
    df["asr_flag"] = df.judge_unsafe.astype(float)
    df["strict_flag"] = (df.judge_unsafe & ~df.refusal).astype(float)
    df["refusal_flag"] = df.refusal.astype(float)
    return df


def check_complete(df: pd.DataFrame) -> None:
    n = df.groupby(["model", "precision", "dataset"]).size().unstack("dataset")
    expected = n.max()
    short = n[(n < expected).any(axis=1)]
    if len(short):
        print("WARNING: incomplete cells (numbers are not comparable until these finish):\n", short)


# ---------------------------------------------------------------- metrics
METRICS = {
    "asr_all": (lambda d: d.harmful, "asr_flag"),
    "asr_direct": (lambda d: d.harmful & (d.variant == "direct"), "asr_flag"),
    "asr_template": (lambda d: d.harmful & d.variant.str.startswith("tmpl_"), "asr_flag"),
    "asr_strict": (lambda d: d.harmful, "strict_flag"),
    "refusal_harmful": (lambda d: d.harmful, "refusal_flag"),
    "overrefusal_xstest": (lambda d: d.dataset == "xstest", "refusal_flag"),
    "overrefusal_jbb_benign": (lambda d: d.dataset == "jbb_benign", "refusal_flag"),
}


class Metric:
    """Estimate and bootstrap distribution for every (model, precision) on one metric."""

    def __init__(self, df, mask_fn, flag, n_boot, seed):
        sub = df[mask_fn(df)]
        self.est, self.dist, self.n = {}, {}, {}
        if sub.empty:
            return
        clusters = sorted(sub.behavior_id.unique())
        idx = boot_indices(len(clusters), n_boot, seed)    # shared across conditions, so comparisons stay paired
        for (m, p), g in sub.groupby(["model", "precision"]):
            grp = g.groupby("behavior_id")[flag]
            s = grp.sum().reindex(clusters).fillna(0).to_numpy(float)
            c = grp.count().reindex(clusters).fillna(0).to_numpy(float)
            self.est[(m, p)] = 100 * ratio(s, c)
            self.dist[(m, p)] = 100 * ratio(s, c, idx)
            self.n[(m, p)] = int(c.sum())

    def ci(self, m, p, alpha=0.05):
        return percentile_ci(self.dist[(m, p)], alpha)


def summary_table(df, metrics: dict[str, Metric], perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, p) in sorted({k for mt in metrics.values() for k in mt.est}, key=lambda k: (k[0], PREC_ORDER.index(k[1]))):
        r = {"model": m, "precision": p}
        for name, mt in metrics.items():
            if (m, p) in mt.est:
                lo, hi = mt.ci(m, p)
                r.update({name: mt.est[(m, p)], f"{name}_lo": lo, f"{name}_hi": hi, f"{name}_n": mt.n[(m, p)]})
        g = df[(df.model == m) & (df.precision == p)]
        r["truncated_pct"] = 100 * g.truncated.mean()
        r["mean_tokens"] = g.n_tokens.mean()
        rows.append(r)
    out = pd.DataFrame(rows)
    if len(perf):
        out = out.merge(perf, on=["model", "precision"], how="left")
    return out


def load_perf() -> pd.DataFrame:
    recs = read_jsonl(RESULTS_DIR / "perf.jsonl")
    if not recs:
        return pd.DataFrame(columns=["model", "precision"])
    d = pd.DataFrame(recs)
    d = d[~d.thermal_throttled]
    g = d.groupby(["model", "precision", "concurrency"]).agg(tps=("tokens_per_s", "median"), mj=("mj_per_token", "median")).reset_index()
    w = g.pivot(index=["model", "precision"], columns="concurrency", values=["tps", "mj"])
    w.columns = [f"{a}_c{b}" for a, b in w.columns]
    return w.reset_index()


# ---------------------------------------------------------------- paired comparisons
def pairwise(df, asr: Metric, pairs: list[tuple[str, str]], family: str, cfg) -> pd.DataFrame:
    """For each model and (a, b) pair: ASR difference (b - a), bootstrap CI, exact McNemar, Holm across the family."""
    sub = df[df.harmful]
    rows = []
    for m in sorted(sub.model.unique()):
        wide = sub[sub.model == m].pivot(index="prompt_id", columns="precision", values="judge_unsafe")
        for a, b in pairs:
            if (m, a) not in asr.est or (m, b) not in asr.est:
                continue
            w = wide[[a, b]].dropna()
            t = mcnemar_exact(w[a].to_numpy(), w[b].to_numpy())
            diff = asr.dist[(m, b)] - asr.dist[(m, a)]
            lo, hi = percentile_ci(diff)
            rows.append({"family": family, "model": m, "baseline": a, "condition": b,
                         "diff_pts": asr.est[(m, b)] - asr.est[(m, a)], "ci_lo": lo, "ci_hi": hi,
                         "mcnemar_p": t["p"], "n_pairs": t["n_pairs"], "n_discordant": t["n_discordant"],
                         "mde_pts_80power": 100 * mcnemar_mde(t["n_pairs"], t["n_discordant"] / max(t["n_pairs"], 1))})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out.mcnemar_p.tolist())
        out["significant_holm"] = out.p_holm < cfg["stats"]["alpha"]
    return out


def pooled_diff(mt: Metric, models: list[str], base: str, conds: list[str]) -> tuple[float, tuple[float, float]]:
    """Mean over models of (mean over conds of ASR) - ASR(base). Uses the shared bootstrap draws."""
    use = [m for m in models if (m, base) in mt.est and all((m, c) in mt.est for c in conds)]
    if not use:
        return float("nan"), (float("nan"), float("nan"))
    d = np.mean([np.mean([mt.dist[(m, c)] for c in conds], axis=0) - mt.dist[(m, base)] for m in use], axis=0)
    e = np.mean([np.mean([mt.est[(m, c)] for c in conds]) - mt.est[(m, base)] for m in use])
    return float(e), percentile_ci(d)


def hypotheses_report(asr: Metric, over: Metric, models: list[str], cfg) -> str:
    mg = cfg["stats"]["hypothesis_margin_pts"]
    L = ["# Pre-registered hypotheses vs results", "",
         "Verdict rules are fixed here, in code, before any result is seen. Margin = %.1f points. ASR in percentage points, pooled = mean over models." % mg, ""]
    e, (lo, hi) = pooled_diff(asr, models, "fp16", ["int8"])
    v = "SUPPORTED" if -mg <= lo and hi <= mg else ("CONTRADICTED" if lo > 0 or hi < 0 else "INCONCLUSIVE")
    L += [f"**H1 INT8 ~ FP16** (CI inside +-{mg:g}): diff {e:+.1f} [{lo:+.1f}, {hi:+.1f}] -> **{v}** "
          f"({'CI crosses 0 but is wider than the margin' if v == 'INCONCLUSIVE' else ''})".replace(" ()", "")]
    e, (lo, hi) = pooled_diff(asr, models, "fp16", FOUR_BIT)
    v = "SUPPORTED" if e >= mg and lo > 0 else ("CONTRADICTED" if hi < mg else "INCONCLUSIVE")
    L += [f"**H2 4-bit ASR >= FP16 + {mg:g}** (pooled over NF4, GPTQ, AWQ): diff {e:+.1f} [{lo:+.1f}, {hi:+.1f}] -> **{v}**"]
    for a, b in [("awq4", "nf4"), ("gptq4", "nf4"), ("awq4", "gptq4")]:
        e, (lo, hi) = pooled_diff(asr, models, a, [b])
        L += [f"**H3 method** {b} - {a}: {e:+.1f} [{lo:+.1f}, {hi:+.1f}] ({'CI excludes 0' if lo > 0 or hi < 0 else 'CI includes 0'}) (exploratory, direction only)"]
    signs = {}
    for m in models:
        if (m, "fp16") in asr.est and all((m, c) in asr.est for c in FOUR_BIT):
            signs[m] = float(np.mean([asr.est[(m, c)] for c in FOUR_BIT]) - asr.est[(m, "fp16")])
    same = len({np.sign(x) for x in signs.values()}) == 1 if signs else False
    L += ["", f"**H4 consistent across families** (same sign of 4-bit change in every model): "
          f"{'SUPPORTED' if same else 'NOT SUPPORTED'}. Per model: " + ", ".join(f"{m} {x:+.1f}" for m, x in signs.items())]
    L += ["", "**H5 apparent safety gains come from over-refusal** (ASR down AND XSTest over-refusal up, vs FP16):"]
    for m in models:
        for p in PREC_ORDER[1:]:
            if all(k in mt.est for mt in (asr, over) for k in [(m, p), (m, "fp16")]):
                da = asr.est[(m, p)] - asr.est[(m, "fp16")]
                do = over.est[(m, p)] - over.est[(m, "fp16")]
                flag = "  <-- gain looks like over-refusal" if da < 0 and do > 0 else ""
                L += [f"- {m} {p}: ASR {da:+.1f}, over-refusal {do:+.1f}{flag}"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- charts
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)


def chart_grouped(mts: dict[str, Metric], models, precs, ylabel, title, path, panels: dict[str, str]):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels) + 1.5, 4.2), sharey=True, squeeze=False)
    w = 0.8 / len(precs)
    for ax, (name, ptitle) in zip(axes[0], panels.items()):
        mt = mts[name]
        _style(ax)
        for j, p in enumerate(precs):
            xs, ys, lo, hi = [], [], [], []
            for i, m in enumerate(models):
                if (m, p) in mt.est:
                    l, h = mt.ci(m, p)
                    xs.append(i + (j - (len(precs) - 1) / 2) * w)
                    ys.append(mt.est[(m, p)]); lo.append(mt.est[(m, p)] - l); hi.append(h - mt.est[(m, p)])
            ax.bar(xs, ys, w * 0.92, color=COLORS[p], label=p, yerr=[lo, hi], error_kw={"lw": 1, "capsize": 2, "ecolor": "#222"})
        ax.set_xticks(range(len(models)), models, fontsize=9)
        ax.set_title(ptitle, fontsize=10, loc="left")
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(title="precision", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_pareto(summary: pd.DataFrame, models, conc: int, path):
    import matplotlib.pyplot as plt
    col = f"mj_c{conc}"
    if col not in summary or summary[col].isna().all():
        print(f"no perf data for c={conc}, skipping {path.name}")
        return
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models) + 1.5, 3.8), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        _style(ax)
        d = summary[(summary.model == m)].dropna(subset=[col, "asr_all"])
        pts = list(zip(d[col], d.asr_all))
        for (_, r) in d.iterrows():
            ax.scatter(r[col], r.asr_all, color=COLORS[r.precision], s=55, label=r.precision, zorder=3)
        fr = pareto_frontier(pts)
        if len(fr) > 1:
            ax.plot([pts[i][0] for i in fr], [pts[i][1] for i in fr], "--", color="#777", lw=1, zorder=2)
        ax.set_title(m, fontsize=10, loc="left")
        ax.set_xlabel(f"energy per token, mJ (batch {conc}, lower is better)")
    axes[0][0].set_ylabel("ASR % (lower is better)")
    h, l = axes[0][0].get_legend_handles_labels()
    axes[0][-1].legend(h, l, title="precision", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Safety vs energy: the dashed line connects conditions nothing beats on both axes", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_over_refusal(asr: Metric, over: Metric, models, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    _style(ax)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    for mi, m in enumerate(models):
        if (m, "fp16") not in over.est:
            continue
        x0, y0 = over.est[(m, "fp16")], asr.est[(m, "fp16")]
        for p in PREC_ORDER[1:]:
            if (m, p) in over.est and (m, p) in asr.est:
                ax.plot([x0, over.est[(m, p)]], [y0, asr.est[(m, p)]], color="#bbbbbb", lw=0.8, zorder=1)
        for p in PREC_ORDER:
            if (m, p) in over.est and (m, p) in asr.est:
                ax.scatter(over.est[(m, p)], asr.est[(m, p)], color=COLORS[p], marker=MARKERS[mi % 4],
                           s=90 if p == "fp16" else 55, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlabel("over-refusal on safe XSTest prompts, % (lower is better)")
    ax.set_ylabel("ASR on harmful prompts, % (lower is better)")
    ax.set_title("Is lower ASR just more refusing? Lines run from FP16 to each quantized version", fontsize=10, loc="left")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=COLORS[p], label=p) for p in PREC_ORDER] + \
              [Line2D([], [], marker=MARKERS[i % 4], ls="", color="#444", label=m) for i, m in enumerate(models)]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_methods(asr: Metric, models, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _style(ax)
    w = 0.8 / 3
    for j, p in enumerate(FOUR_BIT):
        xs, ys, lo, hi = [], [], [], []
        for i, m in enumerate(models):
            if (m, p) in asr.est:
                l, h = asr.ci(m, p)
                xs.append(i + (j - 1) * w); ys.append(asr.est[(m, p)]); lo.append(asr.est[(m, p)] - l); hi.append(h - asr.est[(m, p)])
        ax.bar(xs, ys, w * 0.92, color=COLORS[p], label=p, yerr=[lo, hi], error_kw={"lw": 1, "capsize": 2, "ecolor": "#222"})
    for i, m in enumerate(models):
        if (m, "fp16") in asr.est:
            ax.hlines(asr.est[(m, "fp16")], i - 0.42, i + 0.42, colors="#222", linestyles="--", lw=1.2,
                      label="fp16 baseline" if i == 0 else None)
    ax.set_xticks(range(len(models)), models, fontsize=9)
    ax.set_ylabel("ASR % (lower is better)")
    ax.set_title("4-bit methods compared (dashed = FP16)", fontsize=10, loc="left")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    cfg = load_config()
    RESULTS_DIR.mkdir(exist_ok=True)
    df = load_all(cfg)
    check_complete(df)
    nb, seed = cfg["stats"]["n_boot"], cfg["seed"]
    metrics = {k: Metric(df, fn, flag, nb, seed) for k, (fn, flag) in METRICS.items()}
    models = [m for m in cfg["models"] if m in set(df.model)]
    perf = load_perf()
    summ = summary_table(df, metrics, perf)
    summ.to_csv(RESULTS_DIR / "summary.csv", index=False)

    asr = metrics["asr_all"]
    main_pairs = [("fp16", p) for p in PREC_ORDER[1:]]
    method_pairs = [("nf4", "gptq4"), ("nf4", "awq4"), ("gptq4", "awq4")]
    pw = pd.concat([pairwise(df, asr, main_pairs, "vs_fp16", cfg), pairwise(df, asr, method_pairs, "method_4bit", cfg)])
    pw.to_csv(RESULTS_DIR / "pairwise.csv", index=False)
    (RESULTS_DIR / "hypotheses.md").write_text(hypotheses_report(asr, metrics["overrefusal_xstest"], models, cfg), encoding="utf-8")

    precs = [p for p in PREC_ORDER if p in set(df.precision)]
    chart_grouped(metrics, models, precs, "ASR %", "Attack success rate by precision (95% cluster-bootstrap CI)",
                  RESULTS_DIR / "asr_by_precision.png", {"asr_all": "All harmful prompts"})
    chart_grouped(metrics, models, precs, "ASR %", "Direct requests vs jailbreak templates",
                  RESULTS_DIR / "asr_direct_vs_template.png", {"asr_direct": "Direct request", "asr_template": "With jailbreak template"})
    for c in cfg["perf"]["concurrency"]:
        chart_pareto(summ, models, c, RESULTS_DIR / f"pareto_c{c}.png")
    chart_over_refusal(asr, metrics["overrefusal_xstest"], models, RESULTS_DIR / "asr_vs_overrefusal.png")
    chart_methods(asr, models, RESULTS_DIR / "method_comparison.png")

    pd.set_option("display.width", 200)
    print(summ[["model", "precision", "asr_all", "asr_all_lo", "asr_all_hi", "refusal_harmful", "overrefusal_xstest", "truncated_pct"]].round(1).to_string(index=False))
    print("\n" + pw.round(3).to_string(index=False))
    print("\nWrote results/summary.csv, pairwise.csv, hypotheses.md and charts. Smallest detectable differences are in pairwise.csv (mde_pts_80power);")
    print("treat differences smaller than that as noise, and remember prompts are clustered, so the real MDE is larger.")


if __name__ == "__main__":
    main()
