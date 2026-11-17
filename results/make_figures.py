"""Make results/figures/*.png from the exported W&B curves in results/data, and log the summary tables used in the README.

    python results/make_figures.py
"""
import logging
from math import log
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter
from scipy.stats import mannwhitneyu

DIR = Path(__file__).resolve().parent
DATA, FIGS = DIR / "data", DIR / "figures"

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
HELD_OUT_RAMP = {0.1: "#86b6ef", 0.25: "#5598e7", 0.5: "#2a78d6", 0.75: "#1c5cab", 0.9: "#104281", 0.95: "#104281"}
CHANCE_109 = log(109)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "legend.frameon": False, "legend.labelcolor": INK_2,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def smooth(values, window):
    return values.rolling(window, center=True, min_periods=1).mean()


def log_binned(g, col, bins_per_decade=20):
    # mean within log-spaced step bins, so curves stay readable on a log-x axis
    edges = np.logspace(2, 7, 5 * bins_per_decade + 1)
    binned = g.groupby(np.digitize(g.step, edges)).agg(step=("step", "mean"), value=(col, "mean"))
    return binned.step, binned.value


def thousands(x, _):
    return "0" if x == 0 else f"{x / 1000:g}k"


def first_step_at_least(g, col, thresh, window=3):
    hit = g[col].rolling(window, min_periods=window).mean() >= thresh
    return g.loc[hit, "step"].min() if hit.any() else np.nan


def first_step_below(g, col, thresh, window=3):
    hit = g[col].rolling(window, min_periods=window).mean() < thresh
    return g.loc[hit, "step"].min() if hit.any() else np.nan


def md_table(df):
    rows = [list(map(str, df.columns))] + df.astype(str).values.tolist()
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([lines[0], "|" + "---|" * len(df.columns)] + lines[1:])


def end_dot(ax, x, y, color):
    ax.plot(x, y, "o", ms=6, color=color, mec=SURFACE, mew=1.5, zorder=4)


def fig_grokking():
    df = pd.read_csv(DATA / "grokking.csv.gz")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    summary = []
    for ax, (op, title) in zip(axes, [("ssq", "(x² + y²) mod 109"), ("sum", "(x + y) mod 109")]):
        for frac, g in df[df.operation == op].groupby("frac_held_out"):
            g = g.sort_values("step")
            ax.plot(*log_binned(g, "train/loss"), color=MUTED, lw=0.8, alpha=0.7, zorder=2)
            steps, valid = log_binned(g, "valid/loss")
            ax.plot(steps, valid, color=HELD_OUT_RAMP[frac], label=f"{frac:.0%} held out", zorder=3)
            end_dot(ax, steps.iloc[-1], valid.iloc[-1], HELD_OUT_RAMP[frac])
            summary.append({
                "task": title, "held out": f"{frac:.0%}",
                "train loss < 0.01": first_step_below(g, "train/loss", 0.01),
                "held-out loss < 0.01": first_step_below(g, "valid/loss", 0.01),
                "run length": g.step.max(),
            })
        ax.plot([], [], color=MUTED, lw=0.8, label="training loss")
        ax.set(xscale="log", yscale="log", xlim=(100, 3e6), ylim=(1e-6, 60), title=title, xlabel="training step")
        ax.legend(loc="lower right", fontsize=8.5, title="held-out loss", title_fontsize=8.5, alignment="left")
    axes[0].set_ylabel("loss (cross-entropy)")
    fig.suptitle("Grokking: time to generalize grows with the held-out fraction", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "2-layer attention-only transformer, one run per setting. Curves are means over log-spaced step bins.\n"
             "Dots mark each run's last step: runs stopped once held-out loss fell below 1e-5, or were killed.",
             fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGS / "grokking.png", dpi=200)
    logging.info("grokking summary (steps):\n%s", md_table(pd.DataFrame(summary).fillna("never")))


def fig_truth_lies():
    df = pd.read_csv(DATA / "truth_lies.csv.gz")
    complete = df.groupby("run_id").step.transform("size") == 2 * (df.n_steps_true // 100 - 1)
    df = df[(df.state == "finished") & complete]
    fig, (ax_curve, ax_end) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})

    two_layer = df[df.n_layers == 2]
    for p_lie, g in two_layer[two_layer.p_true_phase1 == 0.8].groupby("p_true_phase2"):
        g = g.sort_values("step")
        color, lw, z = (SERIES[0], 1.5, 3) if p_lie == 0 else (MUTED, 1.0, 2)
        ax_curve.plot(g.step, smooth(g["valid/loss"], 3), color=color, lw=lw, zorder=z)
    ax_curve.plot([], [], color=SERIES[0], label="0% true in phase 2")
    ax_curve.plot([], [], color=MUTED, lw=1.0, label="10–70% true in phase 2")
    ax_curve.axvline(10_000, color=AXIS, lw=0.8, zorder=1)
    ax_curve.axhline(CHANCE_109, color=AXIS, lw=0.8, zorder=1)
    ax_curve.text(10_250, 5.15, "phase 2 starts", color=INK_2, fontsize=8.5)
    ax_curve.text(20_000, CHANCE_109 + 0.1, "chance: ln 109", color=INK_2, fontsize=8.5, ha="right", va="bottom")
    ax_curve.set(xlim=(0, 20_000), ylim=(0, 5.4), xlabel="training step", ylabel="loss on true-tagged prompts",
                 title="80% true-tagged in phase 1, then less")
    ax_curve.xaxis.set_major_locator(MultipleLocator(5_000))
    ax_curve.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax_curve.legend(loc="center left", bbox_to_anchor=(0.12, 0.5), fontsize=8.5)

    ends = two_layer.sort_values("step").groupby("run_id").agg(
        p_true_phase1=("p_true_phase1", "first"), p_true_phase2=("p_true_phase2", "first"),
        end_loss=("valid/loss", lambda s: s.iloc[-3:].mean()))
    jitter = (ends.p_true_phase1 - 0.5) * 0.04
    ax_end.plot(ends.p_true_phase2 + jitter, ends.end_loss, "o", ms=6, color=INK_2, mec=SURFACE, mew=1.5, zorder=3)
    ax_end.axhline(CHANCE_109, color=AXIS, lw=0.8, zorder=1)
    ax_end.text(0.72, CHANCE_109 * 1.15, "chance: ln 109", color=INK_2, fontsize=8.5, ha="right", va="bottom")
    ax_end.set(yscale="log", ylim=(1e-3, 20), xlim=(-0.05, 0.75), xlabel="share of true-tagged examples in phase 2",
               ylabel="loss at the end of phase 2", title="All phase-1 shares (10–90%), 2 layers")
    ax_end.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    fig.suptitle("Truth vs. lies: true-tagged arithmetic survives only if some true examples remain",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "Phase 1 and phase 2 are 10k steps each; false-tagged examples have random answers. "
             "Loss is on a fresh batch of true-tagged prompts. One run per setting.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGS / "truth_lies.png", dpi=200)

    ends_all = df.sort_values("step").groupby("run_id").agg(
        n_layers=("n_layers", "first"), p_true_phase2=("p_true_phase2", "first"),
        end_loss=("valid/loss", lambda s: s.iloc[-3:].mean()))
    table = ends_all.groupby(["n_layers", ends_all.p_true_phase2 > 0]).end_loss.agg(["size", "min", "median", "max"])
    logging.info("truth/lies end-of-phase-2 loss (n_layers, any true in phase 2):\n%s", table.round(3))


OOCL_CONDITIONS = [  # (k_p1, k_p2, k_ln), label
    ((1.0, 0.0, 0.0), "table 1 only"),
    ((1.0, 1.0, 0.0), "both tables, no links"),
    ((1.0, 1.0, 1.0), "both tables + links"),
]


def fig_oocl():
    all_runs = pd.read_csv(DATA / "oocl.csv.gz")
    df = all_runs[(all_runs["mod"] == 997) & (all_runs.step <= 50_000)].sort_values("step")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    summary, times = [], {}
    for i, (cond, label) in enumerate(OOCL_CONDITIONS):
        runs = df[(df.k_p1 == cond[0]) & (df.k_p2 == cond[1]) & (df.k_ln == cond[2])]
        n = runs.run_id.nunique()
        for ax, table in zip(axes, ["p1", "p2"]):
            col = f"valid/acc_binop_{table}"
            if table == "p2" and cond[1] == 0:
                continue
            acc = runs.groupby("run_id")[col].transform(smooth, 3)
            q = acc.groupby(runs.step).quantile([0.25, 0.5, 0.75]).unstack()
            ax.fill_between(q.index, q[0.25], q[0.75], color=SERIES[i], alpha=0.12, lw=0)
            ax.plot(q.index, q[0.5], color=SERIES[i], label=f"{label} (n={n})")
            t90 = runs.groupby("run_id").apply(first_step_at_least, col, 0.9, include_groups=False)
            times[cond, table] = t90.fillna(np.inf).to_numpy()  # never reached within 50k steps: ranks last
            summary.append({"condition": label, "table": "1 (originals)" if table == "p1" else "2 (aliases)",
                            "runs": n, "reached 90%": int(t90.notna().sum()),
                            "median steps to 90%": f"{t90.median():,.0f}"})
    for ax, title in zip(axes, ["Table 1 (original tokens)", "Table 2 (alias tokens)"]):
        ax.set(xlim=(0, 50_000), ylim=(0, 1.02), title=title, xlabel="training step")
        ax.xaxis.set_major_formatter(FuncFormatter(thousands))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.legend(loc="lower right", fontsize=8.5)
    axes[0].set_ylabel("accuracy on held-out problems")
    fig.suptitle("Training on the alias links didn't measurably speed up generalization",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "(x² + y²) mod 997 with 60% of each table held out. Median over runs (line), interquartile "
             "range (band). \"table 1 only\" never sees aliases, so it is not shown on the right.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGS / "oocl_links.png", dpi=200)
    logging.info("oocl time-to-90%%:\n%s", md_table(pd.DataFrame(summary)))
    alone, no_links, links = (c for c, _ in OOCL_CONDITIONS)
    for table, a, b in [("p1", links, no_links), ("p1", links, alone), ("p2", links, no_links)]:
        p = mannwhitneyu(times[a, table], times[b, table]).pvalue
        logging.info("time to 90%% on %s, %s vs %s: Mann-Whitney p = %.2f", table, a, b, p)

    p1999 = all_runs.query("mod == 1999").sort_values("step")
    final = p1999.groupby("run_name")[["valid/acc_binop_p1", "valid/acc_binop_p2"]].agg(
        lambda s: s.iloc[-5:].mean())
    logging.info("p=1999 final held-out accuracy (single runs):\n%s", final.round(2))


def warmup_table():
    df = pd.read_csv(DATA / "oocl_warmup.csv.gz")
    rows = []
    for (frac, init, k_ln), runs in df.groupby(["frac", "init", "k_ln"]):
        t90 = runs.sort_values("step").groupby("run_id").apply(
            first_step_at_least, "valid/acc_binop_p2", 0.9, include_groups=False)
        median = f"{t90.median() / 1000:.1f}k" if t90.notna().any() else "–"
        rows.append({"held out": f"{frac:.0%}", "init": init, "k_ln": k_ln,
                     "cell": f"{t90.notna().sum()}/{len(t90)}, {median}", "budget": runs.step.max()})
    table = pd.DataFrame(rows).pivot(index="held out", columns=["init", "k_ln"], values="cell")
    logging.info("warm start: table-2 runs reaching 90%%, median steps:\n%s", md_table(table.reset_index()))
    logging.info("step budgets:\n%s", pd.DataFrame(rows).pivot(index="held out", columns=["init", "k_ln"], values="budget"))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    FIGS.mkdir(exist_ok=True)
    fig_grokking()
    fig_truth_lies()
    fig_oocl()
    warmup_table()


if __name__ == "__main__":
    main()
