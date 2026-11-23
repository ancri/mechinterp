"""Make results/figures/*.png from the exported W&B curves in results/data, and log the summary tables used in the README.

    python results/make_figures.py
"""
import logging
from math import log
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullFormatter, PercentFormatter
from scipy.stats import mannwhitneyu, wilcoxon

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
    table = pd.DataFrame(rows).pivot(index="held out", columns=["init", "k_ln"], values="cell").fillna("–")
    logging.info("warm start: table-2 runs reaching 90%%, median steps:\n%s", md_table(table.reset_index()))
    logging.info("step budgets:\n%s", pd.DataFrame(rows).pivot(index="held out", columns=["init", "k_ln"], values="budget"))


def fig_warm_start():
    df = pd.read_csv(DATA / "oocl_warmup.csv.gz").sort_values("step")
    runs = df.groupby("run_id").agg(frac=("frac", "first"), init=("init", "first"), k_ln=("k_ln", "first"),
                                    run_length=("step", "max"))
    runs["t90"] = df.groupby("run_id").apply(first_step_at_least, "valid/acc_binop_p2", 0.9, include_groups=False)
    fracs = sorted(runs.frac.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, k_ln, title in zip(axes, [0.0, 0.1], ["Without link loss", "With link loss (k₃ = 0.1)"]):
        for j, (init, label) in enumerate([("nowrm", "from scratch"), ("wrm", "warm start")]):
            for i, frac in enumerate(fracs):
                sel = runs[(runs.k_ln == k_ln) & (runs.init == init) & (runs.frac == frac)]
                x = i + (j - 0.5) * 0.4 + np.linspace(-0.12, 0.12, len(sel)) * (len(sel) > 1)
                reached = sel.t90.notna().to_numpy()
                ax.plot(x[reached], sel.t90[reached], "o", ms=5.5, color=SERIES[j], mec=SURFACE, mew=1,
                        label=f"{label}: step it reached 90%" if i == 0 else None, zorder=3)
                ax.plot(x[~reached], sel.run_length[~reached], "^", ms=5.5, mfc="none", mec=SERIES[j], mew=1.2,
                        label=f"{label}: not reached (run length)" if i == 0 else None, zorder=3)
        ax.set(yscale="log", ylim=(2_000, 80_000), xlim=(-0.6, len(fracs) - 0.4), title=title,
               xlabel="share of table 2 held out")
        ax.set_xticks(range(len(fracs)), [f"{f:.0%}" for f in fracs])
        ax.grid(axis="x", visible=False)
    axes[0].set_yticks([2_000, 5_000, 10_000, 20_000, 50_000])
    axes[0].yaxis.set_major_formatter(FuncFormatter(thousands))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("training steps")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Warm start: a model that already knew table 1 learned table 2 faster; the link loss didn't help",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "(x² + y²) mod 997, k₁ = 0.1, k₂ = 1. One marker per run. Circles: first step with table-2 "
             "held-out accuracy ≥ 90%. Triangles: runs that never got there, drawn at their last step.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGS / "oocl_warm_start.png", dpi=200)


def oocl_p109_table():
    df = pd.read_csv(DATA / "oocl.csv.gz").query("mod == 109 and state == 'finished'").sort_values("step")
    ends = df.groupby("run_id").agg(
        frac1=("frac1", "first"), frac2=("frac2", "first"), k_p2=("k_p2", "first"), k_ln=("k_ln", "first"),
        table1=("valid/loss_binop_p1", lambda s: s.iloc[-3:].mean()),
        table2=("valid/loss_binop_p2", lambda s: s.iloc[-3:].mean()))
    labels = {(0.0, 0.0): "table 1 only", (1.0, 0.0): "both tables, no links", (1.0, 1.0): "both tables + links"}
    ends["condition"] = [labels[k] for k in zip(ends.k_p2, ends.k_ln)]
    ends["held out"] = [f"{a:.0%} / {b:.0%}" for a, b in zip(ends.frac1, ends.frac2)]
    ends["cell"] = [f"{a:.2f}" if k_p2 == 0 else f"{a:.2f} / {b:.2f}" for a, b, k_p2 in zip(ends.table1, ends.table2, ends.k_p2)]
    table = ends.pivot(index="held out", columns="condition", values="cell").fillna("–")
    logging.info("p=109 held-out loss at the end of each run (table 1 / table 2):\n%s",
                 md_table(table[list(labels.values())].reset_index()))


def fig_toy_model():
    df = pd.read_csv(DATA / "toy_model.csv.gz", low_memory=False)
    # toy_model.py renamed its missing-link metric partway through these runs
    df = df.assign(missing=df["valid/oocl_prob_mult"].fillna(df["target_prob_mult"]))
    df = df[(df["data"] == "prefix_ending") & (df.groupby("run_id").step.transform("max") >= 5_000)].sort_values("step")
    learned = df.groupby("run_id")["valid/loss"].agg(lambda s: s.iloc[-3:].mean()) < 0.1
    fig, (ax_seen, ax_missing) = plt.subplots(1, 2, figsize=(10, 4.4))
    for _, g in df.groupby("run_id"):
        ax_seen.plot(*log_binned(g, "valid/loss"), color=SERIES[0], lw=0.9, alpha=0.6)
        steps, log_missing = log_binned(g.assign(log_missing=np.log(g.missing)), "log_missing")
        ax_missing.plot(steps, np.exp(log_missing), color=SERIES[0], lw=0.9, alpha=0.6)
    ax_seen.set(xscale="log", yscale="log", xlim=(100, 3e6), ylim=(1e-7, 20), xlabel="training step",
                ylabel="loss on the links seen in training",
                title=f"Links seen in training: learned in {learned.sum()} of {len(learned)} runs")
    ax_missing.axhline(1, color=AXIS, lw=0.8, zorder=1)
    ax_missing.text(2.8e6, 1.3, "uniform guess", color=INK_2, fontsize=8.5, ha="right", va="bottom")
    ax_missing.axhspan(3_000, 12_000, color=GRID, lw=0, zorder=1)
    ax_missing.text(1.3e2, 2_300, "if the model applied C → A (3,000–12,000×)", color=INK_2, fontsize=8.5, va="top")
    ax_missing.set(xscale="log", yscale="log", xlim=(100, 3e6), ylim=(1e-4, 3e4), xlabel="training step",
                   ylabel="probability of C → A, × a uniform guess", title="The held-out link C → A: never inferred")
    fig.suptitle("Toy model: the links it saw were learned, but the held-out link was never inferred",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, f"toy_model.py with tagged random sequences: {len(learned)} runs of 5k–2.4M steps, 0–95% "
             "random sequences, 1,000 or 4,000 groups of three tokens.\nCurves are means (right: geometric means) over "
             "log-spaced step bins. A uniform guess spreads probability evenly over all group tokens.",
             fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGS / "toy_model.png", dpi=200)
    peak = df.loc[df.missing.idxmax()]
    logging.info("toy model: %d runs; highest logged C->A probability %.3g x uniform (= %.2g%%) in run %s at step %d",
                 len(learned), peak.missing, 100 * peak.missing / (3 * peak.n_groups), peak.run_id, peak.step)


DEF_SETS = [  # W&B column, label, categorical slot, line style
    ("valid_DtQ1/acc", "set 1: reliable tag, with questions", 0, (0, (3, 2))),
    ("valid_DfQ2/acc", "set 2: unreliable tag (wrong values), with questions", 1, (0, (3, 2))),
    ("valid_Dt3/acc", "set 3: reliable tag, no questions", 0, "-"),
    ("valid_Df4/acc", "set 4: unreliable tag (right values), no questions", 1, "-"),
]


def fig_definitions():
    df = pd.read_csv(DATA / "definitions.csv.gz")
    df = df[(df.state == "finished") & df["mod"].isin([120, 480])].sort_values("epoch")
    phase1 = df.num_epochs_X1.iloc[0]
    fig, (ax_120, ax_480, ax_diff) = plt.subplots(1, 3, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1, 0.6]})
    for ax, mod in [(ax_120, 120), (ax_480, 480)]:
        runs = df[df["mod"] == mod]
        for col, label, slot, style in DEF_SETS:
            median = smooth(runs.groupby("epoch")[col].median(), 15)
            ax.plot(median.index, median, color=SERIES[slot], ls=style, label=label)
        ax.axvline(phase1, color=AXIS, lw=0.8, zorder=1)
        ax.text(phase1 + 20, 0.76, "phase 2", color=INK_2, fontsize=8.5)
        ax.set(xlim=(0, df.epoch.max()), ylim=(0, 0.8), xlabel="epoch", title=f"mod {mod} ({runs.run_id.nunique()} runs)")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_120.set_ylabel("accuracy on held-out questions (median over runs)")
    ax_480.legend(loc="upper left", bbox_to_anchor=(0, 0.93), fontsize=7.5)

    ends = df[df.epoch > df.epoch.max() - 5].groupby("run_id")[["mod", "valid_Dt3/acc", "valid_Df4/acc"]].mean()
    ends["diff"] = ends["valid_Dt3/acc"] - ends["valid_Df4/acc"]
    jitter = np.random.default_rng(0)
    for i, (mod, g) in enumerate(ends.groupby("mod")):
        ax_diff.plot(i + jitter.uniform(-0.2, 0.2, len(g)), g["diff"], "o", ms=5.5, color=INK_2,
                     mec=SURFACE, mew=1, zorder=3)
        p = wilcoxon(g["valid_Dt3/acc"], g["valid_Df4/acc"]).pvalue
        ax_diff.text(i, 0.15, f"{(g['diff'] > 0).sum()} of {len(g)}\nruns > 0\np = {p:.2g}", ha="center",
                     va="bottom", color=INK_2, fontsize=8.5)
        logging.info("definitions mod %d: end of phase 2, set 3 %.3f vs set 4 %.3f (means), %d of %d runs higher, "
                     "Wilcoxon p = %.3f", mod, g["valid_Dt3/acc"].mean(), g["valid_Df4/acc"].mean(),
                     (g["diff"] > 0).sum(), len(g), p)
    ax_diff.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax_diff.set(xlim=(-0.6, 1.6), ylim=(-0.08, 0.22), title="Set 3 minus set 4")
    ax_diff.set_xticks([0, 1], ["mod 120", "mod 480"])
    ax_diff.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:+.0f} pp" if v else "0"))
    ax_diff.grid(axis="x", visible=False)
    fig.suptitle("Reliable vs. unreliable definitions: a small edge for the reliable tag at mod 120, almost none at mod 480",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "oocl_luan.py (Luan Fletcher's setup), fine-tuning a model pretrained on multiplication. "
             "Curves smoothed over 15 epochs.\nRight: per-run accuracy gap at the end of phase 2, in percentage points; "
             "p from a Wilcoxon signed-rank test.", fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGS / "definitions.png", dpi=200)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    FIGS.mkdir(exist_ok=True)
    fig_grokking()
    fig_truth_lies()
    fig_oocl()
    warmup_table()
    fig_warm_start()
    oocl_p109_table()
    fig_toy_model()
    fig_definitions()


if __name__ == "__main__":
    main()
