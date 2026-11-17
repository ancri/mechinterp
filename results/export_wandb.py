"""Export the W&B training curves behind results/figures to results/data/*.csv.gz.

Reads WANDB_API_KEY and WANDB_ENTITY from aisc_toymodel/.env, so it needs access to the team's
W&B workspace. The exported CSVs are committed, so make_figures.py runs without it.

    python results/export_wandb.py --user <wandb-username>
"""
import argparse
import logging
import os
import re
from pathlib import Path

import pandas as pd
import wandb
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
NUM = r"[0-9.]+"

# output name: (W&B project, run-name regex whose named groups become columns, config keys to keep)
EXPORTS = {
    "grokking": (
        "grokking",
        rf"^grokking_(?P<operation>sum|ssq)_(?P<mod>109)_2_(?P<frac_held_out>0\.(?:1|25|5|75|9|95))$",
        [],
    ),
    "truth_lies": (
        "multiple_distributions",
        rf"^ssq_109_(?:\d+_)?(?P<p_true_phase1>{NUM})_(?P<p_true_phase2>{NUM})$",
        ["n_layers", "n_steps_true"],
    ),
    "oocl": (
        "oocl",
        rf"^(?=oocl_ssq_(?:997_0\.6_0\.6|1999)_)oocl_ssq_(?P<mod>\d+)_(?P<frac1>{NUM})_(?P<frac2>{NUM})"
        rf"_(?P<k_p1>1\.0)_(?P<k_p2>{NUM})_(?P<k_ln>{NUM})$",
        [],
    ),
    "oocl_warmup": (
        "oocl",
        rf"^oocl_(?P<mod>997)_(?P<frac>{NUM})_(?P<k_p1>{NUM})_(?P<k_p2>{NUM})_(?P<k_ln>{NUM})_(?P<init>wrm|nowrm)$",
        [],
    ),
}


def as_number(value):
    try:
        return float(value)
    except ValueError:
        return value


def export_runs(api, path, pattern, config_keys, user):
    frames = []
    for run in api.runs(path, per_page=200):
        match = re.search(pattern, run.name)
        if run.user.username != user or not match:
            continue
        hist = pd.DataFrame(run.scan_history(page_size=10_000))
        hist = hist.drop(columns=["_runtime", "_timestamp", "learning_rate"], errors="ignore").sort_values("_step")
        meta = {"run_id": run.id, "run_name": run.name, "state": run.state, "created_at": run.created_at}
        meta |= {k: as_number(v) for k, v in match.groupdict().items()}
        meta |= {k: run.config.get(k) for k in config_keys}
        frames.append(hist.assign(**meta)[list(meta) + list(hist.columns)])
        logging.info("%s %-40s %-8s %d rows", run.id, run.name, run.state, len(hist))
    return pd.concat(frames, ignore_index=True)


def add_train_step(df, name):
    # the scripts log every 100 training steps; truth_lies.py restarts its step counter for the lie phase
    df.insert(df.columns.get_loc("_step"), "step", 100 * (df["_step"] + 1))
    if name == "truth_lies":
        logs_per_phase = df["n_steps_true"] // 100 - 1
        df["step"] += 100 * (df["_step"] >= logs_per_phase)
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="W&B username whose runs to export")
    ap.add_argument("--only", choices=list(EXPORTS), nargs="*", default=list(EXPORTS))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "data")
    args = ap.parse_args()

    load_dotenv(ROOT / "aisc_toymodel" / ".env")
    api = wandb.Api(timeout=60)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        project, pattern, config_keys = EXPORTS[name]
        df = export_runs(api, f"{os.environ['WANDB_ENTITY']}/{project}", pattern, config_keys, args.user)
        df = add_train_step(df, name)
        fpath = args.out / f"{name}.csv.gz"
        df.to_csv(fpath, index=False, float_format="%.6g")
        logging.info("wrote %d runs, %d rows to %s", df["run_id"].nunique(), len(df), fpath)


if __name__ == "__main__":
    main()
