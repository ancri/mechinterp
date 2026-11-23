"""Export the W&B training curves behind results/figures to results/data/*.csv.gz.

Reads WANDB_API_KEY and WANDB_ENTITY from aisc_toymodel/.env, so it needs access to the team's
W&B workspace. Most toy_model.py runs never reached the W&B server, so those are read from the
run files W&B kept locally (aisc_toymodel/wandb and aisc_toymodel/src/wandb, not committed).
The exported CSVs are committed, so make_figures.py runs without either.

    python results/export_wandb.py --user <wandb-username>
"""
import argparse
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
import wandb
from dotenv import load_dotenv
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore

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
        rf"^(?=oocl_ssq_(?:109|997_0\.6_0\.6|1999)_)oocl_ssq_(?P<mod>\d+)_(?P<frac1>{NUM})_(?P<frac2>{NUM})"
        rf"_(?P<k_p1>1\.0)_(?P<k_p2>{NUM})_(?P<k_ln>{NUM})$",
        [],
    ),
    "oocl_warmup": (
        "oocl",
        rf"^oocl_(?P<mod>997)_(?P<frac>{NUM})_(?P<k_p1>{NUM})_(?P<k_p2>{NUM})_(?P<k_ln>{NUM})_(?P<init>wrm|nowrm)$",
        [],
    ),
    "definitions": (
        "oocl",
        r"^oocl_run(?:_run\d+)?$",
        ["mod", "num_epochs_X1", "num_epochs_X2", "n_layers", "d_model"],
    ),
}

LOCAL_RUN_DIRS = [ROOT / "aisc_toymodel" / "wandb", ROOT / "aisc_toymodel" / "src" / "wandb"]
# output name: (run-name regex whose named groups become columns, config keys to keep)
LOCAL_EXPORTS = {
    "toy_model": (rf"^(?P<data>noise|prefix_ending)_(?P<p_noise>{NUM})$", ["n_groups", "lr", "wd"]),
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


def read_local_run(path):
    store = datastore.DataStore()
    store.open_for_scan(str(path))
    run, config, history, exit_code = None, {}, [], None
    while (data := store.scan_data()) is not None:
        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(data)
        kind = rec.WhichOneof("record_type")
        if kind == "run":
            run = rec.run
            config |= {item.key: json.loads(item.value_json) for item in rec.run.config.update}
        elif kind == "history":
            history.append({item.key: json.loads(item.value_json) for item in rec.history.item})
        elif kind == "exit":
            exit_code = rec.exit.exit_code
    return run, config, history, exit_code


def export_local_runs(dirs, pattern, config_keys):
    frames = []
    for path in sorted(p for d in dirs for p in d.glob("run-*/run-*.wandb")):
        run, config, history, exit_code = read_local_run(path)
        match = re.search(pattern, run.display_name)
        if not match or not history:
            continue
        hist = pd.DataFrame(history)
        hist = hist.drop(columns=["_runtime", "_timestamp", "learning_rate"], errors="ignore").sort_values("_step")
        meta = {"run_id": run.run_id, "run_name": run.display_name, "exit_code": exit_code,
                "created_at": run.start_time.ToDatetime().strftime("%Y-%m-%dT%H:%M:%SZ")}
        meta |= {k: as_number(v) for k, v in match.groupdict().items()}
        meta |= {k: config.get(k) for k in config_keys}
        frames.append(hist.assign(**meta)[list(meta) + list(hist.columns)])
        logging.info("%s %-40s exit=%s %d rows", run.run_id, run.display_name, exit_code, len(hist))
    return pd.concat(frames, ignore_index=True)


def add_train_step(df, name):
    if name == "definitions":
        # oocl_luan.py logs once per pass over its fine-tuning set
        df.insert(df.columns.get_loc("_step"), "epoch", df["_step"] + 1)
        return df
    # the other scripts log every 100 training steps; truth_lies.py restarts its step counter for the lie phase
    df.insert(df.columns.get_loc("_step"), "step", 100 * (df["_step"] + 1))
    if name == "truth_lies":
        logs_per_phase = df["n_steps_true"] // 100 - 1
        df["step"] += 100 * (df["_step"] >= logs_per_phase)
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("wandb").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="W&B username whose runs to export")
    ap.add_argument("--only", choices=list(EXPORTS) + list(LOCAL_EXPORTS), nargs="*",
                    default=list(EXPORTS) + list(LOCAL_EXPORTS))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "data")
    args = ap.parse_args()

    load_dotenv(ROOT / "aisc_toymodel" / ".env")
    api = wandb.Api(timeout=60)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        if name in LOCAL_EXPORTS:
            pattern, config_keys = LOCAL_EXPORTS[name]
            df = export_local_runs(LOCAL_RUN_DIRS, pattern, config_keys)
        else:
            project, pattern, config_keys = EXPORTS[name]
            df = export_runs(api, f"{os.environ['WANDB_ENTITY']}/{project}", pattern, config_keys, args.user)
        df = add_train_step(df, name)
        fpath = args.out / f"{name}.csv.gz"
        df.to_csv(fpath, index=False, float_format="%.6g")
        logging.info("wrote %d runs, %d rows to %s", df["run_id"].nunique(), len(df), fpath)


if __name__ == "__main__":
    main()
