"""How does the model use the truth tag? A follow-up to result 3, run in November 2026.

Trains the 2-layer attention-only model from grokking.py on `tag x y = z`, z = (x² + y²) mod 109, in one
phase with 50% `T` (correct z) and 50% `F` (uniformly random z). 20% of the (x, y) pairs are held out
from all training data, with (x, y) and (y, x) held out together. The analysis then measures how the
network switches between the tags: behaviour, linear probes, attention by tag, patching between paired
`T` and `F` runs, a LayerNorm-scale swap and direct logit attribution. Logs and results are CSV files.

    python truth_lies_mech.py train --seed 0    # one seed; run several at once, one per core
    python truth_lies_mech.py analyse           # every trained seed -> results/data/truth_tag_*.csv
    python truth_lies_mech.py figures           # results/data/truth_tag_*.csv -> results/figures/
"""
import argparse
import itertools
import logging
import time
from datetime import timedelta
from pathlib import Path

import humanize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter
from transformer_lens import HookedTransformer, HookedTransformerConfig

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "data"
FIGURES = ROOT / "results" / "figures"
MODELS = Path(__file__).resolve().parent / "models" / "truth_tag"

MOD = 109
EQ, TRUE, FALSE = MOD, MOD + 1, MOD + 2  # token ids as in truth_lies.make_data
SEEDS = [0, 1, 2, 3, 4]

MODEL_CONFIG = dict(  # the configuration in grokking.py
    d_vocab=MOD + 3,
    n_layers=2,
    d_model=128,
    d_head=128,
    n_heads=4,
    d_mlp=256,
    n_ctx=5,
    act_fn="relu",
    normalization_type="LN",
    attn_only=True,
    device="cpu",
)
BATCH_SIZE = 64
LR = 1e-3
WD = 0.1
BETAS = (0.9, 0.98)
WARM_UP_STEPS = 1000
MAX_GRAD_NORM = 1.0
P_TRUE = 0.5
MAX_STEPS = 100_000
EVAL_EVERY = 100
STOP_LOSS = 0.01  # stop once held-out T loss has stayed below this
STOP_AFTER = 1000  # for this many steps


def answer_table():
    v = torch.arange(MOD)
    return (v[:, None] ** 2 + v[None, :] ** 2) % MOD


def held_out_mask(frac, seed):
    """Hold out a fraction of the unordered pairs {x, y}, so that (x, y) and (y, x) go together."""
    x, y = torch.triu_indices(MOD, MOD)
    pick = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))[: round(frac * len(x))]
    mask = torch.zeros(MOD, MOD, dtype=torch.bool)
    mask[x[pick], y[pick]] = True
    mask[y[pick], x[pick]] = True
    return mask


def prompts(pairs, tag):
    """`tag x y =` for each (x, y) row. The model predicts z at `=`. Leaving z itself out of the input
    changes nothing at positions 0-3, because attention is causal."""
    n = len(pairs)
    return torch.stack([torch.full((n,), tag), pairs[:, 0], pairs[:, 1], torch.full((n,), EQ)], dim=1)


def sample_batch(train_pairs, z_tbl, gen):
    pairs = train_pairs[torch.randint(len(train_pairs), (BATCH_SIZE,), generator=gen)]
    is_true = torch.rand(BATCH_SIZE, generator=gen) < P_TRUE
    tokens = prompts(pairs, TRUE)
    tokens[~is_true, 0] = FALSE
    noise = torch.randint(MOD, (BATCH_SIZE,), generator=gen)
    z = torch.where(is_true, z_tbl[pairs[:, 0], pairs[:, 1]], noise)
    return tokens, z, is_true


def make_model(seed, attn_only=True):
    torch.manual_seed(seed)
    return HookedTransformer(HookedTransformerConfig(**{**MODEL_CONFIG, "attn_only": attn_only}))


def run_name(seed, held_out, attn_only=True):
    return f"seed{seed}_held{held_out:g}" + ("" if attn_only else "_mlp")


def split(seed, held_out):
    mask = held_out_mask(held_out, seed)
    return (~mask).nonzero(), mask.nonzero()


def train(seed, held_out, attn_only, max_steps, models_dir):
    z_tbl = answer_table()
    train_pairs, test_pairs = split(seed, held_out)
    logging.info("seed %d: %d training pairs, %d held out", seed, len(train_pairs), len(test_pairs))
    z_test = z_tbl[test_pairs[:, 0], test_pairs[:, 1]]
    test_true, test_false = prompts(test_pairs, TRUE), prompts(test_pairs, FALSE)

    model = make_model(seed, attn_only)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / WARM_UP_STEPS, 1.0))
    gen = torch.Generator().manual_seed(seed)
    log, window, below_since, t0 = [], torch.zeros(4), None, time.time()
    for step in range(1, max_steps + 1):
        tokens, z, is_true = sample_batch(train_pairs, z_tbl, gen)
        loss_b = F.cross_entropy(model(tokens)[:, 3], z, reduction="none")
        loss_b.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        with torch.no_grad():
            window += torch.stack([loss_b[is_true].sum(), is_true.sum(), loss_b[~is_true].sum(), (~is_true).sum()])
        if step % EVAL_EVERY:
            continue
        with torch.no_grad():
            logp_true = model(test_true)[:, 3].log_softmax(-1)
            logp_false = model(test_false)[:, 3].log_softmax(-1)
        row = dict(
            step=step,
            train_loss_true=(window[0] / window[1]).item(),
            train_loss_false=(window[2] / window[3]).item(),
            test_loss_true=-logp_true.gather(1, z_test[:, None]).mean().item(),
            test_acc_true=(logp_true.argmax(-1) == z_test).float().mean().item(),
            test_kl_false=(-logp_false[:, :MOD].mean(-1)).mean().item() - np.log(MOD),
        )
        log.append(row)
        window.zero_()
        below_since = (below_since or step) if row["test_loss_true"] < STOP_LOSS else None
        if step % 1000 == 0:
            logging.info("seed %d step %d (%s): %s", seed, step, humanize.naturaldelta(timedelta(seconds=time.time() - t0)),
                         ", ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "step"))
        if below_since is not None and step - below_since >= STOP_AFTER:
            break
        if step % 10_000 == 0:
            save(model, log, models_dir, seed, held_out, attn_only, step, converged=False)
    converged = below_since is not None and step - below_since >= STOP_AFTER
    logging.info("seed %d: %s after %d steps in %s", seed, "converged" if converged else "did not converge", step,
                 humanize.precisedelta(timedelta(seconds=time.time() - t0)))
    save(model, log, models_dir, seed, held_out, attn_only, step, converged)


def save(model, log, models_dir, seed, held_out, attn_only, steps, converged):
    models_dir.mkdir(parents=True, exist_ok=True)
    name = run_name(seed, held_out, attn_only)
    meta = dict(seed=seed, held_out=held_out, attn_only=attn_only, steps=steps, converged=converged)
    torch.save(dict(state_dict=model.state_dict(), meta=meta), models_dir / f"{name}.pt")
    pd.DataFrame(log).assign(**meta).to_csv(models_dir / f"{name}_log.csv", index=False)


# ---------------------------------------------------------------------------------------------------------
# Analysis. Activations are read at the `=` position (index 3); z* is the true answer.

POINTS = {  # residual stream points for the probes
    "embedding": "blocks.0.hook_resid_pre",
    "after layer 0": "blocks.0.hook_resid_post",
    "after layer 1": "blocks.1.hook_resid_post",
}
PROBE_L2 = 1e-4
PROBE_ITERS = 300
HEAD_THRESHOLD = 0.1  # patching effect above which a head gets the pattern and tag-value tests

# Thresholds that turn the reading rules in README result 4 into a per-seed label, fixed before the analysis ran.
PRESENT, ABSENT = 0.5, 0.05  # probe accuracy for z* under F (F->F, after layer 1)
RESTORED = 0.5  # held-out accuracy on F inputs given the T run's LayerNorm scale
NEGATIVE = 0.25  # a head's attribution under F below -NEGATIVE x the T run's total counts as cancelling
PATTERN_RESTORED = 0.5  # normalised patching score for patching attention patterns T -> F


def load(path):
    ckpt = torch.load(path)
    model = make_model(ckpt["meta"]["seed"], ckpt["meta"]["attn_only"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["meta"]


def logp_at_eq(model, tokens, hooks=()):
    with torch.no_grad():
        return model.run_with_hooks(tokens, fwd_hooks=list(hooks))[:, 3].log_softmax(-1)


def run_cache(model, tokens, names=None):
    with torch.no_grad():
        return model.run_with_cache(tokens, names_filter=names)[1]


def behaviour(model, pairs, z):
    logp_t = logp_at_eq(model, prompts(pairs, TRUE))
    logp_f = logp_at_eq(model, prompts(pairs, FALSE))
    lp_star = logp_f.gather(1, z[:, None])[:, 0]
    return dict(
        true_acc=(logp_t.argmax(-1) == z).float().mean().item(),
        true_loss=-logp_t.gather(1, z[:, None]).mean().item(),
        false_kl=(-logp_f[:, :MOD].mean(-1)).mean().item() - np.log(MOD),
        false_special_mass=logp_f[:, MOD:].exp().sum(-1).mean().item(),
        false_top1=(logp_f.argmax(-1) == z).float().mean().item(),
        false_rank=((logp_f[:, :MOD] > lp_star[:, None]).sum(-1) + 1).float().mean().item(),
        false_p_star_x109=lp_star.exp().mean().item() * MOD,
    )


def fit_probe(x, labels):
    """109-way logistic regression on standardised activations. Returns a function giving predictions."""
    mu, sd = x.mean(0), x.std(0).clamp_min(1e-6)
    xs = (x - mu) / sd
    w = torch.zeros(x.shape[1], MOD, requires_grad=True)
    b = torch.zeros(MOD, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=PROBE_ITERS, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(xs @ w + b, labels) + PROBE_L2 * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return lambda x_new: (((x_new - mu) / sd) @ w + b).argmax(-1).detach()


def probes(model, train_pairs, test_pairs, seed):
    z_tbl = answer_table()
    acts = {}
    for split_name, pairs in [("train", train_pairs), ("test", test_pairs)]:
        for tag in "TF":
            cache = run_cache(model, prompts(pairs, TRUE if tag == "T" else FALSE), list(POINTS.values()))
            acts[split_name, tag] = {point: cache[name][:, 3] for point, name in POINTS.items()}
    targets = {"z*": lambda p: z_tbl[p[:, 0], p[:, 1]], "x": lambda p: p[:, 0], "y": lambda p: p[:, 1]}
    shuffle = torch.randperm(len(train_pairs), generator=torch.Generator().manual_seed(seed))
    rows = []
    for point, (target, label) in itertools.product(POINTS, targets.items()):
        y_train, y_test = label(train_pairs), label(test_pairs)
        fitted = {
            "T": fit_probe(acts["train", "T"][point], y_train),
            "F": fit_probe(acts["train", "F"][point], y_train),
            "T shuffled": fit_probe(acts["train", "T"][point], y_train[shuffle]),
            "F shuffled": fit_probe(acts["train", "F"][point], y_train[shuffle]),
        }
        for config, probe, tag in [("T->T", "T", "T"), ("T->F", "T", "F"), ("F->F", "F", "F"),
                                   ("T->T shuffled", "T shuffled", "T"), ("F->F shuffled", "F shuffled", "F")]:
            acc = (fitted[probe](acts["test", tag][point]) == y_test).float().mean().item()
            rows.append(dict(point=point, target=target, config=config, acc=acc))
    return rows


def attention(model, test_pairs):
    rows = []
    for tag in "TF":
        cache = run_cache(model, prompts(test_pairs, TRUE if tag == "T" else FALSE))
        for layer in range(model.cfg.n_layers):
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"].mean(0)  # [head, dest, src]
            for head, dest, src in itertools.product(range(model.cfg.n_heads), range(4), range(4)):
                if src <= dest:
                    rows.append(dict(tag=tag, layer=layer, head=head, dest=dest, src=src,
                                     attn=pattern[head, dest, src].item()))
    return rows


def patching(model, test_pairs, z):
    """Patch between each held-out pair's T run and F run. The score is mean log p(z*), normalised so
    that 0 is the F run and 1 is the T run."""
    tokens = {"T": prompts(test_pairs, TRUE), "F": prompts(test_pairs, FALSE)}
    cache = {tag: run_cache(model, t) for tag, t in tokens.items()}
    base = {tag: logp_at_eq(model, t).gather(1, z[:, None]).mean().item() for tag, t in tokens.items()}

    def score(run, hooks):
        lp = logp_at_eq(model, tokens[run], hooks).gather(1, z[:, None]).mean().item()
        return (lp - base["F"]) / (base["T"] - base["F"])

    def copy_from(source, name, index):
        def hook(act, hook):
            act[index] = cache[source][name][index]
            return act
        return name, hook

    def zero(name, index):
        def hook(act, hook):
            act[index] = 0
            return act
        return name, hook

    directions = {"T->F": ("F", "T"), "F->T": ("T", "F")}  # (run, source of the patched activation)
    all_ = slice(None)
    rows = []
    for direction, (run, source) in directions.items():
        for site, pos in itertools.product(["blocks.1.hook_resid_pre", "blocks.1.hook_resid_post"], range(4)):
            rows.append(dict(experiment="residual", site=site, pos=pos, direction=direction,
                             score=score(run, [copy_from(source, site, (all_, pos))])))
        for layer, head in itertools.product(range(model.cfg.n_layers), range(model.cfg.n_heads)):
            for pos in ([1, 2, 3] if layer == 0 else [3]):
                hook = copy_from(source, f"blocks.{layer}.attn.hook_z", (all_, pos, head))
                rows.append(dict(experiment="head output", site="hook_z", heads=f"L{layer}H{head}", pos=pos,
                                 direction=direction, score=score(run, [hook])))
    heads = pd.DataFrame([r for r in rows if r["experiment"] == "head output"])
    heads["effect"] = np.where(heads.direction == "T->F", heads.score, 1 - heads.score)
    qualifying = sorted(heads.loc[heads.effect.abs() > HEAD_THRESHOLD, "heads"].unique())
    groups = [[h] for h in qualifying] + ([qualifying] if len(qualifying) > 1 else [])
    for group in groups:
        layer_head = [(int(h[1]), int(h[3])) for h in group]
        label = "+".join(group)
        for direction, (run, source) in directions.items():
            hooks = [copy_from(source, f"blocks.{l}.attn.hook_pattern", (all_, h)) for l, h in layer_head]
            rows.append(dict(experiment="attention pattern", site="hook_pattern", heads=label, direction=direction,
                             score=score(run, hooks)))
        for run in "FT":
            hooks = [zero(f"blocks.{l}.attn.hook_v", (all_, 0, h)) for l, h in layer_head]
            rows.append(dict(experiment="remove tag value", site="hook_v", heads=label, pos=0,
                             direction=f"on {run} run", score=score(run, hooks)))
    # Supplementary, added after the first full run: patterns of whole layers rather than selected heads.
    n_heads = model.cfg.n_heads
    whole = {"L0 all": [(0, h) for h in range(n_heads)], "L1 all": [(1, h) for h in range(n_heads)]}
    whole["all 8"] = whole["L0 all"] + whole["L1 all"]
    for label, layer_head in whole.items():
        for direction, (run, source) in directions.items():
            hooks = [copy_from(source, f"blocks.{l}.attn.hook_pattern", (all_, h)) for l, h in layer_head]
            rows.append(dict(experiment="attention pattern, whole layers", site="hook_pattern", heads=label,
                             direction=direction, score=score(run, hooks)))
    hooks = [copy_from("T", f"blocks.{l}.attn.hook_pattern", (all_, h)) for l, h in whole["all 8"]]
    hooks.append(copy_from("T", "blocks.1.attn.hook_v", (all_, 0)))
    rows.append(dict(experiment="attention pattern, whole layers, + layer-1 tag value", site="hook_pattern+hook_v",
                     heads="all 8", direction="T->F", score=score("F", hooks)))
    return rows, base


def layernorm(model, test_pairs, z):
    tokens = {"T": prompts(test_pairs, TRUE), "F": prompts(test_pairs, FALSE)}
    cache = {tag: run_cache(model, t) for tag, t in tokens.items()}
    # Directions in the residual stream that change the number tokens' logits relative to each other.
    w_eff = model.ln_final.w[:, None] * model.W_U[:, :MOD]
    u, s, _ = torch.linalg.svd(w_eff - w_eff.mean(1, keepdim=True), full_matrices=False)
    basis = u[:, s > s.max() * 1e-5].detach()
    rows = []
    for tag in "TF":
        resid = cache[tag]["blocks.1.hook_resid_post"][:, 3]
        centred = resid - resid.mean(-1, keepdim=True)
        inside = centred @ basis @ basis.T
        rows.append(dict(kind="norms", tag=tag, resid_norm=resid.norm(dim=-1).mean().item(),
                         centred_norm=centred.norm(dim=-1).mean().item(),
                         number_subspace_norm=inside.norm(dim=-1).mean().item(),
                         other_norm=(centred - inside).norm(dim=-1).mean().item(),
                         ln_scale=cache[tag]["ln_final.hook_scale"][:, 3, 0].mean().item(),
                         subspace_dim=basis.shape[1]))
        for point, pos in itertools.product(["after layer 0", "after layer 1"], range(4)):
            norm = cache[tag][POINTS[point]][:, pos].norm(dim=-1).mean().item()
            rows.append(dict(kind="position norms", tag=tag, point=point, pos=pos, resid_norm=norm))
    for run, source in [("T", None), ("F", None), ("F", "T"), ("T", "F")]:
        hooks = []
        if source is not None:
            def hook(act, hook, source=source):
                act[:, 3] = cache[source]["ln_final.hook_scale"][:, 3]
                return act
            hooks = [("ln_final.hook_scale", hook)]
        logp = logp_at_eq(model, tokens[run], hooks)
        rows.append(dict(kind="swap", tag=run, scale_from=source or run,
                         acc=(logp.argmax(-1) == z).float().mean().item(),
                         p_star=logp.gather(1, z[:, None]).exp().mean().item(),
                         logp_star=logp.gather(1, z[:, None]).mean().item()))
    return rows


def logit_attribution(model, test_pairs, z):
    """Split logit(z*) minus the mean number-token logit into contributions, with the final LayerNorm
    scale frozen at its value in each run."""
    direction = model.W_U[:, z] - model.W_U[:, :MOD].mean(1, keepdim=True)  # [d_model, batch]
    const = model.ln_final.b @ direction + model.b_U[z] - model.b_U[:MOD].mean()
    rows = []
    for tag in "TF":
        tokens = prompts(test_pairs, TRUE if tag == "T" else FALSE)
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)
        scale = cache["ln_final.hook_scale"][:, 3]

        def attribute(c):
            c = c - c.mean(-1, keepdim=True)
            return ((c * model.ln_final.w / scale) * direction.T).sum(-1).detach()

        parts = {("embedding", None): attribute(cache["blocks.0.hook_resid_pre"][:, 3])}
        for layer in range(model.cfg.n_layers):
            v = cache[f"blocks.{layer}.attn.hook_v"]  # [batch, src, head, d_head]
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][:, :, 3]  # [batch, head, src]
            for head in range(model.cfg.n_heads):
                for src in range(4):
                    out = pattern[:, head, src, None] * (v[:, src, head] @ model.W_O[layer, head])
                    parts[f"L{layer}H{head}", src] = attribute(out)
            if not model.cfg.attn_only:
                parts[f"MLP{layer}", None] = attribute(cache[f"blocks.{layer}.hook_mlp_out"][:, 3])
        parts["biases", None] = attribute(model.b_O.sum(0).expand(len(z), -1)) + const.detach()
        total = sum(parts.values())
        actual = logits[:, 3, :][torch.arange(len(z)), z] - logits[:, 3, :MOD].mean(-1)
        assert torch.allclose(total, actual, atol=1e-3), (total - actual).abs().max()
        for (component, src), value in parts.items():
            rows.append(dict(tag=tag, component=component, src=src, attribution=value.mean().item()))
        rows.append(dict(tag=tag, component="total", src=None, attribution=actual.mean().item()))
    return rows


def classify(probe, ln, dla, patch):
    """Apply the reading rules to one seed, with the thresholds above."""
    acc_f = probe.query("point == 'after layer 1' and target == 'z*' and config == 'F->F'").acc.item()
    restored = ln.query("kind == 'swap' and tag == 'F' and scale_from == 'T'").acc.item()
    heads = dla[(dla.component.str.startswith("L")) & (dla.tag == "F")].groupby("component").attribution.sum()
    total_t = dla.query("component == 'total' and tag == 'T'").attribution.item()
    pattern = patch.query("experiment == 'attention pattern' and direction == 'T->F'").score
    pattern_best = pattern.max()
    if acc_f >= PRESENT and restored >= RESTORED:
        label = "H1a rescale"
    elif acc_f >= PRESENT and heads.min() < -NEGATIVE * total_t:
        label = "H1b cancel"
    elif acc_f <= ABSENT and pattern_best >= PATTERN_RESTORED:
        label = "H2 gate"
    else:
        label = "none of the three"
    return dict(probe_z_false=acc_f, ln_swap_acc=restored, most_negative_head_false=heads.min(),
                most_negative_head=heads.idxmin(), total_true=total_t, best_pattern_patch=pattern_best, label=label)


def analyse(held_out, attn_only, models_dir, data_dir, seeds):
    z_tbl = answer_table()
    out = {k: [] for k in ["training", "behaviour", "probes", "attention", "patching", "layernorm", "dla",
                           "classification"]}
    for seed in seeds:
        name = run_name(seed, held_out, attn_only)
        model, meta = load(models_dir / f"{name}.pt")
        logging.info("seed %d: %d steps, converged %s", seed, meta["steps"], meta["converged"])
        out["training"].append(pd.read_csv(models_dir / f"{name}_log.csv"))
        train_pairs, test_pairs = split(seed, held_out)
        z_test = z_tbl[test_pairs[:, 0], test_pairs[:, 1]]
        z_train = z_tbl[train_pairs[:, 0], train_pairs[:, 1]]
        for split_name, pairs, z in [("held out", test_pairs, z_test), ("training", train_pairs, z_train)]:
            out["behaviour"].append(dict(seed=seed, steps=meta["steps"], converged=meta["converged"],
                                         pairs=split_name, **behaviour(model, pairs, z)))
        t0 = time.time()
        frames = {
            "probes": pd.DataFrame(probes(model, train_pairs, test_pairs, seed)),
            "attention": pd.DataFrame(attention(model, test_pairs)),
            "layernorm": pd.DataFrame(layernorm(model, test_pairs, z_test)),
            "dla": pd.DataFrame(logit_attribution(model, test_pairs, z_test)),
        }
        rows, base = patching(model, test_pairs, z_test)
        frames["patching"] = pd.DataFrame(rows).assign(logp_true_run=base["T"], logp_false_run=base["F"])
        for k, df in frames.items():
            out[k].append(df.assign(seed=seed))
        out["classification"].append(dict(seed=seed, converged=meta["converged"],
                                          **classify(frames["probes"], frames["layernorm"], frames["dla"],
                                                     frames["patching"])))
        logging.info("seed %d analysed in %s: %s", seed, humanize.naturaldelta(timedelta(seconds=time.time() - t0)),
                     out["classification"][-1])
    data_dir.mkdir(parents=True, exist_ok=True)
    prefix = "truth_tag" + ("" if attn_only else "_mlp")
    for k, v in out.items():
        df = pd.concat(v) if isinstance(v[0], pd.DataFrame) else pd.DataFrame(v)
        df.to_csv(data_dir / (f"{prefix}_{k}.csv" + (".gz" if k == "training" else "")), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    logging.info("behaviour:\n%s", pd.DataFrame(out["behaviour"]).round(4).to_string(index=False))
    logging.info("classification:\n%s", pd.DataFrame(out["classification"]).round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------------------
# Figures and README tables, from the CSV files written by analyse. Same palette as results/make_figures.py.

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TAG_COLOR = {"T": SERIES[0], "F": SERIES[1]}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "legend.frameon": False, "legend.labelcolor": INK_2,
    })


def bars_with_seeds(ax, x, values, color, width, label=None, horizontal=False, center=np.mean):
    """A bar at the centre of the seeds' values (the mean by default), with one dot per seed."""
    bar = ax.barh if horizontal else ax.bar
    bar(x, center(values), width, color=color, label=label, zorder=2)
    jitter = np.linspace(-width / 4, width / 4, len(values)) if len(values) > 1 else [0]
    xs, ys = np.asarray(x) + jitter, np.asarray(values)
    ax.plot(*((ys, xs) if horizontal else (xs, ys)), "o", ms=3.5, color=INK, mec=SURFACE, mew=0.6, zorder=3)


def fig_probes(data_dir, out):
    df = pd.read_csv(data_dir / "truth_tag_probes.csv")
    configs = [("T->T", "probe trained on T, tested on T"), ("T->F", "trained on T, tested on F"),
               ("F->F", "trained on F, tested on F")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    width = 0.26
    for ax, (target, title) in zip(axes, [("z*", "The answer z*"), ("x", "The first number x"),
                                          ("y", "The second number y")]):
        d = df[df.target == target]
        for j, point in enumerate(POINTS):
            for i, (config, label) in enumerate(configs):
                acc = d[(d.config == config) & (d.point == point)].acc
                bars_with_seeds(ax, j + (i - 1) * width, acc, SERIES[i], width * 0.92, label if j == 0 else None)
            control = d[d.config.str.endswith("shuffled") & (d.point == point)].acc.max()
            ax.hlines(control, j - 1.6 * width, j + 1.6 * width, color=INK_2, lw=1, ls=(0, (2, 2)), zorder=4,
                      label="shuffled-label control (highest)" if j == 0 else None)
        ax.set_xticks(range(len(POINTS)), ["after the\nembedding", "after\nlayer 0", "after\nlayer 1"])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1.02)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_ylabel("held-out accuracy of a linear probe at =")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 2, 3, 0]  # the three probes, then the control
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_mechanism(data_dir, out):
    attn = pd.read_csv(data_dir / "truth_tag_attention.csv")
    ln = pd.read_csv(data_dir / "truth_tag_layernorm.csv")
    patch = pd.read_csv(data_dir / "truth_tag_patching.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), gridspec_kw=dict(width_ratios=[1, 0.62, 1.05]),
                             layout="constrained")

    ax = axes[0]
    from_eq = attn[(attn.layer == 0) & (attn.dest == 3)].groupby(["seed", "tag", "src"]).attn.mean()
    for i, tag in enumerate("TF"):
        for src in range(4):
            bars_with_seeds(ax, src + (i - 0.5) * 0.38, from_eq.xs((tag, src), level=["tag", "src"]).values,
                            TAG_COLOR[tag], 0.36, f"{tag} prompts" if src == 0 else None)
    ax.set_xticks(range(4), ["tag", "x", "y", "="])
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_title("Layer 0: where = attends")
    ax.set_ylabel("attention from =, mean of 4 heads")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    norms = ln[ln.kind == "position norms"].query("pos == 3")
    for j, point in enumerate(["after layer 0", "after layer 1"]):
        for i, tag in enumerate("TF"):
            vals = norms[(norms.point == point) & (norms.tag == tag)].resid_norm.values
            bars_with_seeds(ax, j + (i - 0.5) * 0.38, vals, TAG_COLOR[tag], 0.36,
                            center=lambda v: np.exp(np.log(v).mean()))
    ax.set_yscale("log")
    ax.set_ylim(1.5, 800)
    ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100, 200, 500]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter("{x:g}")
    ax.set_xticks(range(2), ["after\nlayer 0", "after\nlayer 1"])
    ax.set_title("Size of the residual stream at =")
    ax.set_ylabel("norm, geometric mean over seeds")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    ln_score = ln.query("kind == 'swap' and tag == 'F' and scale_from == 'T'").set_index("seed").logp_star
    base = patch.groupby("seed")[["logp_true_run", "logp_false_run"]].first()
    ln_score = (ln_score - base.logp_false_run) / (base.logp_true_run - base.logp_false_run)
    groups = patch.heads.str.contains("+", regex=False)
    tests = [
        ("T residual at =,\ninput to layer 1", patch.query(
            "experiment == 'residual' and site == 'blocks.1.hook_resid_pre' and pos == 3 and direction == 'T->F'")),
        ("T attention patterns,\nheads with effect > 0.1", patch[
            (patch.experiment == "attention pattern") & (patch.direction == "T->F") & groups]),
        ("T attention patterns, all\nlayer-0 heads (added later)", patch.query(
            "experiment == 'attention pattern, whole layers' and heads == 'L0 all' and direction == 'T->F'")),
        ("tag's value removed,\nheads with effect > 0.1", patch[
            (patch.experiment == "remove tag value") & (patch.direction == "on F run") & groups]),
    ]
    for k, (label, rows) in enumerate(tests):
        bars_with_seeds(ax, k, rows.score.values, SERIES[2], 0.6, horizontal=True)
    bars_with_seeds(ax, len(tests), ln_score.values, SERIES[2], 0.6, horizontal=True)
    ax.set_yticks(range(len(tests) + 1), [t[0] for t in tests] + ["T final\nLayerNorm scale"])
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.axvline(1, color=AXIS, lw=0.8, ls=(0, (2, 2)))
    ax.set_xlabel("log p(z*) rescaled: 0 = F run, 1 = T run")
    ax.set_title("What restores z* in F runs")
    ax.grid(axis="y", visible=False)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def md_table(df):
    rows = [list(map(str, df.columns))] + df.astype(str).values.tolist()
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([lines[0], "|" + "---|" * len(df.columns)] + lines[1:])


def tables(data_dir):
    beh = pd.read_csv(data_dir / "truth_tag_behaviour.csv").query("pairs == 'held out'")
    t = pd.DataFrame({
        "seed": beh.seed,
        "steps": beh.steps.map(lambda s: f"{s / 1000:g}k"),
        "T: accuracy": beh.true_acc.map("{:.1%}".format),
        "T: loss": beh.true_loss.map("{:.4f}".format),
        "F: KL(uniform ‖ model)": beh.false_kl.map("{:.4f}".format),
        "F: mass on special tokens": beh.false_special_mass.map("{:.2%}".format),
        "F: top-1 = z*": beh.false_top1.map("{:.2%}".format),
        "F: mean rank of z*": beh.false_rank.map("{:.1f}".format),
        "F: p(z*) × 109": beh.false_p_star_x109.map("{:.3f}".format),
    })
    logging.info("behaviour on held-out pairs (chance under F: top-1 0.92%%, rank 55, p x 109 = 1):\n%s", md_table(t))
    probe = pd.read_csv(data_dir / "truth_tag_probes.csv")
    logging.info("probe accuracy, mean (min-max) over seeds:\n%s", probe.groupby(["target", "point", "config"]).acc.agg(
        lambda a: f"{a.mean():.1%} ({a.min():.1%}-{a.max():.1%})").unstack().to_string())
    logging.info("classification:\n%s", pd.read_csv(data_dir / "truth_tag_classification.csv").round(3).to_string(index=False))


def figures(data_dir, figures_dir):
    style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_probes(data_dir, figures_dir / "truth_tag_probes.png")
    fig_mechanism(data_dir, figures_dir / "truth_tag_mechanism.png")
    tables(data_dir)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train", help="train one seed")
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p_analyse = sub.add_parser("analyse", help="run every measurement on every seed")
    p_analyse.add_argument("--data", type=Path, default=DATA, help="output directory for the CSV files")
    p_analyse.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p_figures = sub.add_parser("figures", help="figures and README tables from the CSV files")
    p_figures.add_argument("--data", type=Path, default=DATA)
    p_figures.add_argument("--figures", type=Path, default=FIGURES)
    for p in [p_train, p_analyse]:
        p.add_argument("--held-out", type=float, default=0.2)
        p.add_argument("--mlp", action="store_true", help="add MLPs (attn_only=False)")
        p.add_argument("--models", type=Path, default=MODELS, help="checkpoint directory")
        p.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.command == "figures":
        figures(args.data, args.figures)
        return
    torch.set_num_threads(args.threads)
    if args.command == "train":
        train(args.seed, args.held_out, not args.mlp, args.max_steps, args.models)
    elif args.command == "analyse":
        analyse(args.held_out, not args.mlp, args.models, args.data, args.seeds)


if __name__ == "__main__":
    main()
