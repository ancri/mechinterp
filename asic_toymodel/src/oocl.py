"""
Test OOCL.

Vocabulary is going to be:
[0, 1, 2, 3, ..., mod - 1] -- first n=mod tokens, "originals"
U
[mod, mod + 1, ... 2 * mod - 1] -- next n=mod tokens, "aliases" of originals
[equal, true, false, define] -- next 4 tokens

total of 2 * mod + 4 tokens

there are 2 phases of training:
1. train on originals. save model model_original
2. train on linkages + aliases. test unseen aliases
"""


import logging
import torch
import wandb
from transformer_lens import HookedTransformer, HookedTransformerConfig
from dataclasses import dataclass, asdict
import numpy as np
import time
import os
from tqdm.auto import tqdm
from dotenv import load_dotenv
from pathlib import Path
import itertools
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.truth_lies import get_device, loss_fn_z, get_acc


@dataclass
class DataParams:
    mod: int = 997
    n_groups: int = 2
    operation: str = "ssq"


@dataclass
class Tokens:
    # diffs from nv
    true: int = 1
    false: int = 2
    equal: int = 0
    define: int = 3


@dataclass
class TrainParams:
    n_steps: int = 5e4
    n_repetitions: int = 1  # how many times to train the same loop
    batch_size: int = 2**8
    lr: float = 1.4e-3
    wd: float = 0.1
    betas: tuple = (0.9, 0.98)
    max_grad_norm: float = 1.0
    warm_up_steps: int = 1000
    save_every: int = 10000  # save every this many steps
    early_stop_valid_loss: float = 1e-5
    n_steps_epoch: int = 100  # validate / log once every this many steps
    k_p1: float = 1.0
    k_p2: float = 1.0
    k_ln: float = 1.0


default_transformer_config = dict(
    d_vocab=512,
    n_layers=2,  # first layer could learn to do the replacement, 2nd layer could learn the lookup
    d_model=2**7,
    d_head=2**7,
    n_heads=4,
    d_mlp=2**8,
    n_ctx=5,
    act_fn="gelu",
    normalization_type="LN",
    attn_only=True,
)


def loss_fn_linkages(logits, tokens):
    # only compare the z position i.e. index 4: [T/F | x | y | = | z]
    # logit shape: [batch, pos, vocab]
    # token shape: [batch, pos]
    logits = logits[:, 3].unsqueeze(1)
    tokens = tokens[:, 4].unsqueeze(1)
    log_probs = logits.log_softmax(-1)
    correct_log_probs = log_probs.gather(-1, tokens[..., None])[..., 0]
    return -correct_log_probs.mean()


def make_tbl_mask(mod=17, method="sum", frac_held_out=0.05):
    # Todo: 
    #   this seems very slow for large values of mod
    #   perhaps do it in numpy and move to gpu at the end 
    tbl_vv = torch.empty((mod, mod), dtype=torch.long)
    nv = mod
    for v0 in range(nv):
        for v1 in range(v0, nv):
            if method == "sum":
                tbl_vv[v0, v1] = (v0 + v1) % mod
                tbl_vv[v1, v0] = tbl_vv[v0, v1]
            elif method == "ssq":
                tbl_vv[v0, v1] = (v0**2 + v1**2) % mod
                tbl_vv[v1, v0] = tbl_vv[v0, v1]
            else:
                raise ValueError(f"Unknown method {method}")
    train_vv = torch.randperm(nv * nv).reshape(nv, nv) > (frac_held_out * nv * nv)
    valid_vv = ~train_vv
    assert torch.equal((train_vv & valid_vv).any(), torch.tensor(False))  # train and valid are distinct
    x_vv = torch.arange(nv).repeat(nv, 1).T
    y_vv = torch.arange(nv).repeat(nv, 1)
    return x_vv, y_vv, tbl_vv, train_vv, valid_vv


def make_data(batch_size, x_vv, y_vv, z_vv, m_vv, seed=1337):
    """Sample only where m_vv is True.
    """
    # torch.manual_seed(seed)
    nv = x_vv.shape[0]
    nb = batch_size
    nV = nv * nv
    x_V = x_vv.reshape(nV)
    y_V = y_vv.reshape(nV)
    z_V = z_vv.reshape(nV)
    m_V = m_vv.reshape(nV)
    nM = m_V.sum().item()
    while True:
        # generate a batch of data of shape [batch_size, 5]
        # each datapoint looks like: t | x | y | = | z
        x_bt = torch.empty((nb, 5), dtype=torch.long)
        i = torch.where(m_V)[0][torch.randint(0, nM, (nb,))]  # choose only masked elements
        assert torch.equal(m_V[i].all(), torch.tensor(True))  # ensure they are masked
        x_bt[:, 0] = 2 * nv + Tokens.true   # redundant prefix
        x_bt[:, 1] = x_V[i]             # x
        x_bt[:, 2] = y_V[i]             # y
        x_bt[:, 3] = 2 * nv + Tokens.equal  # equal sign
        x_bt[:, 4] = z_V[i]             # z
        yield x_bt


def make_data_linkage(batch_size, mod):
    nv = mod
    nb = batch_size
    while True:
        # generate a batch of data of shape [batch_size, 5]
        # each datapoint looks like: d | d | x | = | y
        # where d is define tag, x and y are from the original and aliases groups
        i = torch.randint(0, nv, (nb,))
        g = torch.randint(0, 2, (nb,))
        x_bt = torch.empty((nb, 5), dtype=torch.long)
        x_bt[:,:2] = 2 * nv + Tokens.define  # define, define
        x_bt[:, 2] = g * nv + i              # x
        x_bt[:, 3] = 2 * nv + Tokens.equal   # equal sign
        x_bt[:, 4] = (1 - g) * nv + i        # y
        yield x_bt


def train_phase1(model, train_loader, valid_loader, nsteps, lr, betas, max_grad_norm, wd, **kwargs):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    warm_up_steps = kwargs.get("warm_up_steps", 1000)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / warm_up_steps, 1.0))
    losses = []
    # for epoch in tqdm(range(nsteps_true), desc="Epoch Tru"):
    for epoch in range(int(nsteps)):
        # tokens = next(train_loader_tru)
        tokens = next(train_loader)
        tokens = tokens.to(DEVICE)
        logits = model(tokens)
        loss = loss_fn_z(logits, tokens)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        losses.append(loss.item())
        step_epoch = kwargs.get("n_steps_epoch", 100)
        if (epoch > 0) & (epoch % step_epoch == 0):
            # validation is unseen data
            losses = losses[-step_epoch:]
            train_loss = np.mean(losses)
            model.eval()
            with torch.no_grad():
                # logging.info(tokens)
                tokens = next(valid_loader)
                tokens = tokens.to(DEVICE)
                logits = model(tokens)
                loss = loss_fn_z(logits, tokens,)
                valid_loss = loss.item()
                lr_curr = scheduler.get_last_lr()[0]
                # lr_curr = lr
                logging.info(
                    f"Epoch: {epoch}, "
                    f"train_loss: {train_loss:.5f}, "
                    f"valid_loss: {valid_loss:.5f}, "
                    f"lr: {lr_curr:.5f}",
                )
                wandb.log({
                    "train/loss": train_loss,
                    "valid/loss": valid_loss,
                    "learning_rate": lr_curr,
                })

            # potentially save model
            save_every = kwargs.get("save_every", None)
            model_name = kwargs.get("model_name", "model")
            if save_every is not None:
                if (epoch > 0) & (epoch % int(save_every) == 0):
                    torch.save(model.state_dict(), os.path.join(dir_models, f"{model_name}_{epoch:010}.pt"))
            early_stop_valid_loss = kwargs.get("early_stop_valid_loss", None)
            if early_stop_valid_loss is not None and valid_loss < early_stop_valid_loss:
                logging.info(f"Early stopping due to valid loss limit of {early_stop_valid_loss} at epoch {epoch}")
                break
            model.train()


def train_phase2(
        model, 
        train_loader_p1, train_loader_p2, 
        valid_loader_p1, valid_loader_p2, 
        loader_linkages, 
        nsteps, lr, betas, max_grad_norm, wd,
        **kwargs,
        ):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    warm_up_steps = kwargs.get("warm_up_steps", 1000)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / warm_up_steps, 1.0))
    losses_binop_p1, losses_binop_p2, losses_linkages = [], [], []

    k_p1 = kwargs.get("k_p1")
    k_p2 = kwargs.get("k_p2")
    k_ln = kwargs.get("k_ln")
    for epoch in range(int(nsteps)):
        # tokens = next(train_loader_tru)
        tokens_binop_p1 = next(train_loader_p1).to(DEVICE)
        tokens_binop_p2 = next(train_loader_p2).to(DEVICE)
        tokens_linkages = next(loader_linkages).to(DEVICE)
        logits_binop_p1 = model(tokens_binop_p1)
        logits_binop_p2 = model(tokens_binop_p2)
        logits_linkages = model(tokens_linkages)
        loss_binop_p1 = loss_fn_z(logits_binop_p1, tokens_binop_p1)
        loss_binop_p2 = loss_fn_z(logits_binop_p2, tokens_binop_p2)
        loss_linkages = loss_fn_linkages(logits_linkages, tokens_linkages)
        loss = k_p1 * loss_binop_p1 + k_p2 * loss_binop_p2 + k_ln * loss_linkages
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        losses_binop_p1.append(loss_binop_p1.item())
        losses_binop_p2.append(loss_binop_p2.item())
        losses_linkages.append(loss_linkages.item())
        step_epoch = kwargs.get("n_steps_epoch", 100)
        if (epoch > 0) & (epoch % step_epoch == 0):
            # validation is unseen data
            losses_binop_p1 = losses_binop_p1[-step_epoch:]
            losses_binop_p2 = losses_binop_p2[-step_epoch:]
            losses_linkages = losses_linkages[-step_epoch:]
            train_loss_binop_p1 = np.mean(losses_binop_p1)
            train_loss_binop_p2 = np.mean(losses_binop_p2)
            train_loss_linkages = np.mean(losses_linkages)
            model.eval()
            with torch.no_grad():
                # logging.info(tokens)
                tokens_binop_p1 = next(valid_loader_p1).to(DEVICE)
                tokens_binop_p2 = next(valid_loader_p2).to(DEVICE)
                # tokens_linkages = next(loader_linkages).to(DEVICE)
                logits_p1 = model(tokens_binop_p1)
                logits_p2 = model(tokens_binop_p2)
                loss_binop_p1 = loss_fn_z(logits_p1, tokens_binop_p1,)
                loss_binop_p2 = loss_fn_z(logits_p2, tokens_binop_p2,)
                accuracy_binop_p1 = get_acc(logits_p1, tokens_binop_p1)
                accuracy_binop_p2 = get_acc(logits_p2, tokens_binop_p2)
                # loss_linkages = loss_fn_linkages(logits_linkages, tokens_linkages)
                valid_loss_binop_p1 = loss_binop_p1.item()
                valid_loss_binop_p2 = loss_binop_p2.item()

                lr_curr = scheduler.get_last_lr()[0]
                logging.info(
                    f"Epoch: {epoch}, "
                    f"train_loss_binop_p1: {train_loss_binop_p1:.5f}, "
                    f"train_loss_binop_p2: {train_loss_binop_p2:.5f}, "
                    f"train_loss_linkages: {train_loss_linkages:.5f}, "
                    f"valid_loss_binop_p1: {valid_loss_binop_p1:.5f}, "
                    f"valid_loss_binop_p2: {valid_loss_binop_p2:.5f}, "
                    f"valid_acc_binop_p1: {accuracy_binop_p1:.5f}, "
                    f"valid_acc_binop_p2: {accuracy_binop_p2:.5f}, "
                    f"lr: {lr_curr:.5f}",
                )
                wandb.log({
                    "train/loss_binop_p1": train_loss_binop_p1,
                    "train/loss_binop_p2": train_loss_binop_p2,
                    "train/loss_linkages": train_loss_linkages,
                    "valid/loss_binop_p1": valid_loss_binop_p1,
                    "valid/loss_binop_p2": valid_loss_binop_p2,
                    "valid/acc_binop_p1": accuracy_binop_p1,
                    "valid/acc_binop_p2": accuracy_binop_p2,
                    "learning_rate": lr_curr,
                })

            # potentially save model
            save_every = kwargs.get("save_every", None)
            model_name = kwargs.get("model_name", "model")
            if save_every is not None:
                if (epoch > 0) & (epoch % int(save_every) == 0):
                    ts_start = kwargs.get("ts_start_training", 0)
                    torch.save(model.state_dict(), os.path.join(dir_models, f"{model_name}.{ts_start}.{epoch:010}.pt"))
            early_stop_valid_loss = kwargs.get("early_stop_valid_loss", None)
            # if early_stop_valid_loss is not None and valid_loss < early_stop_valid_loss:
            #     logging.info(f"Early stopping due to valid loss limit of {early_stop_valid_loss} at epoch {epoch}")
            #     break
            model.train()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    DEVICE = get_device()
    logging.info(f"using device: {DEVICE}")
    torch.set_default_device(DEVICE)

    data_params = DataParams()
    tokens = Tokens()
    transformer_config = default_transformer_config
    transformer_config.update(dict(
        d_vocab=2 * data_params.mod + 4,  # originals, aliases, and 4 special tokens: end, random, not-random, define
    ))
    train_params = TrainParams()

    # load wandb
    assert load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    # prep model saving directory
    dir_models = "models/transformers/"  # save models here
    Path(dir_models).mkdir(exist_ok=True, parents=True)

    cfg = HookedTransformerConfig(**transformer_config)
    # model.load_state_dict(torch.load(os.path.join(dir_models, "interrupted.pt")))
    for _ in range(train_params.n_repetitions):
        for frac_held_out_phase1, frac_held_out_phase2, k_p1, k_p2, k_ln in [
            (0.60, 0.60, 1.0, 0.0, 0.0),
            (0.60, 0.60, 1.0, 1.0, 0.0),
            (0.60, 0.60, 1.0, 1.0, 1.0),
            ]:
            train_params.k_p1 = k_p1
            train_params.k_p2 = k_p2
            train_params.k_ln = k_ln

            x_vv, y_vv, z_vv, train_vv, valid_vv = make_tbl_mask(
                mod=data_params.mod, method=data_params.operation, frac_held_out=frac_held_out_phase1,
            )
            _, _, _, train2_vv, valid2_vv = make_tbl_mask(
                mod=data_params.mod, method=data_params.operation, frac_held_out=frac_held_out_phase2,
            )

            logging.info(
                f"dataset has "
                f"{train_vv.sum().item()} training examples and "
                f"{valid_vv.sum().item()} validation examples."
            )
            model = HookedTransformer(cfg)
            name = (
                f"oocl_{data_params.operation}_{data_params.mod}_"
                f"{round(frac_held_out_phase1, 2)}_{round(frac_held_out_phase2, 2)}_"
                f"{round(train_params.k_p1, 1)}_{round(train_params.k_p2, 1)}_{round(train_params.k_ln, 1)}"
                )
            logging.info(f"model / run named: {name}")
            train_loader_phase1 = make_data(train_params.batch_size, x_vv, y_vv, z_vv, train_vv)
            valid_loader_phase1 = make_data(train_params.batch_size, x_vv, y_vv, z_vv, valid_vv)

            x2_vv = x_vv + data_params.mod
            y2_vv = y_vv + data_params.mod
            z2_vv = z_vv + data_params.mod
            train_loader_phase2 = make_data(train_params.batch_size, x2_vv, y2_vv, z2_vv, train2_vv)
            valid_loader_phase2 = make_data(train_params.batch_size, x2_vv, y2_vv, z2_vv, valid2_vv)

            loader_linkages = make_data_linkage(train_params.batch_size, data_params.mod)
            ts_start_training = int(time.time())
            wandb.init(
                # set the wandb project where this run will be logged
                project="oocl",
                entity=os.getenv("WANDB_ENTITY"),
                name=name,
                # track hyperparameters and run metadata
                config={
                    "ts_start": ts_start_training,  # use to map between model files and wandb runs
                    "frac1": frac_held_out_phase1,
                    "frac2": frac_held_out_phase2,
                    **asdict(data_params),
                    **asdict(train_params),
                    **transformer_config,
                }
            )
            # try:
            #     train_phase1(
            #         model, train_loader_phase1, valid_loader_phase1, train_params.n_steps, model_name=f"{name}_phase1",
            #         **asdict(train_params), **asdict(data_params),
            #     )
            # except KeyboardInterrupt:
            #     torch.save(model.state_dict(), os.path.join(dir_models, "phase1_interrupted.pt"))
            #     #  do not wandb.finish() on purpose
            #     raise KeyboardInterrupt
            try:
                train_phase2(
                    model,
                    train_loader_phase1, train_loader_phase2,
                    valid_loader_phase1, valid_loader_phase2,
                    loader_linkages,
                    train_params.n_steps, model_name=f"{name}_phase2",
                    ts_start_training=ts_start_training,
                    **asdict(train_params), **asdict(data_params),
                )
            except KeyboardInterrupt:
                torch.save(model.state_dict(), os.path.join(dir_models, "phase2_interrupted.pt"))
                #  do not wandb.finish() on purpose
                raise KeyboardInterrupt
            ts_finish_training = time.time()
            fpath_model = os.path.join(dir_models, f"{name}.{ts_start_training}.final.pt")
            logging.info(f"training n_layers={model.cfg.n_layers} took {(ts_finish_training - ts_start_training)//60} minutes")
            logging.info(f"saving model to {fpath_model}")
            torch.save(model.state_dict(), fpath_model)
            wandb.finish()
