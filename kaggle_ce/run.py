#!/usr/bin/env python
"""Cross-encoder member of the Business Entity Resolution ensemble (Kaggle T4 / P100 / CPU smoke).

A transformer reads both records' raw text ("name | address") and scores P(same business). It is trained on the hard
pairs of the training S1s and scores the uncertain holdout / test pairs; the LightGBM pipeline fuses these scores
(stacking fitted on the holdout), so the transformer only has to be good where LightGBM is unsure.

Stages, every one resumable (re-run the same command after a crash / new session):
  train   fine-tune; checkpoint every --ckpt-every steps (model + optimizer + scheduler + scaler + position in the
          epoch), evaluation every --eval-every steps, best model (validation logloss) kept in best/
  score   best model on val / holdout / test pairs, in shards of --shard pairs (finished shards are skipped)

    python run.py all --model intfloat/multilingual-e5-small --tag e5s --gpu 0
    python run.py all --model FacebookAI/xlm-roberta-base   --tag xlmr --gpu 1 --bs 64 --lr 2e-5
    python run.py all --smoke --model FacebookAI/xlm-roberta-base --tag smoke     # tiny CPU run

Outputs in OUT/TAG/: progress.json (live), train_log.csv, metrics.json, ckpt/last.pt, best/, scores_<split>.parquet
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["train", "score", "all"])
    ap.add_argument("--data", default=None, help="folder with train_pairs/holdout_pairs/test_pairs/records parquet "
                                                 "(default: found under /kaggle/input or ./data)")
    ap.add_argument("--out", default="/kaggle/working/ce_work" if os.path.isdir("/kaggle/working") else "ce_work")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--tag", default=None, help="output sub-folder (default: derived from --model)")
    ap.add_argument("--gpu", default=None, help="GPU index for this process (sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max-len", dest="max_len", type=int, default=96)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=2000)
    ap.add_argument("--ckpt-every", dest="ckpt_every", type=int, default=1000)
    ap.add_argument("--eval-n", dest="eval_n", type=int, default=60000, help="val pairs used at periodic evals")
    ap.add_argument("--shard", type=int, default=250000)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume-from", dest="resume_from", default=None,
                    help="a previous run's output folder (e.g. an earlier notebook version added as input)")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, few steps (CPU test)")
    a = ap.parse_args(argv)
    if a.tag is None:
        a.tag = a.model.rstrip("/").split("/")[-1].replace(".", "-")
    if a.smoke:
        a.epochs, a.bs, a.max_len, a.eval_every, a.ckpt_every, a.eval_n, a.shard = 1, 8, 48, 8, 5, 200, 300
        a.max_steps = a.max_steps or 16
    return a


A = parse_args() if __name__ == "__main__" else None
if A is not None and A.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(A.gpu)          # before torch is imported
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402


# ------------------------------------------------------------------ utilities
def log(msg, out=None):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if out is not None:
        with open(out / "run.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")


def atomic_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, path)


def find_data(arg):
    if arg:
        return Path(arg)
    cands = [Path("data")] + (sorted(Path("/kaggle/input").rglob("train_pairs.parquet"))
                              if os.path.isdir("/kaggle/input") else [])
    for c in cands:
        c = c.parent if c.name == "train_pairs.parquet" else c
        if (c / "train_pairs.parquet").exists():
            return c
    raise SystemExit("data folder with train_pairs.parquet not found (use --data)")


def binary_metrics(y, logit):
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
    y = np.asarray(y)
    p = 1 / (1 + np.exp(-np.clip(np.asarray(logit, np.float64), -30, 30)))
    out = {"n": int(len(y)), "pos_rate": float(y.mean()) if len(y) else 0.0}
    if len(y) and 0 < y.mean() < 1:
        out.update(auc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                   logloss=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
                   acc=float(((p > 0.5) == y).mean()))
    return out


def gbdt_logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Texts:
    """id -> 'name | address' for S1 records and '[S2] name | address' for pool records."""

    def __init__(self, data):
        r = pd.read_parquet(data / "records.parquet")
        r = r.drop_duplicates("id")
        name = r["name"].fillna("").astype(str).str.strip()
        addr = r["addr"].fillna("").astype(str).str.strip()
        src = r["id"].astype(str).str[:2]
        body = name + " | " + addr
        self.text = np.where(src == "S1", body, "[" + src + "] " + body)
        self.index = pd.Index(r["id"].astype(str))

    def pairs(self, df):
        i = self.index.get_indexer(df["s1_id"].astype(str))
        j = self.index.get_indexer(df["cand_id"].astype(str))
        if (i < 0).any() or (j < 0).any():
            raise SystemExit(f"{int((i < 0).sum() + (j < 0).sum())} pair ids missing from records.parquet")
        return self.text[i], self.text[j]


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def autocast():
    import contextlib
    if device() == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_scaler():
    on = device() == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=on)
    except Exception:                                         # older torch
        return torch.cuda.amp.GradScaler(enabled=on)


def load_model(name_or_path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(name_or_path, num_labels=1,
                                                               ignore_mismatched_sizes=True)
    return tok, model


def batches(order, lengths, bs, bucket=64):
    """Batches over `order`; inside windows of `bucket` batches rows are sorted by length (less padding)."""
    win = bs * bucket
    out = []
    for w in range(0, len(order), win):
        chunk = order[w:w + win]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        bl = [chunk[i:i + bs] for i in range(0, len(chunk), bs)]
        rng = np.random.RandomState(w)
        out.extend(bl[k] for k in rng.permutation(len(bl)))
    return out


@torch.no_grad()
def predict(model, tok, ta, tb, bs, max_len, progress=None):
    model.eval()
    dev = device()
    n = len(ta)
    out = np.zeros(n, np.float32)
    lengths = np.array([len(x) + len(y) for x, y in zip(ta, tb)])
    order = np.argsort(lengths, kind="stable")
    t0 = time.time()
    for b, i in enumerate(range(0, n, bs)):
        idx = order[i:i + bs]
        enc = tok([ta[j] for j in idx], [tb[j] for j in idx], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt")
        enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
        with autocast():
            out[idx] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
        if progress and (b % 50 == 0 or i + bs >= n):
            progress(min(i + bs, n), n, time.time() - t0)
    model.train()
    return out


# ------------------------------------------------------------------ training
def train(a, data, out):
    texts = Texts(data)
    tr = pd.read_parquet(data / "train_pairs.parquet")
    if a.smoke:
        tr = pd.concat([tr[~tr.is_val].head(400), tr[tr.is_val].head(200)], ignore_index=True)
    fit, val = tr[~tr.is_val].reset_index(drop=True), tr[tr.is_val].reset_index(drop=True)
    ta, tb = texts.pairs(fit)
    y = fit["y"].to_numpy(np.float32)
    rs = np.random.RandomState(a.seed)
    ev = val.iloc[rs.permutation(len(val))[:a.eval_n]].reset_index(drop=True)
    eva, evb = texts.pairs(ev)
    lengths = np.array([len(x) + len(z) for x, z in zip(ta, tb)])
    steps_per_epoch = math.ceil(len(fit) / a.bs)
    total = int(steps_per_epoch * a.epochs)
    if a.max_steps:
        total = min(total, a.max_steps)
    warm = max(1, int(a.warmup * total))
    log(f"[{a.tag}] train {len(fit):,} pairs (pos {y.mean():.3f}), val {len(val):,}, eval subset {len(ev):,} | "
        f"{steps_per_epoch:,} steps/epoch, {total:,} steps, device {device()} "
        f"{torch.cuda.get_device_name(0) if device() == 'cuda' else ''}", out)

    tok, model = load_model(a.model)
    model.to(device())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    scaler = make_scaler()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    ck = out / "ckpt" / "last.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    step, best, loss_ema, hist = 0, {"logloss": float("inf")}, None, []
    if a.resume_from and not ck.exists():                    # continue a run from an earlier session's output
        src = Path(a.resume_from) / a.tag
        for sub in ("ckpt", "best"):
            if (src / sub).exists():
                shutil.copytree(src / sub, out / sub, dirs_exist_ok=True)
        for f in ("train_log.csv",):
            if (src / f).exists():
                shutil.copy(src / f, out / f)
        log(f"[{a.tag}] copied checkpoints from {src}", out)
    if ck.exists():
        state = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        step, best, loss_ema = state["step"], state["best"], state["loss_ema"]
        log(f"[{a.tag}] resumed from step {step:,} (best val logloss {best['logloss']:.4f})", out)
    if (out / "train_log.csv").exists():
        hist = pd.read_csv(out / "train_log.csv").to_dict("records")
        hist = [h for h in hist if h["step"] <= step]

    def save_ckpt():
        tmp = Path(str(ck) + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "best": best, "loss_ema": loss_ema}, tmp)
        os.replace(tmp, ck)

    def evaluate():
        nonlocal best
        lg = predict(model, tok, eva, evb, a.bs * 4, a.max_len)
        m = binary_metrics(ev["y"].to_numpy(), lg)
        g = binary_metrics(ev["y"].to_numpy(), gbdt_logit(ev["p_gbdt"]))
        m["gbdt_auc"], m["gbdt_logloss"] = g.get("auc"), g.get("logloss")
        improved = m.get("logloss", float("inf")) < best["logloss"]
        if improved:
            best = {**m, "step": step}
            model.save_pretrained(out / "best")
            tok.save_pretrained(out / "best")
        hist.append({"step": step, "loss_ema": loss_ema, "lr": sched.get_last_lr()[0], **{f"val_{k}": v for k, v in
                                                                                          m.items()}})
        pd.DataFrame(hist).to_csv(out / "train_log.csv", index=False)
        log(f"[{a.tag}] eval step {step:,}: val AUC {m.get('auc', float('nan')):.5f} logloss "
            f"{m.get('logloss', float('nan')):.4f} (GBDT AUC {m['gbdt_auc'] or float('nan'):.5f}) "
            f"{'*best*' if improved else ''}", out)
        return m

    model.train()
    t0, t_last, done_at_start = time.time(), time.time(), step
    last_eval = hist[-1] if hist else {}
    while step < total:
        epoch = step // steps_per_epoch
        order = np.random.RandomState(a.seed + epoch).permutation(len(fit))
        bl = batches(order, lengths, a.bs)
        for bi in range(step % steps_per_epoch, len(bl)):
            idx = bl[bi]
            enc = tok([ta[j] for j in idx], [tb[j] for j in idx], truncation=True, max_length=a.max_len,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(device(), non_blocking=True) for k, v in enc.items()}
            yb = torch.from_numpy(y[idx]).to(device())
            with autocast():
                logits = model(**enc).logits.squeeze(-1)
            loss = loss_fn(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            lv = float(loss)
            loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv
            if step % a.eval_every == 0 or step == total:
                last_eval = evaluate()
            if step % a.ckpt_every == 0 or step == total:
                save_ckpt()
            if time.time() - t_last > 20 or step == total:
                el = time.time() - t0
                rate = (step - done_at_start) / max(el, 1e-9)
                atomic_json(out / "progress.json", {
                    "tag": a.tag, "model": a.model, "stage": "train", "step": step, "total": total,
                    "pct": round(100 * step / total, 2), "epoch": round(step / steps_per_epoch, 3),
                    "loss_ema": loss_ema, "lr": sched.get_last_lr()[0], "it_per_s": rate,
                    "pairs_per_s": rate * a.bs, "eta_min": (total - step) / max(rate, 1e-9) / 60,
                    "gpu_mem_gb": (torch.cuda.max_memory_allocated() / 2 ** 30) if device() == "cuda" else 0,
                    "last_eval": last_eval, "best": best, "updated": time.strftime("%H:%M:%S")})
                t_last = time.time()
            if step >= total:
                break
    if not (out / "best").exists():                          # no evaluation happened (tiny runs)
        evaluate()
    (out / "TRAIN_DONE").write_text(json.dumps(best, default=float), encoding="utf-8")
    log(f"[{a.tag}] training done: best val logloss {best['logloss']:.4f} AUC {best.get('auc', float('nan')):.5f} "
        f"at step {best.get('step')}", out)


# ------------------------------------------------------------------ scoring
def score(a, data, out):
    texts = Texts(data)
    tok, model = load_model(out / "best")
    model.to(device())
    metrics = json.loads((out / "metrics.json").read_text()) if (out / "metrics.json").exists() else {}
    best = json.loads((out / "TRAIN_DONE").read_text()) if (out / "TRAIN_DONE").exists() else {}
    for split in ("val", "holdout", "test"):
        final = out / f"scores_{split}.parquet"
        if final.exists():
            continue
        if split == "val":
            df = pd.read_parquet(data / "train_pairs.parquet")
            df = df[df.is_val].reset_index(drop=True)
        else:
            df = pd.read_parquet(data / f"{split}_pairs.parquet")
        if a.smoke:
            df = df.head(a.shard * 2 + 17)
        n_sh = max(1, math.ceil(len(df) / a.shard))
        t0 = time.time()
        for k in range(n_sh):
            sp = out / "shards" / f"{split}_{k:04d}.parquet"
            if sp.exists():
                continue
            sp.parent.mkdir(parents=True, exist_ok=True)
            part = df.iloc[k * a.shard:(k + 1) * a.shard]
            ta, tb = texts.pairs(part)

            def prog(i, n, el, k=k):
                done = k * a.shard + i
                atomic_json(out / "progress.json", {
                    "tag": a.tag, "model": a.model, "stage": f"score_{split}", "shard": k + 1, "shards": n_sh,
                    "pairs_done": done, "pairs": len(df), "pct": round(100 * done / max(len(df), 1), 2),
                    "pairs_per_s": i / max(el, 1e-9), "eta_min": (len(df) - done) / max(i / max(el, 1e-9), 1e-9) / 60,
                    "best": best, "updated": time.strftime("%H:%M:%S")})

            lg = predict(model, tok, ta, tb, a.bs * 4, a.max_len, prog)
            keep = [c for c in ("s1_id", "cand_id", "cty", "y", "p_gbdt", "fold") if c in part.columns]
            res = part[keep].copy()
            res[f"ce_{a.tag}"] = lg
            tmp = Path(str(sp) + ".tmp")
            res.to_parquet(tmp, index=False)
            os.replace(tmp, sp)
            log(f"[{a.tag}] scored {split} shard {k + 1}/{n_sh} ({len(part):,} pairs, {time.time() - t0:.0f}s)", out)
        allp = pd.concat([pd.read_parquet(out / "shards" / f"{split}_{k:04d}.parquet") for k in range(n_sh)],
                         ignore_index=True)
        allp.to_parquet(final, index=False)
        if "y" in allp:
            metrics[split] = {"ce": binary_metrics(allp["y"].to_numpy(), allp[f"ce_{a.tag}"].to_numpy()),
                              "gbdt": binary_metrics(allp["y"].to_numpy(), gbdt_logit(allp["p_gbdt"]))}
            log(f"[{a.tag}] {split}: CE {metrics[split]['ce']} | GBDT {metrics[split]['gbdt']}", out)
        atomic_json(out / "metrics.json", metrics)
    atomic_json(out / "progress.json", {"tag": a.tag, "model": a.model, "stage": "done", "pct": 100.0,
                                        "best": best, "updated": time.strftime("%H:%M:%S")})
    (out / "SCORE_DONE").write_text("ok", encoding="utf-8")
    log(f"[{a.tag}] scoring done", out)


def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    data = find_data(a.data)
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "config.json", vars(a))
    try:
        if a.stage in ("train", "all") and not (out / "TRAIN_DONE").exists():
            train(a, data, out)
        if a.stage in ("score", "all") and not (out / "SCORE_DONE").exists():
            score(a, data, out)
    except Exception as e:
        atomic_json(out / "progress.json", {"tag": a.tag, "stage": "FAILED", "error": repr(e)[:500],
                                            "updated": time.strftime("%H:%M:%S")})
        raise


if __name__ == "__main__":
    main(A)
