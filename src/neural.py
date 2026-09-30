"""Optional GPU boosters (plan section 11): a cross-encoder pair feature and a dense blocking channel.

Both are gated: they run only when torch sees a CUDA device that actually executes kernels
(Kaggle P100 = sm_60 may be unsupported by recent torch builds), transformers is importable and the
model loads (local path, a copy under /kaggle/input, or a Hugging Face download). Any failure is
logged and the pipeline continues without the booster. Models used are MIT / Apache-2.0 (see README).
"""
from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("ber")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")      # no fork warnings next to joblib workers


def torch_cuda_status() -> dict:
    try:
        import torch
    except Exception:
        return {"ok": False, "why": "torch not installed"}
    if not torch.cuda.is_available():
        return {"ok": False, "why": "torch.cuda.is_available() is False"}
    try:
        n = torch.cuda.device_count()
        info = {"n_gpu": n, "names": [torch.cuda.get_device_name(i) for i in range(n)],
                "capability": [list(torch.cuda.get_device_capability(i)) for i in range(n)],
                "torch": torch.__version__}
        x = torch.randn(256, 256, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = torch.nn.functional.gelu(x @ x)
        torch.cuda.synchronize()
        float(y.float().mean())
        return {"ok": True, **info}
    except Exception as e:  # e.g. "no kernel image is available for execution on the device"
        return {"ok": False, "why": repr(e)[:300]}


def resolve_model(name_or_path: str, roots=("/kaggle/input",)) -> str:
    p = Path(name_or_path)
    if p.exists() and (p / "config.json").exists():
        return str(p)
    tail = name_or_path.rstrip("/").split("/")[-1].lower()
    for r in roots:
        if not os.path.isdir(r):
            continue
        for cfgf in sorted(Path(r).rglob("config.json")):
            d = cfgf.parent
            if tail in str(d).lower() and any((d / f).exists() for f in
                                              ("model.safetensors", "pytorch_model.bin")):
                return str(d)
    if _hf_cached(name_or_path):
        return name_or_path
    import socket
    try:                                                    # fail fast instead of hanging on retries offline
        socket.create_connection(("huggingface.co", 443), timeout=6).close()
    except OSError:
        raise RuntimeError(f"model {name_or_path!r} is not available locally and huggingface.co is unreachable "
                           "(Kaggle internet off?). Enable internet, attach the model as a Kaggle input, "
                           "or set neural.cross_encoder=off")
    return name_or_path


def _hf_cached(repo_id):
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)
    except Exception:
        return False


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _autocast():
    """fp16 autocast on GPU, a no-op context on CPU."""
    import contextlib

    import torch
    if _device() == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _scaler():
    import torch
    on = _device() == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=on)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=on)


def _load(model_path, kind):
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    if kind == "cls":
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1,
                                                                   ignore_mismatched_sizes=True)
    else:
        model = AutoModel.from_pretrained(model_path)
    return tok, model


def _wrap(model):
    import torch
    model.to(_device())
    if _device() == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def train_cross_encoder(model_path, ta, tb, y, nc, seed):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok, model = _load(model_path, "cls")
    model = _wrap(model)
    dev = _device()
    bs, L, n = nc["ce_batch"], nc["ce_max_len"], len(ta)
    steps = max(1, nc["ce_epochs"] * math.ceil(n / bs))
    warm = max(1, int(0.06 * steps))
    opt = torch.optim.AdamW(model.parameters(), lr=nc["ce_lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    scaler = _scaler()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(seed)
    y = np.asarray(y, np.float32)
    model.train()
    step, t0 = 0, time.time()
    for _ in range(nc["ce_epochs"]):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            enc = tok([ta[j] for j in idx], [tb[j] for j in idx], truncation=True, max_length=L,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            yb = torch.from_numpy(y[idx]).to(dev)
            with _autocast():
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
            if step % 250 == 0 or step == steps:
                el = time.time() - t0
                log.info("    ce step %d/%d loss=%.4f  %.1f it/s  elapsed %.0fs  ETA %.0fs", step, steps, float(loss),
                         step / max(el, 1e-9), el, el / step * (steps - step))
    return model, tok


def score_cross_encoder(model, tok, ta, tb, nc):
    import torch
    model.eval()
    dev = _device()
    n = len(ta)
    out = np.zeros(n, np.float32)
    order = np.argsort([len(a) + len(b) for a, b in zip(ta, tb)], kind="stable")   # length bucketing
    bs = nc["ce_batch"] * 4
    t0 = time.time()
    n_batches = (n + bs - 1) // bs
    with torch.no_grad():
        for b, i in enumerate(range(0, n, bs)):
            idx = order[i:i + bs]
            enc = tok([ta[j] for j in idx], [tb[j] for j in idx], truncation=True, max_length=nc["ce_max_len"],
                      padding=True, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            with _autocast():
                out[idx] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            if (b + 1) % 200 == 0:
                el = time.time() - t0
                log.info("    ce scoring %d/%d pairs  %.0f pairs/s  ETA %.0fs", i + len(idx), n, (i + len(idx)) / el,
                         el / (b + 1) * (n_batches - b - 1))
    log.info("    ce scored %d pairs in %.0fs", n, time.time() - t0)
    return out


def _cuda_gc():
    import gc

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def record_text(n):
    return [f"{a} | {b}" for a, b in zip(n["raw_name"], n["raw_addr"])]


def cross_encoder_feature(tr, te, nc, seed=42, ckpt=None, ckpt_key=None):
    """2-fold (by S1 fold parity) cross-encoder logits for the pairs pass-1 finds plausible.
    tr/te: dict(pairs, n1, npool, p1, rank, [y, fold]). Returns (ce_train, ce_test) with NaN = not scored.
    Each of the two fold-models is checkpointed (ckpt), so an interrupted run resumes after the first."""
    t0 = time.time()
    model_path = None
    txt = {}
    for name, d in (("tr", tr), ("te", te)):
        t1, tp = record_text(d["n1"]), record_text(d["npool"])
        s1, cand = d["pairs"]["s1"].to_numpy(), d["pairs"]["cand"].to_numpy()
        txt[name] = ([t1[i] for i in s1], [tp[j] for j in cand])
    m_tr = (tr["p1"] >= nc["ce_min_p1"]) | (tr["rank"] <= nc["ce_top_rank"])
    m_te = (te["p1"] >= nc["ce_min_p1"]) | (te["rank"] <= nc["ce_top_rank"])
    ce_tr = np.full(len(m_tr), np.nan, np.float32)
    ce_te = np.zeros(len(m_te), np.float32)
    parity = tr["fold"] % 2
    rng = np.random.RandomState(seed)
    ite = np.flatnonzero(m_te)
    for m in (0, 1):
        fit = np.flatnonzero(m_tr & (parity == m))
        pos, neg = fit[tr["y"][fit] > 0], fit[tr["y"][fit] <= 0]
        budget = max(nc["ce_max_train"] - len(pos), 0)
        if len(neg) > budget:
            neg = rng.choice(neg, budget, replace=False)
        fit = np.sort(np.concatenate([pos, neg]))
        oth = np.flatnonzero(m_tr & (parity != m))
        mkey = f"{ckpt_key}-model{m}" if (ckpt is not None and ckpt_key) else None
        saved = ckpt.load(mkey) if mkey else None
        if saved is None:
            if model_path is None:
                model_path = resolve_model(nc["ce_model"])
                log.info("cross-encoder: model %s", model_path)
            log.info("  ce model %d/2: train on %d pairs (%d positive), then score %d train + %d test pairs",
                     m + 1, len(fit), len(pos), len(oth), len(ite))
            model, tok = train_cross_encoder(model_path, [txt["tr"][0][i] for i in fit],
                                             [txt["tr"][1][i] for i in fit], tr["y"][fit], nc, seed + m)
            saved = {"tr": score_cross_encoder(model, tok, [txt["tr"][0][i] for i in oth],
                                               [txt["tr"][1][i] for i in oth], nc),
                     "te": score_cross_encoder(model, tok, [txt["te"][0][i] for i in ite],
                                               [txt["te"][1][i] for i in ite], nc)}
            del model, tok
            _cuda_gc()
            if mkey:
                ckpt.save(mkey, saved)
        ce_tr[oth] = saved["tr"]
        ce_te[ite] += saved["te"] / 2
    ce_te[~m_te] = np.nan
    log.info("cross-encoder done: scored %d train / %d test pairs in %.0fs", int(m_tr.sum()), int(m_te.sum()),
             time.time() - t0)
    return ce_tr, ce_te


def dense_embed(n, nc):
    """Mean-pooled, L2-normalised multilingual-e5 embeddings of 'query: name, address'."""
    import torch
    model_path = resolve_model(nc["dense_model"])
    tok, model = _load(model_path, "enc")
    model = _wrap(model).eval()
    dev = _device()
    texts = [f"query: {a}, {b}" for a, b in zip(n["raw_name"], n["raw_addr"])]
    out = np.zeros((len(texts), model.module.config.hidden_size if hasattr(model, "module")
                    else model.config.hidden_size), np.float32)
    order = np.argsort([len(t) for t in texts], kind="stable")
    bs = nc["dense_batch"]
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=64, padding=True, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            with _autocast():
                h = model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            out[idx] = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
    del model, tok
    _cuda_gc()
    return out


def dense_topk(E1, E2, k, chunk=4096):
    """Top-k inner products (cosines) of each row of E1 among E2 -> (rows, cols, scores)."""
    k = min(k, len(E2))
    if k <= 0 or len(E1) == 0:
        z = np.zeros(0, np.int64)
        return z, z.copy(), np.zeros(0, np.float32)
    R, C, S = [], [], []
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError
        B = torch.from_numpy(E2).to("cuda", torch.float16)
        for s in range(0, len(E1), chunk):
            A = torch.from_numpy(E1[s:s + chunk]).to("cuda", torch.float16)
            v, i = torch.topk((A @ B.T).float(), k, dim=1)
            R.append(np.repeat(np.arange(s, s + len(A)), k))
            C.append(i.cpu().numpy().ravel())
            S.append(v.cpu().numpy().ravel())
        del B
    except Exception:
        R, C, S = [], [], []
        step = max(16, int(5e7 // max(len(E2), 1)))
        for s in range(0, len(E1), step):
            M = E1[s:s + step] @ E2.T
            i = np.argpartition(-M, k - 1, axis=1)[:, :k]
            R.append(np.repeat(np.arange(s, s + len(M)), k))
            C.append(i.ravel())
            S.append(np.take_along_axis(M, i, 1).ravel())
    return (np.concatenate(R).astype(np.int64), np.concatenate(C).astype(np.int64),
            np.concatenate(S).astype(np.float32))
