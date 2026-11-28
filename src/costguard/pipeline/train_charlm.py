"""A real (small) training run that produces real loss curves for the CostGuard demo.

Character-level MLP language model (Bengio-style: 8-char context -> embedding -> hidden -> next char)
trained on Tiny Shakespeare with SGD + momentum and a linear learning-rate warmup. Pure numpy, CPU.

Nothing here is scripted: if the learning rate is too high the run really diverges, the loss really
becomes NaN, and the checkpoints written after that point really contain NaN weights.

    python pipeline/train_charlm.py --lr 0.3 --steps 60000 --out runs/a     # one run
    python pipeline/train_charlm.py --lr 0.1 --resume runs/a/ckpt/step_42000.npz --out runs/a_fix

Writes <out>/log.jsonl (one line per logged step / eval / checkpoint) and <out>/ckpt/step_N.npz.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

CTX, EMB, HID = 8, 24, 256


def load_text(path):
    text = Path(path).read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int32)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def init_params(V, rng):
    return {
        "C": rng.normal(0, 1.0, (V, EMB)).astype(np.float32),
        "W1": (rng.normal(0, 1.0, (CTX * EMB, HID)) / np.sqrt(CTX * EMB)).astype(np.float32),
        "b1": np.zeros(HID, np.float32),
        "W2": (rng.normal(0, 1.0, (HID, V)) / np.sqrt(HID) * 0.1).astype(np.float32),
        "b2": np.zeros(V, np.float32),
    }


def batch(data, B, rng):
    ix = rng.integers(0, len(data) - CTX - 1, B)
    X = np.stack([data[i:i + CTX] for i in ix])
    Y = data[ix + CTX]
    return X, Y


ACT = "relu"
FP16 = False  # True mimics mixed-precision training without loss scaling: every matmul accumulates in fp32
#               (as tensor cores do) and its output is stored in fp16, so anything above 65504 becomes inf.


def mm(a, b):
    out = a @ b
    if FP16:
        with np.errstate(all="ignore"):
            out = out.astype(np.float16).astype(np.float32)
    return out


def forward(P, X, Y):
    with np.errstate(all="ignore"):
        emb = P["C"][X].reshape(len(X), -1)
        h_pre = mm(emb, P["W1"]) + P["b1"]
        h = np.tanh(h_pre) if ACT == "tanh" else np.maximum(h_pre, 0)
        logits = mm(h, P["W2"]) + P["b2"]
    m = logits.max(1, keepdims=True)
    with np.errstate(all="ignore"):
        lse = m + np.log(np.exp(logits - m).sum(1, keepdims=True))
        loss = float((lse[:, 0] - logits[np.arange(len(Y)), Y]).mean())
    return loss, (X, emb, h_pre, h, logits, lse)


def backward(P, cache, Y):
    X, emb, h_pre, h, logits, lse = cache
    B = len(Y)
    with np.errstate(all="ignore"):
        dlogits = np.exp(logits - lse)
        dlogits[np.arange(B), Y] -= 1
        dlogits /= B
        g = {"W2": mm(h.T, dlogits), "b2": dlogits.sum(0)}
        dh = mm(dlogits, P["W2"].T)
        if ACT == "tanh":
            dh *= 1 - h ** 2
        else:
            dh[h_pre <= 0] = 0
        g["W1"] = mm(emb.T, dh)
        g["b1"] = dh.sum(0)
        demb = mm(dh, P["W1"].T).reshape(B, CTX, EMB)
        dC = np.zeros_like(P["C"])
        np.add.at(dC, X, demb)
        g["C"] = dC
    return g


def evaluate(P, data, rng, batches):
    return float(np.mean([forward(P, *batch(data, 512, rng))[0] for _ in range(batches)]))


def save_ckpt(P, M, Vv, step, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, step=step, **{k: v for k, v in P.items()}, **{"m_" + k: v for k, v in M.items()},
             **{"v_" + k: v for k, v in Vv.items()})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "costguard-datasets/shakespeare/input.txt"))
    ap.add_argument("--lr", type=float, required=True, help="peak learning rate")
    ap.add_argument("--warmup", type=int, default=50000)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--act", choices=["relu", "tanh"], default="relu")
    ap.add_argument("--optim", choices=["sgd", "adam"], default="sgd")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    global ACT, FP16
    ACT, FP16 = a.act, a.precision == "fp16"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, val, V = load_text(a.data)
    rng = np.random.default_rng(a.seed)
    P = init_params(V, rng)
    M = {k: np.zeros_like(v) for k, v in P.items()}
    Vv = {k: np.zeros_like(v) for k, v in P.items()}
    start = 0
    if a.resume:
        z = np.load(a.resume)
        start = int(z["step"])
        P = {k: z[k] for k in P}
        M = {k: z["m_" + k] for k in M}
        Vv = {k: z["v_" + k] for k in Vv} if "v_C" in z.files else Vv
        rng = np.random.default_rng(a.seed + start)
    log = open(out / "log.jsonl", "w")

    def emit(**kw):
        log.write(json.dumps(kw) + "\n")

    emit(kind="config", **vars(a), vocab=V, params=int(sum(v.size for v in P.values())))
    t0 = time.time()
    for step in range(start + 1, a.steps + 1):
        lr = a.lr * min(1.0, step / a.warmup)
        X, Y = batch(train, a.batch, rng)
        loss, cache = forward(P, X, Y)
        g = backward(P, cache, Y)
        with np.errstate(all="ignore"):
            gnorm = float(np.sqrt(sum(float((v.astype(np.float64) ** 2).sum()) for v in g.values())))
            for k in P:
                if a.optim == "adam":
                    M[k] = 0.9 * M[k] + 0.1 * g[k]
                    Vv[k] = 0.999 * Vv[k] + 0.001 * g[k] ** 2
                    mh, vh = M[k] / (1 - 0.9 ** step), Vv[k] / (1 - 0.999 ** step)
                    P[k] = P[k] - lr * mh / (np.sqrt(vh) + 1e-8)
                else:
                    M[k] = a.momentum * M[k] + g[k]
                    P[k] = P[k] - lr * M[k]
        if step % a.log_every == 0:
            emit(kind="train", step=step, loss=None if not np.isfinite(loss) else round(loss, 4),
                 loss_raw=str(loss), grad_norm=None if not np.isfinite(gnorm) else round(gnorm, 3), lr=round(lr, 6))
        if step % a.ckpt_every == 0:
            p = out / "ckpt" / f"step_{step}.npz"
            sha = save_ckpt(P, M, Vv, step, p)
            has_nan = any(not np.isfinite(v).all() for v in P.values())
            emit(kind="ckpt", step=step, file=p.name, sha256=sha, bytes=p.stat().st_size, params_finite=not has_nan)
        if step % a.eval_every == 0:
            emit(kind="eval", step=step, val_loss=evaluate(P, val, rng, a.eval_batches))
    emit(kind="done", step=a.steps, seconds=round(time.time() - t0, 1))
    log.close()


if __name__ == "__main__":
    main()
