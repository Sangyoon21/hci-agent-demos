"""Build data/scenario.json for the overnight demo from REAL data.

Inputs (made by the other pipeline scripts; see pipeline/README.md):
  runs/<name>/log.jsonl + ckpt/*.npz   real training runs (train_charlm.py, Tiny Shakespeare)
  philly/extract/*                     real per-minute GPU utilization (philly_extract.py, Microsoft Philly trace)

What is real and what is mapped:
  * loss, val loss, gradient norm, learning rate, checkpoint files, checkpoint health: real, from the runs.
  * GPU utilization: real per-minute nvidia-smi from three Philly jobs (Oct-Nov 2017), replayed from their
    own launch minute onto a 10:14 PM launch.
  * Clock: the runs are small CPU runs, so training steps are spread linearly over the night
    (a fixed steps-per-minute per run). This mapping is the one thing chosen by hand, and it is printed.
  * AI answers: written in advance (no API key in the demo), but every number in them is computed here.

    python pipeline/build_scenario.py --datasets ../costguard-datasets --out data/scenario.json
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import train_charlm as T  # noqa: E402

NIGHT_MIN = 540           # 10:14 PM -> 7:14 AM
PRICE_HR = 27.45          # AWS p4de.24xlarge (8x A100 80GB) on-demand, us-east-1
CHANCE_BAND = 0.15        # "stuck at chance": smoothed loss within this of the letter-frequency loss
STUCK_MIN = 10            # ... for this many minutes
SILENT_MIN = 20           # no train-loss line for this long -> flag
IDLE_PCT, IDLE_MIN = 5, 30


def load_log(p):
    return [json.loads(l) for l in open(p)]


def ema(xs, a=0.1):
    out, e = [], None
    for x in xs:
        e = x if e is None else (1 - a) * e + a * x
        out.append(e)
    return out


def philly_series(E, cand, cands, util):
    c = cands[cand]
    u = util[util.cand == cand].sort_values("time")
    s = pd.Timestamp(c["start"]).floor("min")
    m = ((u.time - s).dt.total_seconds() // 60).astype(int)
    ser = pd.Series(u.gpu_mean.values, index=m.values).groupby(level=0).mean()
    ser = ser.reindex(range(0, ser.index.max() + 1)).interpolate(limit=3)
    return ser, c


def ckpt_health(path, val, rng_seed=0):
    z = np.load(path)
    P = {k: z[k] for k in ("C", "W1", "b1", "W2", "b2")}
    finite = all(bool(np.isfinite(v).all()) for v in P.values())
    rng = np.random.default_rng(rng_seed)
    X, Y = T.batch(val, 4096, rng)
    emb = P["C"][X].reshape(len(X), -1)
    h = np.maximum(emb @ P["W1"] + P["b1"], 0)
    dead = float((h.max(0) == 0).mean())
    loss, _ = T.forward(P, X, Y)
    return {"step": int(z["step"]), "finite": finite, "dead_pct": round(100 * dead, 1), "val_loss": round(loss, 3),
            "bytes": Path(path).stat().st_size}


def hm(minute):
    t = (22 * 60 + 14 + int(round(minute))) % 1440
    h, mm = divmod(t, 60)
    return f"{(h % 12) or 12}:{mm:02d} {'AM' if h < 12 else 'PM'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = Path(a.datasets)
    E = D / "philly" / "extract"
    cands = json.load(open(E / "candidates.json"))
    util = pd.read_csv(E / "candidate_util.csv", parse_dates=["time"])
    train, val, V = T.load_text(D / "shakespeare" / "input.txt")
    p = np.bincount(train, minlength=V) / len(train)
    chance = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    runs = []

    # ================= A: a100-train-2 — learning rate too high, model collapses ======================
    LA = load_log(D / "runs/relu01/log.jsonl")
    trA = [x for x in LA if x["kind"] == "train"]
    smA = ema([x["loss"] for x in trA])
    # onset = first step after which the smoothed loss never leaves the chance band for STUCK_MIN minutes;
    # place it at 1:36 AM (minute 202) as in the journey map, which fixes steps/minute for this run.
    band = [s >= chance - CHANCE_BAND for s in smA]
    on_i = next(i for i in range(len(band)) if all(band[i:i + 60]))
    kA = trA[on_i]["step"] / 202.0
    fireA = 202 + STUCK_MIN
    ckA = sorted(glob.glob(str(D / "runs/relu01/ckpt/*.npz")), key=lambda s: int(re.findall(r"\d+", Path(s).name)[0]))
    healthA = [dict(ckpt_health(f, val), minute=round(int(re.findall(r"\d+", Path(f).name)[0]) / kA, 1),
                    file=f"step_{re.findall(r'[0-9]+', Path(f).name)[0]}.pt") for f in ckA]
    healthA = [h for h in healthA if h["minute"] <= NIGHT_MIN]
    gpuA, cA = philly_series(E, 0, cands, util)
    lr_rise = next(x for x in trA if x["step"] >= 6000)["lr"]
    evA = [x for x in LA if x["kind"] == "eval"]
    fixL = load_log(D / "runs/relu01_fix_lr0003/log.jsonl")
    runs.append({
        "id": "a100-train-2", "title": "char-LM run 7 (lr 0.1)", "story": "collapse",
        "cmd": "python train.py --lr 0.1", "steps_per_min": round(kA, 2), "total_steps": trA[-1]["step"],
        "loss": [[round(x["step"] / kA, 2), x["loss"], round(s, 4), x["grad_norm"], x["lr"]] for x, s in zip(trA, smA)
                 if x["step"] / kA <= NIGHT_MIN],
        "val": [[round(x["step"] / kA, 1), round(x["val_loss"], 3)] for x in evA if x["step"] / kA <= NIGHT_MIN],
        "checkpoints": healthA,
        "gpu": [None if np.isnan(v) else round(float(v), 1) for v in gpuA.reindex(range(NIGHT_MIN + 1)).values],
        "philly": {k: cA[k] for k in ("jobid", "status", "machine", "start", "end", "hours")},
        "flag": {"minute": fireA, "rule": "stuck_at_chance", "onset": 202,
                 "detail": f"smoothed loss ≥ {chance - CHANCE_BAND:.2f} for {STUCK_MIN} min (chance level {chance:.2f})"},
        "fix": {"from": "--lr 0.1", "to": "--lr 3e-3 --resume ckpt/step_6000.pt",
                "val": [[x["step"], round(x["val_loss"], 3)] for x in fixL if x["kind"] == "eval"],
                "note": "Real run: resumed from step_6000 with peak lr 3e-3 and trained to step 60000."},
        "lr_at_rise": lr_rise,
    })

    # ================= B: a100-train-1 — healthy run, slow full-validation pass =========================
    gpuB, cB = philly_series(E, 163, cands, util)
    g = gpuB.values
    dip0 = next(i for i in range(140, len(g)) if g[i] < 30)
    dip1 = next(i for i in range(dip0, len(g)) if g[i] > 70)
    LB = load_log(D / "runs/relu0003/log.jsonl")
    trB = [x for x in LB if x["kind"] == "train"]
    eval_step = 2000 * round(100 * dip0 / 2000)
    kB = eval_step / dip0
    dur = dip1 - dip0

    def tB(step):
        m = step / kB
        return m if step <= eval_step else m + dur
    zB = np.load(D / f"runs/relu0003/ckpt/step_{eval_step}.npz")
    PB = {k: zB[k] for k in ("C", "W1", "b1", "W2", "b2")}
    nb = (len(val) - T.CTX - 1) // 512
    prog, tot = [], 0.0
    for b in range(nb):
        ix = np.arange(b * 512, (b + 1) * 512)
        X = np.stack([val[i:i + T.CTX] for i in ix]); Y = val[ix + T.CTX]
        l, _ = T.forward(PB, X, Y); tot += l
        if (b + 1) % 10 == 0 or b == nb - 1:
            prog.append([round(dip0 + dur * (b + 1) / nb, 1), b + 1, nb, round(tot / (b + 1), 4)])
    smB = ema([x["loss"] for x in trB])
    runs.append({
        "id": "a100-train-1", "title": "char-LM run 6 (lr 3e-3)", "story": "slow_eval",
        "cmd": "python train.py --lr 3e-3 --full-eval-every 30000", "steps_per_min": round(kB, 2),
        "loss": [[round(tB(x["step"]), 2), x["loss"], round(s, 4), x["grad_norm"], x["lr"]] for x, s in zip(trB, smB)
                 if tB(x["step"]) <= NIGHT_MIN],
        "eval": {"step": eval_step, "start": dip0, "end": dip1, "batches": nb, "progress": prog,
                 "val_loss": round(tot / nb, 4)},
        "checkpoints": [dict(ckpt_health(f, val), minute=round(tB(int(re.findall(r"\d+", Path(f).name)[0])), 1),
                             file=f"step_{re.findall(r'[0-9]+', Path(f).name)[0]}.pt")
                        for f in sorted(glob.glob(str(D / "runs/relu0003/ckpt/*.npz")),
                                        key=lambda s: int(re.findall(r"\d+", Path(s).name)[0]))
                        if tB(int(re.findall(r"\d+", Path(f).name)[0])) <= NIGHT_MIN][-3:],
        "gpu": [None if np.isnan(v) else round(float(v), 1) for v in gpuB.reindex(range(NIGHT_MIN + 1)).values],
        "philly": {k: cB[k] for k in ("jobid", "status", "machine", "start", "end", "hours")},
        "flag": {"minute": dip0 + SILENT_MIN, "rule": "silent", "onset": dip0,
                 "detail": f"no training-loss line for {SILENT_MIN} min and GPU fell below 30%"},
        "naive_kill": dip0 + 15,
    })

    # ================= C: a100-train-3 — finished at night, GPUs idle until morning =====================
    gpuC, cC = philly_series(E, 66, cands, util)
    gc = gpuC.values
    busy = next(i for i in range(len(gc)) if gc[i] < IDLE_PCT)
    LC = load_log(D / "runs/tanh0003/log.jsonl")
    trC = [x for x in LC if x["kind"] == "train"]
    kC = trC[-1]["step"] / busy
    smC = ema([x["loss"] for x in trC])
    ckC = sorted(glob.glob(str(D / "runs/tanh0003/ckpt/*.npz")), key=lambda s: int(re.findall(r"\d+", Path(s).name)[0]))
    T.ACT = "tanh"
    lastC = ckpt_health(ckC[-1], val)
    T.ACT = "relu"
    lastC["dead_pct"] = None  # tanh units do not die; health = finite + val loss
    lastC.update(minute=round(busy, 1), file=f"step_{lastC['step']}.pt")
    idle_from = busy
    runs.append({
        "id": "a100-train-3", "title": "char-LM run 5 (tanh, lr 3e-3)", "story": "finished",
        "cmd": "python train.py --act tanh --lr 3e-3 --steps 60000", "steps_per_min": round(kC, 2),
        "loss": [[round(x["step"] / kC, 2), x["loss"], round(s, 4), x["grad_norm"], x["lr"]] for x, s in zip(trC, smC)],
        "val": [[round(x["step"] / kC, 1), round(x["val_loss"], 3)] for x in LC if x["kind"] == "eval"],
        "checkpoints": [lastC],
        "finished_at": busy,
        "gpu": [None if np.isnan(v) else round(float(v), 1) for v in gpuC.reindex(range(NIGHT_MIN + 1)).values],
        "philly": {k: cC[k] for k in ("jobid", "status", "machine", "start", "end", "hours")},
        "flag": {"minute": idle_from + IDLE_MIN, "rule": "idle", "onset": idle_from,
                 "detail": f"GPU below {IDLE_PCT}% for {IDLE_MIN} min and the training process has exited"},
    })

    stats = json.load(open(E / "job_stats.json"))
    out = {
        "night": {"start": "10:14 PM", "minutes": NIGHT_MIN, "price_hr": PRICE_HR,
                  "price_note": "AWS p4de.24xlarge (8x A100 80GB) on-demand, us-east-1, after AWS's June 2025 P4/P5 price cut. Check aws.amazon.com/ec2/pricing before quoting."},
        "chance_loss": round(chance, 3),
        "rules": {"chance_band": CHANCE_BAND, "stuck_min": STUCK_MIN, "silent_min": SILENT_MIN,
                  "idle_pct": IDLE_PCT, "idle_min": IDLE_MIN, "healthy_dead_pct_max": 1.0,
                  "min_confidence": 0.8},
        "philly_stats": {**stats, "overnight": {"jobs": 154, "idle_hour_jobs": 87, "idle_minute_share": 0.295,
                                                 "robust_jobs": 140, "robust_idle_hour_jobs": 73,
                                                 "robust_idle_minute_share": 0.246}},
        "runs": runs,
    }
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", a.out, Path(a.out).stat().st_size, "bytes")
    for r in runs:
        print(f"{r['id']}: {r['steps_per_min']} steps/min, flag at {hm(r['flag']['minute'])} ({r['flag']['rule']})")
    print("A healthy ckpts:", [(h["file"], h["dead_pct"], h["val_loss"]) for h in healthA if h["dead_pct"] < 1])
    print("A latest ckpt before flag:", [(h["file"], h["dead_pct"], h["val_loss"]) for h in healthA if h["minute"] <= fireA][-1])
    print("B eval:", hm(dip0), "->", hm(dip1), "step", eval_step, "val", round(tot / nb, 3))
    print("C finished:", hm(busy), "final", lastC)


if __name__ == "__main__":
    main()
