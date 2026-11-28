"""Pull real overnight GPU-utilization traces out of the Microsoft Philly trace (ATC'19).

Source: https://github.com/msr-fiddle/philly-traces  (Jeon et al., "Analysis of Large-Scale Multi-Tenant GPU
Clusters for DNN Training Workloads", USENIX ATC 2019). 117,325 jobs, Aug-Dec 2017, per-minute nvidia-smi.

The 6.6 GB archive is never unpacked in full: cluster_gpu_util is streamed out of the .tar.gz and only the rows
for the selected jobs' machines and time windows are kept.

    python pipeline/philly_extract.py --philly ../costguard-datasets/philly --out data/philly
Writes:
    <out>/job_stats.json        GPU-hours by final job status (whole trace)
    <out>/candidates.json       overnight single-node 8-GPU job attempts
    <out>/candidate_util.csv    per-minute util (mean of the job's 8 GPUs) for each candidate
"""
import argparse
import csv
import io
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

FMT = "%Y-%m-%d %H:%M:%S"


def parse(t):
    return datetime.strptime(t, FMT) if t and t != "None" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--philly", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.philly), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = json.load(open(root / "trace-data" / "cluster_job_log"))

    # ---- whole-trace stats: GPU-hours by final status --------------------------------------------
    gh = defaultdict(float)
    n = defaultdict(int)
    for j in jobs:
        n[j["status"]] += 1
        for at in j["attempts"]:
            s, e = parse(at.get("start_time")), parse(at.get("end_time"))
            if not s or not e:
                continue
            gpus = sum(len(d["gpus"]) for d in at["detail"])
            gh[j["status"]] += gpus * (e - s).total_seconds() / 3600
    tot = sum(gh.values())
    stats = {"jobs": dict(n), "gpu_hours": {k: round(v) for k, v in gh.items()},
             "gpu_hours_share": {k: round(v / tot, 4) for k, v in gh.items()}, "total_gpu_hours": round(tot)}
    (out / "job_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

    # ---- candidates: one attempt on one machine using all 8 GPUs, launched 20:00-23:59, 5-30 h long --------
    cands = []
    for j in jobs:
        for k, at in enumerate(j["attempts"]):
            s, e = parse(at.get("start_time")), parse(at.get("end_time"))
            if not s or not e or len(at["detail"]) != 1 or len(at["detail"][0]["gpus"]) != 8:
                continue
            dur = (e - s).total_seconds() / 3600
            if 20 <= s.hour <= 23 and 5 <= dur <= 30:
                cands.append({"jobid": j["jobid"], "status": j["status"], "attempt": k, "n_attempts": len(j["attempts"]),
                              "machine": at["detail"][0]["ip"], "start": at["start_time"], "end": at["end_time"],
                              "hours": round(dur, 2), "user": j["user"], "vc": j["vc"]})
    (out / "candidates.json").write_text(json.dumps(cands, indent=1))
    print(len(cands), "candidates", {s: sum(c["status"] == s for c in cands) for s in ("Pass", "Failed", "Killed")})

    # ---- stream per-minute util for candidate windows --------------------------------------------------------
    win = defaultdict(list)  # machine -> [(start, end, idx)]
    for i, c in enumerate(cands):
        win[c["machine"]].append((parse(c["start"]), parse(c["end"]) + timedelta(hours=1), i))
    proc = subprocess.Popen(["tar", "-xzOf", str(root / "trace-data.tar.gz"), "trace-data/cluster_gpu_util"],
                            stdout=subprocess.PIPE)
    w = csv.writer(open(out / "candidate_util.csv", "w", newline=""))
    w.writerow(["cand", "time", "gpu_mean", "gpu_min", "gpu_max", "n_na"])
    kept = 0
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace"):
        parts = line.rstrip("\n").split(",")
        if len(parts) < 10 or parts[1] not in win:
            continue
        try:
            t = datetime.strptime(parts[0][:19], FMT)
        except ValueError:
            continue
        for s, e, i in win[parts[1]]:
            if s <= t <= e:
                vals = [float(v) for v in parts[2:10] if v not in ("NA", "")]
                if vals:
                    w.writerow([i, parts[0][:19], round(sum(vals) / len(vals), 1), round(min(vals), 1),
                                round(max(vals), 1), 8 - len(vals)])
                    kept += 1
    proc.wait()
    print("kept", kept, "util rows")


if __name__ == "__main__":
    main()
