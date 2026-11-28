"""A SIMULATED cloud. No real AWS/GCP API is ever called.

The only operations that exist are:
    pause(id)  restart(id)  get_metrics(id)  get_logs(id)
There is deliberately NO delete (or resize, or terminate) function: the "cloud
permission" CostGuard holds literally cannot destroy anything.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta

import config

PERMISSIONS = ("pause", "restart", "get_metrics", "get_logs")  # no delete


class Cloud:
    def __init__(self):
        data = json.loads((config.DATA_DIR / "machines.json").read_text())
        self.clock = datetime.fromisoformat(config.START_TIME)
        self.machines: dict[str, dict] = {}
        for raw in data["machines"]:
            self.machines[raw["id"]] = self._build(copy.deepcopy(raw))

    # ------------------------------------------------------------ setup
    def _build(self, m: dict) -> dict:
        now = self.clock
        n = len(m["history"]["gpu"])
        m["times"] = [now - timedelta(hours=n - 1 - i) for i in range(n)]
        lines = [(now - timedelta(minutes=l["min_ago"]), l["msg"]) for l in m.pop("logs")]
        for t, g, c, net, d in zip(m["times"], m["history"]["gpu"], m["history"]["cpu"],
                                   m["history"]["net"], m["history"]["disk"]):
            lines.append((t, self._monitor_line(g, c, net, d)))
        m["log"] = sorted(lines, key=lambda x: x[0])
        ck = m.pop("last_checkpoint_min_ago")
        m["last_checkpoint"] = now - timedelta(minutes=ck) if ck is not None else None
        m["commits"] = [{"time": now - timedelta(minutes=c["min_ago"]), "author": c["author"], "msg": c["msg"]}
                        for c in m["commits"]]
        m["has_gpu"] = any(g is not None for g in m["history"]["gpu"])
        m["state"] = "running"
        m["paused_at"] = None
        m["paused_by"] = None
        m["paused_minutes"] = 0.0
        m["step"] = 0            # index into future / after_restart script
        m["restarted"] = False
        return m

    @staticmethod
    def _monitor_line(g, c, net, d) -> str:
        gpu = "n/a" if g is None else f"{g}%"
        return f"[monitor] gpu_util={gpu} cpu={c}% net={net}MB/s disk_write={d}MB/s"

    # -------------------------------------------------------- time
    def advance_to(self, t: datetime):
        """Move the clock forward, accruing paused time (= money saved)."""
        if t <= self.clock:
            return
        mins = (t - self.clock).total_seconds() / 60
        for m in self.machines.values():
            if m["state"] == "paused":
                m["paused_minutes"] += mins
        self.clock = t

    def record_hour(self):
        """Append one hourly metrics sample per machine (called at the end of each simulated hour)."""
        for m in self.machines.values():
            if m["state"] == "paused":
                s = {"gpu": 0 if m["has_gpu"] else None, "cpu": 0, "net": 0.0, "disk": 0, "logs": []}
            else:
                script = m["after_restart"] if m["restarted"] else m["future"]
                s = script[min(m["step"], len(script) - 1)]
                m["step"] += 1
            for k in ("gpu", "cpu", "net", "disk"):
                m["history"][k] = m["history"][k][1:] + [s[k]]
            m["times"] = m["times"][1:] + [self.clock]
            for i, line in enumerate(s.get("logs", [])):
                m["log"].append((self.clock - timedelta(minutes=50 - 7 * i), line))
                if "[ckpt]" in line and ("complete" in line or "saved" in line):
                    m["last_checkpoint"] = self.clock - timedelta(minutes=50 - 7 * i)
            if m["state"] == "running":
                m["log"].append((self.clock, self._monitor_line(s["gpu"], s["cpu"], s["net"], s["disk"])))

    # ------------------------------------------------- the 4 operations
    def get_metrics(self, mid: str) -> dict:
        m = self.machines[mid]
        return {
            "id": mid, "state": m["state"], "env": m["env"], "price_hr": m["price_hr"],
            "times": list(m["times"]), "gpu": list(m["history"]["gpu"]), "cpu": list(m["history"]["cpu"]),
            "net": list(m["history"]["net"]), "disk": list(m["history"]["disk"]),
            "last_checkpoint": m["last_checkpoint"], "commits": list(m["commits"]), "tags": dict(m["tags"]),
        }

    def get_logs(self, mid: str, n: int = config.LOG_LINES_FOR_AI) -> list[str]:
        return [f"{t:%Y-%m-%d %H:%M} {msg}" for t, msg in self.machines[mid]["log"][-n:]]

    def pause(self, mid: str, by: str) -> bool:
        m = self.machines[mid]
        if m["state"] == "paused":
            return False
        m["state"], m["paused_at"], m["paused_by"] = "paused", self.clock, by
        m["log"].append((self.clock, f"[cloud] instance stopped by {by}; disk and checkpoints kept"))
        return True

    def restart(self, mid: str, by: str) -> bool:
        m = self.machines[mid]
        if m["state"] == "running":
            return False
        m["state"], m["paused_at"], m["paused_by"] = "running", None, None
        m["restarted"], m["step"] = True, 0
        m["log"].append((self.clock, f"[cloud] instance started by {by}"))
        return True

    # ---------------------------------------------------- derived
    def saved_usd(self, mid: str) -> float:
        m = self.machines[mid]
        return m["paused_minutes"] / 60 * m["price_hr"]
