"""CostGuard agent loop. Runs once per simulated hour:

  1. WATCH   plain rules, no AI: GPU < 5% for 2+ hours and no network traffic -> flagged
  2. JUDGE   AI, only for flagged machines -> {machine_id, verdict, action, reason, confidence, est_savings}
  3. ACT     policy.py decides (not the AI); every pause/restart passes policy.check_execution
  4. REPORT  savings = hours paused x hourly price (weekly summary panel)

The World object holds the simulated cloud, the simulated Slack channel and the trace.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Callable, Optional

import config
import llm
import mock_responses as mocks
import policy
import prompts
from cloud_sim import Cloud

BOT = "CostGuard"


def fmt_t(t: datetime) -> str:
    return t.strftime("%-I:%M %p")


def fmt_ago(minutes: float) -> str:
    h = minutes / 60
    if h < 1:
        return f"{int(minutes)} min ago"
    return f"{h:.0f} hour{'s' if round(h) != 1 else ''} ago"


def money(x: float) -> str:
    x = 0.0 if abs(x) < 0.005 else x
    return f"${x:,.2f}" if abs(x) < 1000 else f"${x:,.0f}"


class World:
    def __init__(self, backend: Optional[str] = None):
        self.backend = backend or config.BACKEND
        self.cloud = Cloud()
        self.messages: list[dict] = []
        self.trace: list[dict] = []      # one entry per hourly run
        self.pending_events: list[dict] = []  # things that happened since the last run
        self.scheduled: list[dict] = []
        self.ai_calls: list[dict] = []
        self.blocked: list[dict] = []
        self.hours_run = 0
        self._seq = 0
        self.p = {}
        for mid, m in self.cloud.machines.items():
            self.p[mid] = {"undo_count": len(m.get("undo_history", [])), "rule_on": True, "snoozed_until": None,
                           "held": False, "approval_msg": None, "pending_pause_msg": None,
                           "paused_msg": None, "last_verdict": "—", "keep_posted": False}
        self.event(f"Simulation started. CostGuard's cloud permission: {', '.join(config.ALLOWED_ACTIONS)}, "
                   "read metrics, read logs. There is no delete.", "info")

    # ----------------------------------------------------------- helpers
    @property
    def now(self) -> datetime:
        return self.cloud.clock

    def m(self, mid):
        return self.cloud.machines[mid]

    def event(self, text, level="info", **kw):
        self.pending_events.append({"time": self.now, "text": text, "level": level, **kw})

    def post(self, kind, mid=None, author=BOT, **fields) -> dict:
        self._seq += 1
        msg = {"id": self._seq, "time": self.now, "author": author, "kind": kind, "mid": mid,
               "title": "", "lines": [], "note": "", "rollback": [], "buttons": [], "resolved": None, **fields}
        self.messages.append(msg)
        return msg

    def msg(self, msg_id) -> Optional[dict]:
        return next((x for x in self.messages if x["id"] == msg_id), None)

    def resolve(self, msg_id, text):
        x = self.msg(msg_id)
        if x and not x["resolved"]:
            x["resolved"], x["buttons"] = text, []

    def _eod(self) -> datetime:
        return self.now.replace(hour=23, minute=59, second=0, microsecond=0)

    # ------------------------------------------------- the only way to act
    def execute(self, action: str, mid: str, actor: str) -> bool:
        ok, why = policy.check_execution(action, self.m(mid), actor)
        if not ok:
            self.blocked.append({"time": self.now, "mid": mid, "why": why})
            self.event(f"POLICY BLOCKED {action}({mid}) by {actor}: {why}", "block")
            return False
        done = getattr(self.cloud, action)(mid, by=actor)
        self.event(f"cloud.{action}({mid}) by {actor}: {'done' if done else 'no change'}", "act")
        return done

    # --------------------------------------------------------- WATCH
    def watch(self, mid) -> dict:
        mt = self.cloud.get_metrics(mid)
        low = 0
        for g, c, n in zip(reversed(mt["gpu"]), reversed(mt["cpu"]), reversed(mt["net"])):
            busy_signal = (g if g is not None else c)
            if busy_signal < config.IDLE_GPU_PCT and n < config.IDLE_MAX_NET_MBPS:
                low += 1
            else:
                break
        gpus = [g for g in mt["gpu"][len(mt["gpu"]) - low:]] if low else []
        flagged = low >= config.IDLE_MIN_HOURS
        if mt["gpu"][-1] is None:
            ev = f"CPU at {mt['cpu'][-1]}%, network {mt['net'][-1]} MB/s"
        elif low:
            lo, hi = min(gpus), max(gpus)
            ev = f"GPU at {lo}% for {low} hours" if lo == hi else f"GPU at {lo}-{hi}% for {low} hours"
        else:
            ev = f"GPU at {mt['gpu'][-1]}%"
        if mt["last_checkpoint"]:
            ev += f", last save {fmt_ago((self.now - mt['last_checkpoint']).total_seconds() / 60)}"
        if low and mt["net"][-1] < config.IDLE_MAX_NET_MBPS:
            ev += ", no network traffic"
        return {"mid": mid, "low_hours": low, "flagged": flagged, "evidence": ev,
                "gpu_now": mt["gpu"][-1], "net_now": mt["net"][-1]}

    def _skip_reason(self, mid) -> Optional[str]:
        p, m = self.p[mid], self.m(mid)
        if m["state"] == "paused":
            return "already paused"
        if p["held"]:
            return "held for human review (policy block)"
        if not p["rule_on"]:
            return f"auto-pause rule switched off after {p['undo_count']} undos"
        if p["snoozed_until"] and self.now < p["snoozed_until"]:
            return f"owner said keep; snoozed until {fmt_t(p['snoozed_until'])}"
        if p["approval_msg"] and not self.msg(p["approval_msg"])["resolved"]:
            return "approval request already pending"
        if p["pending_pause_msg"] and not self.msg(p["pending_pause_msg"])["resolved"]:
            return "pause already scheduled"
        return None

    # --------------------------------------------------------- JUDGE
    def _judge_prompt(self, mid) -> tuple[str, list[str]]:
        m, mt = self.m(mid), self.cloud.get_metrics(mid)
        logs = self.cloud.get_logs(mid)
        rows = "\n".join(
            f"{t:%H:%M}  gpu={'n/a' if g is None else str(g) + '%'}  cpu={c}%  net={n}  disk={d}"
            for t, g, c, n, d in zip(mt["times"], mt["gpu"], mt["cpu"], mt["net"], mt["disk"]))
        ck = mt["last_checkpoint"]
        commits = "\n".join(f"- {c['time']:%Y-%m-%d %H:%M} {c['author']}: {c['msg']}" for c in mt["commits"]) or "- none"
        user = prompts.JUDGE_USER.format(
            id=mid, name=m["name"], hardware=m["hardware"], env=m["env"], owner=m["owner"], price_hr=m["price_hr"],
            tags=json.dumps(m["tags"]), now=f"{self.now:%Y-%m-%d %H:%M}", metrics_table=rows,
            last_checkpoint=f"{ck:%Y-%m-%d %H:%M} ({fmt_ago((self.now - ck).total_seconds() / 60)})" if ck else "none recorded",
            commits=commits, n_logs=len(logs), logs="\n".join(logs))
        input_texts = logs + [f"{k}: {v}" for k, v in m["tags"].items()] + [c["msg"] for c in mt["commits"]]
        return user, input_texts

    def _ai(self, kind, mid, system, user, mock) -> tuple[llm.LLMResult, Optional[dict]]:
        res = llm.call(system, user, mock=mock, backend=self.backend)
        self.ai_calls.append({"time": self.now, "kind": kind, "mid": mid, "cost": res.cost_usd,
                              "in": res.input_tokens, "out": res.output_tokens, "model": res.model})
        return res, llm.parse_json(res.text)

    # ----------------------------------------------------------- run
    def run_hour(self, on_step: Optional[Callable[[str], None]] = None) -> dict:
        note = on_step or (lambda s: None)
        end = self.now + timedelta(hours=1)
        # Scheduled things inside this hour happen in time order (pauses after warnings, owner replies, restarts).
        while True:
            due = sorted([e for e in self.scheduled if e["time"] <= end], key=lambda e: e["time"])
            if not due:
                break
            e = due[0]
            self.scheduled.remove(e)
            self.cloud.advance_to(e["time"])
            self._fire(e)
        self.cloud.advance_to(end)
        self.cloud.record_hour()
        self.hours_run += 1

        entry = {"time": self.now, "events": self.pending_events, "watch": [], "judge": []}
        self.pending_events = []
        note("1. WATCH: plain rules on all machines")
        flagged = []
        for mid in self.cloud.machines:
            w = self.watch(mid)
            w["skip"] = self._skip_reason(mid) if w["flagged"] else None
            entry["watch"].append(w)
            if w["flagged"] and not w["skip"]:
                flagged.append(w)

        for w in flagged:
            mid, m = w["mid"], self.m(w["mid"])
            note(f"2. JUDGE: asking the AI about {mid}")
            user, input_texts = self._judge_prompt(mid)
            res, parsed = self._ai("judge", mid, prompts.JUDGE_SYSTEM, user,
                                   mock=lambda m=m, mid=mid: mocks.judge(m, self.cloud.get_logs(mid)))
            note(f"3. ACT: policy check for {mid}")
            dec = policy.decide(m, parsed, res.error, res.text, self.cloud.machines, input_texts)
            entry["judge"].append({"mid": mid, "system": prompts.JUDGE_SYSTEM, "user": user, "raw": res.text,
                                   "parsed": parsed, "error": res.error, "in": res.input_tokens,
                                   "out": res.output_tokens, "cost": res.cost_usd, "latency": res.latency_s,
                                   "model": res.model, "policy": dec})
            self._act(m, w, parsed or {}, dec)
        entry["events"] += self.pending_events
        self.pending_events = []
        self.trace.append(entry)
        return entry

    def _act(self, m, w, ai, dec):
        mid, p, price = m["id"], self.p[m["id"]], m["price_hr"]
        reason = str(ai.get("reason", "")).strip()
        try:
            conf = float(ai.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        cost_line = f"costs {money(price)} an hour ({money(price * 24)} a day)"
        o = dec["outcome"]
        verdict_label = {"pause_after_warning": "idle: pause", "keep": "busy: keep",
                         "approval_request": "idle: approval", "ask_owner": "unsure: ask",
                         "blocked": "🛑 blocked"}[o]
        p["last_verdict"] = verdict_label
        if o == "pause_after_warning":
            at = self.now + timedelta(minutes=config.PAUSE_GRACE_MINUTES)
            msg = self.post("warn", mid, to=m["owner"], title=f"{mid} looks idle",
                            lines=[("Evidence", w["evidence"]), ("Cost", cost_line),
                                   ("Action", f"I'll pause it at **{fmt_t(at)}** unless you say keep. Disk and checkpoints are kept.")],
                            note=f"AI ({conf:.0%} sure): {reason}",
                            buttons=[("keep", "Keep it", "secondary"), ("pause_now", "Pause now", "primary"),
                                     ("not_mine", "Not mine", "tertiary")])
            p["pending_pause_msg"] = msg["id"]
            self.scheduled.append({"time": at, "kind": "pause", "mid": mid, "msg_id": msg["id"]})
            self.event(f"{mid}: warned @{m['owner']}; pause scheduled for {fmt_t(at)}", "act")
        elif o == "keep":
            if not p["keep_posted"]:
                self.post("keep", mid, title=f"{mid} looked idle, but it's busy",
                          lines=[("Evidence", w["evidence"] + " (that's what tripped the rule)"),
                                 ("Cost", cost_line),
                                 ("Action", "**Keeping it running.** No action needed.")],
                          note=f"AI ({conf:.0%} sure): {reason}")
                p["keep_posted"] = True
            self.event(f"{mid}: AI says busy, keep running", "pass")
        elif o == "approval_request":
            until = datetime.combine(self.now.date(), datetime.strptime(config.PROD_RESTART_BEFORE, "%H:%M").time())
            if until <= self.now:
                until += timedelta(days=1)
            hrs = (until - self.now).total_seconds() / 3600
            msg = self.post("approval", mid, to=config.APPROVER, title=f"Approval needed: {mid} is production",
                            lines=[("Evidence", w["evidence"]), ("Cost", cost_line),
                                   ("Action", f"I **can't pause production** myself. Approve pausing it until "
                                              f"**{fmt_t(until)}**? Saves about {money(price * hrs)}.")],
                            note=f"AI ({conf:.0%} sure): {reason}",
                            rollback=[f"Restart {mid} automatically at {fmt_t(until)}, before morning traffic.",
                                      f"Anyone can click **Roll back** to call restart({mid}) immediately (warm-up ~4 min).",
                                      "After restart: check /healthz, confirm it re-registers with the load balancer.",
                                      "Nothing is deleted; disks, config and model weights stay as they are."],
                            buttons=[("approve", "Approve", "primary"), ("reject", "Reject", "secondary")],
                            until=until)
            p["approval_msg"] = msg["id"]
            self.event(f"{mid}: production, approval request sent to @{config.APPROVER}", "warn")
        elif o == "ask_owner":
            self.post("ask", mid, to=m["owner"], title=f"Is {mid} still in use?",
                      lines=[("Evidence", w["evidence"]), ("Cost", cost_line),
                             ("Action", "I'm **not sure** it's idle, so I won't touch it. Want me to pause it?")],
                      note=f"AI ({conf:.0%} sure): {reason or 'no usable answer'}",
                      buttons=[("pause_now", "Pause now", "primary"), ("keep", "Keep it", "secondary"),
                               ("not_mine", "Not mine", "tertiary")])
            p["snoozed_until"] = self.now + timedelta(hours=3)
            self.event(f"{mid}: unsure, asked @{m['owner']}", "warn")
        elif o == "blocked":
            p["held"] = True
            self.blocked.append({"time": self.now, "mid": mid, "why": dec.get("blocked_reason", "")})
            ai_wanted = f"{ai.get('action', '?')} **{ai.get('machine_id', '?')}**"
            lines = [("AI asked to", ai_wanted + (f": “{reason}”" if reason else "")),
                     ("Blocked because", "; ".join(c["msg"] for c in dec["checks"] if c["ok"] is False))]
            hits = policy.scan_for_injection(self._judge_prompt(mid)[1])
            if hits:
                lines.append(("Source", f"text inside {mid}'s logs/tags: `{hits[0][:110]}`"))
            lines.append(("Action", f"**Nothing was changed.** {mid} is held for human review. "
                                    f"@{m['owner']}, please check where that text came from."))
            self.post("blocked", mid, to=m["owner"], title="Policy blocked an action", lines=lines,
                      buttons=[("release", "Reviewed, release hold", "tertiary")])
            self.event(f"{mid}: POLICY BLOCKED ({dec.get('blocked_reason')})", "block")

    # ------------------------------------------------- scheduled events
    def _fire(self, e):
        mid = e.get("mid")
        if e["kind"] == "pause":
            warn = self.msg(e["msg_id"])
            if not warn or warn["resolved"]:
                return
            if self.execute("pause", mid, BOT):
                self.resolve(e["msg_id"], f"Paused at {fmt_t(self.now)}")
                self._post_paused(mid, by=BOT)
        elif e["kind"] == "owner_reply":
            self.reply(e["author"], e["text"])
        elif e["kind"] == "restart":
            self.p[mid]["snoozed_until"] = self._eod()
            if self.m(mid)["state"] == "paused" and self.execute("restart", mid, BOT):
                pm = self.p[mid]["paused_msg"]
                if pm:
                    self.resolve(pm, f"Restarted at {fmt_t(self.now)}")
                self.post("restarted", mid, title=f"Restarted ✓ {mid}",
                          lines=[("Action", f"Restarted at {fmt_t(self.now)} as planned, ahead of morning traffic. "
                                            f"Saved {money(self.cloud.saved_usd(mid))} overnight.")])

    def _post_paused(self, mid, by):
        m = self.m(mid)
        who = "I paused" if by == BOT else f"Paused by @{by}."
        msg = self.post("paused", mid, title=f"Paused ✓ {mid}",
                        lines=[("Action", f"{who} {mid} at {fmt_t(self.now)}. Saving {money(m['price_hr'])} an hour. "
                                          "Disk and checkpoints are kept. Reply **undo** to restart it.")],
                        buttons=[("undo", "Undo", "secondary")])
        self.p[mid]["paused_msg"] = msg["id"]
        script = m.get("owner_script")
        if script and by == BOT:
            self.scheduled.append({"time": self.now + timedelta(minutes=script["after_pause_min"]),
                                   "kind": "owner_reply", "author": m["owner"], "text": script["text"]})

    # ------------------------------------------------- human actions
    def button(self, msg_id: int, action: str):
        x = self.msg(msg_id)
        if not x or x["resolved"] is not None and action not in ("undo", "rollback", "rule_on"):
            return
        mid = x["mid"]
        actor = x.get("to") or self.m(mid)["owner"]
        if action == "approve":
            self.resolve(msg_id, f"Approved by @{actor} at {fmt_t(self.now)}")
            if self.execute("pause", mid, actor):
                until = x["until"]
                self.scheduled.append({"time": until, "kind": "restart", "mid": mid})
                pm = self.post("paused", mid, title=f"Paused ✓ {mid} (approved by @{actor})",
                               lines=[("Action", f"Paused at {fmt_t(self.now)}. I'll restart it at {fmt_t(until)}. "
                                                 f"Saving {money(self.m(mid)['price_hr'])} an hour.")],
                               buttons=[("rollback", "Roll back now", "secondary")])
                self.p[mid]["paused_msg"] = pm["id"]
                self.p[mid]["snoozed_until"] = self._eod()
            self.event(f"{mid}: approved by @{actor}", "act")
        elif action == "reject":
            self.resolve(msg_id, f"Rejected by @{actor}")
            self.p[mid]["snoozed_until"] = self._eod()
            self.post("info", mid, title=f"OK, leaving {mid} running",
                      lines=[("Action", "I won't ask about it again today.")])
            self.event(f"{mid}: approval rejected by @{actor}", "info")
        elif action == "rollback":
            self._restart(mid, actor, counts_as_undo=False, msg_id=msg_id)
        elif action == "release":
            self.resolve(msg_id, f"Reviewed by @{actor}; hold released")
            self.p[mid]["held"] = False
            self.p[mid]["snoozed_until"] = self._eod()
        elif action == "rule_on":
            self.p[mid]["rule_on"], self.p[mid]["undo_count"] = True, 0
            self.resolve(msg_id, f"Rule switched back on by @{actor}")
            self.event(f"{mid}: auto-pause rule switched back on", "info")
        else:  # keep / pause_now / not_mine / undo: same path as a typed reply
            self.reply(actor, {"keep": "keep it", "pause_now": "pause now", "not_mine": "not mine",
                               "undo": "undo"}[action], machine=mid, via_button=True)

    _PATTERNS = [
        ("not_mine", r"\bnot (mine|my)\b|\bisn'?t mine\b|\bwrong (person|owner)\b"),
        ("undo", r"\bundo\b|\brestart\b|\bunpause\b|\bbring (it )?back\b|\bturn (it )?back on\b|\bresume\b"),
        ("keep", r"\bkeep\b|\bdon'?t (pause|touch|stop)\b|\bleave it\b|\bstill (using|need)\b|\bneed it\b|\bin use\b"),
        ("pause_now", r"\bpause( it)?( now)?\b|\bgo ahead\b|\bshut (it )?down\b|\bstop it\b"),
    ]

    def reply(self, author: str, text: str, machine: Optional[str] = None, via_button=False):
        """An owner's reply in Slack (typed, scripted, or a button)."""
        text = text.strip()
        if not text:
            return
        self.post("human", machine, author=author, title="", lines=[], text=text, via_button=via_button)
        owned = [mid for mid, m in self.cloud.machines.items() if m["owner"] == author]
        named = next((mid for mid in self.cloud.machines if mid in text), None)
        ctx = machine or named or next((x["mid"] for x in reversed(self.messages)
                                        if x["author"] == BOT and x["mid"] in owned), None)
        intent, how = None, "keyword rule"
        for name, pat in self._PATTERNS:
            if re.search(pat, text, flags=re.IGNORECASE):
                intent = name
                break
        if intent is None:  # AI helps with free text; its answer is checked like everything else
            user = prompts.INTERPRET_USER.format(owned=", ".join(owned) or "none", context_machine=ctx or "none", text=text)
            res, parsed = self._ai("interpret", ctx, prompts.INTERPRET_SYSTEM, user,
                                   mock=lambda: mocks.interpret(text, ctx))
            parsed = parsed or {}
            intent = parsed.get("intent") if parsed.get("intent") in ("undo", "keep", "pause_now", "not_mine") else "unclear"
            if parsed.get("machine_id") in self.cloud.machines:
                ctx = parsed["machine_id"]
            how = "AI interpretation"
            self.event(f"Reply from @{author} interpreted by AI as {intent!r}", "info",
                       ai={"system": prompts.INTERPRET_SYSTEM, "user": user, "raw": res.text, "in": res.input_tokens,
                           "out": res.output_tokens, "cost": res.cost_usd, "error": res.error})
        else:
            self.event(f"Reply from @{author} matched {how}: {intent!r}" + (f" on {ctx}" if ctx else ""), "info")

        if intent == "unclear" or not ctx:
            self.post("info", ctx, title="Sorry, I didn't catch that",
                      lines=[("Try", "**keep**, **undo**, **pause now** or **not mine** (you can name the machine).")])
            return
        m = self.m(ctx)
        if m["owner"] != author:
            self.post("info", ctx, title=f"Only @{m['owner']} can change {ctx} from chat",
                      lines=[("Action", "No change made.")])
            self.event(f"Reply from @{author} ignored: not the owner of {ctx}", "warn")
            return
        state, p = m["state"], self.p[ctx]
        pending = p["pending_pause_msg"] if p["pending_pause_msg"] and not self.msg(p["pending_pause_msg"])["resolved"] else None
        if intent in ("undo", "keep") and state == "paused":
            self._restart(ctx, author, counts_as_undo=(m["paused_by"] == BOT))
        elif intent in ("undo", "keep"):
            for mid_ in (pending, p.get("approval_msg")):
                if mid_:
                    self.resolve(mid_, f"Kept by @{author}")
            self.scheduled = [e for e in self.scheduled if not (e["kind"] == "pause" and e["mid"] == ctx)]
            p["snoozed_until"] = self._eod()
            p["last_verdict"] = "owner: keep"
            self.post("info", ctx, title=f"OK, keeping {ctx} running",
                      lines=[("Action", "I won't touch it again today.")])
        elif intent == "pause_now":
            if m["env"] == "production":
                self.post("info", ctx, title="Production needs the Approve button",
                          lines=[("Action", f"I can't pause {ctx} from chat. @{config.APPROVER} has to approve it.")])
                return
            if pending:
                self.resolve(pending, f"Paused early by @{author}")
            self.scheduled = [e for e in self.scheduled if not (e["kind"] == "pause" and e["mid"] == ctx)]
            if self.execute("pause", ctx, author):
                self._post_paused(ctx, by=author)
        elif intent == "not_mine":
            if pending:
                self.resolve(pending, f"@{author}: not mine")
            self.scheduled = [e for e in self.scheduled if not (e["kind"] == "pause" and e["mid"] == ctx)]
            p["snoozed_until"] = self._eod()
            self.post("info", ctx, title="Thanks, I'll find the real owner",
                      lines=[("Action", f"I won't pause a machine with an unclear owner. I've asked #infra-leads who owns {ctx}.")])

    def _restart(self, mid, actor, counts_as_undo: bool, msg_id=None):
        paused_at = self.m(mid)["paused_at"]
        if not self.execute("restart", mid, actor):
            return
        p = self.p[mid]
        if p["paused_msg"]:
            self.resolve(p["paused_msg"], f"Undone by @{actor} at {fmt_t(self.now)}")
        self.scheduled = [e for e in self.scheduled if not (e["kind"] == "restart" and e["mid"] == mid)]
        p["snoozed_until"] = self._eod()
        p["last_verdict"] = "undone"
        self.post("restarted", mid, title=f"Restarted ✓ {mid}",
                  lines=[("Action", f"Back up at {fmt_t(self.now)}. I won't pause it again today.")])
        if counts_as_undo:
            p["undo_count"] += 1
            self.m(mid).setdefault("undo_history", []).append(
                f"{self.now:%a %b %d}: paused at {fmt_t(paused_at)}, undone by {actor} at {fmt_t(self.now)}")
            self.event(f"{mid}: undo #{p['undo_count']}", "warn")
            if p["undo_count"] >= config.UNDO_LIMIT and p["rule_on"]:
                p["rule_on"] = False
                p["last_verdict"] = "rule off"
                self.post("rule_off", mid, title=f"I've switched off auto-pause for {mid}",
                          lines=[("Evidence", f"My pause has been undone {p['undo_count']} times:"),
                                 ("History", "; ".join(self.m(mid)["undo_history"][-config.UNDO_LIMIT:])),
                                 ("Action", f"I'll keep reporting {mid}'s cost in the weekly summary, "
                                            "but I won't pause it again unless you turn the rule back on.")],
                          buttons=[("rule_on", "Turn rule back on", "tertiary")])
                self.event(f"{mid}: rule switched itself off after {p['undo_count']} undos", "block")

    # ------------------------------------------------------ REPORT
    def totals(self) -> dict:
        saved = sum(self.cloud.saved_usd(mid) for mid in self.cloud.machines)
        ai = sum(c["cost"] for c in self.ai_calls)
        per_hour = ai / self.hours_run if self.hours_run else 0
        return {
            "saved": saved,
            "paused": sum(1 for m in self.cloud.machines.values() if m["state"] == "paused"),
            "approvals": sum(1 for x in self.messages if x["kind"] == "approval" and not x["resolved"]),
            "blocked": len(self.blocked),
            "ai_cost": ai, "ai_calls": len(self.ai_calls),
            "ai_month_rate": per_hour * config.HOURS_PER_MONTH,
            "burn_hr": sum(m["price_hr"] for m in self.cloud.machines.values() if m["state"] == "running"),
        }

    def weekly(self) -> list[dict]:
        rows = []
        for mid, m in self.cloud.machines.items():
            if m["paused_minutes"] > 0 or m["state"] == "paused":
                rows.append({"Machine": mid, "Env": m["env"], "Hours paused": round(m["paused_minutes"] / 60, 2),
                             "$/hr": m["price_hr"], "Saved": round(self.cloud.saved_usd(mid), 2),
                             "Now": m["state"]})
        return rows
