"""The policy layer. Plain code, no AI. It alone decides what CostGuard may do.

    test + idle + high confidence  -> warn the owner, pause after 30 simulated minutes unless they say keep
    production                     -> never pause; post an approval request with a rollback plan
    AI output names another machine, or touches production, or asks for an action
    other than pause/restart       -> BLOCKED and logged
    busy                           -> keep
    unsure / low confidence        -> ask the owner instead of acting
"""
from __future__ import annotations

import json
import re

import config


def _chk(ok, rule, msg):
    return {"ok": ok, "rule": rule, "msg": msg}


def scan_for_injection(texts: list[str]) -> list[str]:
    """Return the input lines that contain instruction-like text aimed at the agent."""
    hits = []
    for t in texts:
        if any(re.search(p, t, flags=re.IGNORECASE) for p in config.INJECTION_PATTERNS):
            hits.append(t)
    return hits


def decide(machine: dict, ai: dict | None, ai_error: str | None, raw_text: str,
           all_machines: dict, input_texts: list[str]) -> dict:
    """Return {"outcome", "allowed", "checks", "summary", ...}. Outcomes:
    pause_after_warning | approval_request | ask_owner | keep | blocked"""
    mid, checks = machine["id"], []

    def out(outcome, summary, **kw):
        allowed = outcome != "blocked"
        return {"outcome": outcome, "allowed": allowed, "checks": checks, "summary": summary, **kw}

    # 1. Is the AI output usable at all?
    if ai_error or not isinstance(ai, dict):
        checks.append(_chk(False, "AI output", f"unusable ({ai_error or 'not valid JSON'}); acting would be a guess"))
        return out("ask_owner", "AI output was unusable, so CostGuard asks the owner instead of acting.")
    verdict = str(ai.get("verdict", "")).lower()
    action = str(ai.get("action", "")).lower()
    try:
        conf = float(ai.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    if verdict not in ("idle", "busy", "unsure"):
        checks.append(_chk(False, "AI output", f"unknown verdict {verdict!r}; treated as unsure"))
        verdict = "unsure"
    else:
        checks.append(_chk(True, "AI output", f"valid JSON: verdict={verdict}, action={action or '-'}, confidence={conf:.2f}"))

    # 2. Allowed actions only. The AI may recommend pause / keep / ask_owner; nothing else exists.
    if action and action not in ("pause", "keep", "ask_owner"):
        checks.append(_chk(False, "Allowed actions",
                           f"AI asked for {action!r}; the only actions that exist are {', '.join(config.ALLOWED_ACTIONS)}"))
        return out("blocked", f"AI asked for a forbidden action ({action}).", blocked_reason="forbidden action")
    checks.append(_chk(True, "Allowed actions", f"only {' and '.join(config.ALLOWED_ACTIONS)} are possible"))

    # 3. Target check: the output must be about THIS machine.
    target = str(ai.get("machine_id", "")).strip()
    if target != mid:
        tgt = all_machines.get(target)
        extra = f" ({tgt['env']})" if tgt else ""
        checks.append(_chk(False, "Target check", f"AI output names {target or 'no machine'}{extra}, but it was judging {mid}"))
        if tgt and tgt["env"] == "production":
            checks.append(_chk(False, "Production check", f"{target} is production; CostGuard may never pause production"))
        return out("blocked", f"AI output tried to act on {target or 'another machine'} instead of {mid}.",
                   blocked_reason="names a different machine", named=target)
    checks.append(_chk(True, "Target check", f"output is about {mid}"))

    # 4. Does the output mention touching any OTHER production machine?
    other_prod = [o for o, m in all_machines.items() if m["env"] == "production" and o != mid and o in raw_text]
    if other_prod:
        checks.append(_chk(False, "Production check", f"AI output mentions production machine(s) {', '.join(other_prod)}"))
        return out("blocked", f"AI output referred to production machine {other_prod[0]}.",
                   blocked_reason="asks to touch production", named=other_prod[0])

    # 5. Inputs that try to instruct the agent: hold the machine for a human, act on nothing.
    hits = scan_for_injection(input_texts)
    if hits:
        checks.append(_chk(False, "Injection scan", f"{len(hits)} line(s) in logs/tags read like instructions to the agent"))
        return out("blocked", f"{mid}'s logs or tags contain instructions aimed at CostGuard; held for human review.",
                   blocked_reason="instructions found in machine data", injection=hits[0])
    checks.append(_chk(True, "Injection scan", "no instruction-like text in logs or tags"))

    # 6. Savings sanity check (never trust the AI's arithmetic).
    max_sav = round(machine["price_hr"] * 24, 2)
    try:
        sav = float(ai.get("est_savings", 0))
    except (TypeError, ValueError):
        sav = 0.0
    if sav > max_sav or sav < 0:
        checks.append(_chk(None, "Savings check", f"AI estimated ${sav:,.2f}; max possible is ${max_sav:,.2f}. Using ${max_sav:,.2f}"))
        sav = max_sav
    else:
        checks.append(_chk(True, "Savings check", f"${sav:,.2f}/24h is within the possible ${max_sav:,.2f}"))

    # 7. Decide.
    if verdict == "busy":
        checks.append(_chk(True, "Decision", "AI says busy: keep it running"))
        return out("keep", "AI judged it busy, so it stays on.", est_savings=sav)
    if verdict == "unsure" or conf < config.AUTO_PAUSE_MIN_CONFIDENCE:
        checks.append(_chk(None, "Confidence", f"{verdict}, confidence {conf:.2f} < {config.AUTO_PAUSE_MIN_CONFIDENCE}: ask the owner"))
        return out("ask_owner", "Not sure enough to act; asking the owner.", est_savings=sav)
    if machine["env"] == "production":
        checks.append(_chk(None, "Production check", "production is never paused automatically; post an approval request"))
        return out("approval_request", "Production: approval request posted, nothing paused.", est_savings=sav)
    checks.append(_chk(True, "Decision", f"test + idle + confidence {conf:.2f} >= {config.AUTO_PAUSE_MIN_CONFIDENCE}: "
                                         f"warn, pause in {config.PAUSE_GRACE_MINUTES} min unless the owner says keep"))
    return out("pause_after_warning", f"Warn owner, pause in {config.PAUSE_GRACE_MINUTES} min.", est_savings=sav)


def check_execution(action: str, machine: dict, actor: str) -> tuple[bool, str]:
    """Final gate in front of the cloud API. Every pause/restart goes through here."""
    if action not in config.ALLOWED_ACTIONS:
        return False, f"{action!r} is not an allowed action"
    if action == "pause" and machine["env"] == "production" and actor == "CostGuard":
        return False, "CostGuard may never pause production on its own; a human must approve"
    return True, "allowed"


def summarize_json(d) -> str:
    return json.dumps(d, indent=2, default=str)
