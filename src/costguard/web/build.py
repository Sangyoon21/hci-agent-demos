"""Build web/costguard.html (browser version for sharing) from the real data, prompts and config.

    python web/build.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
import prompts  # noqa: E402

data = {
    "machines": json.loads((ROOT / "data" / "machines.json").read_text())["machines"],
    "prompts": {k: getattr(prompts, k) for k in ("JUDGE_SYSTEM", "JUDGE_USER", "INTERPRET_SYSTEM", "INTERPRET_USER")},
    "config": {
        "IDLE_GPU_PCT": config.IDLE_GPU_PCT, "IDLE_MIN_HOURS": config.IDLE_MIN_HOURS,
        "IDLE_MAX_NET_MBPS": config.IDLE_MAX_NET_MBPS, "LOG_LINES_FOR_AI": config.LOG_LINES_FOR_AI,
        "AUTO_PAUSE_MIN_CONFIDENCE": config.AUTO_PAUSE_MIN_CONFIDENCE,
        "PAUSE_GRACE_MINUTES": config.PAUSE_GRACE_MINUTES, "UNDO_LIMIT": config.UNDO_LIMIT,
        "PROD_RESTART_BEFORE": config.PROD_RESTART_BEFORE, "ALLOWED_ACTIONS": list(config.ALLOWED_ACTIONS),
        "INJECTION_PATTERNS": config.INJECTION_PATTERNS, "APPROVER": config.APPROVER,
        "HOURS_PER_MONTH": config.HOURS_PER_MONTH, "PRICE_IN": config.PRICES_PER_MTOK["mock"]["input"],
        "PRICE_OUT": config.PRICES_PER_MTOK["mock"]["output"], "MODEL": config.ANTHROPIC_MODEL,
    },
}
tpl = (ROOT / "web" / "template.html").read_text()
out = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
(ROOT / "web" / "costguard.html").write_text(out)
print("wrote", ROOT / "web" / "costguard.html", len(out), "bytes")
