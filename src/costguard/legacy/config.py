"""Central configuration for CostGuard. Everything tunable lives here."""
import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

# ---- LLM backend --------------------------------------------------------
# COSTGUARD_BACKEND: "anthropic" or "mock" (offline, canned). If unset, uses
# anthropic when ANTHROPIC_API_KEY is present, otherwise mock.
BACKEND = (os.environ.get("COSTGUARD_BACKEND")
           or ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "mock")).lower()

# Small, cheap hosted model by default.
ANTHROPIC_MODEL = os.environ.get("COSTGUARD_MODEL", "claude-haiku-5-5")
ANTHROPIC_EFFORT = os.environ.get("COSTGUARD_EFFORT", "low")  # keeps cost and latency down
MAX_OUTPUT_TOKENS = 2000
MOCK_LATENCY_SECONDS = float(os.environ.get("COSTGUARD_MOCK_LATENCY", "0.3"))

# ---- Prices (USD per 1M tokens) ----------------------------------------
# Anthropic list prices, Nov 2026 (prompts <= 100K tokens); check before quoting.
PRICES_PER_MTOK = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
    "mock": {"input": 0.10, "output": 0.50},  # priced like Haiku so the cost column is realistic
}

# ---- Simulated cloud ----------------------------------------------------
START_TIME = "2026-11-24T01:15"   # simulated clock start (Tue, 1:15 AM)
HOURS_PER_MONTH = 730

# ---- 1. WATCH (plain rules, no AI) --------------------------------------
IDLE_GPU_PCT = 5          # GPU below this ...
IDLE_MIN_HOURS = 2        # ... for at least this many consecutive hours ...
IDLE_MAX_NET_MBPS = 1.0   # ... and network traffic below this = flagged

# ---- 2. JUDGE (AI) ------------------------------------------------------
LOG_LINES_FOR_AI = 50

# ---- 3. ACT (policy layer) ----------------------------------------------
AUTO_PAUSE_MIN_CONFIDENCE = 0.80   # below this, ask the owner instead of acting
PAUSE_GRACE_MINUTES = 30           # warn, then pause after this unless the owner says "keep"
UNDO_LIMIT = 3                     # after this many undos on one machine, its auto-pause rule switches off
PROD_RESTART_BEFORE = "06:30"      # approved production pauses are undone automatically at this time
ALLOWED_ACTIONS = ("pause", "restart")  # the only things CostGuard can ever do to a machine

# Text in logs/tags that looks like instructions aimed at the agent.
INJECTION_PATTERNS = [
    r"\bsystem\s*:", r"ignore (all |your |previous |the )*(rules|instructions)", r"you are now",
    r"\bassistant\s*:", r"disregard (the |your )?(policy|rules)",
]

APPROVER = "dana"  # on-call SRE who approves production changes
