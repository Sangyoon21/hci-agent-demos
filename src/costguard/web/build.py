"""Build web/costguard.html (one self-contained page) from web/template.html + data/scenario.json.

    python web/build.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "data" / "scenario.json").read_text())
tpl = (ROOT / "web" / "template.html").read_text()
assert "/*__DATA__*/null" in tpl
out = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
(ROOT / "web" / "costguard.html").write_text(out)
print("wrote", ROOT / "web" / "costguard.html", len(out), "bytes")
