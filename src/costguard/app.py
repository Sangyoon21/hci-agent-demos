"""CostGuard demo UI.  Run:  streamlit run app.py"""
from __future__ import annotations

import html
import os
import re

import pandas as pd
import streamlit as st

import agent
import cloud_sim
import config

st.set_page_config(page_title="CostGuard", page_icon="🛡", layout="wide", initial_sidebar_state="expanded")

ACCENT = {"warn": "#d97706", "ask": "#d97706", "keep": "#16a34a", "approval": "#2563eb", "blocked": "#dc2626",
          "paused": "#64748b", "restarted": "#16a34a", "rule_off": "#7c3aed", "info": "#94a3b8"}
ICON = {"warn": "⏸", "ask": "❔", "keep": "✅", "approval": "🔐", "blocked": "🛑", "paused": "⏸",
        "restarted": "▶", "rule_off": "🔕", "info": "💬"}
PEOPLE = {"maya": "#db2777", "leo": "#0891b2", "priya": "#7c3aed", "chen": "#ea580c", "ana": "#16a34a",
          "jordan": "#2563eb", "kofi": "#ca8a04", "dana": "#dc2626"}

st.markdown("""
<style>
.block-container {padding-top: 3.0rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
.appbar {display:flex; align-items:center; justify-content:space-between; margin-bottom:.4rem; flex-wrap:wrap; gap:.5rem;}
.appbar h1 {font-size:1.55rem; margin:0; padding:0;}
.appbar .sub {color:#475569; font-size:.92rem;}
.clock {font-variant-numeric: tabular-nums; font-weight:700; font-size:1.05rem; background:#0f172a; color:#f8fafc;
        padding:.35rem .7rem; border-radius:8px;}
.badge {display:inline-block; padding:.18rem .55rem; border-radius:999px; font-size:.78rem; font-weight:600;}
.b-mock {background:#e2e8f0; color:#334155;} .b-live {background:#dcfce7; color:#166534;}
.chan {display:flex; align-items:center; gap:.5rem; font-weight:700; font-size:1.05rem; padding:.2rem 0 .5rem;
       border-bottom:1px solid #e2e8f0; margin-bottom:.4rem;}
.chan .muted {font-weight:400; color:#64748b; font-size:.85rem;}
[class*="st-key-msg_"] {border-left:3px solid transparent; padding:.15rem .2rem .25rem .55rem; border-radius:6px; gap:.25rem !important;}
[class*="st-key-msg_"]:hover {background:#f8fafc;}
.msg {display:flex; gap:.6rem; margin-bottom:.75rem;}
.av {flex:0 0 34px; height:34px; border-radius:7px; color:#fff; font-weight:700; display:flex; align-items:center;
     justify-content:center; font-size:.85rem;}
.mbody {flex:1; min-width:0;}
.mhead {font-size:.9rem;} .mhead b {font-size:.93rem;} .mhead .t {color:#64748b; font-size:.78rem; margin-left:.35rem;}
.app {background:#e2e8f0; color:#475569; font-size:.62rem; font-weight:700; padding:.05rem .3rem; border-radius:3px; margin-left:.3rem;}
.mtitle {font-weight:700; margin:.1rem 0 .15rem;}
.mline {font-size:.9rem; line-height:1.45; margin:.05rem 0; display:grid; grid-template-columns:6.3rem 1fr; column-gap:.4rem;}
.mline.plain {display:block;}
.mline .lb {color:#64748b; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.03em; padding-top:.15rem;}
.mnote {font-size:.83rem; color:#475569; background:#f1f5f9; border-radius:6px; padding:.3rem .5rem; margin-top:.3rem;}
.mroll {font-size:.85rem; border:1px dashed #93c5fd; background:#eff6ff; border-radius:6px; padding:.35rem .6rem; margin-top:.3rem;}
.mroll ol {margin:.1rem 0 0 1.1rem; padding:0;}
.res {display:inline-block; font-size:.78rem; background:#f1f5f9; color:#334155; border-radius:999px; padding:.1rem .55rem; margin-top:.25rem;}
.mline code, .mnote code {font-size:.78rem; background:#fee2e2; color:#991b1b; padding:.05rem .25rem; border-radius:3px; white-space:normal;}
.daybreak {text-align:center; color:#64748b; font-size:.75rem; margin:.5rem 0 .2rem; border-top:1px solid #e2e8f0; padding-top:.3rem;}
.empty {color:#64748b; text-align:center; padding:3rem 1rem;}
.perm {font-size:.85rem; line-height:1.7;}
.ev {font-size:.85rem; margin:.1rem 0; font-family: ui-monospace, Menlo, monospace;}
.ev-block {color:#b91c1c; font-weight:600;} .ev-act {color:#1d4ed8;} .ev-warn {color:#b45309;} .ev-pass {color:#15803d;} .ev-info {color:#475569;}
.chk {font-size:.86rem; margin:.08rem 0;}
.panel-h {font-weight:700; font-size:1.0rem; margin:.2rem 0 .3rem;}
[class*="st-key-msg_"] .stButton button {padding: .1rem .7rem; min-height: 1.9rem; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ state
def world() -> agent.World:
    if "world" not in st.session_state:
        st.session_state.world = agent.World(st.session_state.get("backend", config.BACKEND))
    return st.session_state.world


def md(s: str) -> str:
    s = html.escape(str(s))
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


W = world()
T = W.totals()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🛡 CostGuard")
    st.caption("Finds idle cloud machines. Pauses test machines on its own. Asks a person before touching production.")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    labels = {"mock": "Offline (mock)", "anthropic": "Claude (live)"}
    choice = st.segmented_control("AI runs on", options=list(labels), format_func=labels.get,
                                  default=W.backend, key="backend_ctl")
    if choice and choice != W.backend:
        W.backend = choice
        st.session_state.backend = choice
    if W.backend == "anthropic" and not has_key:
        st.warning("ANTHROPIC_API_KEY is not set, so live AI calls will fail and the policy layer will fall back to "
                   "asking owners. Switch to Offline for the demo.")
    st.caption(f"Live model: `{config.ANTHROPIC_MODEL}`")
    st.divider()
    st.markdown("**Cloud permission held by CostGuard**")
    st.markdown("<div class='perm'>✅ pause(id)<br>✅ restart(id)<br>✅ get_metrics(id)<br>✅ get_logs(id)<br>"
                "❌ <b>delete</b>: this function does not exist</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**Rules**")
    st.caption(f"WATCH: GPU < {config.IDLE_GPU_PCT}% for {config.IDLE_MIN_HOURS}+ h and no traffic → ask the AI.  \n"
               f"ACT: test + idle + ≥{config.AUTO_PAUSE_MIN_CONFIDENCE:.0%} sure → warn, pause after "
               f"{config.PAUSE_GRACE_MINUTES} min. Production → approval only. "
               f"{config.UNDO_LIMIT} undos → rule switches off.")
    st.divider()
    if st.button("↺ Reset demo", width="stretch"):
        st.session_state.pop("world", None)
        st.rerun()

# ---------------------------------------------------------------- app bar
badge = ("<span class='badge b-live'>● Claude " + config.ANTHROPIC_MODEL + "</span>" if W.backend == "anthropic"
         else "<span class='badge b-mock'>● Offline mock AI</span>")
st.markdown(f"""<div class='appbar'>
  <div><h1>🛡 CostGuard</h1><div class='sub'>Simulated cloud · fake Slack · the AI judges, the policy decides</div></div>
  <div style='display:flex;gap:.6rem;align-items:center'>{badge}
  <span class='clock'>🕑 {W.now:%a %b %d} · {agent.fmt_t(W.now)}</span></div></div>""", unsafe_allow_html=True)

c = st.columns(5)
c[0].metric("Money saved so far", agent.money(T["saved"]), border=True,
            help="Hours paused × hourly price, summed over all machines.")
c[1].metric("Machines paused", f"{T['paused']} of {len(W.cloud.machines)}", border=True,
            help=f"Running machines currently cost ${T['burn_hr']:,.2f}/hour.")
c[2].metric("Approvals pending", T["approvals"], border=True)
c[3].metric("Blocked actions", T["blocked"], border=True, help="Actions the policy layer refused.")
c[4].metric("AI cost this month", f"${T['ai_cost']:.4f}", border=True,
            delta=f"≈ ${T['ai_month_rate']:.2f}/mo at this rate" if W.hours_run else None, delta_color="off",
            help=f"{T['ai_calls']} AI calls so far. AI is only called for machines the plain rule flags.")

bc = st.columns([1.3, 3.7])
run = bc[0].button(f"▶ Advance 1 hour  →  {agent.fmt_t(W.now + pd.Timedelta(hours=1))}", type="primary",
                   width="stretch")
nxt = sorted(W.scheduled, key=lambda e: e["time"])
if nxt:
    bits = []
    for e in nxt[:4]:
        what = {"pause": f"pause {e.get('mid')}", "restart": f"restart {e.get('mid')}",
                "owner_reply": f"@{e.get('author')} replies"}[e["kind"]]
        bits.append(f"{agent.fmt_t(e['time'])} {what}")
    bc[1].caption("Coming up in the next hour or later: " + " · ".join(bits))
elif not W.hours_run:
    bc[1].caption("The agent runs once every simulated hour. Click **Advance 1 hour** to run it for the first time.")
if run:
    with st.status(f"Agent run at {agent.fmt_t(W.now + pd.Timedelta(hours=1))}", expanded=True) as s:
        W.run_hour(on_step=lambda t: s.write(t))
        s.update(label="Agent run complete", state="complete")
    st.rerun()

left, right = st.columns([0.95, 1.05], gap="medium")


# ------------------------------------------------------------ fake Slack
def render_msg(x: dict):
    with st.container(key=f"msg_{x['id']}"):
        if x["author"] != agent.BOT:
            col = PEOPLE.get(x["author"], "#475569")
            via = " <span class='t'>(button)</span>" if x.get("via_button") else ""
            st.markdown(f"""<div class='msg'><div class='av' style='background:{col}'>{x['author'][0].upper()}</div>
              <div class='mbody'><div class='mhead'><b>@{x['author']}</b><span class='t'>{agent.fmt_t(x['time'])}</span>{via}</div>
              <div class='mline plain'>{md(x['text'])}</div></div></div>""", unsafe_allow_html=True)
            return
        acc = ACCENT.get(x["kind"], "#94a3b8")
        st.markdown(f"<style>.st-key-msg_{x['id']} {{border-left-color:{acc} !important;}}</style>", unsafe_allow_html=True)
        to = f"@{x['to']} " if x.get("to") else ""
        lines = "".join(f"<div class='mline'><span class='lb'>{html.escape(k)}</span><span>{md(v)}</span></div>" for k, v in x["lines"])
        note = f"<div class='mnote'>🤖 {md(x['note'])}</div>" if x["note"] else ""
        roll = ("<div class='mroll'><b>Rollback plan</b><ol>" + "".join(f"<li>{md(r)}</li>" for r in x["rollback"])
                + "</ol></div>") if x["rollback"] else ""
        res = f"<div class='res'>✓ {html.escape(x['resolved'])}</div>" if x["resolved"] else ""
        st.markdown(f"""<div class='msg'><div class='av' style='background:#4f46e5'>🛡</div><div class='mbody'>
          <div class='mhead'><b>CostGuard</b><span class='app'>APP</span><span class='t'>{agent.fmt_t(x['time'])}</span></div>
          <div class='mtitle'>{ICON.get(x['kind'], '')} {html.escape(to)}{md(x['title'])}</div>
          {lines}{note}{roll}{res}</div></div>""", unsafe_allow_html=True)
        if x["buttons"]:
            with st.container(horizontal=True, gap="small"):
                for act, label, kind in x["buttons"]:
                    st.button(label, key=f"b_{x['id']}_{act}", type=kind, on_click=W.button, args=(x["id"], act),
                              width="content")


with left:
    st.markdown("<div class='chan'># cloud-costs <span class='muted'>· CostGuard + 8 machine owners (simulated Slack)</span></div>",
                unsafe_allow_html=True)
    feed = st.container(height=660, autoscroll=True, border=False)
    with feed:
        if not W.messages:
            st.markdown("<div class='empty'>No messages yet.<br>Click <b>▶ Advance 1 hour</b> to let CostGuard look at the cloud.</div>",
                        unsafe_allow_html=True)
        last_run = None
        for x in W.messages:
            run_hour = x["time"].replace(minute=15) if x["time"].minute >= 15 else (x["time"] - pd.Timedelta(hours=1)).replace(minute=15)
            if run_hour != last_run:
                st.markdown(f"<div class='daybreak'>{agent.fmt_t(x['time'])}</div>", unsafe_allow_html=True)
                last_run = run_hour
            render_msg(x)
    owners = sorted({m["owner"] for m in W.cloud.machines.values()}, key=lambda o: (o != "jordan", o))
    with st.form("reply", clear_on_submit=True, border=False):
        fc = st.columns([1.1, 3.4, 0.9], vertical_alignment="bottom")
        who = fc[0].selectbox("Reply as", owners, format_func=lambda o: f"@{o}")
        text = fc[1].text_input("Message #cloud-costs", placeholder="e.g. undo, I need it today  ·  keep it  ·  pause now  ·  not mine")
        sent = fc[2].form_submit_button("Send", width="stretch")
    if sent and text.strip():
        W.reply(who, text)
        st.rerun()
    st.caption("Simple replies are matched by keyword rules; anything else is interpreted by the AI, and the result "
               "still goes through the same checks (only the owner can change their machine).")

# ------------------------------------------------------------- machines
with right:
    st.markdown("<div class='panel-h'>Machines</div>", unsafe_allow_html=True)
    rows = []
    for mid, m in W.cloud.machines.items():
        g = m["history"]["gpu"]
        rows.append({
            "Machine": mid,
            "Env · state": ("🔴 prod" if m["env"] == "production" else "🧪 test") + "  " +
                           ("⏸ paused" if m["state"] == "paused" else "▶ running"),
            "GPU %": f"{g[-1]}%" if g[-1] is not None else "no GPU",
            "GPU, last 12 h": [v or 0 for v in g] if m["has_gpu"] else None,
            "$/hr": m["price_hr"],
            "Agent's last verdict": W.p[mid]["last_verdict"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=330,
                 column_config={
                     "GPU %": st.column_config.TextColumn("GPU", width=58),
                     "GPU, last 12 h": st.column_config.LineChartColumn("GPU, 12 h", y_min=0, y_max=100, width=72),
                     "$/hr": st.column_config.NumberColumn("$/hr", format="$%.2f", width=68),
                     "Machine": st.column_config.TextColumn(width=104),
                     "Env · state": st.column_config.TextColumn(width=132),
                     "Agent's last verdict": st.column_config.TextColumn("Verdict", width=105),
                 })

    with st.container(border=True):
        st.markdown("<div class='panel-h'>📈 Weekly savings report</div>", unsafe_allow_html=True)
        wk = W.weekly()
        net = T["saved"] - T["ai_cost"]
        k = st.columns(3)
        k[0].metric("Saved this week", agent.money(T["saved"]))
        k[1].metric("AI cost", f"${T['ai_cost']:.4f}")
        k[2].metric("Net", agent.money(net))
        if wk:
            st.dataframe(pd.DataFrame(wk), hide_index=True, width="stretch",
                         column_config={"$/hr": st.column_config.NumberColumn(format="$%.2f"),
                                        "Saved": st.column_config.NumberColumn(format="$%.2f")})
            paused_now = [m for m in W.cloud.machines.values() if m["state"] == "paused"]
            if paused_now:
                rate = sum(m["price_hr"] for m in paused_now)
                st.caption(f"Machines paused right now save ${rate:,.2f}/hour, or about ${rate * 24 * 7:,.0f} "
                           "if they stay paused for a week.")
        else:
            st.caption("Savings = hours paused × hourly price. Nothing has been paused yet.")
        rule_off = [mid for mid, p in W.p.items() if not p["rule_on"]]
        if rule_off:
            st.caption("Auto-pause switched off (too many undos): " + ", ".join(rule_off) +
                       f". Still costs ${sum(W.m(x)['price_hr'] for x in rule_off):,.2f}/hour.")

# ------------------------------------------------------------ agent trace
st.write("")
with st.expander(f"🔎 Agent trace · {len(W.trace)} hourly run(s) · {T['ai_calls']} AI call(s) · ${T['ai_cost']:.4f}",
                 expanded=False):
    if not W.trace and not W.pending_events:
        st.caption("Nothing yet.")
    tabs_src = list(reversed(W.trace))
    names = ([f"Since last run ({len(W.pending_events)})"] if W.pending_events else []) + \
            [f"{agent.fmt_t(e['time'])} run" + (" 🛑" if any(j['policy']['outcome'] == 'blocked' for j in e['judge']) else "")
             for e in tabs_src]
    if names:
        tabs = st.tabs(names)
        ti = 0

        def events_block(evs):
            for ev in evs:
                st.markdown(f"<div class='ev ev-{ev['level']}'>{agent.fmt_t(ev['time'])} · {html.escape(ev['text'])}</div>",
                            unsafe_allow_html=True)
                if ev.get("ai"):
                    a = ev["ai"]
                    st.caption(f"AI interpret call: {a['in']} in / {a['out']} out tokens · ${a['cost']:.5f}"
                               + (f" · ERROR {a['error']}" if a["error"] else ""))
                    cc = st.columns(2)
                    cc[0].code(a["system"] + "\n\n" + a["user"], language=None, wrap_lines=True, height=200)
                    cc[1].code(a["raw"] or "(no output)", language="json", wrap_lines=True, height=200)

        if W.pending_events:
            with tabs[0]:
                st.caption("Button clicks, replies and scheduled actions since the last hourly run.")
                events_block(W.pending_events)
            ti = 1
        for e, tab in zip(tabs_src, tabs[ti:]):
            with tab:
                if e["events"]:
                    st.markdown("**Events this hour**")
                    events_block(e["events"])
                st.markdown("**1 · WATCH (plain rules, no AI)**")
                st.dataframe(pd.DataFrame([{
                    "Machine": w["mid"], "Low hours": w["low_hours"],
                    "Flagged": "🚩 yes" if w["flagged"] else "no",
                    "Sent to AI?": ("yes" if w["flagged"] and not w["skip"] else (f"no: {w['skip']}" if w["skip"] else "no")),
                    "Evidence": w["evidence"]} for w in e["watch"]]), hide_index=True, width="stretch")
                if not e["judge"]:
                    st.caption("No machine needed an AI judgment this hour, so no AI cost.")
                for j in e["judge"]:
                    pol = j["policy"]
                    head = {"blocked": "🛑 BLOCKED", "pause_after_warning": "✅ allowed: warn, then pause",
                            "keep": "✅ allowed: keep", "approval_request": "🔐 approval request",
                            "ask_owner": "❔ ask owner"}[pol["outcome"]]
                    st.markdown(f"**2 · JUDGE + 3 · ACT: `{j['mid']}`** — policy decision: **{head}**")
                    mc = st.columns(4)
                    mc[0].metric("Input tokens", j["in"])
                    mc[1].metric("Output tokens", j["out"])
                    mc[2].metric("Cost", f"${j['cost']:.5f}")
                    mc[3].metric("Latency", f"{j['latency']:.1f}s")
                    for chk in pol["checks"]:
                        mark = "✅" if chk["ok"] is True else ("🛑" if chk["ok"] is False else "⚠️")
                        st.markdown(f"<div class='chk'>{mark} <b>{html.escape(chk['rule'])}</b>: {html.escape(chk['msg'])}</div>",
                                    unsafe_allow_html=True)
                    cc = st.columns(2)
                    cc[0].caption(f"Prompt sent to {j['model']}")
                    cc[0].code(j["system"] + "\n\n---\n\n" + j["user"], language=None, wrap_lines=True, height=260)
                    cc[1].caption("Raw AI output" + (f" · ERROR: {j['error']}" if j["error"] else ""))
                    cc[1].code(j["raw"] or "(no output)", language="json", wrap_lines=True, height=260)
                    st.divider()
