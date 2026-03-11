"""
TAS AutoBD — Streamlit Application (v2)
========================================
Agentic LLM-powered automatic business development tool.

v2 UI improvements
-------------------
1. Contact Quality Panel   — shows scored/ranked contacts with role badges.
   Generic emails (info@, support@) are visually flagged as low-quality.
2. Insight Highlights      — surfaces the key_insights and competitive_gaps
   sections from research as highlighted callouts, not buried in a text block.
3. Outcome Tracker         — simple "did this lead respond?" toggle after send,
   stored in ./data/outcomes.json so quality can be measured over time.
4. Pre-send Deliverability Warning — warns if the only available email is a
   generic/filtered address before sending.

Run with:
    streamlit run app.py
"""

import json
import logging
import os
from pathlib import Path

import streamlit as st

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from config import validate_config, DATA_DIR
from utils import clean_html_fences
from email_utils import send_email, add_email_manually

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

OUTCOMES_FILE = os.path.join(DATA_DIR, "outcomes.json")


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TAS AutoBD",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #4A0E8F;
        font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
    }
    .stButton > button {
        background-color: #FFFFFF !important;
        color: #4A0E8F !important;
        border: 2px solid #4A0E8F !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.4rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #4A0E8F !important;
        color: #FFFFFF !important;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #F4EFFC !important;
        color: #1A0030 !important;
        border: 1px solid #4A0E8F !important;
        border-radius: 6px !important;
    }
    .stSelectbox > div > div {
        background-color: #F4EFFC !important;
        color: #1A0030 !important;
        border: 1px solid #4A0E8F !important;
    }
    h1, h2, h3, h4 { color: #4A0E8F !important; }
    .stProgress > div > div > div > div {
        background-color: #4A0E8F !important;
    }
    [data-testid="stSidebar"] {
        background-color: #4A0E8F !important;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background-color: #FFFFFF !important;
        color: #4A0E8F !important;
        border: none !important;
    }
    .step-card {
        background: #F9F6FF;
        border-left: 4px solid #4A0E8F;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    .insight-card {
        background: #FFF8E1;
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .gap-card {
        background: #FEF2F2;
        border-left: 4px solid #EF4444;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .contact-exec   { color: #16a34a; font-weight: bold; }
    .contact-tech   { color: #2563eb; font-weight: bold; }
    .contact-biz    { color: #7c3aed; font-weight: bold; }
    .contact-gen    { color: #6b7280; }
    .contact-bad    { color: #dc2626; text-decoration: line-through; }
    .badge-ok       { color: #16a34a; font-weight: bold; }
    .badge-warn     { color: #dc2626; font-weight: bold; }
    .tool-log-entry {
        font-family: monospace;
        font-size: 0.85rem;
        padding: 0.2rem 0;
        border-bottom: 1px solid #e8e0f7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Outcome tracking ───────────────────────────────────────────────────────────

def _load_outcomes() -> list:
    try:
        if os.path.exists(OUTCOMES_FILE):
            with open(OUTCOMES_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_outcome(company: str, email: str, responded: bool) -> None:
    outcomes = _load_outcomes()
    outcomes.append({
        "company": company,
        "email": email,
        "responded": responded,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
    })
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(OUTCOMES_FILE, "w") as f:
            json.dump(outcomes, f, indent=2)
    except Exception as exc:
        logger.warning("Could not save outcome: %s", exc)


def _render_outcomes_summary() -> None:
    outcomes = _load_outcomes()
    if not outcomes:
        return
    total = len(outcomes)
    responded = sum(1 for o in outcomes if o.get("responded"))
    rate = responded / total * 100 if total else 0
    st.markdown(
        f"📊 **Pipeline:** {total} proposals sent · {responded} responses · "
        f"**{rate:.0f}% response rate**"
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🚀 TAS AutoBD")
        st.markdown("*Agentic AI Business Development*")
        st.divider()

        st.markdown("### API Status")
        cfg = validate_config()
        icons = {True: "🟢", False: "🔴"}

        provider = cfg["provider"].capitalize()
        model = cfg["model"]
        st.markdown(f"{icons[cfg['llm']]} **{provider}** ({model}) — {'Configured' if cfg['llm'] else 'Missing'}")
        st.markdown(f"{icons[cfg['tavily']]} **Tavily Search** — {'Configured' if cfg['tavily'] else 'Missing'}")
        st.markdown(f"{icons[cfg['sendgrid']]} **SendGrid Email** — {'Configured' if cfg['sendgrid'] else 'Missing (optional)'}")
        st.markdown(f"{icons[cfg['github']]} **GitHub Token** — {'Configured' if cfg['github'] else 'Anonymous (60 req/hr)'}")

        if not all([cfg["llm"], cfg["tavily"]]):
            st.warning("Add missing API keys to your `.env` file and restart.")

        st.divider()
        st.markdown("### How it works")
        steps = [
            ("1️⃣", "Agent researches company + competitive gaps"),
            ("2️⃣", "Review insights & gap analysis, generate idea"),
            ("3️⃣", "Build knowledge base & draft evidence-backed proposal"),
            ("4️⃣", "Review, refine & send to ranked contacts"),
        ]
        for icon, label in steps:
            st.markdown(f"{icon} {label}")

        st.divider()
        current_step = st.session_state.get("step", 1)
        st.markdown(f"**Current step:** {current_step} / 4")
        st.progress((current_step - 1) / 3)

        if current_step > 1:
            if st.button("↩ Start Over"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        st.divider()
        _render_outcomes_summary()
        st.divider()
        st.caption("© 2025 TAS Design Group Inc.")


# ── Session state ──────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "step": 1,
        "company_name": "",
        "additional_url": "",
        "characteristics": "",
        "idea": "",
        "keywords": [],
        "email_list": [],
        "docsearch": None,
        "email_proposal": "",
        "processing_error": None,
        "agent_tool_log": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Tool log widget ───────────────────────────────────────────────────────────

_TOOL_ICONS = {
    "web_search": "🔍",
    "fetch_webpage": "🌐",
    "extract_emails": "📧",
    "search_github": "🐙",
    "fetch_readme": "📄",
}


def _render_tool_log(log: list) -> None:
    if not log:
        return
    label = f"🤖 Agent used **{len(log)} tool calls** to research this company"
    with st.expander(label, expanded=False):
        for entry in log:
            tool = entry.get("tool", "unknown")
            inputs = entry.get("inputs", {})
            icon = _TOOL_ICONS.get(tool, "🔧")
            detail = (
                inputs.get("query")
                or inputs.get("url")
                or inputs.get("keyword")
                or inputs.get("repo_full_name")
                or inputs.get("text", "")[:60]
                or str(inputs)[:60]
            )
            st.markdown(
                f'<div class="tool-log-entry">{icon} <b>{tool}</b>: {detail}</div>',
                unsafe_allow_html=True,
            )


# ── Insight & gap highlights ──────────────────────────────────────────────────

def _extract_section(profile: str, section_header: str) -> str:
    """Extract a named === SECTION === block from the profile text."""
    pattern = rf"===\s*{re.escape(section_header)}\s*===\s*\n(.*?)(?====|\Z)"
    import re as _re
    m = _re.search(pattern, profile, _re.DOTALL | _re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _render_insight_panel(profile: str) -> None:
    """Show key insights and competitive gaps as highlighted cards."""
    import re as _re

    insights = _extract_section(profile, "KEY INSIGHTS (Proposal Hooks)")
    gaps = _extract_section(profile, "COMPETITIVE GAPS")

    if not insights and not gaps:
        return

    st.markdown("---")
    if insights:
        st.markdown(
            '<div class="insight-card">'
            '<b>💡 Key Insights (Proposal Hooks)</b><br>'
            + insights.replace("\n", "<br>")
            + "</div>",
            unsafe_allow_html=True,
        )
    if gaps:
        st.markdown(
            '<div class="gap-card">'
            '<b>⚠️ Competitive Gaps</b><br>'
            + gaps.replace("\n", "<br>")
            + "</div>",
            unsafe_allow_html=True,
        )
    st.markdown("---")


# ── Contact quality panel ─────────────────────────────────────────────────────

def _render_contact_panel(profile: str, email_list: list) -> None:
    """
    Parse and display the scored contact table from the profile.
    Shows role badges and highlights generic/filtered addresses.
    """
    import re as _re

    scored_section = _extract_section(profile, "CONTACT QUALITY SCORES")

    if scored_section:
        with st.expander(f"📇 Contact Quality — {len(email_list)} viable address(es)", expanded=True):
            for line in scored_section.splitlines():
                line = line.strip()
                if not line:
                    continue
                if "FILTERED" in line:
                    css = "contact-bad"
                elif "Executive" in line:
                    css = "contact-exec"
                elif "Technical" in line:
                    css = "contact-tech"
                elif "Business" in line:
                    css = "contact-biz"
                else:
                    css = "contact-gen"
                st.markdown(
                    f'<div class="{css}" style="font-size:0.9rem;padding:2px 0">{line}</div>',
                    unsafe_allow_html=True,
                )
            if not email_list:
                st.warning(
                    "All discovered emails were filtered as generic addresses. "
                    "Add a decision-maker email manually below."
                )
    elif email_list:
        st.info(f"📧 Found {len(email_list)} email address(es) automatically.")
    else:
        st.warning("No email addresses found. Add one manually below.")


# ── Pre-send deliverability check ─────────────────────────────────────────────

def _deliverability_check(email: str) -> list[str]:
    """Return a list of warning strings for a given email, or empty if clean."""
    warnings = []
    local = email.split("@")[0].lower() if "@" in email else email.lower()
    generic_locals = {
        "info", "contact", "hello", "support", "help", "admin",
        "webmaster", "noreply", "no-reply", "office", "general",
    }
    if local in generic_locals:
        warnings.append(
            f"'{email}' looks like a generic inbox. Proposals sent here are rarely read. "
            "Add a named contact instead."
        )
    if not email.endswith((".com", ".org", ".net", ".io", ".co", ".jp", ".vn", ".de", ".uk", ".fr")):
        warnings.append(f"Unusual email domain in '{email}' — verify before sending.")
    return warnings


# ── Step rendering functions ──────────────────────────────────────────────────

def _step1() -> None:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.subheader("Step 1 — Research Target Company")
    st.markdown(
        "Enter the company name. The **autonomous research agent** will:\n"
        "- Gather deep company intelligence across 7 research phases\n"
        "- **Hunt for strategic pivots, competitive losses, and recent signals** (the insight layer)\n"
        "- Rank and score all discovered contacts by decision-maker likelihood"
    )

    company_name = st.text_input(
        "Target company name *",
        value=st.session_state.company_name,
        placeholder="e.g. Toyota, Panasonic, Rakuten …",
    )
    additional_url = st.text_input(
        "Additional URL (optional — e.g. company website or press release)",
        value=st.session_state.additional_url,
        placeholder="https://www.example.com",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🔍 Research Company", use_container_width=True):
        if not company_name.strip():
            st.error("Please enter a company name.")
            return

        cfg = validate_config()
        if not cfg["llm"] or not cfg["tavily"]:
            st.error(
                f"{cfg['provider'].capitalize()} API key and Tavily API key are required. "
                "Check the sidebar and your .env file."
            )
            return

        st.session_state.company_name = company_name.strip()
        st.session_state.additional_url = additional_url.strip()
        st.session_state.agent_tool_log = []

        tool_log: list = []

        def _on_tool_call(tool_name: str, tool_inputs: dict) -> None:
            tool_log.append({"tool": tool_name, "inputs": tool_inputs})

        with st.spinner("🤖 Autonomous research agent working … (≈ 60–90 s — deeper research in v2)"):
            try:
                from get_info import get_company_information

                characteristics, emails = get_company_information(
                    st.session_state.company_name,
                    st.session_state.additional_url,
                    on_tool_call=_on_tool_call,
                )
                st.session_state.characteristics = characteristics
                st.session_state.email_list = emails
                st.session_state.agent_tool_log = tool_log
                st.session_state.step = 2
                st.session_state.processing_error = None
            except Exception as exc:
                logger.error("Step 1 error: %s", exc, exc_info=True)
                st.session_state.processing_error = str(exc)
                st.error(f"Research failed: {exc}")
                return

        st.rerun()


def _step2() -> None:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.subheader(f"Step 2 — Company Profile: {st.session_state.company_name}")
    st.markdown(
        "Review the extracted intelligence. The **Key Insights** and **Competitive Gaps** "
        "sections are the proposal hooks — they make your outreach feel researched, not templated."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    _render_tool_log(st.session_state.get("agent_tool_log", []))

    # Show insight highlights as separate callout cards
    _render_insight_panel(st.session_state.characteristics)

    # Show contact quality panel
    _render_contact_panel(st.session_state.characteristics, st.session_state.email_list)

    characteristics = st.text_area(
        "Full Company Intelligence Profile",
        value=st.session_state.characteristics,
        height=350,
        help="Autonomously extracted by the research agent. Includes insights, competitive gaps, and scored contacts.",
    )
    if characteristics != st.session_state.characteristics:
        st.session_state.characteristics = characteristics

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col2:
        generate = st.button("💡 Generate Idea →", use_container_width=True)

    if generate:
        if not st.session_state.characteristics.strip():
            st.error("Characteristics are empty. Please run Step 1 first.")
            return

        with st.spinner("🤔 Analysing competitive gap and generating anchored solution …"):
            try:
                from get_hypo import get_hypothesis_idea

                idea_text, keywords = get_hypothesis_idea(st.session_state.characteristics)
                st.session_state.idea = idea_text
                st.session_state.keywords = keywords
                st.session_state.step = 3
                st.session_state.processing_error = None
            except Exception as exc:
                logger.error("Step 2 error: %s", exc, exc_info=True)
                st.error(f"Idea generation failed: {exc}")
                return

        st.rerun()


def _step3() -> None:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.subheader("Step 3 — Competitive Gap Analysis & Solution")
    st.markdown(
        "The idea now includes a **competitive gap analysis** stage — the solution is anchored "
        "to a specific, evidence-backed weakness, not generic company needs.\n\n"
        "The proposal will enforce an **insight hook** in the opening and require evidence "
        "citations for all metrics. Self-critique runs up to **3 passes** (raised from 1)."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    idea = st.text_area(
        "Competitive Gap Analysis + Product Idea",
        value=st.session_state.idea,
        height=300,
        help="v2: includes gap analysis stage before the solution proposal.",
    )
    if idea != st.session_state.idea:
        st.session_state.idea = idea

    if st.session_state.keywords:
        st.markdown(
            "**Search keywords:** " + " · ".join(f"`{k}`" for k in st.session_state.keywords)
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col2:
        generate = st.button("📝 Generate Proposal →", use_container_width=True)

    if generate:
        if not st.session_state.idea.strip():
            st.error("Idea is empty.")
            return

        with st.spinner("🏗️ Fetching GitHub READMEs and web articles …"):
            try:
                from make_db import make_db

                db = make_db(st.session_state.idea, st.session_state.keywords)
                st.session_state.docsearch = db
            except Exception as exc:
                logger.error("make_db error: %s", exc, exc_info=True)
                st.error(f"Knowledge base build failed: {exc}")
                return

        with st.spinner("✍️ Drafting evidence-backed proposal and running multi-pass critique …"):
            try:
                from get_proposal import make_proposal

                email_html = make_proposal(
                    st.session_state.idea,
                    st.session_state.docsearch,
                    st.session_state.company_name,
                )
                st.session_state.email_proposal = clean_html_fences(email_html)
                st.session_state.step = 4
                st.session_state.processing_error = None
            except Exception as exc:
                logger.error("make_proposal error: %s", exc, exc_info=True)
                st.error(f"Proposal generation failed: {exc}")
                return

        st.rerun()


def _step4() -> None:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.subheader(f"Step 4 — Review & Send Proposal to {st.session_state.company_name}")
    st.markdown(
        "Review the quality-reviewed HTML proposal. "
        "Recipients are **ranked by decision-maker score** — executives first, "
        "generic inboxes filtered out."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    tab_edit, tab_preview = st.tabs(["✏️ Edit HTML", "👁️ Preview"])

    with tab_edit:
        modified_proposal = st.text_area(
            "Email HTML",
            value=st.session_state.email_proposal,
            height=400,
            label_visibility="collapsed",
        )
        if modified_proposal != st.session_state.email_proposal:
            st.session_state.email_proposal = modified_proposal

    with tab_preview:
        if st.session_state.email_proposal:
            st.components.v1.html(st.session_state.email_proposal, height=600, scrolling=True)
        else:
            st.info("No proposal content yet.")

    st.download_button(
        label="⬇️ Download HTML",
        data=st.session_state.email_proposal,
        file_name=f"proposal_{st.session_state.company_name.replace(' ', '_')}.html",
        mime="text/html",
    )

    st.divider()
    st.subheader("Send Email")

    st.session_state.email_list = add_email_manually(st.session_state.email_list)

    if not st.session_state.email_list:
        st.warning("No viable email addresses found. Add a decision-maker email manually above.")
    else:
        selected_email = st.selectbox(
            "Recipient (ranked by decision-maker score — best first)",
            options=st.session_state.email_list,
            key="selected_email",
        )

        # Pre-send deliverability check
        if selected_email:
            warnings = _deliverability_check(selected_email)
            for w in warnings:
                st.warning(f"⚠️ {w}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.step = 3
                st.rerun()
        with col2:
            send = st.button("📨 Send Email", use_container_width=True)

        if send:
            cfg = validate_config()
            if not cfg["sendgrid"]:
                st.error("SENDGRID_API_KEY is not configured. Add it to your .env file.")
                return

            with st.spinner(f"Sending email to {selected_email} …"):
                success, message = send_email(
                    selected_email,
                    f"Business Proposal for {st.session_state.company_name} — TAS Design Group",
                    st.session_state.email_proposal,
                )

            if success:
                st.success(f"✅ Email sent successfully to **{selected_email}**!")
                st.balloons()

                # Outcome tracker
                st.divider()
                st.markdown("### 📊 Track Outcome")
                st.markdown(
                    "Did this lead respond? Recording outcomes helps measure proposal quality "
                    "and improves future targeting."
                )
                col_yes, col_no, col_skip = st.columns(3)
                with col_yes:
                    if st.button("✅ Yes — they responded", use_container_width=True):
                        _save_outcome(st.session_state.company_name, selected_email, responded=True)
                        st.success("Outcome recorded.")
                        for key in list(st.session_state.keys()):
                            del st.session_state[key]
                        st.rerun()
                with col_no:
                    if st.button("❌ No response", use_container_width=True):
                        _save_outcome(st.session_state.company_name, selected_email, responded=False)
                        st.info("Outcome recorded.")
                        for key in list(st.session_state.keys()):
                            del st.session_state[key]
                        st.rerun()
                with col_skip:
                    if st.button("— Skip", use_container_width=True):
                        for key in list(st.session_state.keys()):
                            del st.session_state[key]
                        st.rerun()
            else:
                st.error(f"❌ Failed to send email: {message}")


# ── Main ──────────────────────────────────────────────────────────────────────

import re  # noqa: E402 — needed by _extract_section


def main() -> None:
    _init_state()
    _render_sidebar()

    st.title("🚀 TAS AutoBD")
    st.markdown(
        "*Automated Business Development v2 — insight-first, evidence-backed proposals*"
    )
    st.divider()

    step = st.session_state.step
    if step == 1:
        _step1()
    elif step == 2:
        _step2()
    elif step == 3:
        _step3()
    elif step == 4:
        _step4()
    else:
        st.session_state.step = 1
        st.rerun()


if __name__ == "__main__":
    main()
