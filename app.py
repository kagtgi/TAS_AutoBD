"""
TAS AutoBD — Streamlit Application
=====================================
Agentic LLM-powered automatic business development tool.
Guides the user through a 4-step wizard:

  Step 1 → Autonomous research of target company (ReAct agent)
  Step 2 → Review profile & generate product idea
  Step 3 → Build knowledge DB & generate proposal (with self-reflection)
  Step 4 → Review & send HTML email proposal

Run with:
    streamlit run app.py
"""

import logging
import os

import streamlit as st

# ── Third-party async compat (needed for Streamlit + asyncio) ─────────────────
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# ── Local imports ─────────────────────────────────────────────────────────────
from config import validate_config
from utils import clean_html_fences
from email_utils import send_email, add_email_manually

# Pipeline agents imported lazily inside handlers to keep startup fast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


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
    .badge-ok   { color: #16a34a; font-weight: bold; }
    .badge-warn { color: #dc2626; font-weight: bold; }
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
            ("1️⃣", "Agent researches company autonomously"),
            ("2️⃣", "Review profile & generate idea"),
            ("3️⃣", "Build knowledge base & draft proposal"),
            ("4️⃣", "Review, refine & send email"),
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
        "agent_tool_log": [],   # list of {"tool": str, "inputs": dict}
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Agent tool-call log widget ─────────────────────────────────────────────────

_TOOL_ICONS = {
    "web_search": "🔍",
    "fetch_webpage": "🌐",
    "extract_emails": "📧",
    "search_github": "🐙",
    "fetch_readme": "📄",
}


def _render_tool_log(log: list) -> None:
    """Display the agent's tool-call history in a collapsible expander."""
    if not log:
        return
    label = f"🤖 Agent used **{len(log)} tool calls** to research this company"
    with st.expander(label, expanded=False):
        for entry in log:
            tool = entry.get("tool", "unknown")
            inputs = entry.get("inputs", {})
            icon = _TOOL_ICONS.get(tool, "🔧")
            # Pick the most descriptive input field for display
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


# ── Step rendering functions ──────────────────────────────────────────────────

def _step1() -> None:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.subheader("Step 1 — Research Target Company")
    st.markdown(
        "Enter the company name. The **autonomous research agent** will decide "
        "what to search for, which pages to fetch, and when it has enough "
        "intelligence — no fixed query templates."
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
            """Collect each tool call for display after research completes."""
            tool_log.append({"tool": tool_name, "inputs": tool_inputs})

        with st.spinner("🤖 Autonomous research agent working … (≈ 45–60 s)"):
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
        "Review the extracted intelligence profile. Edit if needed, then click "
        "**Generate Idea** to produce a tailored product hypothesis."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Show agent tool-call log from Step 1
    _render_tool_log(st.session_state.get("agent_tool_log", []))

    characteristics = st.text_area(
        "Company Intelligence Profile",
        value=st.session_state.characteristics,
        height=350,
        help="Autonomously extracted by the research agent from public web sources.",
    )
    if characteristics != st.session_state.characteristics:
        st.session_state.characteristics = characteristics

    if st.session_state.email_list:
        st.info(f"📧 Found {len(st.session_state.email_list)} email address(es) automatically.")

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

        with st.spinner("🤔 Analysing profile and generating product idea …"):
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
    st.subheader("Step 3 — Product Idea & Knowledge Base")
    st.markdown(
        "Review the generated idea. Edit if needed, then click **Generate Proposal** "
        "to build the knowledge base and draft the email. "
        "The proposal agent will **self-critique and refine** its draft before presenting it."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    idea = st.text_area(
        "Product Idea",
        value=st.session_state.idea,
        height=250,
        help="AI-generated product hypothesis based on the company profile.",
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

        with st.spinner("✍️ Drafting proposal and running self-critique …"):
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
        "Review the quality-reviewed HTML proposal below. Edit directly if needed, "
        "then select a recipient and click **Send Email**."
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
        st.warning("No email addresses found. Add one manually above.")
    else:
        selected_email = st.selectbox(
            "Recipient",
            options=st.session_state.email_list,
            key="selected_email",
        )

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
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            else:
                st.error(f"❌ Failed to send email: {message}")


# ── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    _init_state()
    _render_sidebar()

    st.title("🚀 TAS AutoBD")
    st.markdown(
        "*Automated Business Development — powered by agentic AI with native tool use*"
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
