"""
TAS AutoBD — Streamlit Application
=====================================
Agentic LLM-powered automatic business development tool.
Guides the user through a 4-step wizard:

  Step 1 → Research target company
  Step 2 → Review profile & generate product idea
  Step 3 → Build knowledge DB & generate proposal
  Step 4 → Review & send HTML email proposal

Run with:
    streamlit run app.py
"""

import asyncio
import logging
import os

import streamlit as st

# ── Third-party async compat (needed for Streamlit + asyncio) ─────────────────
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # nest_asyncio is optional; most envs will have it

# ── Local imports ─────────────────────────────────────────────────────────────
from config import validate_config
from utils import run_async, clean_html_fences
from email_utils import send_email, add_email_manually

# Pipeline agents imported lazily inside handlers to avoid slow startup
# (their config objects are created only when needed)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Page config (must be first Streamlit call) ────────────────────────────────
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
    /* ── Global ── */
    .stApp {
        background-color: #FFFFFF;
        color: #4A0E8F;
        font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
    }

    /* ── Buttons ── */
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

    /* ── Inputs ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #F4EFFC !important;
        color: #1A0030 !important;
        border: 1px solid #4A0E8F !important;
        border-radius: 6px !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background-color: #F4EFFC !important;
        color: #1A0030 !important;
        border: 1px solid #4A0E8F !important;
    }

    /* ── Headers ── */
    h1, h2, h3, h4 { color: #4A0E8F !important; }

    /* ── Progress bar ── */
    .stProgress > div > div > div > div {
        background-color: #4A0E8F !important;
    }

    /* ── Sidebar ── */
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

    /* ── Step card ── */
    .step-card {
        background: #F9F6FF;
        border-left: 4px solid #4A0E8F;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* ── Badge ── */
    .badge-ok   { color: #16a34a; font-weight: bold; }
    .badge-warn { color: #dc2626; font-weight: bold; }
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

        # API key status
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
            ("1️⃣", "Enter company name"),
            ("2️⃣", "Review profile & generate idea"),
            ("3️⃣", "Build knowledge base"),
            ("4️⃣", "Send proposal email"),
        ]
        for icon, label in steps:
            st.markdown(f"{icon} {label}")

        st.divider()
        current_step = st.session_state.get("step", 1)
        st.markdown(f"**Current step:** {current_step} / 4")
        progress = (current_step - 1) / 3
        st.progress(progress)

        if current_step > 1:
            if st.button("↩ Start Over"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        st.divider()
        st.caption("© 2025 TAS Design Group Inc.")


# ── Session state initialisation ──────────────────────────────────────────────

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
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Step rendering functions ──────────────────────────────────────────────────

def _step1() -> None:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.subheader("Step 1 — Research Target Company")
    st.markdown(
        "Enter the name of the company you want to pitch to. "
        "AutoBD will crawl the web and build a structured intelligence profile."
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

        with st.spinner("🔎 Crawling the web for company intelligence …"):
            try:
                from get_info import get_company_information

                characteristics, emails = get_company_information(
                    st.session_state.company_name,
                    st.session_state.additional_url,
                )
                st.session_state.characteristics = characteristics
                st.session_state.email_list = emails
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

    characteristics = st.text_area(
        "Company Intelligence Profile",
        value=st.session_state.characteristics,
        height=350,
        help="This profile was automatically extracted from public web sources.",
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
        "to build the knowledge base and draft the email."
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

        with st.spinner("🏗️ Building knowledge base from GitHub & web (this takes 1-2 minutes) …"):
            try:
                from make_db import make_db

                db = run_async(
                    make_db(st.session_state.idea, st.session_state.keywords)
                )
                st.session_state.docsearch = db
            except Exception as exc:
                logger.error("make_db error: %s", exc, exc_info=True)
                st.error(f"Knowledge base build failed: {exc}")
                return

        with st.spinner("✍️ Crafting your personalised proposal …"):
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
        "Review the HTML proposal below. Edit directly if needed, then select "
        "a recipient and click **Send Email**."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Email editor
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

    # Download button
    st.download_button(
        label="⬇️ Download HTML",
        data=st.session_state.email_proposal,
        file_name=f"proposal_{st.session_state.company_name.replace(' ', '_')}.html",
        mime="text/html",
    )

    st.divider()
    st.subheader("Send Email")

    # Email address management
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
                # Reset to allow a new proposal run
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
        "*Automated Business Development — powered by agentic AI pipelines*"
    )
    st.divider()

    # Route to the current step
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
