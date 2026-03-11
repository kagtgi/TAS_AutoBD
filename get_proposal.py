"""
TAS AutoBD — Proposal Generation Agent
=========================================
Uses RAG (retrieval-augmented generation) over the FAISS knowledge base
to craft a structured business proposal and a polished HTML email.

Agentic enhancements (v2)
--------------------------
After the initial email is drafted, a self-reflection cycle runs:
  1. Critique  — score the email on 5 sales-quality dimensions (1-10 each)
  2. Refine    — if the overall score is below the quality threshold, apply
                 the listed improvements in one targeted rewrite pass
  3. Return    — the best version of the email (original or refined)

This mirrors a senior BD manager reviewing and red-lining a draft before send.
"""

import re
import logging
from typing import List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import get_llm
from utils import format_docs

logger = logging.getLogger(__name__)

# Minimum quality score (out of 10) to skip refinement
_QUALITY_THRESHOLD = 7.5


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class Feature(BaseModel):
    name: str = Field(description="Feature name (5 words max)")
    description: str = Field(description="Concrete description of the feature and its business benefit (2-3 sentences)")
    impact: str = Field(description="Measurable impact or outcome this feature delivers")


class Proposal(BaseModel):
    name: str = Field(description="Compelling product or solution name (3-6 words)")
    tagline: str = Field(description="One-sentence value proposition for the customer")
    reason: str = Field(description="Why this customer needs this solution right now (specific to their situation)")
    market_context: str = Field(description="Relevant market trends and industry context supporting this solution")
    stakeholder_mapping: str = Field(description="Key stakeholders who will benefit and their specific gains")
    key_features: List[Feature] = Field(description="3-5 key features tailored to the customer's needs")
    competitive_advantage: str = Field(description="Why TAS Design Group is uniquely positioned to deliver this")
    success_metrics: str = Field(description="Specific KPIs and outcomes the customer can expect in 6-12 months")


# ── Prompts — RAG + Proposal + Email ──────────────────────────────────────────

_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Senior Solution Architect at TAS Design Group Inc. "
        "You have retrieved relevant technical evidence and case studies from similar projects. "
        "Synthesise this context into a concise, actionable summary that includes:\n"
        "1. Key capabilities and features proven in similar implementations\n"
        "2. Architecture and technology stack highlights\n"
        "3. Specific business benefits with concrete metrics where available\n"
        "4. Implementation considerations and success factors\n\n"
        "Be specific and evidence-based. Quote metrics when available. "
        "Relate everything to the customer's specific situation.",
    ),
    (
        "human",
        "Retrieved context from similar projects:\n{context}\n\n"
        "Customer situation and proposed solution:\n{question}",
    ),
])

_PROPOSAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Senior Business Developer at TAS Design Group Inc. crafting a formal, "
        "compelling business proposal for an enterprise client.\n\n"
        "Based on the solution research and company context provided, create a comprehensive "
        "proposal structured as a Proposal JSON object. Be specific, concrete, and client-focused.\n\n"
        "Requirements:\n"
        "- The solution name must be memorable and professional\n"
        "- Every claim must be grounded in the research provided\n"
        "- Features must directly address the customer's stated pain points\n"
        "- Success metrics must be realistic and specific (e.g. '30% reduction in manual processing time')\n"
        "- The competitive advantage must highlight TAS Design Group's actual strengths: "
        "AI/ML expertise, Japan+Vietnam presence, rapid delivery methodology",
    ),
    (
        "human",
        "Solution research and evidence:\n{research}\n\n"
        "Target company and proposed idea:\n{idea}",
    ),
])

_EMAIL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Business Developer at TAS Design Group Inc. writing a high-converting "
        "business proposal email. Your goal is to secure a discovery call.\n\n"
        "Email requirements:\n"
        "- Subject line must create urgency and curiosity (include in email body as <h2>)\n"
        "- Opening: personalised hook referencing a specific challenge the company faces\n"
        "- Value proposition: state the transformation, not just the features\n"
        "- Solution section: describe 3-5 key features with their SPECIFIC BUSINESS IMPACT\n"
        "- Social proof: briefly mention TAS Design Group's track record and expertise\n"
        "- Call to action: specific ask (e.g. '30-minute discovery call') with urgency\n"
        "- Closing: professional signature with contact details\n\n"
        "TAS Design Group profile:\n"
        "- IT consulting and data science company based in Japan and Vietnam\n"
        "- Specialises in AI/ML, data engineering, and custom software development\n"
        "- Delivers enterprise solutions with agile, rapid-iteration methodology\n"
        "- Trusted by clients across manufacturing, retail, finance, and logistics\n\n"
        "Formatting requirements:\n"
        "- Return ONLY valid HTML (no markdown, no code fences)\n"
        "- Use inline CSS for clean, professional styling\n"
        "- Color scheme: primary #4A0E8F (TAS purple), white background, dark text\n"
        "- Mobile-responsive layout with max-width: 650px\n"
        "- Use <table> for layout to ensure email client compatibility\n"
        "- Highlight key metrics and features with colored callout boxes",
    ),
    (
        "human",
        "Target company: {customer_name}\n\n"
        "Proposal details:\n{proposal}",
    ),
])


# ── Self-reflection prompts ────────────────────────────────────────────────────

_CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior B2B sales consultant reviewing a business proposal email "
        "before it is sent to a potential enterprise client.\n\n"
        "Evaluate the email on exactly these 5 criteria (score each 1-10):\n"
        "1. Personalization — Does it reference the company's specific situation, pain points, and industry?\n"
        "2. Evidence        — Are claims backed by concrete metrics, data, or case studies?\n"
        "3. Urgency         — Does it create a compelling reason to act NOW?\n"
        "4. Value Clarity   — Is the ROI and business transformation crystal clear?\n"
        "5. CTA Quality     — Is the call to action specific, low-friction, and easy to accept?\n\n"
        "Respond in EXACTLY this format (no extra text before or after):\n"
        "SCORES: personalization=[X]/10, evidence=[X]/10, urgency=[X]/10, value=[X]/10, cta=[X]/10\n"
        "OVERALL: [average]/10\n"
        "TOP IMPROVEMENTS:\n"
        "1. [specific, actionable improvement]\n"
        "2. [specific, actionable improvement]\n"
        "3. [specific, actionable improvement]",
    ),
    ("human", "Email to critique:\n\n{email_html}"),
])

_REFINE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Business Developer at TAS Design Group Inc. improving a B2B sales email.\n"
        "Apply ALL the listed improvements to make the email more compelling.\n\n"
        "Rules:\n"
        "- Keep the HTML structure and TAS Design Group purple (#4A0E8F) branding\n"
        "- Keep max-width: 650px mobile-responsive layout\n"
        "- Do NOT change factual content — only improve framing, specificity, and persuasion\n"
        "- Return ONLY valid HTML — no markdown fences, no preamble, no explanation",
    ),
    (
        "human",
        "Original email:\n{email_html}\n\n"
        "Required improvements:\n{improvements}\n\n"
        "Return the improved HTML email:",
    ),
])


# ── Self-reflection helper ─────────────────────────────────────────────────────

def _critique_and_refine(email_html: str, llm) -> str:
    """
    Run one critique + conditional refinement cycle on an HTML proposal email.

    Scores the email across 5 sales-quality dimensions.  If the overall score
    falls below _QUALITY_THRESHOLD (7.5 / 10), one targeted rewrite pass is
    applied.  Returns the best available version (original or refined).
    """
    try:
        critique_chain = _CRITIQUE_PROMPT | llm | StrOutputParser()
        critique = critique_chain.invoke({"email_html": email_html[:12_000]})
        logger.info("Critique output:\n%s", critique[:400])

        # Parse overall score
        score_match = re.search(
            r"OVERALL:\s*(\d+(?:\.\d+)?)\s*/\s*10", critique, re.IGNORECASE
        )
        score = float(score_match.group(1)) if score_match else 8.0
        logger.info("Proposal quality score: %.1f / 10 (threshold %.1f)", score, _QUALITY_THRESHOLD)

        if score >= _QUALITY_THRESHOLD:
            logger.info("Score %.1f meets threshold — no refinement needed.", score)
            return email_html

        # Extract the improvements section
        imp_match = re.search(
            r"TOP IMPROVEMENTS:(.*?)$", critique, re.DOTALL | re.IGNORECASE
        )
        improvements = imp_match.group(1).strip() if imp_match else critique

        logger.info("Refining proposal (score %.1f < %.1f) …", score, _QUALITY_THRESHOLD)
        refine_chain = _REFINE_PROMPT | llm | StrOutputParser()
        refined = refine_chain.invoke(
            {"email_html": email_html, "improvements": improvements}
        )

        # Basic sanity check: refined output should look like HTML
        if refined.strip().startswith("<"):
            logger.info("Refinement successful (%d chars).", len(refined))
            return refined

        logger.warning("Refined output did not look like HTML — keeping original.")
        return email_html

    except Exception as exc:
        logger.warning("Critique/refine cycle failed (%s) — returning original.", exc)
        return email_html


# ── Main function ─────────────────────────────────────────────────────────────

def make_proposal(idea: str, db, company_name: str) -> str:
    """
    Generate an HTML email proposal for *company_name* using RAG over *db*,
    then run a self-reflection critique+refine cycle.

    Parameters
    ----------
    idea         : the product idea / hypothesis string
    db           : FAISS vector store built by make_db()
    company_name : name of the target company (used in the email salutation)

    Returns
    -------
    HTML string of the ready-to-send, quality-reviewed email
    """
    logger.info("Generating proposal for: %s", company_name)
    llm = get_llm()

    # ── Step 1: RAG retrieval ─────────────────────────────────────────────
    retriever = db.as_retriever(search_kwargs={"k": 5})
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | _RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    rag_summary = rag_chain.invoke(idea)
    logger.info("RAG summary: %d chars", len(rag_summary))

    # ── Step 2: Structured proposal ───────────────────────────────────────
    structured_llm = llm.with_structured_output(Proposal)
    try:
        proposal: Proposal = structured_llm.invoke(
            _PROPOSAL_PROMPT.format_messages(research=rag_summary, idea=idea)
        )
        features_text = "\n".join(
            f"  - {f.name}: {f.description} (Impact: {f.impact})"
            for f in proposal.key_features
        )
        proposal_text = (
            f"Solution: {proposal.name}\n"
            f"Tagline: {proposal.tagline}\n"
            f"Why now: {proposal.reason}\n"
            f"Market context: {proposal.market_context}\n"
            f"Stakeholders: {proposal.stakeholder_mapping}\n"
            f"Key features:\n{features_text}\n"
            f"Competitive advantage: {proposal.competitive_advantage}\n"
            f"Success metrics: {proposal.success_metrics}"
        )
    except Exception as exc:
        logger.warning("Structured proposal failed, using raw RAG summary: %s", exc)
        proposal_text = rag_summary

    logger.info("Structured proposal ready.")

    # ── Step 3: Draft HTML email ──────────────────────────────────────────
    email_chain = _EMAIL_PROMPT | llm | StrOutputParser()
    email_html = email_chain.invoke(
        {"proposal": proposal_text, "customer_name": company_name}
    )
    logger.info("Initial email draft ready (%d chars).", len(email_html))

    # ── Step 4: Self-reflection — critique & conditional refinement ───────
    email_html = _critique_and_refine(email_html, llm)
    logger.info("Proposal finalised for: %s", company_name)

    return email_html
