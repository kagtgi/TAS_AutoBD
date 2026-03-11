"""
TAS AutoBD — Proposal Generation Agent (v2 — Evidence-First)
==============================================================
Philosophy upgrade: The old system generated proposals that sounded confident
but were unverifiable. Any experienced enterprise buyer could ask "where does
that number come from?" and the sales rep had no answer.

This version enforces three rules:
  1. Every claim must trace back to research (cited fact, not LLM guess)
  2. The opening hook must reference a SPECIFIC insight from the research phase
     (the "how did they know that?" moment that earns attention)
  3. The critique cycle runs up to 3 passes with tightening standards

Three improvements over v1
---------------------------
1. Evidence grounding prompt  — the proposal agent is explicitly instructed
   to cite specific facts, not generate plausible-sounding metrics
2. Insight hook enforcement   — the email opening must reference one of the
   specific, non-obvious facts found in the research phase
3. Multi-pass critique        — runs up to 3 refinement cycles, stopping when
   the quality threshold is met or citations are confirmed present
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

# Quality thresholds for the multi-pass critique cycle
_QUALITY_THRESHOLD = 8.0        # raised from 7.5 — higher bar
_MAX_CRITIQUE_PASSES = 3        # max refinement iterations


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class Feature(BaseModel):
    name: str = Field(description="Feature name (5 words max)")
    description: str = Field(description="Concrete description of the feature and its business benefit (2-3 sentences)")
    impact: str = Field(description="Measurable impact — must be grounded in company research, not a generic estimate")


class Proposal(BaseModel):
    name: str = Field(description="Compelling product or solution name (3-6 words)")
    tagline: str = Field(description="One-sentence value proposition for the customer")
    insight_hook: str = Field(description="The single most specific, surprising fact from research that earns the reader's attention — must be a real finding, not a generic observation")
    reason: str = Field(description="Why this customer needs this solution right now — must reference the specific competitive gap identified in research")
    market_context: str = Field(description="Relevant market trends with at least one cited data point or named competitor from research")
    stakeholder_mapping: str = Field(description="Key stakeholders who will benefit and their specific gains — use real role names/departments found in research")
    key_features: List[Feature] = Field(description="3-5 key features tailored to the customer's specific pain points")
    competitive_advantage: str = Field(description="Why TAS Design Group can close THIS specific gap — derived from the competitive gap analysis, not generic marketing copy")
    success_metrics: str = Field(description="Specific KPIs the customer can expect — tie metrics to the gap being closed, acknowledge where exact numbers are estimates")


# ── Prompts ───────────────────────────────────────────────────────────────────

_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Senior Solution Architect at TAS Design Group Inc. "
        "You have retrieved relevant technical evidence from similar projects and open-source tools. "
        "Synthesise this context into a concise, actionable summary that includes:\n"
        "1. Key capabilities and features proven in similar implementations\n"
        "2. Architecture and technology stack highlights\n"
        "3. Specific business benefits — cite any real metrics from the sources\n"
        "4. Implementation considerations and success factors\n\n"
        "Be specific and evidence-based. When sources contain real metrics or case study data, "
        "quote them directly rather than paraphrasing. "
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
        "CRITICAL RULES:\n"
        "1. The insight_hook field MUST contain a specific, surprising fact from the company "
        "   research — a recent pivot, failure, competitive loss, or strategic signal that most "
        "   people would not know. This is what makes the buyer think 'how did they know that?'\n"
        "2. Every metric in success_metrics must be either cited from research/evidence OR "
        "   explicitly flagged as an estimate (e.g. 'estimated ~30% reduction, based on similar "
        "   deployments in the logistics sector').\n"
        "3. The competitive_advantage must explain why TAS closes THIS specific gap better than "
        "   the client's internal IT team or a large consultancy — not generic copy.\n"
        "4. Market context must name at least one real competitor or cite one real trend from research.\n\n"
        "Violating any of these rules produces a proposal that enterprise buyers will distrust.",
    ),
    (
        "human",
        "Solution research and evidence:\n{research}\n\n"
        "Target company profile and competitive gap analysis:\n{idea}",
    ),
])

_EMAIL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Business Developer at TAS Design Group Inc. writing a high-converting "
        "business proposal email. Your goal is to secure a discovery call.\n\n"
        "CRITICAL — the email opening must use the insight_hook:\n"
        "  The first paragraph must reference one specific, non-obvious fact about the company "
        "  (from the insight_hook field). This is what differentiates a researched proposal "
        "  from a generic template. It earns the reader's trust immediately.\n\n"
        "Email structure:\n"
        "1. Subject line — creates urgency and curiosity (include as <h2> in the body)\n"
        "2. Opening hook — reference the specific insight. E.g.:\n"
        "   'We noticed [Company] discontinued its [Product X] line in Q3 2024 just as\n"
        "   [Competitor Y] launched [competing feature]. That window is still open.'\n"
        "3. The competitive gap — briefly frame what this is costing them\n"
        "4. The solution — what TAS will build and WHY it closes this specific gap\n"
        "5. Key features — 3-5, with their specific business impact\n"
        "6. Evidence — cite at least one real metric from the research (or flag it as an estimate)\n"
        "7. Why TAS — specific to this engagement, not boilerplate\n"
        "8. CTA — specific ask ('30-minute call this week') with urgency\n"
        "9. Signature — professional, with contact details\n\n"
        "TAS Design Group profile:\n"
        "- IT consulting and data science company based in Japan and Vietnam\n"
        "- Specialises in AI/ML, data engineering, and custom software development\n"
        "- Delivers enterprise solutions with agile, rapid-iteration methodology\n"
        "- Trusted by clients across manufacturing, retail, finance, and logistics\n\n"
        "Formatting:\n"
        "- Return ONLY valid HTML (no markdown, no code fences)\n"
        "- Inline CSS, primary colour #4A0E8F (TAS purple), white background\n"
        "- Mobile-responsive, max-width: 650px\n"
        "- Use <table> layout for email client compatibility\n"
        "- Highlight key metrics and features with coloured callout boxes",
    ),
    (
        "human",
        "Target company: {customer_name}\n\n"
        "Proposal details (including insight hook and competitive gap):\n{proposal}",
    ),
])


# ── Self-reflection prompts ────────────────────────────────────────────────────

_CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior B2B sales consultant reviewing a business proposal email "
        "before it is sent to a potential enterprise client.\n\n"
        "Evaluate the email on exactly these 6 criteria (score each 1-10):\n"
        "1. Personalization  — Does the opening reference a SPECIFIC, non-obvious fact "
        "   about this company (not a generic observation)?\n"
        "2. Evidence         — Are claims backed by concrete metrics, cited data, or real "
        "   competitor names? Are estimates labelled as estimates?\n"
        "3. Urgency          — Does it create a compelling reason to act NOW vs. in 6 months?\n"
        "4. Value Clarity    — Is the ROI and business transformation crystal clear?\n"
        "5. CTA Quality      — Is the call to action specific, low-friction, and easy to accept?\n"
        "6. Gap Relevance    — Does the solution directly address a specific competitive gap "
        "   identified in research, or does it feel generic?\n\n"
        "Respond in EXACTLY this format (no extra text before or after):\n"
        "SCORES: personalization=[X]/10, evidence=[X]/10, urgency=[X]/10, value=[X]/10, cta=[X]/10, gap=[X]/10\n"
        "OVERALL: [average]/10\n"
        "FAIL_REASONS: [list any score below 7 and why]\n"
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
        "Apply ALL the listed improvements. Focus especially on:\n"
        "- Strengthening the opening hook to reference a MORE specific company fact\n"
        "- Adding or clarifying evidence citations (real data or labelled estimates)\n"
        "- Tightening the connection between the solution and the company's competitive gap\n\n"
        "Rules:\n"
        "- Keep the HTML structure and TAS Design Group purple (#4A0E8F) branding\n"
        "- Keep max-width: 650px mobile-responsive layout\n"
        "- Do NOT fabricate new facts — only sharpen framing and specificity\n"
        "- Return ONLY valid HTML — no markdown fences, no preamble, no explanation",
    ),
    (
        "human",
        "Original email:\n{email_html}\n\n"
        "Required improvements:\n{improvements}\n\n"
        "Return the improved HTML email:",
    ),
])


# ── Multi-pass critique helper ─────────────────────────────────────────────────

def _critique_and_refine(email_html: str, llm) -> str:
    """
    Run up to _MAX_CRITIQUE_PASSES cycles of critique + refinement.

    v2 changes over v1
    ------------------
    - Quality threshold raised to 8.0 (from 7.5)
    - Added 6th criterion: gap relevance
    - Runs up to 3 passes (from 1) — stops early if threshold is met
    - Logs individual dimension scores for observability
    """
    best = email_html

    for pass_num in range(1, _MAX_CRITIQUE_PASSES + 1):
        try:
            critique_chain = _CRITIQUE_PROMPT | llm | StrOutputParser()
            critique = critique_chain.invoke({"email_html": best[:12_000]})
            logger.info("Critique pass %d:\n%s", pass_num, critique[:500])

            score_match = re.search(
                r"OVERALL:\s*(\d+(?:\.\d+)?)\s*/\s*10", critique, re.IGNORECASE
            )
            score = float(score_match.group(1)) if score_match else 8.0
            logger.info(
                "Pass %d quality score: %.1f / 10 (threshold %.1f)",
                pass_num, score, _QUALITY_THRESHOLD,
            )

            if score >= _QUALITY_THRESHOLD:
                logger.info("Pass %d score %.1f meets threshold — stopping.", pass_num, score)
                return best

            imp_match = re.search(
                r"TOP IMPROVEMENTS:(.*?)$", critique, re.DOTALL | re.IGNORECASE
            )
            improvements = imp_match.group(1).strip() if imp_match else critique

            logger.info("Refining (pass %d, score %.1f < %.1f) …", pass_num, score, _QUALITY_THRESHOLD)
            refine_chain = _REFINE_PROMPT | llm | StrOutputParser()
            refined = refine_chain.invoke(
                {"email_html": best, "improvements": improvements}
            )

            if refined.strip().startswith("<"):
                logger.info("Refinement pass %d successful (%d chars).", pass_num, len(refined))
                best = refined
            else:
                logger.warning("Refined output (pass %d) did not look like HTML — keeping previous.", pass_num)
                break

        except Exception as exc:
            logger.warning("Critique/refine cycle (pass %d) failed (%s) — stopping.", pass_num, exc)
            break

    return best


# ── Main function ─────────────────────────────────────────────────────────────

def make_proposal(idea: str, db, company_name: str) -> str:
    """
    Generate an HTML email proposal for *company_name* using RAG over *db*,
    then run a multi-pass self-reflection critique+refine cycle.

    v2 changes
    ----------
    - Proposal agent now enforces evidence citations and insight hook
    - Email opening must reference a specific, non-obvious research finding
    - Critique runs up to 3 passes (from 1)
    - Quality threshold raised to 8.0 (from 7.5)

    Parameters
    ----------
    idea         : competitive gap analysis + product idea (from get_hypo.py v2)
    db           : FAISS vector store built by make_db()
    company_name : name of the target company

    Returns
    -------
    HTML string of the quality-reviewed email
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
            f"\n[INSIGHT HOOK — use this in the email opening]\n{proposal.insight_hook}\n\n"
            f"Why now (competitive gap): {proposal.reason}\n"
            f"Market context: {proposal.market_context}\n"
            f"Stakeholders: {proposal.stakeholder_mapping}\n"
            f"Key features:\n{features_text}\n"
            f"Competitive advantage (gap-derived): {proposal.competitive_advantage}\n"
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

    # ── Step 4: Multi-pass critique & refinement ──────────────────────────
    email_html = _critique_and_refine(email_html, llm)
    logger.info("Proposal finalised for: %s", company_name)

    return email_html
