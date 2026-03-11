"""
TAS AutoBD — Hypothesis & Idea Generation Agent (v2 — Competitive-Gap-First)
=============================================================================
Philosophy upgrade: The old system proposed ONE solution based on company needs.
This version anchors the solution to the company's *specific competitive gaps*
(extracted in get_info.py) and derives TAS's positioning dynamically from
those gaps — not from a hardcoded "AI/ML, Japan, Vietnam" template.

Two-stage process
-----------------
Stage 1 — Competitive Gap Analysis
  Read the key_insights and competitive_gaps sections from the company profile.
  Identify the most urgent, evidence-backed gap where technology could close
  the distance to competitors within 3-6 months.

Stage 2 — Solution Ideation
  Propose ONE solution directly tied to that competitive gap.
  The proposed solution must pass the 5-dimension evaluation.
  The TAS competitive advantage section must be derived from the gap, not
  from generic marketing copy.
"""

import logging
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate

from config import get_llm

logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────────

_GAP_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Senior Competitive Intelligence Analyst at TAS Design Group Inc.\n\n"
        "Your task: Read the company research profile and extract the single most urgent "
        "competitive gap — the area where this company is concretely falling behind its "
        "competitors in a way that technology can close.\n\n"
        "Rules:\n"
        "- Use only information from the profile; never fabricate\n"
        "- The gap must be specific and evidence-backed, not generic\n"
        "  BAD: 'They lack digital transformation'\n"
        "  GOOD: 'Their main competitor launched an AI-powered logistics dashboard in Q3 2024 "
        "while this company still uses a 2017 ERP with no API layer'\n"
        "- Identify WHO in the company feels this gap most acutely (which role/department)\n"
        "- Estimate the business cost of the gap if possible\n\n"
        "Output format:\n"
        "GAP: [one-sentence description]\n"
        "EVIDENCE: [specific facts from the profile that support this]\n"
        "PAIN OWNER: [role/department most affected]\n"
        "BUSINESS COST: [estimated impact if available, or 'not quantified from available data']\n"
        "URGENCY: [why this needs to be addressed now, not in 2 years]",
    ),
    (
        "human",
        "Company profile:\n\n{profile}",
    ),
])


_IDEA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Senior Business Developer at TAS Design Group Inc., an IT consulting "
        "and data science firm specialising in AI, machine learning, and custom software.\n\n"
        "You have been given a company profile AND a specific competitive gap analysis.\n"
        "Your task: propose the single most impactful technology solution that closes this gap.\n\n"
        "Evaluation framework — the solution must score well on all five dimensions:\n"
        "1. Business Impact   — does it directly close the identified competitive gap?\n"
        "2. Technical Feasibility — can TAS deliver this in 3-6 months?\n"
        "3. User Experience   — will it improve daily workflows for the pain owner?\n"
        "4. Ethical Design    — transparent, fair, privacy-respecting, and secure?\n"
        "5. Sustainability    — long-term value without lock-in or waste?\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "- The solution must respond to the SPECIFIC GAP identified, not generic company needs\n"
        "- The competitive advantage section must explain WHY TAS can close THIS gap better\n"
        "  than the company's own IT team or a large consultancy — derive this from the gap\n"
        "  itself, NOT from generic 'AI/ML in Japan and Vietnam' copy\n"
        "- At least one claim about business outcomes must reference the gap evidence\n\n"
        "Output format:\n"
        "**[SOLUTION NAME]**\n\n"
        "WHY NOW: [Why this gap is urgent and why waiting costs them]\n\n"
        "WHAT WE BUILD: [Core features and technology — specific, not generic]\n\n"
        "HOW IT CLOSES THE GAP: [Direct connection between solution and competitive gap evidence]\n\n"
        "BUSINESS OUTCOMES: [Specific, realistic metrics tied to closing this gap]\n\n"
        "WHY TAS: [Derived from the gap and company situation, not boilerplate]\n\n"
        "Recommend EXACTLY ONE solution.",
    ),
    (
        "human",
        "Company profile:\n\n{profile}\n\n"
        "Competitive gap analysis:\n\n{gap_analysis}",
    ),
])

_KEYWORD_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a technical lead at an IT startup. Your task is to generate exactly three "
        "GitHub repository search keywords based on a proposed software solution.\n\n"
        "Rules:\n"
        "- Each keyword must be lowercase with words separated by underscores\n"
        "- Keywords should be generic enough to return many open-source repos "
        "(e.g. 'machine_learning', 'data_pipeline', 'crm_system', 'computer_vision')\n"
        "- Avoid overly niche terms that return few results\n"
        "- Focus on the core technology or domain, not the company-specific use case\n\n"
        "Output ONLY a numbered list — no explanations, no extra text:\n"
        "1. keyword_one\n"
        "2. keyword_two\n"
        "3. keyword_three",
    ),
    (
        "human",
        "Proposed solution:\n\n{idea}",
    ),
])


# ── Main function ─────────────────────────────────────────────────────────────

def get_hypothesis_idea(characteristics: str) -> Tuple[str, List[str]]:
    """
    Generate a product hypothesis grounded in the company's specific competitive gap.

    v2 changes
    ----------
    - Stage 1: Extract the specific competitive gap from the research profile
    - Stage 2: Propose a solution anchored to that gap
    - TAS competitive advantage is derived from the gap, not from hardcoded copy
    - Gap analysis is prepended to idea_text so it flows into get_proposal.py

    Parameters
    ----------
    characteristics : structured company profile string (output of get_info.py)

    Returns
    -------
    (idea_text, keywords)
        idea_text : gap analysis + proposed solution (combined for the proposal agent)
        keywords  : list of 1-3 lowercase underscore-separated GitHub search keywords
    """
    logger.info("Generating competitive gap analysis …")
    llm = get_llm()

    # Stage 1 — Competitive gap analysis
    gap_chain = _GAP_ANALYSIS_PROMPT | llm
    gap_result = gap_chain.invoke({"profile": characteristics})
    gap_analysis: str = gap_result.content
    logger.info("Gap analysis (%d chars):\n%s", len(gap_analysis), gap_analysis[:300])

    # Stage 2 — Solution ideation anchored to the gap
    idea_chain = _IDEA_PROMPT | llm
    idea_result = idea_chain.invoke({
        "profile": characteristics,
        "gap_analysis": gap_analysis,
    })
    idea_text: str = idea_result.content
    logger.info("Idea generated (%d chars)", len(idea_text))

    # Combine gap + idea so the proposal agent has full context
    combined_idea = (
        "=== COMPETITIVE GAP ANALYSIS ===\n"
        + gap_analysis
        + "\n\n=== PROPOSED SOLUTION ===\n"
        + idea_text
    )

    # Extract GitHub keywords from the solution
    keyword_chain = _KEYWORD_PROMPT | llm
    kw_result = keyword_chain.invoke({"idea": idea_text})

    keywords: List[str] = []
    for line in kw_result.content.splitlines():
        cleaned = line.strip().lstrip("0123456789. ").strip()
        if cleaned and "@" not in cleaned and len(cleaned) > 2:
            keywords.append(cleaned.lower().replace(" ", "_"))

    if not keywords:
        logger.warning("Keyword extraction produced no results — using fallback.")
        keywords = ["software_solution"]

    keywords = keywords[:3]
    logger.info("Keywords: %s", keywords)
    return combined_idea, keywords
