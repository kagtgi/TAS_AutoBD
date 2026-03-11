"""
TAS AutoBD — Hypothesis & Idea Generation Agent
=================================================
Analyses a company's characteristics with a human-centric lens and
proposes a tailored software / AI / IT solution.  Also extracts
GitHub-searchable keywords for the knowledge-base builder.
"""

import logging
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate

from config import get_llm

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_IDEA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Senior Business Developer at TAS Design Group Inc., an IT consulting "
        "and data science firm specialising in AI, machine learning, and custom software. "
        "Your role is to identify the single most impactful technology solution for a "
        "prospective client based on their company profile.\n\n"
        "Evaluation framework — assess the proposed solution against all five dimensions:\n"
        "1. Business Impact — does it directly address a key pain point or unlock new revenue?\n"
        "2. Technical Feasibility — can TAS Design Group realistically build this in 3-6 months?\n"
        "3. User Experience — will it genuinely improve daily workflows for end users?\n"
        "4. Ethical Design — is it transparent, fair, privacy-respecting, and secure?\n"
        "5. Sustainability — does it deliver long-term value without creating dependency or waste?\n\n"
        "Output format:\n"
        "- Start with the SOLUTION NAME in bold\n"
        "- Write 2-3 sentences on WHY this company needs it right now\n"
        "- Describe WHAT TAS will build (core features, technology used)\n"
        "- Explain HOW it will transform their business operations\n"
        "- State the expected BUSINESS OUTCOMES (efficiency, cost savings, revenue)\n\n"
        "Be specific and concrete. Avoid generic recommendations. "
        "Recommend EXACTLY ONE solution.",
    ),
    (
        "human",
        "Company profile:\n\n{profile}",
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
    Generate a product hypothesis and GitHub search keywords.

    Parameters
    ----------
    characteristics : structured company profile string

    Returns
    -------
    (idea_text, keywords)
        idea_text : the proposed solution description as a string
        keywords  : list of 1-3 lowercase underscore-separated GitHub search keywords
    """
    logger.info("Generating hypothesis from company characteristics …")
    llm = get_llm()

    # Chain 1 — generate the product idea
    idea_chain = _IDEA_PROMPT | llm
    idea_result = idea_chain.invoke({"profile": characteristics})
    idea_text: str = idea_result.content
    logger.info("Idea generated (%d chars)", len(idea_text))

    # Chain 2 — extract GitHub keywords from the idea
    keyword_chain = _KEYWORD_PROMPT | llm
    kw_result = keyword_chain.invoke({"idea": idea_text})

    keywords: List[str] = []
    for line in kw_result.content.splitlines():
        cleaned = line.strip().lstrip("0123456789. ").strip()
        # Ignore empty lines and lines that look like email addresses
        if cleaned and "@" not in cleaned and len(cleaned) > 2:
            keywords.append(cleaned.lower().replace(" ", "_"))

    if not keywords:
        logger.warning("Keyword extraction produced no results — using fallback.")
        keywords = ["software_solution"]

    keywords = keywords[:3]
    logger.info("Keywords: %s", keywords)
    return idea_text, keywords
