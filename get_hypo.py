"""
TAS AutoBD — Hypothesis & Idea Generation Agent
=================================================
Analyses a company's characteristics with a human-centric lens and
proposes a tailored software / AI / IT solution.  Also extracts
GitHub-searchable keywords for the knowledge-base builder.
"""

import logging
from typing import List, Tuple

from langchain.prompts import PromptTemplate

from config import get_llm

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_IDEA_PROMPT = PromptTemplate(
    template=(
        "You are a Senior Business Developer at TAS Design Group Inc., an IT consulting "
        "and data science firm. Your role is to identify the most impactful software, "
        "AI, or ML solution for a prospective client.\n\n"
        "Apply a human-centric lens:\n"
        "1. Ethical implications — does the solution align with societal values?\n"
        "2. User experience — will it genuinely improve people's day-to-day work?\n"
        "3. Sustainability — what is the long-term environmental and social impact?\n"
        "4. Inclusivity — does it serve diverse user groups, including those with "
        "special needs?\n"
        "5. Privacy & security — how is user data protected?\n\n"
        "Based on the company characteristics below:\n"
        "- Identify the core human needs and pain points.\n"
        "- Recommend ONE focused software/AI/ML/IT solution TAS Design Group can build.\n"
        "- Explain how it will positively impact users' lives and business outcomes.\n\n"
        "Company Characteristics:\n{query}"
    ),
    input_variables=["query"],
)

_KEYWORD_PROMPT = PromptTemplate(
    template=(
        "You are a technical lead at an IT startup. Based on the proposed solution below, "
        "generate EXACTLY THREE simple, general keywords suitable for searching "
        "relevant open-source repositories on GitHub.\n\n"
        "Rules:\n"
        "- Each keyword must be lowercase with spaces replaced by underscores.\n"
        "- Keep keywords generic enough to return many results (e.g. 'machine_learning', "
        "'data_visualization', 'crm_system').\n"
        "- Output ONLY a numbered list — no explanations.\n\n"
        "Format:\n"
        "1. keyword_one\n"
        "2. keyword_two\n"
        "3. keyword_three\n\n"
        "Proposed solution:\n{query}"
    ),
    input_variables=["query"],
)


# ── Main function ─────────────────────────────────────────────────────────────

async def get_hypothesis_idea(characteristics: str) -> Tuple[object, List[str]]:
    """
    Generate a product hypothesis and GitHub search keywords.

    Parameters
    ----------
    characteristics : structured company profile string

    Returns
    -------
    (idea_message, keywords)
        idea_message : LangChain AIMessage with .content (str)
        keywords     : list of 1-3 lowercase underscore-separated strings
    """
    logger.info("Generating hypothesis from company characteristics …")
    llm = get_llm()

    # Chain 1 — generate the product idea
    idea_chain = _IDEA_PROMPT | llm
    idea = idea_chain.invoke({"query": characteristics})
    logger.info("Idea generated (%d chars)", len(idea.content))

    # Chain 2 — extract GitHub keywords from the idea
    keyword_chain = _KEYWORD_PROMPT | llm
    kw_result = keyword_chain.invoke({"query": idea.content})

    keywords: List[str] = []
    for line in kw_result.content.splitlines():
        cleaned = line.strip().lstrip("0123456789. ").strip()
        # Ignore empty lines and lines that look like email addresses
        if cleaned and "@" not in cleaned:
            keywords.append(cleaned.lower().replace(" ", "_"))

    if not keywords:
        logger.warning("Keyword extraction produced no results — using fallback.")
        keywords = ["software_solution"]

    keywords = keywords[:3]
    logger.info("Keywords: %s", keywords)
    return idea, keywords
