"""
TAS AutoBD — Company Research Agent (v2 — Insight-First)
==========================================================
Philosophy upgrade: The old system gathered facts. This system hunts for
*insight* — the specific, surprising, or overlooked information that makes
a buyer think "how did they know that?"

Three research upgrades over v1
--------------------------------
1. Insight Discovery Phase  — explicitly searches for strategic pivots, failures,
   competitive pressures, and recent leadership signals that most generic research
   misses. These become the "personalisation hooks" in the proposal.

2. Competitive Intelligence Phase — researches the prospect's main competitors
   deeply to understand where the company is falling behind. This feeds dynamic
   competitive positioning (not hardcoded "we do AI/ML in Japan").

3. Contact Quality Scoring  — all extracted emails are scored and ranked by
   decision-maker likelihood using contact_scorer.py. Generic addresses
   (info@, support@) are flagged, not silently passed to the sales rep.

Public API (unchanged signature)
---------------------------------
    get_company_information(company_name, another_url="", on_tool_call=None)
        → (profile_text: str, email_list: List[str], scored_contacts: List[ScoredContact])

Note: The function still returns a 2-tuple for backwards compatibility with the
existing app.py step handler. The scored_contacts are embedded in profile_text as
a special section that the UI parses.
"""

import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Tuple

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config import (
    ANTHROPIC_API_KEY,
    LLM_PROVIDER,
    get_llm,
    get_text_splitter,
    get_tavily_client,
)
from contact_scorer import score_contacts, ScoredContact

logger = logging.getLogger(__name__)


# ── Pydantic schema ────────────────────────────────────────────────────────────

class Company(BaseModel):
    name: str = Field(description="Company name")
    email: List[str] = Field(default_factory=list, description="Company email addresses found")
    website: str = Field(default="", description="Company website URL")
    phone_number: str = Field(default="", description="Company phone number")
    general_information: str = Field(description="General background, founding story, size, and key facts")
    industry_information: str = Field(description="Industry sector, market position, and competitive landscape")
    current_needs: str = Field(description="Active business needs, strategic priorities, and growth goals")
    lacking_areas: str = Field(description="Technology gaps, pain points, and areas ripe for improvement")
    software_production: str = Field(default="", description="Current software, IT systems, and tech stack in use")
    financial_highlights: str = Field(default="", description="Revenue, funding rounds, growth metrics, or financial health")
    sales_staff_info: str = Field(default="", description="Key decision-makers, sales team, or contact persons")
    contact_information: str = Field(default="", description="Contact details from the company's contact page")
    # v2 additions
    key_insights: str = Field(default="", description="3-5 specific, non-obvious facts: recent pivots, failures, competitive pressures, or strategic signals that most people would not know")
    competitive_gaps: str = Field(default="", description="Where this company is concretely falling behind its competitors — specific, evidence-backed weaknesses")
    summary: str = Field(description="2-3 sentence executive summary highlighting the biggest BD opportunity")


# ── Consolidation prompt ───────────────────────────────────────────────────────

_CONSOLIDATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Senior Business Developer at TAS Design Group Inc.\n"
        "Convert the research report below into a clean, structured company profile.\n"
        "Preserve ALL email addresses found. Translate everything to English.\n"
        "Focus on information that is actionable for a B2B pitch.\n"
        "Never fabricate data — only use what is in the report.\n\n"
        "Pay special attention to:\n"
        "- key_insights: pull out 3-5 specific, non-obvious, surprising facts\n"
        "  (recent leadership changes, failed product lines, public complaints,\n"
        "   lost market share, strategic pivots, regulatory issues, etc.)\n"
        "- competitive_gaps: identify concrete areas where competitors are ahead\n"
        "  and this company is visibly struggling or behind.",
    ),
    ("human", "Research report:\n\n{report}"),
])


# ═══════════════════════════════════════════════════════════════════════════════
# PATH A — Agentic (Anthropic / ReAct loop)
# ═══════════════════════════════════════════════════════════════════════════════

_RESEARCH_SYSTEM = """\
You are a Senior Business Developer at TAS Design Group Inc., an IT consulting \
and data science company specialising in AI/ML and custom software.

Your task: conduct deep market intelligence on a TARGET COMPANY to identify \
business-development opportunities for TAS Design Group.

RESEARCH METHODOLOGY — work through ALL phases with the available tools:

Phase 1 — Broad Overview (2-3 searches)
  Understand the company: size, industry, history, business model.

Phase 2 — Website Deep-Dive
  Fetch the official website. Look for product lines, recent announcements,
  and any technology language they use.

Phase 3 — Strategic Context (critical)
  Search for: recent strategic pivots, new CEO/CTO appointments, M&A activity,
  failed products, discontinued services, or markets they have exited.
  These signal pain points and open doors.

Phase 4 — Competitive Intelligence (critical)
  Identify the company's top 2-3 competitors. Search for evidence that the
  TARGET company is losing ground: market share loss, negative analyst reports,
  customer complaints about their technology, competitor announcements that
  directly threaten them. Be specific.

Phase 5 — Technology Audit
  Search for their current IT stack, software vendors they use, digital
  transformation initiatives, and any known tech debt or outages.

Phase 6 — Financial Signals
  Revenue, funding rounds, cost-cutting news, or growth signals from the
  last 12-24 months.

Phase 7 — Contact Discovery
  Fetch the /contact, /about, /team, /leadership pages. Use extract_emails
  on any page with staff information. Also search LinkedIn-style for
  "[company] CTO email" or "[company] VP Technology contact".

REQUIRED OUTPUT SECTIONS:
## Company Overview
## Industry & Market Position
## Strategic Pivots & Recent Signals
[3-5 specific, non-obvious facts — recent failures, pivots, leadership changes,
 or competitive pressures. These are the "hooks" that make a proposal feel
 researched rather than templated.]
## Competitive Gaps & Weaknesses
[Where are they concretely falling behind competitors? Be specific and
 evidence-backed. Never fabricate.]
## Technology Gaps & Pain Points
## Current Software & IT Systems
## Financial Highlights
## Key Contacts & Decision-Makers
[List names, titles, and any email addresses. Prioritise C-suite, VP, Director,
 CTO-level contacts over generic addresses.]
## Email Addresses Found
[List every email on its own line. Mark generic ones with [GENERIC] so they
 can be filtered.]
## Executive Summary
[2-3 sentences: biggest BD opportunity for TAS, referencing a specific
 competitive gap or strategic pivot you found.]

Translate all content to English. Never fabricate data.\
"""


def _agentic_research(
    company_name: str,
    another_url: str,
    on_tool_call: Optional[Callable],
) -> str:
    """Run the autonomous ReAct research loop. Returns raw research report text."""
    from agent_runner import run_agent
    from tools import RESEARCH_TOOL_SCHEMAS

    user_message = (
        f"Research this company for B2B business development: **{company_name}**\n\n"
        f"Be especially thorough on:\n"
        f"1. Recent strategic pivots, failures, or leadership changes (last 12-24 months)\n"
        f"2. Where they are losing ground to competitors\n"
        f"3. Direct contact emails for C-suite, VP, or Director-level people"
    )
    if another_url and another_url.startswith("http"):
        user_message += f"\n\nAlso include this URL in your research: {another_url}"

    return run_agent(
        system_prompt=_RESEARCH_SYSTEM,
        user_message=user_message,
        tools=RESEARCH_TOOL_SCHEMAS,
        max_iterations=15,   # increased from 12 to allow deeper research
        on_tool_call=on_tool_call,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PATH B — Classic (OpenAI or no Anthropic key)
# ═══════════════════════════════════════════════════════════════════════════════

_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Senior Business Developer at TAS Design Group Inc., an IT consulting "
        "and data science company. Extract structured intelligence about the target company "
        "from the web document provided. Translate all content to English. Focus on:\n"
        "- Company identity, size, and background\n"
        "- Active business needs and strategic priorities\n"
        "- Technology gaps and digital transformation challenges\n"
        "- Email addresses of sales, marketing, or executive staff\n"
        "- Current software, platforms, and IT infrastructure\n"
        "- Industry context, competitors, and market position\n"
        "- Financial health, funding, or growth signals\n"
        "- Recent pivots, failures, leadership changes, or competitive pressures\n\n"
        "If information is not found in the document, use an empty string. "
        "Never fabricate data.",
    ),
    ("human", "Target company: {company_name}\n\nWeb document:\n{content}"),
])

_SUMMARISE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Senior Business Developer at TAS Design Group Inc.\n"
        "Consolidate multiple raw intelligence extracts about a single company into one clean, "
        "structured company profile.\n"
        "- Merge and deduplicate information across all sources\n"
        "- Preserve all email addresses found\n"
        "- Resolve contradictions by preferring the most specific or recent data\n"
        "- Format output as a structured profile with clear section headers\n"
        "- Focus on what is actionable for a B2B pitch\n"
        "- Write in clear, professional English\n"
        "- Include a 'Strategic Pivots & Recent Signals' section with 3-5 specific facts\n"
        "- Include a 'Competitive Gaps & Weaknesses' section with evidence-backed observations",
    ),
    ("human", "Raw extractions from multiple sources about the target company:\n\n{extractions}"),
])


def _classic_research(company_name: str, another_url: str) -> str:
    """Parallel-crawler implementation with expanded query set. Returns consolidated profile."""
    tavily_client = get_tavily_client()
    llm = get_llm()
    text_splitter = get_text_splitter()

    queries = [
        f"{company_name} company overview history mission size",
        f"{company_name} annual report business strategy 2024 2025",
        f"{company_name} digital transformation technology challenges problems",
        f"{company_name} contact email team directory CTO VP director",
        f"{company_name} software IT systems technology stack vendors",
        f"{company_name} industry competitors market share competition",
        f"{company_name} news expansion growth investment funding 2024",
        # v2 additions — insight and competitive intelligence
        f"{company_name} problems failures discontinued product pivot restructure",
        f"{company_name} vs competitor losing market share falling behind",
        f"{company_name} CTO CEO director email LinkedIn contact",
    ]

    def _fetch_urls(query: str) -> List[str]:
        try:
            response = tavily_client.search(
                query, search_depth="advanced", topic="general", max_results=2
            )
            return [x["url"] for x in response.get("results", [])]
        except Exception as exc:
            logger.warning("Tavily search failed for %r: %s", query, exc)
            return []

    web_paths: List[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for urls in pool.map(_fetch_urls, queries):
            web_paths.extend(urls)

    web_paths = list(set(web_paths))
    if another_url and another_url.startswith("http"):
        web_paths.insert(0, another_url)

    logger.info("Found %d unique URLs to process", len(web_paths))
    extract_chain = _EXTRACT_PROMPT | llm

    def _process_url(url: str) -> str:
        from langchain_community.document_loaders import WebBaseLoader
        try:
            loader = WebBaseLoader(web_paths=(url,))
            docs = loader.load()
        except Exception as exc:
            logger.debug("Failed to load %s: %s", url, exc)
            return ""

        results = []
        for doc in docs:
            content = doc.page_content.strip()
            if not content:
                continue
            deadline = time.time() + 20
            try:
                if len(content) >= 128_000:
                    chunks = text_splitter.create_documents([content])
                    for chunk in chunks:
                        if time.time() > deadline:
                            break
                        try:
                            res = extract_chain.invoke(
                                {"content": chunk.page_content, "company_name": company_name}
                            )
                            results.append(res.content)
                        except Exception as exc:
                            logger.debug("Chunk extraction failed: %s", exc)
                else:
                    res = extract_chain.invoke(
                        {"content": content, "company_name": company_name}
                    )
                    results.append(res.content)
            except Exception as exc:
                logger.debug("Document processing failed for %s: %s", url, exc)
        return "\n".join(results)

    url_results: List[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_process_url, url): url for url in web_paths}
        for future in as_completed(futures):
            result = future.result()
            if result:
                url_results.append(result)

    combined = "\n\n---\n\n".join(url_results)
    summarise_chain = _SUMMARISE_PROMPT | llm
    result = summarise_chain.invoke({"extractions": combined[:60_000]})
    return result.content


# ═══════════════════════════════════════════════════════════════════════════════
# Public function
# ═══════════════════════════════════════════════════════════════════════════════

def get_company_information(
    company_name: str,
    another_url: str = "",
    on_tool_call: Optional[Callable] = None,
) -> Tuple[str, List[str]]:
    """
    Research *company_name* on the web and return a structured intelligence profile.

    v2 changes
    ----------
    - Research prompt now explicitly hunts for strategic pivots, failures, and
      competitive gaps (the "insight" layer that makes proposals feel researched)
    - All extracted emails are scored by contact_scorer and appended to the
      profile as a ranked, role-annotated contact table
    - Generic emails (info@, support@) are flagged and ranked last
    - max_iterations increased from 12 → 15

    Returns
    -------
    (profile_text, email_list)
        profile_text : structured company profile including scored contact table
        email_list   : viable emails only, ranked best-first by decision-maker score
    """
    logger.info("Starting research on: %s", company_name)

    use_agentic = bool(ANTHROPIC_API_KEY) and LLM_PROVIDER == "anthropic"

    if use_agentic:
        logger.info("Using agentic ReAct research loop (Anthropic).")
        raw_report = _agentic_research(company_name, another_url, on_tool_call)
    else:
        logger.info("Using classic parallel-crawler (OpenAI / fallback).")
        raw_report = _classic_research(company_name, another_url)

    logger.info("Raw research report: %d chars", len(raw_report))

    # ── Consolidate into structured Company Pydantic object ───────────────
    llm = get_llm()
    structured_llm = llm.with_structured_output(Company)
    try:
        company: Company = structured_llm.invoke(
            _CONSOLIDATE_PROMPT.format_messages(report=raw_report[:60_000])
        )
        profile_lines = [
            f"Company: {company.name}",
            f"Website: {company.website}",
            f"Phone: {company.phone_number}",
            f"Email(s): {', '.join(company.email) if company.email else 'Not found'}",
            "",
            f"General Information: {company.general_information}",
            f"Industry: {company.industry_information}",
            f"Current Needs: {company.current_needs}",
            f"Lacking Areas / Pain Points: {company.lacking_areas}",
            f"Software & IT Systems: {company.software_production}",
            f"Financial Highlights: {company.financial_highlights}",
            f"Sales / Key Contacts: {company.sales_staff_info}",
            f"Contact Information: {company.contact_information}",
        ]

        # v2: add insight and competitive gap sections if available
        if company.key_insights:
            profile_lines += ["", "=== KEY INSIGHTS (Proposal Hooks) ===", company.key_insights]
        if company.competitive_gaps:
            profile_lines += ["", "=== COMPETITIVE GAPS ===", company.competitive_gaps]

        profile_lines += ["", f"Executive Summary: {company.summary}"]

        profile = "\n".join(profile_lines)
        raw_emails = list(set(company.email))
    except Exception as exc:
        logger.warning("Structured consolidation failed, using raw report: %s", exc)
        profile = raw_report
        raw_emails = []

    # ── Augment emails with regex scan over the full report ───────────────
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    regex_emails = re.findall(email_pattern, raw_report)
    all_emails = list(set(raw_emails + regex_emails))

    # ── Score and rank contacts ───────────────────────────────────────────
    scored = score_contacts(all_emails)
    viable_emails = [c.email for c in scored if c.is_viable]
    all_contact_emails = [c.email for c in scored]  # includes blacklisted for display

    # Append a scored contact table to the profile so the UI can surface it
    if scored:
        contact_table_lines = ["", "=== CONTACT QUALITY SCORES ==="]
        for c in scored:
            status = "✓ VIABLE" if c.is_viable else "✗ FILTERED"
            contact_table_lines.append(
                f"{c.badge} [{c.tier_label}] {c.email}  (role: {c.role}, score: {c.score}/100) — {status}"
            )
        profile += "\n" + "\n".join(contact_table_lines)

    logger.info(
        "Research complete. %d total emails, %d viable.",
        len(all_emails),
        len(viable_emails),
    )
    return profile, viable_emails
