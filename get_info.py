"""
TAS AutoBD — Company Research Agent (Agentic)
===============================================
Uses Claude's native tool-use (ReAct loop) to autonomously research a target
company.  The agent decides which search queries to run, which pages to fetch,
and when it has gathered enough intelligence — no more hard-coded 7-query lists.

When ANTHROPIC_API_KEY is available the full agentic path is used.
When only OPENAI_API_KEY is available the system falls back to the classic
LangChain-based parallel crawler (identical to the original implementation).

Public API (unchanged)
----------------------
    get_company_information(company_name, another_url="", on_tool_call=None)
        → (profile_text: str, email_list: List[str])
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
    summary: str = Field(description="2-3 sentence executive summary highlighting the biggest BD opportunity")


# ── Consolidation prompt (shared by both paths) ───────────────────────────────

_CONSOLIDATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Senior Business Developer at TAS Design Group Inc.\n"
        "Convert the research report below into a clean, structured company profile.\n"
        "Preserve ALL email addresses found. Translate everything to English.\n"
        "Focus on information that is actionable for a B2B pitch.\n"
        "Never fabricate data — only use what is in the report.",
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

RESEARCH METHODOLOGY — work through these phases with the available tools:
1. Broad overview   — 2-3 web searches to understand the company (size, industry, history)
2. Website deep-dive — fetch the company's official website for authoritative details
3. Strategic context — search for recent news, strategic plans, and upcoming initiatives
4. Technology audit  — search for current IT stack, software used, digital transformation pain points
5. Financial signals — search for revenue, funding rounds, investment news
6. Contacts          — search for and fetch contact/team pages; use extract_emails on them
7. Competitive lens  — identify main competitors and how the company is positioned

REQUIRED INFORMATION (keep researching until you have ALL of these):
• Company background, founding story, size (employees, offices)
• Industry position and main competitors
• Current strategic priorities and business goals (2024-2025)
• Technology gaps and digital transformation challenges
• Current software / IT systems in use
• Financial highlights (revenue, funding, growth signals)
• Key decision-makers and their roles
• ALL email addresses found (especially CTO, IT, marketing, executive contacts)
• Company website URL

OUTPUT FORMAT
When you have gathered sufficient information, write a comprehensive profile
with these sections:

## Company Overview
## Industry & Market Position
## Strategic Priorities & Business Needs
## Technology Gaps & Pain Points
## Current Software & IT Systems
## Financial Highlights
## Key Contacts & Decision-Makers
## Email Addresses Found
[list every email on its own line]
## Executive Summary
[2-3 sentences: biggest BD opportunity for TAS Design Group]

Translate all content to English.  Never fabricate data."""


def _agentic_research(
    company_name: str,
    another_url: str,
    on_tool_call: Optional[Callable],
) -> str:
    """Run the autonomous ReAct research loop. Returns raw research report text."""
    from agent_runner import run_agent
    from tools import RESEARCH_TOOL_SCHEMAS

    user_message = f"Research this company for B2B business development: **{company_name}**"
    if another_url and another_url.startswith("http"):
        user_message += f"\n\nAlso include this URL in your research: {another_url}"

    return run_agent(
        system_prompt=_RESEARCH_SYSTEM,
        user_message=user_message,
        tools=RESEARCH_TOOL_SCHEMAS,
        max_iterations=12,
        on_tool_call=on_tool_call,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PATH B — Classic (OpenAI or no Anthropic key) — original parallel crawler
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
        "- Financial health, funding, or growth signals\n\n"
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
        "- Write in clear, professional English",
    ),
    ("human", "Raw extractions from multiple sources about the target company:\n\n{extractions}"),
])


def _classic_research(company_name: str, another_url: str) -> str:
    """Original parallel-crawler implementation. Returns consolidated profile text."""
    tavily_client = get_tavily_client()
    llm = get_llm()
    text_splitter = get_text_splitter()

    queries = [
        f"{company_name} company overview history mission",
        f"{company_name} annual report business strategy 2024 2025",
        f"{company_name} digital transformation technology challenges",
        f"{company_name} contact email staff directory",
        f"{company_name} software IT systems technology stack",
        f"{company_name} industry competitors market position",
        f"{company_name} news expansion growth investment",
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

    Uses the agentic ReAct loop (Anthropic) when possible; falls back to the
    classic parallel-crawler approach when only OpenAI is configured.

    Parameters
    ----------
    company_name : target company to research
    another_url  : optional additional URL to include in research
    on_tool_call : optional callback(tool_name, tool_inputs) for progress
                   tracking in the UI (only fires in agentic mode)

    Returns
    -------
    (profile_text, email_list)
        profile_text : structured company profile as a plain string
        email_list   : deduplicated list of discovered email addresses
    """
    logger.info("Starting research on: %s", company_name)

    # ── Choose research path ───────────────────────────────────────────────
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
            "",
            f"Executive Summary: {company.summary}",
        ]
        profile = "\n".join(profile_lines)
        email_list = list(set(company.email))
    except Exception as exc:
        logger.warning("Structured consolidation failed, using raw report: %s", exc)
        profile = raw_report
        email_list = []

    # ── Augment emails with regex scan over the full report ───────────────
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    regex_emails = re.findall(email_pattern, raw_report)
    email_list = list(set(email_list + regex_emails))

    logger.info("Research complete. Found %d email address(es).", len(email_list))
    return profile, email_list
