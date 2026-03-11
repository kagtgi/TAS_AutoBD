"""
TAS AutoBD — Company Information Agent
========================================
Crawls the web for a target company using the Tavily search API,
extracts structured characteristics with an LLM, and returns a
human-readable profile along with discovered email addresses.
"""

import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader

from config import (
    get_llm,
    get_text_splitter,
    get_tavily_client,
)

logger = logging.getLogger(__name__)


# ── Pydantic schema for extracted company data ────────────────────────────────

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


# ── Prompts ───────────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Senior Business Developer at TAS Design Group Inc., an IT consulting "
        "and data science company. You are conducting deep market research on a potential "
        "enterprise client to identify business development opportunities.\n\n"
        "Your task: extract structured intelligence about the target company from the web "
        "document provided. Translate all content to English. Focus only on information "
        "relevant to B2B outreach:\n"
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
    (
        "human",
        "Target company: {company_name}\n\nWeb document:\n{content}",
    ),
])

_SUMMARISE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Senior Business Developer at TAS Design Group Inc.\n\n"
        "You have collected multiple raw intelligence extracts about a single company from "
        "different web sources. Your task is to consolidate them into one clean, structured "
        "company profile.\n\n"
        "Requirements:\n"
        "- Merge and deduplicate information across all sources\n"
        "- Preserve all email addresses found\n"
        "- Resolve contradictions by preferring the most specific or recent data\n"
        "- Format output as a structured profile with clear section headers\n"
        "- Focus on what is actionable for a B2B pitch\n"
        "- Write in clear, professional English",
    ),
    (
        "human",
        "Raw extractions from multiple sources about the target company:\n\n{extractions}",
    ),
])


# ── Main function ─────────────────────────────────────────────────────────────

def get_company_information(
    company_name: str, another_url: str = ""
) -> Tuple[str, List[str]]:
    """
    Research *company_name* on the web and return a structured profile.

    Returns
    -------
    (profile_text, email_list)
        profile_text : consolidated company profile as a plain string
        email_list   : deduplicated list of discovered email addresses
    """
    logger.info("Starting research on: %s", company_name)
    tavily_client = get_tavily_client()
    llm = get_llm()
    text_splitter = get_text_splitter()

    # ── Step 1: collect URLs ──────────────────────────────────────────────────
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

    web_paths = list(set(web_paths))  # deduplicate
    if another_url and another_url.startswith("http"):
        web_paths.insert(0, another_url)

    logger.info("Found %d unique URLs to process", len(web_paths))

    # ── Step 2: extract info from each URL ───────────────────────────────────
    extract_chain = _EXTRACT_PROMPT | llm

    def _process_url(url: str) -> str:
        """Load a URL, chunk if needed, run LLM extraction, return raw text."""
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

            deadline = time.time() + 20  # hard per-URL time budget

            try:
                if len(content) >= 128_000:
                    chunks = text_splitter.create_documents([content])
                    for chunk in chunks:
                        if time.time() > deadline:
                            break
                        try:
                            res = extract_chain.invoke({
                                "content": chunk.page_content,
                                "company_name": company_name,
                            })
                            results.append(res.content)
                        except Exception as exc:
                            logger.debug("Chunk extraction failed: %s", exc)
                else:
                    res = extract_chain.invoke({
                        "content": content,
                        "company_name": company_name,
                    })
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
    logger.info("Processed %d URLs; combined text length: %d chars", len(url_results), len(combined))

    # ── Step 3: use structured output to consolidate into Company profile ─────
    structured_llm = llm.with_structured_output(Company)
    try:
        company: Company = structured_llm.invoke(
            _SUMMARISE_PROMPT.format_messages(extractions=combined[:60_000])
        )
        # Format as a readable profile string
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
        logger.warning("Structured output failed, falling back to text summary: %s", exc)
        # Fallback: use text summary and regex email extraction
        summarise_chain = _SUMMARISE_PROMPT | llm
        result = summarise_chain.invoke({"extractions": combined[:60_000]})
        profile = result.content
        email_list = []

    # ── Step 4: augment emails with regex scan ────────────────────────────────
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    regex_emails = re.findall(email_pattern, combined)
    email_list = list(set(email_list + regex_emails))

    logger.info("Research complete. Found %d email address(es).", len(email_list))
    return profile, email_list
