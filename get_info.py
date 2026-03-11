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
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import WebBaseLoader

from config import (
    get_llm,
    get_text_splitter,
    get_semantic_splitter,
    get_tavily_client,
)

logger = logging.getLogger(__name__)

# ── Pydantic schema for extracted company data ────────────────────────────────

class Company(BaseModel):
    name: str = Field(description="Company name")
    email: List[str] = Field(default_factory=list, description="Company email addresses")
    general_information: str = Field(description="General information about the company")
    current_needs: str = Field(description="What the company currently needs")
    lacking_areas: str = Field(description="Areas where the company is lacking or could improve")
    sales_staff_info: str = Field(default="", description="Sales staff information")
    contact_information: str = Field(default="", description="Contact page information")
    software_production: str = Field(default="", description="Software or IT production info")
    industry_information: str = Field(description="Industry and market information")
    financial_highlights: str = Field(default="", description="Financial highlights")
    summary: str = Field(description="Short summary suitable for business development")
    website: str = Field(default="", description="Company website URL")
    phone_number: str = Field(default="", description="Company phone number")


# ── Prompts ───────────────────────────────────────────────────────────────────

_EXTRACT_PROMPT = PromptTemplate(
    template=(
        "You are a Senior Business Developer at TAS Design Group Inc. conducting "
        "market research on a potential customer.\n\n"
        "Extract the following structured information about the company '{company_name}' "
        "from the web document below. Translate all content into English.\n\n"
        "Focus on information relevant to business development:\n"
        "- Company identity and general background\n"
        "- Current business needs and pain points\n"
        "- Technology gaps and lacking areas\n"
        "- Sales / contact staff emails\n"
        "- Software or IT systems already in use\n"
        "- Industry context and competitors\n"
        "- Financial health indicators\n\n"
        "{format_instructions}\n\n"
        "Web document:\n{query}"
    ),
    input_variables=["query", "company_name"],
    partial_variables={},  # filled at call time
)

_SUMMARISE_PROMPT = PromptTemplate(
    template=(
        "You are a Senior Business Developer at TAS Design Group Inc.\n\n"
        "Below are multiple raw extractions about a single company. "
        "Consolidate them into a clean, structured profile. "
        "MUST include the company email(s). "
        "Format each characteristic on its own line as:\n"
        "  CharacteristicName: value\n\n"
        "Raw extractions:\n{query}"
    ),
    input_variables=["query"],
)


# ── Main function ─────────────────────────────────────────────────────────────

def get_company_information(
    company_name: str, another_url: str = ""
) -> Tuple[object, List[str]]:
    """
    Research *company_name* on the web and return:

    Returns
    -------
    (characteristics_message, email_list)
        characteristics_message : LangChain AIMessage with .content (str)
        email_list              : deduplicated list of discovered email addresses
    """
    logger.info("Starting research on: %s", company_name)
    tavily_client = get_tavily_client()
    llm = get_llm()
    text_splitter = get_text_splitter()

    # Try to initialise the semantic splitter; fall back to the basic one if
    # OpenAI embeddings aren't available (e.g., during testing without API key)
    try:
        semantic_splitter = get_semantic_splitter()
    except Exception:
        logger.warning("SemanticChunker unavailable – using RecursiveCharacterTextSplitter")
        semantic_splitter = text_splitter

    # ── Step 1: collect URLs ──────────────────────────────────────────────────
    queries = [
        f"{company_name} company overview",
        f"{company_name} company information",
        f"{company_name} current challenges or needs",
        f"{company_name} contact email staff",
        f"{company_name} software technology systems",
        f"{company_name} industry market competitors",
        f"{company_name} future plans growth",
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

    # ── Step 2: extract structured info from each URL ────────────────────────
    pydantic_parser = PydanticOutputParser(pydantic_object=Company)
    extract_prompt = _EXTRACT_PROMPT.partial(
        format_instructions=pydantic_parser.get_format_instructions()
    )
    extract_chain = extract_prompt | llm

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
                    try:
                        chunks = semantic_splitter.create_documents([content])
                    except Exception:
                        chunks = text_splitter.create_documents([content])
                    for chunk in chunks:
                        if time.time() > deadline:
                            break
                        try:
                            res = extract_chain.invoke(
                                {"query": chunk.page_content, "company_name": company_name}
                            )
                            results.append(res.content)
                        except Exception as exc:
                            logger.debug("Chunk extraction failed: %s", exc)
                else:
                    res = extract_chain.invoke(
                        {"query": content, "company_name": company_name}
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

    combined = "\n\n".join(url_results)
    logger.info("Processed %d URLs; combined text length: %d", len(url_results), len(combined))

    # ── Step 3: consolidate into one structured profile ───────────────────────
    summarise_chain = _SUMMARISE_PROMPT | llm
    characteristics = summarise_chain.invoke({"query": combined})

    # ── Step 4: extract email addresses ───────────────────────────────────────
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    emails = list(set(re.findall(email_pattern, characteristics.content)))

    logger.info("Research complete. Found %d email address(es).", len(emails))
    return characteristics, emails
