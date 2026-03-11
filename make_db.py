"""
TAS AutoBD — Knowledge Database Builder (v2)
=============================================
Downloads relevant GitHub repository READMEs and web articles based on the
generated keywords, then builds a FAISS vector store for RAG retrieval.

Pipeline (simplified from v1 — no ZIP downloads)
-------------------------------------------------
  1. GitHub search  — find top repos per keyword via the GitHub REST API
  2. README fetch   — download README.md directly from raw.githubusercontent.com
                      (parallel, using the tools module; replaces ZIP download)
  3. Web articles   — Tavily search + WebBaseLoader per keyword
  4. Summarise      — LLM summarisation of all collected documents (capped at 20)
  5. FAISS index    — build and return an in-memory vector store

This module is fully synchronous; the async infrastructure from v1 has been
replaced by ThreadPoolExecutor calls via the tools module.
"""

import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

from config import (
    get_llm,
    get_hf_embeddings,
    get_text_splitter,
    get_tavily_client,
)

logger = logging.getLogger(__name__)


# ── LLM Summarisation Prompts ─────────────────────────────────────────────────

_WEB_SUMMARISE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Senior Business Developer at TAS Design Group Inc. preparing research "
        "for a client proposal. Summarise the key insights from the web article below.\n\n"
        "Focus on:\n"
        "1. The core problem this technology/product/approach solves\n"
        "2. The technical approach and key capabilities\n"
        "3. Measurable business outcomes, ROI, or performance metrics\n"
        "4. Real-world adoption or case studies mentioned\n\n"
        "Keep the summary concise (150-250 words). Translate everything to English. "
        "Avoid marketing fluff — stick to facts and concrete benefits.",
    ),
    ("human", "{content}"),
])

_REPO_SUMMARISE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Business Developer at TAS Design Group Inc. analysing open-source "
        "repositories to gather evidence for a client proposal.\n\n"
        "Extract and summarise the key information with a business focus:\n"
        "1. Core problem the project solves and its solution approach\n"
        "2. Key technical capabilities and architecture\n"
        "3. Adoption signals (stars, contributors, production use cases)\n"
        "4. Business benefits and measurable outcomes if mentioned\n"
        "5. Integration complexity and deployment requirements\n\n"
        "Keep the summary concise (150-250 words). "
        "Do NOT include source code. Focus on business relevance.",
    ),
    ("human", "{content}"),
])


# ── GitHub README fetcher (via tools module) ───────────────────────────────────

def _fetch_github_readmes(keywords: List[str], repos_per_keyword: int = 5) -> List[str]:
    """
    Search GitHub for top repositories per keyword and fetch their READMEs
    directly from raw.githubusercontent.com.

    Returns a list of README text strings (non-empty only).
    Uses ThreadPoolExecutor for parallel HTTP requests.
    """
    from tools import search_github, fetch_readme

    seen_repos: set = set()
    repo_queue: List[str] = []

    for keyword in keywords:
        result = search_github(keyword, min_stars=5, max_results=repos_per_keyword)
        repos = result.get("repos", [])
        logger.info("GitHub search '%s': %d repos found.", keyword, len(repos))
        for repo in repos:
            full_name = repo.get("full_name", "")
            if full_name and full_name not in seen_repos:
                seen_repos.add(full_name)
                repo_queue.append(full_name)

    if not repo_queue:
        logger.warning("No GitHub repos found for keywords: %s", keywords)
        return []

    readmes: List[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_readme, name): name for name in repo_queue}
        for future in as_completed(futures):
            result = future.result()
            if result.get("found") and result.get("content"):
                readmes.append(result["content"])
                logger.debug("README fetched: %s (%d chars)", futures[future], len(result["content"]))

    logger.info("Fetched %d READMEs from %d repos.", len(readmes), len(repo_queue))
    return readmes


# ── Web article fetcher ────────────────────────────────────────────────────────

def _fetch_web_docs(keywords: List[str], llm, text_splitter) -> List[str]:
    """
    Search Tavily for web pages related to each keyword, load and summarise
    them.  Returns a list of summary strings.
    """
    tavily_client = get_tavily_client()
    summarise_chain = _WEB_SUMMARISE_PROMPT | llm

    urls_seen: set = set()
    summaries: List[str] = []

    for keyword in keywords[:3]:
        try:
            response = tavily_client.search(
                f"product use case or research about: {keyword}",
                search_depth="advanced",
                topic="general",
                max_results=2,
            )
        except Exception as exc:
            logger.warning("Tavily search failed for keyword=%r: %s", keyword, exc)
            continue

        for result in response.get("results", []):
            url = result.get("url", "")
            if not url or url in urls_seen:
                continue
            urls_seen.add(url)

            try:
                loader = WebBaseLoader(web_paths=(url,))
                docs = loader.load()
            except Exception as exc:
                logger.debug("WebBaseLoader failed for %s: %s", url, exc)
                continue

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
                            result_text = summarise_chain.invoke({"content": chunk.page_content})
                            summaries.append(result_text.content)
                    else:
                        result_text = summarise_chain.invoke({"content": content})
                        summaries.append(result_text.content)
                except Exception as exc:
                    logger.debug("Summarisation failed for %s: %s", url, exc)

    return summaries


# ── Main function (sync) ───────────────────────────────────────────────────────

def make_db(idea: str, keywords: List[str]):
    """
    Build a FAISS knowledge base from GitHub READMEs and web articles.

    Parameters
    ----------
    idea     : the product idea string (provides business context)
    keywords : list of 1-5 GitHub/Tavily search keywords

    Returns
    -------
    FAISS vector store ready for similarity retrieval
    """
    logger.info("Building knowledge DB for keywords: %s", keywords)
    llm = get_llm()
    text_splitter = get_text_splitter()

    # ── Phase 1: Fetch GitHub READMEs directly ────────────────────────────
    readme_texts = _fetch_github_readmes(keywords, repos_per_keyword=5)
    logger.info("GitHub phase complete: %d READMEs collected.", len(readme_texts))

    # ── Phase 2: Fetch & summarise web articles ───────────────────────────
    web_summaries = _fetch_web_docs(keywords, llm, text_splitter)
    logger.info("Web phase complete: %d article summaries collected.", len(web_summaries))

    all_texts = readme_texts + web_summaries
    logger.info("Total raw texts: %d", len(all_texts))

    # ── Phase 3: LLM-summarise all texts for the vector store ────────────
    repo_chain = _REPO_SUMMARISE_PROMPT | llm

    random.shuffle(all_texts)
    final_docs: List[Document] = []
    processed = 0
    max_docs = 20  # cap to keep latency reasonable

    for text in all_texts:
        if processed >= max_docs:
            break
        if not text.strip():
            continue
        deadline = time.time() + 30
        try:
            if len(text) >= 128_000:
                chunks = text_splitter.create_documents([text])
                for chunk in chunks:
                    if time.time() > deadline:
                        break
                    summary = repo_chain.invoke({"content": chunk.page_content})
                    final_docs.append(Document(page_content=summary.content))
                    processed += 1
            else:
                summary = repo_chain.invoke({"content": text})
                final_docs.append(Document(page_content=summary.content))
                processed += 1
        except Exception as exc:
            logger.warning("Summarisation error: %s", exc)

    if not final_docs:
        logger.warning("No documents summarised — using idea text as fallback.")
        final_docs = [Document(page_content=idea)]

    # ── Phase 4: Build FAISS index ────────────────────────────────────────
    logger.info("Building FAISS index from %d documents …", len(final_docs))
    encoder = get_hf_embeddings()
    db = FAISS.from_documents(documents=final_docs, embedding=encoder)
    logger.info("FAISS index built successfully.")
    return db
