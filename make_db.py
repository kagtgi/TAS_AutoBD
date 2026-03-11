"""
TAS AutoBD — Knowledge Database Builder
=========================================
Downloads relevant GitHub repositories and web articles based on
the generated keywords, then builds a FAISS vector store for RAG.

Pipeline:
  1. Search GitHub for repositories matching each keyword.
  2. Download ZIP archives and extract README files.
  3. Search Tavily for web articles about the keywords.
  4. Summarise all collected documents with an LLM.
  5. Embed summaries and store them in a FAISS index.
"""

import os
import csv
import time
import random
import logging
from typing import List

import aiohttp
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

from config import (
    get_llm,
    get_hf_embeddings,
    get_text_splitter,
    get_tavily_client,
    GITHUB_HEADERS,
    GITHUB_DOWNLOAD_DIR,
    DATA_DIR,
)
from utils import process_zip_files_to_faiss

logger = logging.getLogger(__name__)

# ── GitHub helpers ────────────────────────────────────────────────────────────

async def _fetch_json(session: aiohttp.ClientSession, url: str) -> dict:
    """GET *url* and return parsed JSON. Returns {} on error."""
    try:
        async with session.get(url, headers=GITHUB_HEADERS, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status == 200:
                return await resp.json()
            logger.warning("GitHub API returned %s for %s", resp.status, url)
            return {}
    except Exception as exc:
        logger.warning("HTTP error fetching %s: %s", url, exc)
        return {}


async def _download_zip(session: aiohttp.ClientSession, url: str, dest: str) -> bool:
    """Download a ZIP file from *url* to *dest*. Returns True on success."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status == 200:
                with open(dest, "wb") as f:
                    f.write(await resp.read())
                return True
            return False
    except Exception as exc:
        logger.debug("Download failed %s: %s", url, exc)
        return False


async def _download_repo_zip(
    session: aiohttp.ClientSession, repo_name: str, clone_url: str, dest_path: str
) -> bool:
    """
    Try downloading the repo ZIP from both 'main' and 'master' branches.
    Returns True if either succeeds.
    """
    base_url = clone_url.replace(".git", "")
    for branch in ("main", "master"):
        zip_url = f"{base_url}/archive/refs/heads/{branch}.zip"
        if await _download_zip(session, zip_url, dest_path):
            return True
    logger.debug("Could not download zip for %s (tried main & master)", repo_name)
    return False


async def _search_github_keyword(
    session: aiohttp.ClientSession,
    keyword: str,
    output_folder: str,
    csv_writer: csv.writer,
    repos_per_keyword: int = 5,
) -> None:
    """
    Search GitHub for repositories related to *keyword*, download their ZIPs.
    Uses two date-filtered sub-queries to get both recent and established repos.
    """
    date_filters = ["created:>=2023-01-01", "created:<=2023-12-31 stars:>10"]

    for date_filter in date_filters:
        query = f"{keyword} {date_filter} stars:>5"
        url = (
            f"https://api.github.com/search/repositories"
            f"?q={query}&sort=stars&order=desc&per_page={repos_per_keyword}"
        )
        data = await _fetch_json(session, url)
        items = data.get("items", [])

        if not items:
            logger.info("No GitHub results for keyword=%r filter=%r", keyword, date_filter)
            continue

        tasks = []
        row_buffer = []
        for item in items:
            repo_name = item.get("full_name", "")
            clone_url = item.get("clone_url", "")
            if not clone_url:
                continue

            file_name = repo_name.replace("/", "#") + ".zip"
            dest_path = os.path.join(output_folder, file_name)

            tasks.append(_download_repo_zip(session, repo_name, clone_url, dest_path))
            row_buffer.append([item["owner"]["login"], item["name"], clone_url, "queued"])

        results = await asyncio.gather(*tasks)
        for row, ok in zip(row_buffer, results):
            row[-1] = "downloaded" if ok else "failed"
        csv_writer.writerows(row_buffer)

        logger.info(
            "Keyword=%r filter=%r — %d/%d repos downloaded",
            keyword, date_filter, sum(results), len(results)
        )


# ── Tavily helpers ────────────────────────────────────────────────────────────

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
    (
        "human",
        "{content}",
    ),
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
    (
        "human",
        "{content}",
    ),
])


def _fetch_web_docs(keywords: List[str], llm, text_splitter) -> List[str]:
    """
    Search Tavily for web pages related to each keyword, load and
    summarise them. Returns a list of summary strings.
    """
    tavily_client = get_tavily_client()
    summarise_chain = _WEB_SUMMARISE_PROMPT | llm

    urls_seen = set()
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


# ── Main async function ───────────────────────────────────────────────────────

async def make_db(idea: str, keywords: List[str]):
    """
    Build a FAISS knowledge base from GitHub READMEs and web articles.

    Parameters
    ----------
    idea     : the product idea string (used for context-aware summarisation)
    keywords : list of search keywords

    Returns
    -------
    FAISS vector store ready for retrieval
    """
    logger.info("Building knowledge database for keywords: %s", keywords)
    llm = get_llm()
    text_splitter = get_text_splitter()

    os.makedirs(GITHUB_DOWNLOAD_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "repositories.csv")

    # ── Phase 1: Download GitHub repos ───────────────────────────────────────
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["user", "repository", "clone_url", "status"])

        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                _search_github_keyword(
                    session, kw, GITHUB_DOWNLOAD_DIR, csv_writer, repos_per_keyword=5
                )
                for kw in keywords
            ]
            await asyncio.gather(*tasks)

    logger.info("GitHub download phase complete.")

    # ── Phase 2: Extract READMEs from ZIPs ───────────────────────────────────
    all_texts, _ = process_zip_files_to_faiss(GITHUB_DOWNLOAD_DIR)
    logger.info("Extracted %d README texts from downloaded repos.", len(all_texts))

    # ── Phase 3: Fetch & summarise web articles ───────────────────────────────
    web_summaries = _fetch_web_docs(keywords, llm, text_splitter)
    all_texts.extend(web_summaries)
    logger.info("Total raw texts (READMEs + web): %d", len(all_texts))

    # ── Phase 4: Summarise all texts for the vector store ────────────────────
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
        # Fallback: create a minimal document so FAISS doesn't fail on an empty list
        logger.warning("No documents summarised — using idea text as fallback document.")
        final_docs = [Document(page_content=idea)]

    # ── Phase 5: Build FAISS index ────────────────────────────────────────────
    logger.info("Building FAISS index from %d documents …", len(final_docs))
    encoder = get_hf_embeddings()
    db = FAISS.from_documents(documents=final_docs, embedding=encoder)
    logger.info("FAISS index built successfully.")
    return db
