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
import math
import time
import random
import logging
from typing import List

import aiohttp
import asyncio

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader

from config import (
    get_llm,
    get_hf_embeddings,
    get_text_splitter,
    get_semantic_splitter,
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
    date_filters = ["created:<=2024-03-31", "created:>=2022-01-01"]

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

            zip_url = clone_url.replace(".git", "/archive/refs/heads/master.zip")
            file_name = repo_name.replace("/", "#") + ".zip"
            dest_path = os.path.join(output_folder, file_name)

            tasks.append(_download_zip(session, zip_url, dest_path))
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

def _fetch_web_docs(keywords: List[str], llm, text_splitter, semantic_splitter) -> List[str]:
    """
    Search Tavily for web pages related to each keyword, load and
    summarise them. Returns a list of summary strings.
    """
    tavily_client = get_tavily_client()
    summarise_prompt = PromptTemplate(
        template=(
            "You are a Senior Business Developer at TAS Design Group Inc.\n"
            "Summarise the key ideas from this web article for later use in a "
            "business proposal. Focus on:\n"
            "- The core problem solved\n"
            "- The technology or approach used\n"
            "- Measurable benefits or outcomes\n"
            "Translate everything into English.\n\n"
            "{query}"
        ),
        input_variables=["query"],
    )
    summarise_chain = summarise_prompt | llm

    urls_seen = set()
    summaries: List[str] = []

    for keyword in keywords[:3]:
        try:
            response = tavily_client.search(
                f"product or research about: {keyword}",
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
                        try:
                            chunks = semantic_splitter.create_documents([content])
                        except Exception:
                            chunks = text_splitter.create_documents([content])
                        for chunk in chunks:
                            if time.time() > deadline:
                                break
                            result_text = summarise_chain.invoke(chunk.page_content)
                            summaries.append(result_text.content)
                    else:
                        result_text = summarise_chain.invoke(content)
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
    try:
        semantic_splitter = get_semantic_splitter()
    except Exception:
        logger.warning("SemanticChunker unavailable — using basic splitter")
        semantic_splitter = text_splitter

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
    web_summaries = _fetch_web_docs(keywords, llm, text_splitter, semantic_splitter)
    all_texts.extend(web_summaries)
    logger.info("Total raw texts (READMEs + web): %d", len(all_texts))

    # ── Phase 4: Summarise all texts for the vector store ────────────────────
    repo_summarise_prompt = PromptTemplate(
        template=(
            "You are a Business Developer at TAS Design Group Inc. analysing "
            "open-source repositories and articles to support a business proposal.\n\n"
            "Extract and summarise the key information with a human-impact focus:\n"
            "1. Core problem the project solves and its solution approach.\n"
            "2. Ethical technologies used (privacy, security, transparency).\n"
            "3. Accessibility and inclusivity features.\n"
            "4. Long-term societal or environmental impact.\n"
            "5. Concrete metrics or outcomes if mentioned.\n"
            "Do NOT include source code. Keep output concise and business-relevant.\n\n"
            "{query}"
        ),
        input_variables=["query"],
    )
    repo_chain = repo_summarise_prompt | llm

    random.shuffle(all_texts)
    final_docs: List[Document] = []
    processed = 0
    max_docs = 15  # cap to keep latency reasonable

    for text in all_texts:
        if processed >= max_docs:
            break
        if not text.strip():
            continue

        deadline = time.time() + 30
        try:
            if len(text) >= 128_000:
                try:
                    chunks = semantic_splitter.create_documents([text])
                except Exception:
                    chunks = text_splitter.create_documents([text])
                for chunk in chunks:
                    if time.time() > deadline:
                        break
                    summary = repo_chain.invoke(chunk.page_content)
                    final_docs.append(Document(page_content=summary.content))
                    processed += 1
            else:
                summary = repo_chain.invoke(text)
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
