"""
TAS AutoBD — Utility Functions
================================
Shared helpers used across the pipeline modules.
"""

import os
import re
import shutil
import zipfile
import asyncio
import logging
import requests
from typing import List, Tuple

logger = logging.getLogger(__name__)


# ── Text Utilities ────────────────────────────────────────────────────────────

def remove_first_last_line(text: str) -> str:
    """Strip the first and last lines from *text* (used to clean LLM markdown fences)."""
    lines = text.splitlines()
    return "\n".join(lines[1:-1]) if len(lines) > 2 else text


def format_docs(docs) -> str:
    """Concatenate a list of LangChain Documents into one string."""
    return "\n\n".join(doc.page_content for doc in docs)


def clean_html_fences(text: str) -> str:
    """Remove ```html ... ``` fences that LLMs sometimes wrap HTML output in."""
    text = re.sub(r"^```html\s*", "", text.strip())
    text = re.sub(r"```\s*$", "", text.strip())
    return text.strip()


# ── HTTP Utilities ────────────────────────────────────────────────────────────

def getUrl(url: str, headers: dict = None) -> dict:
    """GET *url* and return the parsed JSON body. Raises on HTTP errors."""
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.json()


# ── Async Utilities ───────────────────────────────────────────────────────────

def run_async(coro):
    """
    Run an async coroutine from a synchronous context (e.g., Streamlit).
    Compatible with environments where an event loop may already exist.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an already-running loop (e.g., Jupyter / Streamlit)
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No current event loop – create a fresh one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# ── File / ZIP Utilities ──────────────────────────────────────────────────────

def process_zip_files_to_faiss(folder_path: str) -> Tuple[List[str], List[str]]:
    """
    Iterate over every *.zip* in *folder_path*, extract README.md files,
    and return their text content together with their paths.

    Returns
    -------
    all_texts  : list of README.md text strings
    file_paths : list of original README paths inside the zip
    """
    all_texts: List[str] = []
    file_paths: List[str] = []

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".zip"):
            continue

        zip_file_path = os.path.join(folder_path, file_name)
        logger.info("Processing zip: %s", zip_file_path)

        temp_dir = os.path.join(folder_path, file_name.replace(".zip", "_tmp"))
        try:
            os.makedirs(temp_dir, exist_ok=True)
            with zipfile.ZipFile(zip_file_path, "r") as zf:
                zf.extractall(temp_dir)

            # Walk the extracted tree and collect README.md files
            for root, _, files in os.walk(temp_dir):
                for fname in files:
                    if fname.lower() in ("readme.md", "readme.rst", "readme.txt"):
                        readme_path = os.path.join(root, fname)
                        try:
                            with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read().strip()
                            if content:
                                all_texts.append(content)
                                file_paths.append(readme_path)
                                logger.debug("Collected README from: %s", readme_path)
                        except OSError as exc:
                            logger.warning("Could not read %s: %s", readme_path, exc)

        except zipfile.BadZipFile:
            logger.warning("Skipping invalid zip: %s", zip_file_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error processing %s: %s", zip_file_path, exc)
        finally:
            # Always clean up the temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("Collected %d README texts from zips in %s", len(all_texts), folder_path)
    return all_texts, file_paths
