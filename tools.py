"""
TAS AutoBD — Tool Definitions & Registry
==========================================
All tool schemas (Anthropic native tool-use format) and their Python
implementations live here.  The agent_runner imports this module to
wire up the ReAct loop.

Available tools
---------------
  web_search     : Tavily web search → titles + snippets + URLs
  fetch_webpage  : Load full text of a single URL
  extract_emails : Regex-extract email addresses from arbitrary text
  search_github  : Query the GitHub repositories search API
  fetch_readme   : Download a repo's README.md from raw.githubusercontent.com
"""

import json
import re
import logging
import requests
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ── Anthropic-compatible Tool Schemas ─────────────────────────────────────────

TOOL_SCHEMAS: List[Dict] = [
    {
        "name": "web_search",
        "description": (
            "Search the web for information about a company, industry, or technology topic. "
            "Returns titles, URLs, and content snippets. Use this to find company overviews, "
            "recent news, contact pages, technology stack details, and financial signals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific search query. Include company name for company research.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return, between 1 and 5. Default: 3.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_webpage",
        "description": (
            "Fetch and extract the full text content of a specific URL. "
            "Use this to read company websites, press releases, contact pages, or news articles "
            "when you need more detail than the search snippet provides."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to fetch (must begin with http:// or https://).",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "extract_emails",
        "description": (
            "Scan a block of text and return all email addresses found in it. "
            "Use this on any fetched webpage content that is likely to contain "
            "contact or staff email addresses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to scan for email addresses.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "search_github",
        "description": (
            "Search GitHub for open-source repositories related to a keyword or technology. "
            "Returns repository names, descriptions, star counts, primary language, and URLs. "
            "Use this to find technical references and evidence for proposals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Search keyword (lowercase, underscores for multi-word terms).",
                },
                "min_stars": {
                    "type": "integer",
                    "description": "Minimum GitHub star count. Default: 10.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Repositories to return, between 1 and 10. Default: 5.",
                },
            },
            "required": ["keyword"],
        },
    },
    {
        "name": "fetch_readme",
        "description": (
            "Fetch the README file of a GitHub repository directly from raw.githubusercontent.com. "
            "Use this after search_github to read the detailed documentation of a repository."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo_full_name": {
                    "type": "string",
                    "description": "Full repository identifier in 'owner/repo' format, e.g. 'microsoft/autogen'.",
                },
            },
            "required": ["repo_full_name"],
        },
    },
]

# Convenience subsets used by different agents
RESEARCH_TOOL_SCHEMAS: List[Dict] = [
    s for s in TOOL_SCHEMAS if s["name"] in {"web_search", "fetch_webpage", "extract_emails"}
]

KNOWLEDGE_TOOL_SCHEMAS: List[Dict] = [
    s for s in TOOL_SCHEMAS if s["name"] in {"search_github", "fetch_readme", "web_search"}
]


# ── Tool Implementations ───────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """Search the web using the Tavily API."""
    from config import get_tavily_client

    try:
        client = get_tavily_client()
        response = client.search(
            query,
            search_depth="advanced",
            topic="general",
            max_results=min(int(max_results), 5),
        )
        results = [
            {
                "title": r.get("title", ""),
                "url": r["url"],
                "snippet": r.get("content", "")[:600],
            }
            for r in response.get("results", [])
        ]
        return {"results": results, "count": len(results)}
    except Exception as exc:
        logger.warning("web_search failed for %r: %s", query, exc)
        return {"results": [], "count": 0, "error": str(exc)}


def fetch_webpage(url: str) -> Dict[str, Any]:
    """Fetch and return text content from a URL."""
    from langchain_community.document_loaders import WebBaseLoader

    if not url.startswith(("http://", "https://")):
        return {"url": url, "content": "", "error": "URL must start with http:// or https://"}

    try:
        loader = WebBaseLoader(web_paths=(url,))
        docs = loader.load()
        content = "\n\n".join(
            doc.page_content.strip() for doc in docs if doc.page_content.strip()
        )
        return {"url": url, "content": content[:8_000], "total_length": len(content)}
    except Exception as exc:
        logger.debug("fetch_webpage failed for %s: %s", url, exc)
        return {"url": url, "content": "", "error": str(exc)}


def extract_emails(text: str) -> Dict[str, Any]:
    """Extract email addresses from text using regex."""
    pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    raw = re.findall(pattern, text)
    # Filter obvious false positives (too short or missing a dot in domain)
    emails = list(
        {e for e in raw if len(e) >= 6 and "." in e.split("@")[-1]}
    )
    return {"emails": emails, "count": len(emails)}


def search_github(keyword: str, min_stars: int = 10, max_results: int = 5) -> Dict[str, Any]:
    """Search GitHub for repositories matching a keyword."""
    from config import GITHUB_HEADERS

    url = (
        f"https://api.github.com/search/repositories"
        f"?q={keyword}+stars:>{int(min_stars)}"
        f"&sort=stars&order=desc&per_page={min(int(max_results), 10)}"
    )
    try:
        resp = requests.get(url, headers=GITHUB_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        repos = [
            {
                "full_name": r["full_name"],
                "description": (r.get("description") or "")[:300],
                "stars": r.get("stargazers_count", 0),
                "language": r.get("language") or "",
                "updated_at": (r.get("updated_at") or "")[:10],
                "url": r.get("html_url", ""),
            }
            for r in data.get("items", [])
        ]
        return {"repos": repos, "count": len(repos)}
    except Exception as exc:
        logger.warning("search_github failed for %r: %s", keyword, exc)
        return {"repos": [], "count": 0, "error": str(exc)}


def fetch_readme(repo_full_name: str) -> Dict[str, Any]:
    """Fetch a GitHub repository's README.md directly from raw content."""
    for branch in ("main", "master", "HEAD"):
        url = f"https://raw.githubusercontent.com/{repo_full_name}/{branch}/README.md"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and resp.text.strip():
                return {
                    "repo": repo_full_name,
                    "content": resp.text[:10_000],
                    "length": len(resp.text),
                    "found": True,
                }
        except Exception:
            pass
    return {"repo": repo_full_name, "content": "", "found": False}


# ── Tool Registry & Dispatcher ────────────────────────────────────────────────

TOOL_REGISTRY: Dict[str, Any] = {
    "web_search": web_search,
    "fetch_webpage": fetch_webpage,
    "extract_emails": extract_emails,
    "search_github": search_github,
    "fetch_readme": fetch_readme,
}


def execute_tool(name: str, inputs: Dict[str, Any]) -> str:
    """
    Execute a registered tool by name and return a JSON-encoded result string.

    Parameters
    ----------
    name   : tool name (must be a key in TOOL_REGISTRY)
    inputs : keyword arguments to pass to the tool function

    Returns
    -------
    JSON string — always returns a valid JSON object even on error
    """
    if name not in TOOL_REGISTRY:
        return json.dumps(
            {"error": f"Unknown tool '{name}'. Available: {sorted(TOOL_REGISTRY)}"}
        )
    try:
        result = TOOL_REGISTRY[name](**inputs)
        return json.dumps(result, ensure_ascii=False, default=str)
    except TypeError as exc:
        logger.error("Tool %s bad arguments %s: %s", name, inputs, exc)
        return json.dumps({"error": f"Invalid arguments for '{name}': {exc}"})
    except Exception as exc:
        logger.error("Tool %s execution error: %s", name, exc)
        return json.dumps({"error": str(exc)})
