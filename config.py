"""
TAS AutoBD — Central Configuration
====================================
All API keys, paths, and model settings are loaded from environment
variables (via .env file). No credentials are ever hardcoded.

Supported LLM providers:
  - "anthropic" (default) → Claude claude-sonnet-4-6  (SOTA reasoning)
  - "openai"              → GPT-4o or any OpenAI model
Set LLM_PROVIDER in your .env to switch providers.
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from the project root
load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)

# ── API Keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
SENDGRID_API_KEY: str = os.getenv("SENDGRID_API_KEY", "")
SENDER_EMAIL: str = os.getenv("SENDER_EMAIL", "noreply@tasdesigngroup.com")
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR: str = os.getenv("DATA_DIR", str(Path(__file__).parent / "data"))
FAISS_INDEX_PATH: str = os.path.join(DATA_DIR, "faiss_index")
GITHUB_DOWNLOAD_DIR: str = os.path.join(DATA_DIR, "repos")

# Ensure working directories exist at import time
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(GITHUB_DOWNLOAD_DIR, exist_ok=True)

# ── Model Configuration ───────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" or "openai"
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
HF_EMBEDDING_MODEL: str = os.getenv(
    "HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
)

# ── GitHub API ────────────────────────────────────────────────────────────────
GITHUB_API_BASE = "https://api.github.com"
GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if GITHUB_TOKEN:
    GITHUB_HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


# ── Lazy Factory Functions ────────────────────────────────────────────────────
def get_llm():
    """
    Return an LLM instance based on LLM_PROVIDER setting.

    Default provider is Anthropic (Claude claude-sonnet-4-6).
    Set LLM_PROVIDER=openai in .env to use OpenAI models instead.
    """
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        if not OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Please add it to your .env file."
            )
        return ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=OPENAI_API_KEY,
        )
    else:
        from langchain_anthropic import ChatAnthropic

        if not ANTHROPIC_API_KEY:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Please add it to your .env file."
            )
        return ChatAnthropic(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=ANTHROPIC_API_KEY,
        )


def get_openai_embeddings():
    """Return OpenAI embeddings (higher quality, requires OpenAI API key)."""
    from langchain_openai import OpenAIEmbeddings

    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)


def get_hf_embeddings():
    """Return HuggingFace BGE embeddings (free, no API key required)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_text_splitter():
    """Return a RecursiveCharacterTextSplitter with tiktoken length function."""
    import tiktoken
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    tokenizer = tiktoken.get_encoding("cl100k_base")

    def _tiktoken_len(text: str) -> int:
        return len(tokenizer.encode(text, disallowed_special=()))

    return RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        length_function=_tiktoken_len,
        separators=["\n\n", "\n", " ", ""],
    )


def get_semantic_splitter():
    """Return a SemanticChunker (requires OpenAI embeddings)."""
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        raise ImportError("Install langchain-experimental: pip install langchain-experimental")

    return SemanticChunker(
        get_openai_embeddings(), breakpoint_threshold_type="percentile"
    )


def get_tavily_client():
    """Return a configured TavilyClient."""
    from tavily import TavilyClient

    if not TAVILY_API_KEY:
        raise EnvironmentError(
            "TAVILY_API_KEY is not set. Please add it to your .env file."
        )
    return TavilyClient(api_key=TAVILY_API_KEY)


# ── Config Validation ─────────────────────────────────────────────────────────
def validate_config() -> dict:
    """Return a dict indicating which services are properly configured."""
    _placeholders = {
        "your_anthropic_api_key_here",
        "your_openai_api_key_here",
        "your_tavily_api_key_here",
        "your_sendgrid_api_key_here",
        "your_github_token_here",
        "",
    }

    def _is_set(val: str) -> bool:
        return bool(val and val not in _placeholders)

    llm_ready = (
        _is_set(ANTHROPIC_API_KEY) if LLM_PROVIDER == "anthropic"
        else _is_set(OPENAI_API_KEY)
    )

    return {
        "llm": llm_ready,
        "anthropic": _is_set(ANTHROPIC_API_KEY),
        "openai": _is_set(OPENAI_API_KEY),
        "tavily": _is_set(TAVILY_API_KEY),
        "sendgrid": _is_set(SENDGRID_API_KEY),
        "github": _is_set(GITHUB_TOKEN),
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
    }
