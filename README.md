# 🚀 TAS AutoBD — Agentic LLM Business Development

> **Automate your B2B outreach with state-of-the-art AI agents**
> © 2024 TAS Design Group Inc. — Japan & Vietnam

TAS AutoBD is a multi-agent LLM pipeline that transforms a company name into a polished, personalised business proposal email — in minutes, not days.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  TAS AutoBD Pipeline                     │
└──────────────────────────────────────────────────────────┘
          │                 │                  │
   ┌──────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
   │ Info Agent  │  │  Hypothesis  │  │  DB Builder  │
   │             │  │    Agent     │  │              │
   │ Tavily API  │→ │   GPT-4o     │→ │ GitHub API   │
   │ Web Crawl   │  │   LLM        │  │ Tavily API   │
   │ Concurrent  │  │              │  │ FAISS Index  │
   └─────────────┘  └──────────────┘  └──────┬───────┘
                                             │
                                     ┌───────▼───────┐
                                     │ Proposal Agent │
                                     │               │
                                     │ RAG Retrieval │
                                     │ + GPT-4o LLM  │
                                     │ → HTML Email  │
                                     └───────────────┘
```

| Agent | Role | Technologies |
|-------|------|-------------|
| **Info Agent** | Research target company from public web | Tavily, LangChain, GPT-4o, Concurrent processing |
| **Hypothesis Agent** | Propose tailored IT/AI solution | GPT-4o, LangChain PromptTemplates |
| **DB Builder** | Build knowledge base from GitHub + web | GitHub API, Tavily, FAISS, HuggingFace embeddings |
| **Proposal Agent** | Generate formal proposal email | RAG, FAISS retrieval, GPT-4o, Pydantic |

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd TAS_AutoBD
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required keys:

| Key | Where to get it |
|-----|----------------|
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| `TAVILY_API_KEY` | https://app.tavily.com |
| `SENDGRID_API_KEY` | https://sendgrid.com (optional, for email sending) |

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 4. Or run the Jupyter notebook tutorial

```bash
jupyter notebook AutoBD_Demo.ipynb
```

---

## The 4-Step Wizard

```
Step 1 → Enter company name
          Tavily crawls 7+ search queries concurrently
          LLM extracts structured profile

Step 2 → Review profile (editable)
          GPT-4o generates product hypothesis
          Extracts 3 GitHub search keywords

Step 3 → Review idea (editable)
          Downloads GitHub repos matching keywords
          Fetches Tavily articles
          Builds FAISS vector store
          RAG + GPT-4o generates HTML email

Step 4 → Preview & edit email
          Download as HTML file
          Send via SendGrid
```

---

## Project Structure

```
TAS_AutoBD/
├── app.py               # Streamlit web application
├── config.py            # Central configuration (env vars, lazy init)
├── get_info.py          # Company info research agent
├── get_hypo.py          # Product hypothesis agent
├── make_db.py           # Knowledge base builder
├── get_proposal.py      # Proposal & email generation agent
├── email_utils.py       # Email sending + UI widget
├── utils.py             # Shared utilities
├── AutoBD_Demo.ipynb    # Full tutorial notebook
├── requirements.txt     # Python dependencies
├── .env.example         # Configuration template
└── data/                # Working directory (auto-created)
    ├── repos/           # Downloaded GitHub ZIP archives
    └── *.html           # Generated proposal files
```

---

## Configuration

All settings are loaded from environment variables via a `.env` file:

```env
# Required
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

# For email sending
SENDGRID_API_KEY=SG....
SENDER_EMAIL=you@company.com

# Optional overrides
LLM_MODEL=gpt-4o              # default
LLM_TEMPERATURE=0.7           # default
GITHUB_TOKEN=ghp_...          # raises rate limit to 5000 req/hr
DATA_DIR=./data               # working directory
```

---

## Key Features

- **Zero-hardcoded credentials** — all API keys from `.env`
- **Concurrent web crawling** — 8 parallel threads for fast research
- **Async GitHub download** — concurrent repo fetching with aiohttp
- **RAG-powered proposals** — FAISS + HuggingFace embeddings
- **Editable at every step** — full human-in-the-loop control
- **HTML preview + download** — review before sending
- **Graceful error handling** — each step reports issues clearly
- **Streamlit + Jupyter** — works both as a web app and notebook

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-4o via LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| Web Search | Tavily Search API |
| Web Crawl | LangChain WebBaseLoader |
| GitHub | GitHub REST API v3 |
| UI | Streamlit |
| Email | SendGrid |
| Async | asyncio + aiohttp + nest-asyncio |

---

## License

Proprietary — © 2024 TAS Design Group Inc. All rights reserved.
