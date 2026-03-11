# TAS AutoBD — Automated Business Development

> **Turn a company name into a board-ready proposal email in under 4 minutes.**

TAS AutoBD is an agentic AI system that automates the most time-consuming parts of B2B business development: market research, competitive analysis, solution ideation, and proposal writing. What normally takes a BD team 2–3 days is reduced to a single click.

---

## Why AutoBD?

| The Old Way | With AutoBD |
|---|---|
| 2–3 days of manual research | ~4 minutes end-to-end |
| Generic templates | Hyper-personalised proposals |
| Junior BD team hours | One senior review + send |
| Inconsistent quality | Consistent, structured output |
| $500+ per proposal | ~$0.30 in API costs |

---

## How It Works

```
  Company Name
       │
       ▼
┌─────────────────────────────┐
│  Step 1 · RESEARCH AGENT    │  Web crawl → 7 parallel Tavily queries
│  ~45 seconds                │  → Structured company intelligence
└──────────────┬──────────────┘
               │ Company profile + emails
               ▼
┌─────────────────────────────┐
│  Step 2 · IDEATION AGENT    │  Claude analyzes pain points
│  ~15 seconds                │  → ONE focused AI/software solution
└──────────────┬──────────────┘
               │ Product idea + 3 keywords
               ▼
┌─────────────────────────────┐
│  Step 3 · KNOWLEDGE AGENT   │  GitHub repos + web articles → FAISS
│  ~90 seconds                │  → RAG vector store
└──────────────┬──────────────┘
               │ Evidence base
               ▼
┌─────────────────────────────┐
│  Step 4 · PROPOSAL AGENT    │  RAG retrieval → structured proposal
│  ~30 seconds                │  → Polished HTML email
└──────────────┬──────────────┘
               │
               ▼
     Ready-to-send proposal email
     (edit, download, or send via SendGrid)
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Primary LLM** | Claude claude-sonnet-4-6 (Anthropic) | Reasoning, extraction, proposal writing |
| **Alt LLM** | GPT-4o (OpenAI) | Drop-in alternative |
| **Embeddings** | BAAI/bge-small-en-v1.5 (HuggingFace) | Free, SOTA open-source embeddings |
| **Vector Store** | FAISS | Fast similarity search for RAG |
| **Web Search** | Tavily Search API | Real-time web research & crawling |
| **Web Scraping** | LangChain WebBaseLoader | HTML → structured text |
| **GitHub** | GitHub REST API v3 | Open-source repo discovery |
| **Framework** | LangChain 0.3+ | Agent orchestration & RAG pipelines |
| **UI** | Streamlit | Interactive web application |
| **Email** | SendGrid | Automated email delivery |
| **Async** | asyncio + aiohttp | Concurrent processing |
| **Data** | Pydantic v2 | Structured outputs & validation |

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/TASDesignGroup/TAS_AutoBD.git
cd TAS_AutoBD
pip install -r requirements.txt
```

> **Python 3.10+ required.** Use a virtual environment for best results.

### 2. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
# Primary LLM — choose one provider
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...    # get at console.anthropic.com

# Alternative: use OpenAI instead
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...          # get at platform.openai.com

# Required: web search
TAVILY_API_KEY=tvly-...          # get at app.tavily.com (free tier available)

# Optional: automated email delivery
SENDGRID_API_KEY=SG....
SENDER_EMAIL=outreach@yourcompany.com

# Optional: higher GitHub rate limits (60 → 5,000 req/hr)
GITHUB_TOKEN=ghp_...
```

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
TAS_AutoBD/
├── app.py              # Streamlit UI — 4-step wizard
├── config.py           # Centralized config & lazy LLM/embedding factories
├── get_info.py         # Step 1: Company Research Agent
├── get_hypo.py         # Step 2: Solution Ideation Agent
├── make_db.py          # Step 3: Knowledge Base Builder (GitHub + web → FAISS)
├── get_proposal.py     # Step 4: Proposal & Email Generation Agent
├── email_utils.py      # SendGrid email sending + UI widget
├── utils.py            # Shared utilities (async, ZIP, text cleaning)
├── AutoBD_Demo.ipynb   # Interactive Jupyter tutorial
├── requirements.txt    # Python dependencies
└── .env.example        # Configuration template
```

---

## Agent Details

### Step 1 — Research Agent (`get_info.py`)
- Fires 7 parallel Tavily search queries (overview, strategy, challenges, contacts, technology, competitors, growth)
- Processes up to 14 web pages concurrently with 8 workers (20s timeout per page)
- Uses Claude's native structured output (`Company` Pydantic schema) for reliable extraction
- Outputs: structured company profile + discovered email addresses

### Step 2 — Ideation Agent (`get_hypo.py`)
- Applies a 5-dimension evaluation: Business Impact, Technical Feasibility, UX, Ethics, Sustainability
- Recommends exactly ONE focused solution TAS Design Group can build in 3–6 months
- Extracts 3 GitHub-searchable keywords for knowledge base construction
- Outputs: solution description + keyword list

### Step 3 — Knowledge Agent (`make_db.py`)
- Searches GitHub for relevant open-source repos (2 date filters, stars > 5)
- Automatically tries both `main` and `master` branch ZIPs
- Fetches and summarises web articles via Tavily for each keyword
- Processes up to 20 documents with business-focused summaries
- Builds FAISS vector index with BAAI/bge-small-en-v1.5 embeddings (free, no extra API key)
- Outputs: FAISS vector store ready for RAG

### Step 4 — Proposal Agent (`get_proposal.py`)
- Retrieves top-5 semantically relevant documents from FAISS
- Generates structured `Proposal` Pydantic object: name, tagline, features with impact, success metrics, competitive advantage
- Drafts professional HTML email with inline CSS, TAS purple branding, mobile-responsive layout
- Outputs: complete HTML email ready to review, download, or send

---

## API Requirements & Costs

| Service | Required | Free Tier | Est. Cost / Proposal |
|---|---|---|---|
| Anthropic (Claude) | ✅ Yes* | — | ~$0.20–0.40 |
| OpenAI (GPT-4o) | ✅ Yes* | — | ~$0.20–0.50 |
| Tavily Search | ✅ Yes | 1,000 searches/month | ~$0.02 |
| SendGrid | ❌ Optional | 100 emails/day | Free |
| GitHub Token | ❌ Optional | 60 req/hr (anonymous) | Free |

*One LLM provider is required. Anthropic Claude is recommended (better instruction-following, more reliable structured output).

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `LLM_MODEL` | `claude-sonnet-4-6` | Model ID for chosen provider |
| `LLM_TEMPERATURE` | `0.3` | Lower = more deterministic and consistent |
| `HF_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Free HuggingFace embedding model |
| `DATA_DIR` | `./data` | Working directory for repos and indexes |

### Model Options

```env
# Anthropic (speed vs. capability tradeoff)
LLM_MODEL=claude-haiku-4-5-20251001   # Fastest, cheapest
LLM_MODEL=claude-sonnet-4-6            # Recommended balance (default)
LLM_MODEL=claude-opus-4-6              # Most capable, slower

# OpenAI (requires LLM_PROVIDER=openai)
LLM_MODEL=gpt-4o-mini                  # Cost-efficient
LLM_MODEL=gpt-4o                       # High capability
```

---

## Jupyter Notebook

For step-by-step exploration, batch processing, or integration into your pipeline:

```bash
jupyter notebook AutoBD_Demo.ipynb
```

The notebook covers installation, running each agent independently, customising prompts, batch processing multiple companies, and inspecting the FAISS vector store.

---

## Batch Processing

```python
from get_info import get_company_information
from get_hypo import get_hypothesis_idea
from make_db import make_db
from get_proposal import make_proposal
from utils import run_async

companies = ["Toyota", "Sony", "Panasonic"]

for company in companies:
    profile, emails = get_company_information(company)
    idea, keywords = get_hypothesis_idea(profile)
    db = run_async(make_db(idea, keywords))
    html = make_proposal(idea, db, company)
    with open(f"proposal_{company}.html", "w") as f:
        f.write(html)
    print(f"Generated proposal for {company} — found emails: {emails}")
```

---

## Extending AutoBD

**Add a new LLM provider:**
1. Install: `pip install langchain-<provider>`
2. Add a factory in `config.py` → `get_llm()`
3. Set `LLM_PROVIDER=<provider>` in `.env`

**Customise prompts:**
All prompts are `ChatPromptTemplate` objects at the top of each agent file. Edit system/human messages to tailor tone, focus, or industry vertical.

**Increase knowledge base size:**
Change `max_docs = 20` in `make_db.py` to include more documents in the FAISS index (trades latency for coverage).

---

## About TAS Design Group

TAS Design Group Inc. is an IT consulting and data science company with operations in **Japan** and **Vietnam**. We specialise in AI/ML solutions, data engineering, and custom software development for enterprise clients across manufacturing, retail, finance, and logistics.

AutoBD is our internal tool — now open-sourced so others can benefit from automated, intelligent business development.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
