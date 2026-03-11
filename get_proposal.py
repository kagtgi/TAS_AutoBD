"""
TAS AutoBD — Proposal Generation Agent
=========================================
Uses RAG (retrieval-augmented generation) over the FAISS knowledge base
to craft a structured business proposal and a polished HTML email.
"""

import logging
from typing import List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import get_llm
from utils import format_docs

logger = logging.getLogger(__name__)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class Feature(BaseModel):
    name: str = Field(description="Feature name (5 words max)")
    description: str = Field(description="Concrete description of the feature and its business benefit (2-3 sentences)")
    impact: str = Field(description="Measurable impact or outcome this feature delivers")


class Proposal(BaseModel):
    name: str = Field(description="Compelling product or solution name (3-6 words)")
    tagline: str = Field(description="One-sentence value proposition for the customer")
    reason: str = Field(description="Why this customer needs this solution right now (specific to their situation)")
    market_context: str = Field(description="Relevant market trends and industry context supporting this solution")
    stakeholder_mapping: str = Field(description="Key stakeholders who will benefit and their specific gains")
    key_features: List[Feature] = Field(description="3-5 key features tailored to the customer's needs")
    competitive_advantage: str = Field(description="Why TAS Design Group is uniquely positioned to deliver this")
    success_metrics: str = Field(description="Specific KPIs and outcomes the customer can expect in 6-12 months")


# ── Prompts ───────────────────────────────────────────────────────────────────

_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Senior Solution Architect at TAS Design Group Inc. "
        "You have retrieved relevant technical evidence and case studies from similar projects. "
        "Synthesise this context into a concise, actionable summary that includes:\n"
        "1. Key capabilities and features proven in similar implementations\n"
        "2. Architecture and technology stack highlights\n"
        "3. Specific business benefits with concrete metrics where available\n"
        "4. Implementation considerations and success factors\n\n"
        "Be specific and evidence-based. Quote metrics when available. "
        "Relate everything to the customer's specific situation.",
    ),
    (
        "human",
        "Retrieved context from similar projects:\n{context}\n\n"
        "Customer situation and proposed solution:\n{question}",
    ),
])

_PROPOSAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Senior Business Developer at TAS Design Group Inc. crafting a formal, "
        "compelling business proposal for an enterprise client.\n\n"
        "Based on the solution research and company context provided, create a comprehensive "
        "proposal structured as a Proposal JSON object. Be specific, concrete, and client-focused.\n\n"
        "Requirements:\n"
        "- The solution name must be memorable and professional\n"
        "- Every claim must be grounded in the research provided\n"
        "- Features must directly address the customer's stated pain points\n"
        "- Success metrics must be realistic and specific (e.g. '30% reduction in manual processing time')\n"
        "- The competitive advantage must highlight TAS Design Group's actual strengths: "
        "AI/ML expertise, Japan+Vietnam presence, rapid delivery methodology",
    ),
    (
        "human",
        "Solution research and evidence:\n{research}\n\n"
        "Target company and proposed idea:\n{idea}",
    ),
])

_EMAIL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Business Developer at TAS Design Group Inc. writing a high-converting "
        "business proposal email. Your goal is to secure a discovery call.\n\n"
        "Email requirements:\n"
        "- Subject line must create urgency and curiosity (include in email body as <h2>)\n"
        "- Opening: personalised hook referencing a specific challenge the company faces\n"
        "- Value proposition: state the transformation, not just the features\n"
        "- Solution section: describe 3-5 key features with their SPECIFIC BUSINESS IMPACT\n"
        "- Social proof: briefly mention TAS Design Group's track record and expertise\n"
        "- Call to action: specific ask (e.g. '30-minute discovery call') with urgency\n"
        "- Closing: professional signature with contact details\n\n"
        "TAS Design Group profile:\n"
        "- IT consulting and data science company based in Japan and Vietnam\n"
        "- Specialises in AI/ML, data engineering, and custom software development\n"
        "- Delivers enterprise solutions with agile, rapid-iteration methodology\n"
        "- Trusted by clients across manufacturing, retail, finance, and logistics\n\n"
        "Formatting requirements:\n"
        "- Return ONLY valid HTML (no markdown, no code fences)\n"
        "- Use inline CSS for clean, professional styling\n"
        "- Color scheme: primary #4A0E8F (TAS purple), white background, dark text\n"
        "- Mobile-responsive layout with max-width: 650px\n"
        "- Use <table> for layout to ensure email client compatibility\n"
        "- Highlight key metrics and features with colored callout boxes",
    ),
    (
        "human",
        "Target company: {customer_name}\n\n"
        "Proposal details:\n{proposal}",
    ),
])


# ── Main function ─────────────────────────────────────────────────────────────

def make_proposal(idea: str, db, company_name: str) -> str:
    """
    Generate an HTML email proposal for *company_name* using RAG over *db*.

    Parameters
    ----------
    idea         : the product idea / hypothesis string
    db           : FAISS vector store built by make_db()
    company_name : name of the target company (used in the email salutation)

    Returns
    -------
    HTML string of the ready-to-send email
    """
    logger.info("Generating proposal for: %s", company_name)
    llm = get_llm()

    # ── Step 1: RAG retrieval — gather relevant evidence ─────────────────────
    retriever = db.as_retriever(search_kwargs={"k": 5})
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | _RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    rag_summary = rag_chain.invoke(idea)
    logger.info("RAG summary generated (%d chars)", len(rag_summary))

    # ── Step 2: structured proposal using with_structured_output ─────────────
    structured_llm = llm.with_structured_output(Proposal)
    try:
        proposal: Proposal = structured_llm.invoke(
            _PROPOSAL_PROMPT.format_messages(research=rag_summary, idea=idea)
        )

        # Format proposal for email prompt
        features_text = "\n".join(
            f"  - {f.name}: {f.description} (Impact: {f.impact})"
            for f in proposal.key_features
        )
        proposal_text = (
            f"Solution: {proposal.name}\n"
            f"Tagline: {proposal.tagline}\n"
            f"Why now: {proposal.reason}\n"
            f"Market context: {proposal.market_context}\n"
            f"Stakeholders: {proposal.stakeholder_mapping}\n"
            f"Key features:\n{features_text}\n"
            f"Competitive advantage: {proposal.competitive_advantage}\n"
            f"Success metrics: {proposal.success_metrics}"
        )
    except Exception as exc:
        logger.warning("Structured proposal failed, using raw RAG summary: %s", exc)
        proposal_text = rag_summary

    logger.info("Structured proposal generated.")

    # ── Step 3: email drafting ────────────────────────────────────────────────
    email_chain = _EMAIL_PROMPT | llm | StrOutputParser()
    email_html = email_chain.invoke({
        "proposal": proposal_text,
        "customer_name": company_name,
    })
    logger.info("Email draft ready for: %s", company_name)
    return email_html
