"""
TAS AutoBD — Proposal Generation Agent
=========================================
Uses RAG (retrieval-augmented generation) over the FAISS knowledge base
to craft a structured business proposal and a polished HTML email.
"""

import logging
from typing import List

from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import get_llm
from utils import format_docs

logger = logging.getLogger(__name__)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class Feature(BaseModel):
    name: str = Field(description="Feature name")
    description: str = Field(description="Detailed feature description")


class Proposal(BaseModel):
    name: str = Field(description="Product / solution name")
    reason: str = Field(description="Why this solution is needed by the customer")
    marketing_trends: str = Field(description="Relevant market trends that support this solution")
    stakeholder_mapping: str = Field(description="Key stakeholders and their roles")
    key_features: List[Feature] = Field(description="List of features tailored to the customer")


# ── Prompts ───────────────────────────────────────────────────────────────────

_RAG_PROMPT = PromptTemplate(
    template=(
        "You are a Senior Solution Architect at TAS Design Group Inc.\n"
        "Using the retrieved context below, provide a concise, actionable summary that includes:\n"
        "1. Key features and capabilities relevant to the customer\n"
        "2. Technical approach or architecture highlights\n"
        "3. Specific benefits for the customer's business context\n"
        "4. Any proven outcomes or metrics from similar projects\n\n"
        "Context:\n{context}\n\n"
        "Customer situation:\n{question}\n\n"
        "Summary:"
    ),
    input_variables=["context", "question"],
)

_PROPOSAL_PROMPT_TEMPLATE = (
    "You are a Senior Business Developer at TAS Design Group Inc. crafting a formal proposal.\n\n"
    "Based on the solution research below, create a comprehensive proposal that includes:\n"
    "- A compelling product/solution name\n"
    "- Clear justification for why this solution fits the customer\n"
    "- Alignment with current market trends\n"
    "- Stakeholder mapping (who benefits and how)\n"
    "- Key features tailored to the customer's needs\n"
    "- Industry trend analysis\n"
    "- Growth opportunities and competitive advantages\n\n"
    "{format_instructions}\n\n"
    "Solution research:\n{query}"
)

_EMAIL_PROMPT = PromptTemplate(
    template=(
        "You are a Business Developer at TAS Design Group Inc. writing a formal, "
        "persuasive business proposal email.\n\n"
        "Write a complete HTML email to {customer_name} based on the proposal below. "
        "The email must:\n"
        "- Use professional, warm, and confident language\n"
        "- Open with a strong value proposition\n"
        "- Clearly explain our proposed solution and its business impact\n"
        "- Highlight 3-5 key features with concrete benefits\n"
        "- Include a compelling call to action\n"
        "- Introduce TAS Design Group Inc. as a Japan- and Vietnam-based consulting "
        "and data science company\n\n"
        "Return ONLY valid HTML (no markdown fences). Use inline CSS for styling.\n\n"
        "Proposal details:\n{query}"
    ),
    input_variables=["customer_name", "query"],
)


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
    retriever = db.as_retriever(search_kwargs={"k": 4})
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | _RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    rag_summary = rag_chain.invoke(idea)
    logger.info("RAG summary generated (%d chars)", len(rag_summary))

    # ── Step 2: structured proposal generation ───────────────────────────────
    pydantic_parser = PydanticOutputParser(pydantic_object=Proposal)
    proposal_prompt = PromptTemplate(
        template=_PROPOSAL_PROMPT_TEMPLATE,
        input_variables=["query"],
        partial_variables={"format_instructions": pydantic_parser.get_format_instructions()},
    )
    proposal_chain = proposal_prompt | llm
    structured_proposal = proposal_chain.invoke({"query": rag_summary})
    logger.info("Structured proposal generated.")

    # ── Step 3: email drafting ────────────────────────────────────────────────
    email_chain = _EMAIL_PROMPT | llm
    email = email_chain.invoke({
        "query": structured_proposal.content,
        "customer_name": company_name,
    })
    logger.info("Email draft ready for: %s", company_name)
    return email.content
