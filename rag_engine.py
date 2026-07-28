"""
rag_engine.py
--------------
All the "brains" of the Plant Assistant live here:
  1. Loading the SOP PDFs for both plants
  2. Splitting them into chunks and embedding them (FAISS + HuggingFace, runs fully local/free)
  3. Retrieving the most relevant chunks for a question
  4. Calling the Groq LLM with a safety-aware system prompt to produce the final answer

Keeping this logic separate from app.py makes the Streamlit file much easier to read
and lets you re-use / test the retrieval logic on its own.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "faiss_index"

# Map a short "plant code" to the PDF file name + a friendly display name.
# Add more plants here later just by adding another entry + dropping the PDF in data/
PLANTS: dict[str, dict] = {
    "WMB": {
        "file": "WMB_SOP.pdf",
        "name": "Wembiyagoda Mini Hydro Power Plant",
        "location": "Kalawana, Rathnapura",
        "emoji": "🏞️",
    },
    "BTO": {
        "file": "BTO_SOP.pdf",
        "name": "Batathota Mini Hydro Power Plant",
        "location": "Kuruwita, Rathnapura",
        "emoji": "⚙️",
    },
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq models currently recommended (llama-3.3-70b-versatile / llama-3.1-8b-instant
# were deprecated by Groq on 2026-06-17). Feel free to add/remove models here -
# whatever you pick just needs to be a valid model id from https://console.groq.com/docs/models
AVAILABLE_MODELS = {
    "openai/gpt-oss-120b (recommended - best quality)": "openai/gpt-oss-120b",
    "openai/gpt-oss-20b (fastest)": "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b (multilingual)": "qwen/qwen3.6-27b",
    "llama-3.3-70b-versatile (legacy)": "llama-3.3-70b-versatile",
}

SYSTEM_PROMPT = """You are the Vidullanka PLC Plant Assistant, an AI expert trained on the official
Standard Operating Procedures (SOPs) of Vidullanka's mini hydro power plants.

Your job is to help engineers, supervisors and operators quickly find:
- Startup / shutdown / emergency-stop procedures
- Troubleshooting guidance (water leakage, electrical fire, abnormal readings, etc.)
- Safety instructions, PPE requirements and lockout-tagout steps
- Responsibilities / approval chains (who signs off on what)
- Equipment-specific operating steps (MIV, governor, crane, trash machine, transformer, HVCB, CT/PT, etc.)

RULES YOU MUST FOLLOW:
1. Answer ONLY using the information given to you in the "CONTEXT" section below. This context comes
   directly from the plant's official SOP documents.
2. If the answer is not contained in the context, clearly say so ("I couldn't find this in the SOP
   documents provided") instead of guessing. Never invent procedures, numbers, or pressure/voltage
   values that are not in the context - this is safety-critical equipment.
3. Always mention which plant(s) the answer applies to (Wembiyagoda / Batathota) when it's clear from
   the context, since some procedures may differ slightly between the two sites.
4. For step-by-step procedures, reproduce the steps in a clear numbered list, in the same order as the
   SOP, and do not skip safety-critical steps (lockout-tagout, PPE, approvals, etc.).
5. If the question touches on an emergency situation (fire, flood, abnormal leakage, uncontrolled
   speed, etc.), begin your answer with a short bolded safety reminder before giving the procedure.
6. Be concise and practical - this is being used on the plant floor, not for casual reading.
7. If useful, note the SOP number / section title your answer is based on (e.g. "SOP-016 - Abnormal
   water leakage in turbine") so the user can look it up in the full document if needed.

CONTEXT:
{context}
"""


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    text: str
    plant_code: str
    plant_name: str
    page: int


@dataclass
class AnswerResult:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Vector store building / loading
# --------------------------------------------------------------------------------------

def _load_and_split_documents() -> list:
    """Load every SOP PDF listed in PLANTS and split into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for code, info in PLANTS.items():
        pdf_path = DATA_DIR / info["file"]
        if not pdf_path.exists():
            continue

        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        chunks = splitter.split_documents(pages)

        for chunk in chunks:
            chunk.metadata["plant_code"] = code
            chunk.metadata["plant_name"] = info["name"]
            # PyPDFLoader gives 0-indexed pages -> make them human friendly
            chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1

        all_chunks.extend(chunks)

    return all_chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    """Local, free embedding model - no API key needed for this part."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_or_load_vectorstore(force_rebuild: bool = False) -> FAISS:
    """
    Build the FAISS index from the SOP PDFs, or load a previously-built one from disk
    so you don't have to re-embed everything on every app restart.
    """
    embeddings = get_embeddings()

    if INDEX_DIR.exists() and not force_rebuild:
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    chunks = _load_and_split_documents()
    if not chunks:
        raise FileNotFoundError(
            f"No SOP PDFs found in '{DATA_DIR}'. Make sure WMB_SOP.pdf and/or BTO_SOP.pdf "
            "are placed in the data/ folder."
        )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    return vectorstore


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------

def retrieve_chunks(
    vectorstore: FAISS,
    query: str,
    plant_filter: str = "BOTH",
    k: int = 5,
) -> list[RetrievedChunk]:
    """Similarity search, optionally restricted to a single plant."""
    # Pull extra candidates so that filtering by plant still leaves us `k` results.
    fetch_k = k * 4 if plant_filter != "BOTH" else k
    raw_docs = vectorstore.similarity_search(query, k=fetch_k)

    if plant_filter != "BOTH":
        raw_docs = [d for d in raw_docs if d.metadata.get("plant_code") == plant_filter]

    raw_docs = raw_docs[:k]

    return [
        RetrievedChunk(
            text=d.page_content,
            plant_code=d.metadata.get("plant_code", "?"),
            plant_name=d.metadata.get("plant_name", "Unknown plant"),
            page=d.metadata.get("page", "?"),
        )
        for d in raw_docs
    ]


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(No relevant SOP content was found for this question.)"
    blocks = []
    for c in chunks:
        blocks.append(f"[{c.plant_name} | Page {c.page}]\n{c.text}")
    return "\n\n---\n\n".join(blocks)


# --------------------------------------------------------------------------------------
# LLM call
# --------------------------------------------------------------------------------------

def get_llm(api_key: str, model_name: str, temperature: float = 0.1) -> ChatGroq:
    return ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=1024,
    )


def ask_question(
    vectorstore: FAISS,
    api_key: str,
    model_name: str,
    question: str,
    chat_history: list[tuple[str, str]] | None = None,
    plant_filter: str = "BOTH",
    k: int = 5,
) -> AnswerResult:
    """
    Full RAG pipeline for a single question:
      retrieve -> build prompt (with short chat history for context) -> call Groq -> return answer + sources
    """
    chunks = retrieve_chunks(vectorstore, question, plant_filter=plant_filter, k=k)
    context = _format_context(chunks)

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

    # Add a little short-term memory (last 3 turns) so follow-up questions work naturally
    if chat_history:
        for role, content in chat_history[-6:]:
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=question))

    llm = get_llm(api_key, model_name)
    response = llm.invoke(messages)

    return AnswerResult(answer=response.content, sources=chunks)
