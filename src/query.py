"""RAG query module for asking questions."""
from __future__ import annotations

import os
from typing import Optional
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

from .vectordb import query_similar

load_dotenv()


def build_context(results: list[dict], max_chars: int = 8000) -> str:
    """Build context string from search results."""
    context_parts = []
    total_chars = 0

    for r in results:
        source = r["metadata"].get("source_file", "unknown")
        text = r["text"]

        part = f"[Source: {source}]\n{text}\n"

        if total_chars + len(part) > max_chars:
            break

        context_parts.append(part)
        total_chars += len(part)

    return "\n---\n".join(context_parts)


def ask_openai(
    question: str,
    context: str,
    model: str = "gpt-4o-mini"
) -> str:
    """Ask a question using OpenAI."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = """You are a helpful assistant that answers questions based on video transcript content about crypto trading.
Use the provided context to answer the question. If the context doesn't contain relevant information, say so.
Always cite which source file the information comes from when possible."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context from video transcripts:\n\n{context}\n\n---\n\nQuestion: {question}"}
        ],
        temperature=0.7,
        max_tokens=1500
    )

    return response.choices[0].message.content


def ask_anthropic(
    question: str,
    context: str,
    model: str = "claude-sonnet-4-20250514"
) -> str:
    """Ask a question using Anthropic Claude."""
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system_prompt = """You are a helpful assistant that answers questions based on video transcript content about crypto trading.
Use the provided context to answer the question. If the context doesn't contain relevant information, say so.
Always cite which source file the information comes from when possible."""

    response = client.messages.create(
        model=model,
        max_tokens=1500,
        system=system_prompt,
        messages=[
            {"role": "user", "content": f"Context from video transcripts:\n\n{context}\n\n---\n\nQuestion: {question}"}
        ]
    )

    return response.content[0].text


def ask(
    question: str,
    provider: str = "anthropic",
    n_results: int = 5,
    db_path: Optional[str] = None,
    model: Optional[str] = None
) -> dict:
    """
    Ask a question against the video knowledge base.

    Args:
        question: The question to ask
        provider: 'openai' or 'anthropic'
        n_results: Number of similar chunks to retrieve
        db_path: Path to ChromaDB database
        model: Optional model override

    Returns:
        Dict with answer and sources
    """
    # Search for relevant chunks
    results = query_similar(question, n_results=n_results, db_path=db_path)

    if not results:
        return {
            "answer": "No content found in the database. Please ingest some videos first.",
            "sources": [],
            "context_chunks": 0
        }

    # Build context from results
    context = build_context(results)

    # Get answer from LLM
    if provider == "openai":
        model = model or "gpt-4o-mini"
        answer = ask_openai(question, context, model)
    else:
        model = model or "claude-sonnet-4-20250514"
        answer = ask_anthropic(question, context, model)

    # Extract unique sources
    sources = list(set(r["metadata"].get("source_file", "unknown") for r in results))

    return {
        "answer": answer,
        "sources": sources,
        "context_chunks": len(results)
    }
