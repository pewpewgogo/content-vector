"""Text chunking for vector embeddings."""
from __future__ import annotations

from typing import Iterator


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> Iterator[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: The text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
    """
    if not text:
        return

    # Clean up the text
    text = text.strip()

    if len(text) <= chunk_size:
        yield text
        return

    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            for sep in ['. ', '? ', '! ', '\n\n', '\n']:
                last_sep = text.rfind(sep, start + chunk_size // 2, end)
                if last_sep != -1:
                    end = last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            yield chunk

        # Move start forward, accounting for overlap
        start = end - overlap
        if start >= len(text):
            break


def chunk_transcript(
    transcript: dict,
    chunk_size: int = 1000,
    overlap: int = 200
) -> list[dict]:
    """
    Chunk a transcript and preserve metadata.

    Returns list of dicts with chunk text and source info.
    """
    chunks = []
    for i, chunk_text_content in enumerate(chunk_text(
        transcript["text"],
        chunk_size=chunk_size,
        overlap=overlap
    )):
        chunks.append({
            "text": chunk_text_content,
            "source_file": transcript["file"],
            "source_path": transcript["path"],
            "chunk_index": i,
            "language": transcript.get("language", "unknown")
        })
    return chunks


def chunk_transcripts(
    transcripts: list[dict],
    chunk_size: int = 1000,
    overlap: int = 200
) -> list[dict]:
    """Chunk multiple transcripts."""
    all_chunks = []
    for transcript in transcripts:
        chunks = chunk_transcript(transcript, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks
