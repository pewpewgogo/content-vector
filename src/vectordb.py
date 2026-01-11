"""Vector database operations using ChromaDB."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings


DEFAULT_DB_PATH = ".chroma_db"
COLLECTION_NAME = "video_transcripts"


def get_client(db_path: Optional[str] = None) -> chromadb.PersistentClient:
    """Get or create a ChromaDB client."""
    path = db_path or DEFAULT_DB_PATH
    Path(path).mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False)
    )


def get_collection(
    client: chromadb.PersistentClient,
    collection_name: str = COLLECTION_NAME
) -> chromadb.Collection:
    """Get or create a collection."""
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def add_chunks(
    chunks: list[dict],
    db_path: Optional[str] = None,
    collection_name: str = COLLECTION_NAME
) -> int:
    """
    Add text chunks to the vector database.

    ChromaDB will automatically create embeddings using its default model.
    """
    client = get_client(db_path)
    collection = get_collection(client, collection_name)

    # Prepare data for ChromaDB
    ids = []
    documents = []
    metadatas = []

    existing_count = collection.count()

    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{existing_count + i}"
        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append({
            "source_file": chunk["source_file"],
            "source_path": chunk["source_path"],
            "chunk_index": chunk["chunk_index"],
            "language": chunk.get("language", "unknown")
        })

    # Add to collection in batches
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )

    return len(ids)


def query_similar(
    query: str,
    n_results: int = 5,
    db_path: Optional[str] = None,
    collection_name: str = COLLECTION_NAME
) -> list[dict]:
    """
    Query the vector database for similar chunks.

    Returns list of results with text, metadata, and distance.
    """
    client = get_client(db_path)
    collection = get_collection(client, collection_name)

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    # Format results
    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return formatted


def get_stats(
    db_path: Optional[str] = None,
    collection_name: str = COLLECTION_NAME
) -> dict:
    """Get database statistics."""
    client = get_client(db_path)
    collection = get_collection(client, collection_name)

    # Get unique source files
    all_data = collection.get(include=["metadatas"])
    source_files = set()
    for metadata in all_data["metadatas"]:
        source_files.add(metadata.get("source_file", "unknown"))

    return {
        "total_chunks": collection.count(),
        "source_files": len(source_files),
        "files": sorted(source_files)
    }


def clear_database(
    db_path: Optional[str] = None,
    collection_name: str = COLLECTION_NAME
) -> None:
    """Clear all data from the collection."""
    client = get_client(db_path)
    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass  # Collection doesn't exist
