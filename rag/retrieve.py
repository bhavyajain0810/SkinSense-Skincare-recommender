"""
Utilities for querying the ChromaDB skincare rules collection.
"""

import os
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer


DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")
DEFAULT_COLLECTION_NAME = "skincare_rules"


_model_cache = None


def _get_model() -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_cache


def get_collection(
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        collection = client.get_collection(collection_name)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Chroma collection '{collection_name}' not found. "
            "Run rag/build_index.py or scripts/bootstrap.py first."
        ) from exc
    return collection


def retrieve_rules(collection, query: str, k: int = 8) -> List[Dict[str, Any]]:
    """
    Retrieve top-k rules from the collection for a free-text query.

    Returns a list of dicts with: id, document, metadata, distance.
    """
    if not query.strip():
        return []

    model = _get_model()
    embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=k,
    )

    out: List[Dict[str, Any]] = []
    ids_list = results.get("ids", [[]])[0]
    docs_list = results.get("documents", [[]])[0]
    metas_list = results.get("metadatas", [[]])[0]
    dists_list = results.get("distances", [[]])[0]

    for rid, doc, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
        out.append(
            {
                "id": rid,
                "document": doc,
                "metadata": meta or {},
                "distance": float(dist),
            }
        )
    return out

