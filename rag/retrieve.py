"""
Utilities for querying the ChromaDB skincare rules collection.
"""

import logging
import os
from typing import Any, Dict, List, Mapping

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from utils.validation import validate_k


logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")
DEFAULT_COLLECTION_NAME = "skincare_rules"
CHROMA_SETTINGS = Settings(anonymized_telemetry=False)


_model_cache = None


class IndexNotReadyError(RuntimeError):
    """Raised when the local Chroma collection cannot be used."""


class RetrievalError(RuntimeError):
    """Raised when Chroma returns an unusable result."""


def _get_model() -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _model_cache


def get_collection(
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    if not os.path.isdir(persist_directory) or not os.listdir(persist_directory):
        raise IndexNotReadyError(
            f"Chroma index is missing at '{os.path.abspath(persist_directory)}'. "
            "Run python scripts/bootstrap.py first."
        )
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=CHROMA_SETTINGS,
    )
    try:
        collection = client.get_collection(collection_name)
    except Exception as exc:
        raise IndexNotReadyError(
            f"Chroma collection '{collection_name}' not found. "
            "Run python scripts/bootstrap.py first."
        ) from exc
    return collection


def retrieve_rules(collection, query: str, k: int = 8) -> List[Dict[str, Any]]:
    """
    Retrieve top-k rules from the collection for a free-text query.

    Returns a list of dicts with: id, document, metadata, distance.
    """
    result_count = validate_k(k)
    if not isinstance(query, str) or not query.strip():
        return []
    if collection is None:
        raise IndexNotReadyError("The Chroma collection is not available.")

    try:
        model = _get_model()
        embedding = model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=embedding,
            n_results=result_count,
        )
    except IndexNotReadyError:
        raise
    except Exception as exc:
        raise RetrievalError(f"Unable to retrieve skincare rules: {exc}") from exc

    if not isinstance(results, Mapping):
        raise RetrievalError("Chroma returned an invalid response.")

    out: List[Dict[str, Any]] = []
    try:
        ids_list = (results.get("ids") or [[]])[0]
        docs_list = (results.get("documents") or [[]])[0]
        metas_list = (results.get("metadatas") or [[]])[0]
        dists_list = (results.get("distances") or [[]])[0]
    except (IndexError, TypeError) as exc:
        raise RetrievalError("Chroma returned an invalid result shape.") from exc

    if not all(isinstance(values, list) for values in (
        ids_list,
        docs_list,
        metas_list,
        dists_list,
    )):
        raise RetrievalError("Chroma returned an invalid result shape.")

    for rid, doc, meta, dist in zip(
        ids_list[:result_count],
        docs_list[:result_count],
        metas_list[:result_count],
        dists_list[:result_count],
    ):
        if not rid or not isinstance(doc, str):
            continue
        try:
            distance = float(dist)
        except (TypeError, ValueError) as exc:
            raise RetrievalError("Chroma returned a non-numeric distance.") from exc
        out.append(
            {
                "id": str(rid),
                "document": doc,
                "metadata": meta if isinstance(meta, dict) else {},
                "distance": distance,
            }
        )
    return out

