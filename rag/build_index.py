"""
Build a ChromaDB index from the skincare rules knowledge base.
"""

import json
import logging
import os
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")
DEFAULT_COLLECTION_NAME = "skincare_rules"
CHROMA_SETTINGS = Settings(anonymized_telemetry=False)


def _validate_rules(data: Any) -> List[Dict[str, str]]:
    if not isinstance(data, list):
        raise ValueError("rules.json must contain a list")
    if not data:
        raise ValueError("rules.json does not contain any rules")

    validated: List[Dict[str, str]] = []
    seen_ids = set()
    for index, rule in enumerate(data, start=1):
        if not isinstance(rule, dict):
            raise ValueError(f"Rule {index} must be an object")
        rid = rule.get("id")
        tags = rule.get("tags")
        text = rule.get("text")
        if not all(isinstance(value, str) and value.strip() for value in (rid, tags, text)):
            raise ValueError(f"Rule {index} is missing id, tags, or text")
        if rid in seen_ids:
            raise ValueError(f"Duplicate rule id: {rid}")
        seen_ids.add(rid)
        validated.append({"id": rid, "tags": tags, "text": text})
    return validated


def load_rules() -> List[Dict[str, str]]:
    here = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(here, "..", "knowledge_base", "rules.json")

    if not os.path.exists(kb_path):
        raise FileNotFoundError(
            "rules.json was not found at "
            f"'{os.path.abspath(kb_path)}'. "
            "Generate it first by running: python knowledge_base/generate_rules.py"
        )

    with open(kb_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _validate_rules(data)


def build_index(
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    os.makedirs(persist_directory, exist_ok=True)

    rules = load_rules()
    if not rules:
        raise RuntimeError("No rules found. Run knowledge_base/generate_rules.py first.")

    print(f"[build_index] Loaded {len(rules)} rules")

    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=CHROMA_SETTINGS,
    )

    # We do not configure an embedding function in Chroma; instead we pass
    # pre-computed embeddings explicitly for both indexing and querying.
    existing_names = [c.name for c in client.list_collections()]
    if collection_name in existing_names:
        client.delete_collection(collection_name)
        print(f"[build_index] Deleted existing collection '{collection_name}'")

    collection = client.create_collection(name=collection_name)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict] = []
    for rule in rules:
        ids.append(rule["id"])
        documents.append(rule["text"])
        metadatas.append({"tags": rule.get("tags", "")})

    print("[build_index] Computing embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    print(f"[build_index] Upserting {len(ids)} items into Chroma")
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"[build_index] Done. Persistent directory: {persist_directory}")


def main() -> None:
    build_index()


if __name__ == "__main__":
    main()

