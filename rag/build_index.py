"""
Build a ChromaDB index from the skincare rules knowledge base.
"""

import json
import os
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer


DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")
DEFAULT_COLLECTION_NAME = "skincare_rules"


def load_rules() -> List[Dict]:
    here = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(here, "..", "knowledge_base", "rules.json")
    with open(kb_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("rules.json is not a list")
    return data


def build_index(
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> None:
    os.makedirs(persist_directory, exist_ok=True)

    rules = load_rules()
    if not rules:
        raise RuntimeError("No rules found. Run knowledge_base/generate_rules.py first.")

    print(f"[build_index] Loaded {len(rules)} rules")

    client = chromadb.PersistentClient(path=persist_directory)

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

