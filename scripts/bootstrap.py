"""
Bootstrap script for SkinSense.

Responsibility:
- Ensure the knowledge base rules.json exists and is populated.
- Ensure the ChromaDB index is built and stays in sync with rules.json.
"""

import json
import os
import sys
from typing import Optional


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_project_root_on_path() -> None:
    root = _project_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def _rules_path() -> str:
    return os.path.join(_project_root(), "knowledge_base", "rules.json")


def _latest_mtime(path: str) -> float:
    """Return the latest modified time for a file or directory tree."""
    if not os.path.exists(path):
        return 0.0

    if os.path.isfile(path):
        return os.path.getmtime(path)

    latest = os.path.getmtime(path)
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for filename in filenames:
            if filename.startswith("."):
                continue
            full_path = os.path.join(dirpath, filename)
            try:
                latest = max(latest, os.path.getmtime(full_path))
            except OSError:
                pass

    return latest


def ensure_rules() -> str:
    _ensure_project_root_on_path()
    from knowledge_base.generate_rules import main as generate_main

    rules_path = _rules_path()

    regenerate = False
    if not os.path.exists(rules_path):
        print("[bootstrap] rules.json not found, generating...")
        regenerate = True
    else:
        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list) or len(data) < 50:
                print(
                    "[bootstrap] rules.json exists but looks empty/small; regenerating..."
                )
                regenerate = True
        except Exception:
            print("[bootstrap] rules.json unreadable; regenerating...")
            regenerate = True

    if regenerate:
        generate_main()
    else:
        print("[bootstrap] rules.json already present and populated.")

    return rules_path


def ensure_chroma_index(rules_path: Optional[str] = None) -> None:
    _ensure_project_root_on_path()
    from rag.build_index import DEFAULT_PERSIST_DIR, build_index

    persist_dir = DEFAULT_PERSIST_DIR
    rules_path = rules_path or _rules_path()

    rebuild = False
    if not os.path.exists(persist_dir):
        print("[bootstrap] ChromaDB directory not found, building index...")
        rebuild = True
    else:
        existing = [f for f in os.listdir(persist_dir) if not f.startswith(".")]
        if not existing:
            print("[bootstrap] ChromaDB directory is empty, building index...")
            rebuild = True
        else:
            rules_mtime = _latest_mtime(rules_path)
            index_mtime = _latest_mtime(persist_dir)

            if rules_mtime > index_mtime:
                print(
                    "[bootstrap] rules.json is newer than the ChromaDB index; rebuilding..."
                )
                rebuild = True
            else:
                print("[bootstrap] ChromaDB directory already populated and up to date.")

    if rebuild:
        build_index()


def main() -> None:
    _ensure_project_root_on_path()
    print("[bootstrap] Starting bootstrap...")
    rules_path = ensure_rules()
    ensure_chroma_index(rules_path)
    print("[bootstrap] Bootstrap complete.")


if __name__ == "__main__":
    main()