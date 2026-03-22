"""
SQLite helpers for logging interactions from the SkinSense app.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
DB_PATH = os.path.join(LOGS_DIR, "interactions.db")


def _ensure_dirs() -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)


def normalize_concerns(concerns: Any) -> List[str]:
    """
    Normalize concerns into a clean list of strings.

    Accepts:
    - a comma-separated string
    - a list/tuple/set of values
    - None
    """
    if concerns is None:
        return []

    if isinstance(concerns, str):
        parts = concerns.split(",")
        return [part.strip() for part in parts if part and part.strip()]

    if isinstance(concerns, (list, tuple, set)):
        normalized: List[str] = []
        for item in concerns:
            if item is None:
                continue
            value = str(item).strip()
            if value:
                normalized.append(value)
        return normalized

    value = str(concerns).strip()
    return [value] if value else []


@contextmanager
def _connect() -> Iterable[sqlite3.Connection]:
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    except sqlite3.Error as exc:
        raise RuntimeError(
            f"Failed to open SQLite database at '{DB_PATH}': {exc}"
        ) from exc

    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """
    Create the interactions table if it does not exist.
    """
    _ensure_dirs()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                attributes_json TEXT NOT NULL,
                retrieved_rule_ids TEXT NOT NULL,
                response_md TEXT NOT NULL,
                feedback TEXT NULL
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_interactions_ts ON interactions (ts)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_interactions_feedback ON interactions (feedback)"
        )
        conn.commit()


def insert_interaction(
    attributes: Dict[str, Any],
    retrieved_rule_ids: List[str],
    response_md: str,
) -> int:
    """
    Insert a new interaction row.

    Returns the inserted row id.
    """
    init_db()
    ts = datetime.now(timezone.utc).isoformat()
    normalized_attributes = dict(attributes)
    normalized_attributes["concerns"] = normalize_concerns(
    normalized_attributes.get("concerns")
)
    attrs_json = json.dumps(normalized_attributes, ensure_ascii=False)
    rules_str = ",".join(retrieved_rule_ids)

    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO interactions (ts, attributes_json, retrieved_rule_ids, response_md, feedback)
            VALUES (?, ?, ?, ?, NULL)
            """,
            (ts, attrs_json, rules_str, response_md),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_feedback(interaction_id: int, feedback: str) -> None:
    """
    Update the feedback field for a given interaction.
    """
    if feedback not in {"helpful", "not_helpful"}:
        raise ValueError("feedback must be 'helpful' or 'not_helpful'")

    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE interactions SET feedback = ? WHERE id = ?",
            (feedback, interaction_id),
        )
        conn.commit()


def fetch_all_interactions() -> List[Dict[str, Any]]:
    """
    Return all interactions as a list of dictionaries.
    """
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, ts, attributes_json, retrieved_rule_ids, response_md, feedback
            FROM interactions
            ORDER BY ts ASC
            """
        )
        rows = cur.fetchall()

    results: List[Dict[str, Any]] = []
    for row in rows:
        rid, ts, attrs_json, rules_str, response_md, feedback = row
        try:
            attrs = json.loads(attrs_json)
        except Exception:
            attrs = {}
        results.append(
            {
                "id": rid,
                "ts": ts,
                "attributes": attrs,
                "retrieved_rule_ids": rules_str,
                "response_md": response_md,
                "feedback": feedback,
            }
        )
    return results


def fetch_recent_interactions(limit: int = 25) -> List[Dict[str, Any]]:
    """
    Return the most recent N interactions (default 25), newest first.
    """
    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, ts, attributes_json, retrieved_rule_ids, response_md, feedback
            FROM interactions
            ORDER BY ts DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()

    results: List[Dict[str, Any]] = []
    for row in rows:
        rid, ts, attrs_json, rules_str, response_md, feedback = row
        try:
            attrs = json.loads(attrs_json)
        except Exception:
            attrs = {}
        results.append(
            {
                "id": rid,
                "ts": ts,
                "attributes": attrs,
                "retrieved_rule_ids": rules_str,
                "response_md": response_md,
                "feedback": feedback,
            }
        )
    return results

def get_aggregate_stats() -> Dict[str, Dict[str, int]]:
    """
    Compute aggregate dashboard stats for skin types, concerns, and feedback.

    Returns:
        {
            "skin_types": {"oily": 3, "dry": 2, ...},
            "concerns": {"acne": 4, "redness": 1, ...},
            "feedback": {"helpful": 5, "not_helpful": 2, "none": 3},
        }
    """
    interactions = fetch_all_interactions()

    skin_type_counts: Dict[str, int] = {}
    concern_counts: Dict[str, int] = {}
    feedback_counts: Dict[str, int] = {
        "helpful": 0,
        "not_helpful": 0,
        "none": 0,
    }

    for item in interactions:
        attrs = item.get("attributes") or {}

        skin_type = str(attrs.get("skin_type") or "").strip()
        if skin_type:
            skin_type_counts[skin_type] = skin_type_counts.get(skin_type, 0) + 1

        concerns = attrs.get("concerns") or []
        if isinstance(concerns, str):
            concerns = [c.strip() for c in concerns.split(",") if c.strip()]

        for concern in concerns:
            concern_str = str(concern).strip()
            if concern_str:
                concern_counts[concern_str] = concern_counts.get(concern_str, 0) + 1

        feedback = item.get("feedback")
        if feedback in {"helpful", "not_helpful"}:
            feedback_counts[feedback] += 1
        else:
            feedback_counts["none"] += 1

    return {
        "skin_types": skin_type_counts,
        "concerns": concern_counts,
        "feedback": feedback_counts,
    }

def get_db_path() -> str:
    """Expose the DB path for troubleshooting / display."""
    return DB_PATH

