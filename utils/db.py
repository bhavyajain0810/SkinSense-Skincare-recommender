"""
SQLite helpers for logging interactions from the SkinSense app.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List

from utils.validation import validate_feedback


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
DB_PATH = os.path.join(LOGS_DIR, "interactions.db")


class DatabaseError(RuntimeError):
    """Raised when an SQLite operation cannot be completed."""


class InteractionNotFoundError(LookupError):
    """Raised when feedback targets an interaction that does not exist."""


def normalize_concerns(concerns: Any) -> List[str]:
    """Normalize stored concern values without rewriting historical labels."""
    if concerns is None:
        return []
    if isinstance(concerns, str):
        items = concerns.split(",")
    elif isinstance(concerns, (list, tuple, set)):
        items = concerns
    else:
        items = [concerns]
    return [
        str(item).strip()
        for item in items
        if item is not None and str(item).strip()
    ]


def _ensure_dirs() -> None:
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    except OSError as exc:
        raise DatabaseError(f"Unable to prepare the database directory: {exc}") from exc


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    _ensure_dirs()
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    except sqlite3.Error as exc:
        raise DatabaseError(
            f"Failed to open SQLite database at '{DB_PATH}': {exc}"
        ) from exc

    try:
        yield conn
    except sqlite3.Error as exc:
        conn.rollback()
        raise DatabaseError(f"SQLite operation failed: {exc}") from exc
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
    if not isinstance(attributes, dict):
        raise ValueError("attributes must be a dictionary")
    normalized_attributes = dict(attributes)
    normalized_attributes["concerns"] = normalize_concerns(
        normalized_attributes.get("concerns")
    )
    attrs_json = json.dumps(normalized_attributes, ensure_ascii=False)
    rules_str = ",".join(str(rule_id) for rule_id in retrieved_rule_ids)

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
    if isinstance(interaction_id, bool) or not isinstance(interaction_id, int):
        raise ValueError("interaction_id must be an integer")
    if interaction_id <= 0:
        raise ValueError("interaction_id must be greater than zero")
    normalized_feedback = validate_feedback(feedback)

    init_db()
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE interactions SET feedback = ? WHERE id = ?",
            (normalized_feedback, interaction_id),
        )
        if cur.rowcount == 0:
            raise InteractionNotFoundError(
                f"Interaction {interaction_id} was not found."
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
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
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

        try:
            concerns = normalize_concerns(attrs.get("concerns"))
        except ValueError:
            concerns = []

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

def vacuum_db() -> None:
    """
    Run SQLite VACUUM to reclaim unused space and compact the database file.

    Use this as an optional maintenance step after many deletes/updates or
    other bulk changes. It is not required for normal app usage.
    """
    init_db()
    with _connect() as conn:
        conn.execute("VACUUM")

def get_db_path() -> str:
    """Expose the DB path for troubleshooting / display."""
    return DB_PATH

