from pathlib import Path

import pytest

from utils import db
from utils.validation import InputValidationError


@pytest.fixture
def temporary_db(monkeypatch, tmp_path: Path):
    path = tmp_path / "logs" / "test_interactions.db"
    monkeypatch.setattr(db, "DB_PATH", str(path))
    return path


def test_insert_fetch_and_feedback_update(temporary_db):
    db.init_db()
    interaction_id = db.insert_interaction(
        {"skin_type": "oily", "concerns": ["acne"], "notes": ""},
        ["R001", "R084"],
        "routine",
    )

    rows = db.fetch_all_interactions()
    assert temporary_db.exists()
    assert rows[0]["id"] == interaction_id
    assert rows[0]["attributes"]["concerns"] == ["acne"]
    assert rows[0]["retrieved_rule_ids"] == "R001,R084"
    assert rows[0]["feedback"] is None

    db.update_feedback(interaction_id, "helpful")
    assert db.fetch_all_interactions()[0]["feedback"] == "helpful"


def test_invalid_feedback_is_rejected(temporary_db):
    with pytest.raises(InputValidationError):
        db.update_feedback(1, "maybe")


def test_unknown_interaction_is_rejected(temporary_db):
    db.init_db()
    with pytest.raises(db.InteractionNotFoundError):
        db.update_feedback(999, "not_helpful")


def test_recent_limit_is_validated(temporary_db):
    with pytest.raises(ValueError):
        db.fetch_recent_interactions(0)


def test_database_open_error_is_wrapped(monkeypatch, temporary_db):
    def fail(*args, **kwargs):
        raise db.sqlite3.OperationalError("locked")

    monkeypatch.setattr(db.sqlite3, "connect", fail)
    with pytest.raises(db.DatabaseError, match="Failed to open"):
        db.init_db()
