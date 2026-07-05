from pathlib import Path

from streamlit.testing.v1 import AppTest

import rag.retrieve as retrieve_module
import utils.llm_client as llm_module
from utils import db


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RETRIEVED_RULES = [
    {
        "id": "R001",
        "document": "Use lightweight cosmetic textures for oily, acne-prone skin.",
        "metadata": {"tags": "skin_type:oily concern:acne routine:am"},
        "distance": 0.12,
    },
    {
        "id": "R084",
        "document": "Patch test one new cosmetic product at a time.",
        "metadata": {"tags": "safety:patch_test routine:any"},
        "distance": 0.18,
    },
]

VALID_RESPONSE = """
A simple cosmetic routine focused on consistency.

## AM Routine
- Cleanse gently.
- Moisturize and use sunscreen according to its label.

## PM Routine
- Cleanse gently.
- Finish with a comfortable moisturizer.

## Extra Tips
- Patch test one new product at a time.

## Why these suggestions?
R001 supports lightweight steps and R084 supports patch testing.

## Citations
Used: R001, R084
""".strip()


def test_full_streamlit_submission_uses_temporary_database(monkeypatch, tmp_path):
    temporary_database = tmp_path / "ui-smoke" / "interactions.db"
    fake_collection = object()

    monkeypatch.setattr(db, "DB_PATH", str(temporary_database))
    monkeypatch.setattr(retrieve_module, "get_collection", lambda: fake_collection)
    monkeypatch.setattr(
        retrieve_module,
        "retrieve_rules",
        lambda collection, query, k: RETRIEVED_RULES,
    )
    monkeypatch.setattr(llm_module, "call_llm", lambda prompt: VALID_RESPONSE)
    monkeypatch.setattr(
        llm_module,
        "check_llm_health",
        lambda: {
            "available": True,
            "backend": "mock",
            "base_url": "http://127.0.0.1:8001",
        },
    )

    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60)
    app.run()
    assert not app.exception

    app.selectbox[0].select("oily")
    app.multiselect[0].select("acne")
    app.run()

    build_buttons = [
        button for button in app.button if button.label == "Build my routine"
    ]
    assert len(build_buttons) == 1
    build_buttons[0].click().run()

    assert not app.exception
    assert temporary_database.exists()
    rows = db.fetch_all_interactions()
    assert len(rows) == 1
    assert rows[0]["attributes"] == {
        "skin_type": "oily",
        "concerns": ["acne"],
        "notes": "",
    }
    assert rows[0]["retrieved_rule_ids"] == "R001,R084"
    assert rows[0]["response_md"] == VALID_RESPONSE
    assert any(
        "A simple plan for morning and evening" in heading.value
        for heading in app.markdown
    )
