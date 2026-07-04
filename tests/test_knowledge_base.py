import json
from pathlib import Path

from knowledge_base.generate_rules import generate_rules
from rag.build_index import _validate_rules


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_generated_rules_match_checked_in_knowledge_base():
    stored = json.loads(
        (PROJECT_ROOT / "knowledge_base" / "rules.json").read_text(encoding="utf-8")
    )
    generated = generate_rules()

    assert generated == stored
    assert len(generated) == 89
    assert len({rule["id"] for rule in generated}) == len(generated)
    assert generated[0]["id"] == "R001"
    assert generated[-1]["id"] == "R089"


def test_rules_cover_profiles_concerns_and_safety():
    rules = generate_rules()
    tags = " ".join(rule["tags"] for rule in rules)

    for skin_type in ("oily", "dry", "combination", "sensitive", "normal"):
        assert f"skin_type:{skin_type}" in tags
    for concern in (
        "acne",
        "pigmentation",
        "dullness",
        "dryness",
        "redness",
        "texture",
        "fine_lines",
        "sun_protection",
    ):
        assert f"concern:{concern}" in tags
    assert "safety:non_medical" in tags
    assert "safety:patch_test" in tags


def test_rule_validation_rejects_duplicate_ids():
    rules = [
        {"id": "R001", "tags": "routine:any", "text": "First."},
        {"id": "R001", "tags": "routine:any", "text": "Second."},
    ]

    try:
        _validate_rules(rules)
    except ValueError as exc:
        assert "Duplicate rule id" in str(exc)
    else:
        raise AssertionError("Duplicate IDs should be rejected")
