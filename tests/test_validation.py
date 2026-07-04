import pytest

from utils.validation import (
    InputValidationError,
    normalize_concerns,
    validate_feedback,
    validate_k,
    validate_skin_profile,
)


def test_profile_is_normalized():
    result = validate_skin_profile(
        {
            "skin_type": " OILY ",
            "concerns": ["Acne", "acne", "texture"],
            "notes": "  Keep it simple.  ",
        }
    )
    assert result == {
        "skin_type": "oily",
        "concerns": ["acne", "texture"],
        "notes": "Keep it simple.",
    }


@pytest.mark.parametrize("skin_type", ["", None, "very_oily", 123])
def test_invalid_skin_profile_is_rejected(skin_type):
    with pytest.raises(InputValidationError):
        validate_skin_profile(
            {"skin_type": skin_type, "concerns": [], "notes": ""}
        )


def test_unknown_concern_is_rejected():
    with pytest.raises(InputValidationError, match="Unsupported concern"):
        normalize_concerns(["acne", "eczema"])


@pytest.mark.parametrize("value, expected", [(1, 1), (8, 8), ("20", 20)])
def test_valid_k(value, expected):
    assert validate_k(value) == expected


def test_feedback_validation():
    assert validate_feedback(" Helpful ") == "helpful"
    with pytest.raises(InputValidationError):
        validate_feedback("yes")
