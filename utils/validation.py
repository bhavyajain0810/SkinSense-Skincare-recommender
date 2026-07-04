"""Shared validation helpers for user-controlled SkinSense inputs."""

from typing import Any, Dict, List


SKIN_TYPES = ("oily", "dry", "combination", "sensitive", "normal")
CONCERNS = (
    "acne",
    "pigmentation",
    "dullness",
    "dryness",
    "redness",
    "texture",
    "fine_lines",
    "sun_protection",
)
MIN_RETRIEVAL_K = 1
MAX_RETRIEVAL_K = 20
MAX_NOTES_LENGTH = 1_000


class InputValidationError(ValueError):
    """Raised when a user-controlled value is not supported."""


def normalize_concerns(value: Any) -> List[str]:
    """Return a de-duplicated, lower-case list of concern values."""
    if value is None:
        return []
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        raise InputValidationError("Concerns must be a list or comma-separated text.")

    normalized: List[str] = []
    seen = set()
    for item in items:
        concern = str(item).strip().lower()
        if not concern or concern in seen:
            continue
        if concern not in CONCERNS:
            raise InputValidationError(f"Unsupported concern: {concern}.")
        normalized.append(concern)
        seen.add(concern)
    return normalized


def validate_skin_profile(attributes: Any) -> Dict[str, Any]:
    """Validate and normalize the profile used by retrieval and logging."""
    if not isinstance(attributes, dict):
        raise InputValidationError("Skin profile must be an object.")

    skin_type = str(attributes.get("skin_type") or "").strip().lower()
    if not skin_type:
        raise InputValidationError("Choose a skin type before building a routine.")
    if skin_type not in SKIN_TYPES:
        raise InputValidationError(f"Unsupported skin type: {skin_type}.")

    concerns = normalize_concerns(attributes.get("concerns"))
    notes = str(attributes.get("notes") or "").strip()
    if len(notes) > MAX_NOTES_LENGTH:
        raise InputValidationError(
            f"Notes must be {MAX_NOTES_LENGTH} characters or fewer."
        )

    return {"skin_type": skin_type, "concerns": concerns, "notes": notes}


def validate_k(value: Any) -> int:
    """Validate the number of vector results requested."""
    if isinstance(value, bool):
        raise InputValidationError("Retrieval count must be a whole number.")
    try:
        k = int(value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError("Retrieval count must be a whole number.") from exc
    if k != value and not isinstance(value, str):
        raise InputValidationError("Retrieval count must be a whole number.")
    if not MIN_RETRIEVAL_K <= k <= MAX_RETRIEVAL_K:
        raise InputValidationError(
            f"Retrieval count must be between {MIN_RETRIEVAL_K} and "
            f"{MAX_RETRIEVAL_K}."
        )
    return k


def validate_feedback(value: Any) -> str:
    """Validate a feedback value before it reaches SQLite."""
    feedback = str(value or "").strip().lower()
    if feedback not in {"helpful", "not_helpful"}:
        raise InputValidationError("Feedback must be 'helpful' or 'not_helpful'.")
    return feedback
