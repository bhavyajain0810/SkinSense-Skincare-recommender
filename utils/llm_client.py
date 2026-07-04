"""Client for calling a configurable OpenAI-compatible local endpoint."""

import os
import re
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests

SYSTEM_MESSAGE = (
    "You are SkinSense, a friendly skincare routine assistant. "
    "You only provide cosmetic, non-medical suggestions, and you never diagnose or "
    "treat medical conditions. You must follow the user's prompt instructions exactly."
)


class LLMConfigurationError(Exception):
    pass


class LLMServiceError(RuntimeError):
    pass


class LLMResponseError(RuntimeError):
    pass


REQUIRED_HEADINGS = (
    "## AM Routine",
    "## PM Routine",
    "## Extra Tips",
    "## Why these suggestions?",
    "## Citations",
)


def _get_base_url() -> str:
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:8001").strip()
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LLMConfigurationError(
            "LLM_BASE_URL must be a valid http:// or https:// URL."
        )
    return base_url.rstrip("/")


def _get_timeout() -> float:
    raw_timeout = os.getenv("LLM_TIMEOUT_SECONDS", "20")
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise LLMConfigurationError("LLM_TIMEOUT_SECONDS must be numeric.") from exc
    if timeout <= 0:
        raise LLMConfigurationError("LLM_TIMEOUT_SECONDS must be greater than zero.")
    return timeout


def validate_answer_structure(content: Any) -> str:
    """Return a normalized answer or raise for malformed model output."""
    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError("The LLM returned an empty response.")
    normalized = content.strip()
    missing = [heading for heading in REQUIRED_HEADINGS if heading not in normalized]
    if missing:
        raise LLMResponseError(
            "The LLM response was missing required routine sections."
        )
    citation_section = normalized.split("## Citations", maxsplit=1)[-1]
    if not re.search(r"\bR\d{3}\b", citation_section):
        raise LLMResponseError("The LLM response did not include rule citations.")
    return normalized


def call_llm(prompt: str) -> str:
    """Call the configured endpoint and return a validated skincare routine."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    base_url = _get_base_url()
    api_key = os.getenv("LLM_API_KEY", "dummy").strip() or "dummy"
    model = os.getenv("LLM_MODEL", "skinsense-local").strip() or "skinsense-local"
    payload = {
        "model": model,
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=_get_timeout(),
        )
    except requests.RequestException as exc:
        raise LLMServiceError(
            f"The LLM service is unavailable at {base_url}."
        ) from exc

    if response.status_code != 200:
        raise LLMServiceError(
            f"The LLM service returned HTTP {response.status_code}."
        )
    try:
        data = response.json()
        choices = data["choices"]
        content = choices[0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMResponseError(
            "The LLM service returned an incompatible response."
        ) from exc
    return validate_answer_structure(content)


def check_llm_health() -> Dict[str, Any]:
    """Return a small status object without raising into the Streamlit UI."""
    try:
        base_url = _get_base_url()
        response = requests.get(f"{base_url}/health", timeout=min(_get_timeout(), 3))
        if response.status_code != 200:
            return {"available": False, "backend": "unavailable", "base_url": base_url}
        data = response.json()
        return {
            "available": data.get("status") == "ok",
            "backend": str(data.get("backend") or "unknown"),
            "base_url": base_url,
        }
    except (requests.RequestException, ValueError, LLMConfigurationError):
        return {
            "available": False,
            "backend": "unavailable",
            "base_url": os.getenv("LLM_BASE_URL", "http://localhost:8001"),
        }


def fallback_answer(rule_ids: List[str]) -> str:
    """
    Provide a deterministic, local-only markdown response if the LLM is not available.
    """
    valid_rule_ids = list(
        dict.fromkeys(
            rid for rid in rule_ids if isinstance(rid, str) and re.fullmatch(r"R\d{3}", rid)
        )
    )
    rule_list = ", ".join(valid_rule_ids) if valid_rule_ids else "N/A"
    md = f"""
_A local fallback routine is shown because the configured language service is unavailable._

**Retrieved rule IDs:** {rule_list}

## AM Routine
- Rinse face with lukewarm water and a gentle, non-stripping cleanser.
- Pat skin dry and apply a hydrating serum.
- Follow with a comfortable moisturizer that feels good on your skin.
- Finish with a broad SPF 30+ sunscreen according to the label instructions.

## PM Routine
- Gently cleanse to remove sunscreen, makeup, and daily buildup.
- Apply a hydrating serum or essence, focusing on areas that feel dry or tight.
- Seal in hydration with a moisturizer that does not feel heavy or irritating.

## Extra Tips
- Keep your routine simple and consistent instead of frequently changing products.
- If a product stings, burns, or makes your skin very uncomfortable, stop using it and return to a very basic routine.
- Introduce only one product at a time so you can clearly see how your skin responds.

## Why these suggestions?
This routine focuses on gentle cleansing, comfortable hydration, and everyday sun protection.
It stays away from strong or complicated product combinations so it can fit many skin types
and concerns while you learn what your skin enjoys.

## Citations
Used: {rule_list}
"""
    return md.strip()
