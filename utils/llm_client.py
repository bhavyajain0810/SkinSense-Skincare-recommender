"""
Client for calling an OpenAI-compatible chat completions endpoint.
"""

import os
from typing import List

import requests


SYSTEM_MESSAGE = (
    "You are SkinSense, a friendly skincare routine assistant. "
    "You only provide cosmetic, non-medical suggestions, and you never diagnose or "
    "treat medical conditions. You must follow the user's prompt instructions exactly."
)


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM endpoint is not configured."""


def _get_env(var: str, default: str | None = None) -> str | None:
    return os.getenv(var, default)


def call_llm(prompt: str) -> str:
    """
    Call the OpenAI-compatible endpoint and return the assistant content.

    Raises LLMConfigurationError if configuration is missing and RuntimeError
    if the call itself fails.
    """
    base_url = _get_env("LLM_BASE_URL")
    api_key = _get_env("LLM_API_KEY")
    model = _get_env("LLM_MODEL", "skinsense-local")

    if not base_url or not api_key or not model:
        raise LLMConfigurationError(
            "LLM endpoint is not fully configured. "
            "Set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL."
        )

    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
    except Exception as exc:  # pragma: no cover - network defensive
        raise RuntimeError(f"Failed to reach LLM endpoint at {url}: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"LLM endpoint error {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("LLM response missing 'choices'")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("LLM response missing message content")

    return content


def fallback_answer(rule_ids: List[str]) -> str:
    """
    Provide a deterministic, local-only markdown response if the LLM is not available.
    """
    rule_list = ", ".join(rule_ids) if rule_ids else "N/A"

    md = f"""
_A local fallback template is being used because the live LLM endpoint is not available._

**Retrieved rule IDs:** {rule_list}

## AM Routine
- Rinse face with lukewarm water and a gentle, non-stripping cleanser.
- Pat skin dry and apply a light hydrating layer such as a toner, essence, or serum.
- Follow with a comfortable moisturizer that feels good on your skin.
- Finish with a broad cosmetic sunscreen according to the label instructions.

## PM Routine
- Gently cleanse to remove sunscreen, makeup, and daily buildup.
- Apply a hydrating serum or essence, focusing on areas that feel dry or tight.
- Seal in hydration with a moisturizer that does not feel heavy or irritating.

## Extra Tips
- Keep your routine simple and consistent instead of frequently changing products.
- If a product stings, burns, or makes your skin very uncomfortable, stop using it and return to a very basic routine.
- Introduce only one new cosmetic product at a time so you can clearly see how your skin responds.

## Why these suggestions?
This routine focuses on gentle cleansing, comfortable hydration, and everyday cosmetic sun protection. It stays away from strong or complicated product combinations so it can fit many skin types and concerns while you learn what your skin enjoys.

## Citations
Used rules: {rule_list}
"""
    return md.strip()

