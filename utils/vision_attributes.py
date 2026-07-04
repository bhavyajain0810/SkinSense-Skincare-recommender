"""
Optional vision attribute extraction for uploaded images.

If the environment variable VISION_ATTR_URL is set, we send a JSON payload:
    {"image_base64": "..."}
and expect a JSON response like:
    {"skin_type": "oily", "concerns": ["acne", "texture"], "notes": "T-zone looks shiny"}

If anything fails, we return None and the Streamlit app can safely ignore it.
"""

import base64
import os
from typing import Any, Dict, Optional

import requests

from utils.validation import CONCERNS, SKIN_TYPES


ALLOWED_CONCERNS = set(CONCERNS)


def _normalize_concerns(value: Any) -> list[str]:
    """
    Normalize concerns from the vision API into a filtered list of known values.
    """
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, list):
        raw_items = value
    else:
        return []

    normalized: list[str] = []
    seen: set[str] = set()

    for item in raw_items:
        concern = str(item).strip().lower()
        if not concern or concern not in ALLOWED_CONCERNS or concern in seen:
            continue
        normalized.append(concern)
        seen.add(concern)

    return normalized

def detect_from_image(image_bytes: bytes) -> Optional[Dict[str, Any]]:
    url = os.getenv("VISION_ATTR_URL")
    if not url:
        return None

    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"image_base64": b64}
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not isinstance(data, dict):
            return None
        
        skin_type = data.get("skin_type")
        concerns = data.get("concerns")
        notes = data.get("notes")
        
        concerns_list = _normalize_concerns(concerns)

        result: Dict[str, Any] = {}
        normalized_skin_type = str(skin_type or "").strip().lower()
        if normalized_skin_type in SKIN_TYPES:
            result["skin_type"] = normalized_skin_type
        if concerns_list:
            result["concerns"] = concerns_list
        if isinstance(notes, str) and notes.strip():
            result["notes"] = notes.strip()

        if not result:
            return None
        return result
    except Exception:
        # Fail silently so the rest of the app still works.
        return None

