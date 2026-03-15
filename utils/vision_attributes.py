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

        skin_type = data.get("skin_type")
        concerns = data.get("concerns")
        notes = data.get("notes")

        # Normalize concerns to a list of strings
        if isinstance(concerns, str):
            concerns_list = [c.strip() for c in concerns.split(",") if c.strip()]
        elif isinstance(concerns, list):
            concerns_list = [str(c).strip() for c in concerns if str(c).strip()]
        else:
            concerns_list = []

        result: Dict[str, Any] = {}
        if skin_type:
            result["skin_type"] = str(skin_type).strip().lower()
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

