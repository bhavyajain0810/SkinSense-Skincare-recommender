import base64

import requests

from utils import vision_attributes


class FakeVisionResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


def test_vision_integration_is_optional(monkeypatch):
    monkeypatch.delenv("VISION_ATTR_URL", raising=False)
    monkeypatch.setattr(
        vision_attributes.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("HTTP should not be called")
        ),
    )

    assert vision_attributes.detect_from_image(b"image") is None


def test_vision_response_is_normalized_and_filtered(monkeypatch):
    recorded = {}

    def fake_post(url, **kwargs):
        recorded["url"] = url
        recorded["payload"] = kwargs["json"]
        recorded["timeout"] = kwargs["timeout"]
        return FakeVisionResponse(
            {
                "skin_type": " OILY ",
                "concerns": ["Acne", "texture", "unknown", "acne"],
                "notes": "  Keep it simple.  ",
            }
        )

    monkeypatch.setenv("VISION_ATTR_URL", "http://vision.test/analyze")
    monkeypatch.setattr(vision_attributes.requests, "post", fake_post)

    result = vision_attributes.detect_from_image(b"local-image")

    assert recorded == {
        "url": "http://vision.test/analyze",
        "payload": {
            "image_base64": base64.b64encode(b"local-image").decode("utf-8")
        },
        "timeout": 30,
    }
    assert result == {
        "skin_type": "oily",
        "concerns": ["acne", "texture"],
        "notes": "Keep it simple.",
    }


def test_invalid_vision_profile_returns_only_safe_values(monkeypatch):
    monkeypatch.setenv("VISION_ATTR_URL", "http://vision.test/analyze")
    monkeypatch.setattr(
        vision_attributes.requests,
        "post",
        lambda *args, **kwargs: FakeVisionResponse(
            {
                "skin_type": "medical-condition",
                "concerns": ["unknown"],
                "notes": "",
            }
        ),
    )

    assert vision_attributes.detect_from_image(b"image") is None


def test_vision_network_failure_is_non_fatal(monkeypatch):
    monkeypatch.setenv("VISION_ATTR_URL", "http://vision.test/analyze")
    monkeypatch.setattr(
        vision_attributes.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            requests.ConnectionError("offline")
        ),
    )

    assert vision_attributes.detect_from_image(b"image") is None
