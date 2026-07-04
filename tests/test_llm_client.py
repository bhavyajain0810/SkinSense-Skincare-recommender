from types import SimpleNamespace

import pytest
import requests

from utils import llm_client


VALID_ANSWER = """
One gentle routine.

## AM Routine
- Cleanse.

## PM Routine
- Cleanse.

## Extra Tips
- Patch test.

## Why these suggestions?
Rule R001 supports a simple routine.

## Citations
Used: R001
""".strip()


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data


def test_call_llm_uses_configured_base_url(monkeypatch):
    recorded = {}

    def fake_post(url, **kwargs):
        recorded["url"] = url
        recorded["payload"] = kwargs["json"]
        return FakeResponse({"choices": [{"message": {"content": VALID_ANSWER}}]})

    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8001/")
    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    answer = llm_client.call_llm("Use R001.")

    assert answer == VALID_ANSWER
    assert recorded["url"] == "http://127.0.0.1:8001/v1/chat/completions"
    assert recorded["payload"]["messages"][1]["content"] == "Use R001."


def test_unavailable_service_is_wrapped(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(llm_client.requests, "post", fail)
    with pytest.raises(llm_client.LLMServiceError, match="unavailable"):
        llm_client.call_llm("Use R001.")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": "No structured sections"}}]},
    ],
)
def test_malformed_response_is_rejected(monkeypatch, payload):
    monkeypatch.setattr(
        llm_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(payload),
    )
    with pytest.raises(llm_client.LLMResponseError):
        llm_client.call_llm("Use R001.")


def test_invalid_base_url_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "not-a-url")
    with pytest.raises(llm_client.LLMConfigurationError):
        llm_client.call_llm("Use R001.")


def test_fallback_has_required_structure_and_citations():
    answer = llm_client.fallback_answer(["R001", "bad", "R084", "R001"])
    assert llm_client.validate_answer_structure(answer) == answer
    for heading in llm_client.REQUIRED_HEADINGS:
        assert heading in answer
    assert "Used: R001, R084" in answer
    assert "medical" not in answer.lower()
