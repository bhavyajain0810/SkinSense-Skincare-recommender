import requests
from fastapi.testclient import TestClient

import llm_api.main as api_module
from llm_api.main import app


client = TestClient(app)


def test_health_reports_mock_backend(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "mock")
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "backend": "mock"}


def test_chat_completion_has_openai_compatible_shape(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "mock")
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "skinsense-local",
            "messages": [
                {"role": "system", "content": "Stay cosmetic-only."},
                {"role": "user", "content": "Use rule cards R001 and R084."},
            ],
            "temperature": 0.4,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "skinsense-local"
    assert isinstance(body["created"], int)
    assert body["choices"][0]["index"] == 0
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "## AM Routine" in body["choices"][0]["message"]["content"]
    assert "Used: R001, R084" in body["choices"][0]["message"]["content"]
    assert set(body["usage"]) == {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }


def test_chat_completion_rejects_empty_messages(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "mock")
    response = client.post(
        "/v1/chat/completions",
        json={"model": "skinsense-local", "messages": []},
    )
    assert response.status_code == 422


def test_ollama_backend_forwards_request_and_wraps_response(monkeypatch):
    recorded = {}

    class FakeOllamaResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"message": {"role": "assistant", "content": "Local response."}}

    def fake_post(url, **kwargs):
        recorded["url"] = url
        recorded["payload"] = kwargs["json"]
        recorded["timeout"] = kwargs["timeout"]
        return FakeOllamaResponse()

    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "llama-test")
    monkeypatch.setattr(api_module.requests, "post", fake_post)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "skinsense-local",
            "messages": [{"role": "user", "content": "Use R001."}],
            "temperature": 0.4,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert recorded["url"] == "http://ollama.test:11434/api/chat"
    assert recorded["payload"] == {
        "model": "llama-test",
        "messages": [{"role": "user", "content": "Use R001."}],
        "stream": False,
    }
    assert recorded["timeout"] == 120
    assert body["model"] == "llama-test"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Local response.",
    }


def test_ollama_backend_unavailable_returns_502(monkeypatch):
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setattr(api_module.requests, "post", unavailable)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "skinsense-local",
            "messages": [{"role": "user", "content": "Use R001."}],
        },
    )

    assert response.status_code == 502
    assert "not reachable" in response.json()["detail"]
