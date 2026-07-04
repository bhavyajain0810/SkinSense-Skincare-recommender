from fastapi.testclient import TestClient

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
