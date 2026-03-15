import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


load_dotenv()  # Load llm_api/.env when running locally; Docker uses env_file.

app = FastAPI(title="SkinSense LLM API", version="1.0.0")


def get_backend() -> str:
    return os.getenv("LLM_BACKEND", "mock").lower()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.4)


class ChatCompletionChoiceMessage(BaseModel):
    role: str
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Simple health endpoint that reports the active backend.
    """
    backend = get_backend()
    return {"status": "ok", "backend": backend}


def _extract_rule_ids(prompt: str) -> List[str]:
    """
    Best-effort extraction of rule IDs like R001 from the prompt.
    """
    return sorted(set(re.findall(r"R\d{3}", prompt)))


def _mock_completion(req: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Deterministic, local-only backend that returns a safe skincare routine.
    """
    user_content = ""
    for msg in req.messages:
        if msg.role == "user":
            user_content = msg.content
    rule_ids = _extract_rule_ids(user_content)
    rule_str = ", ".join(rule_ids) if rule_ids else "N/A"

    content = f"""
Here is a gentle, cosmetic-only routine based on the rule cards you provided.

## AM Routine
- Rinse your face with lukewarm water and a mild, non-stripping cleanser.
- Pat the skin dry with a soft towel, avoiding harsh rubbing.
- Apply a light hydrating layer such as a toner, essence, or serum.
- Follow with a comfortable moisturizer that does not feel heavy.
- Finish with a broad cosmetic sunscreen according to the product label.

## PM Routine
- Gently cleanse to remove sunscreen, makeup, and daily buildup.
- If your skin feels comfortable, apply a hydrating serum or essence.
- Seal everything in with a moisturizer that leaves your skin feeling soft but not greasy.

## Extra Tips
- Introduce only one new cosmetic product at a time so you can clearly see how your skin reacts.
- If any product stings, burns, or makes your skin very uncomfortable, stop using it and return to a very simple routine.
- Avoid over-exfoliating; if your skin looks shiny, tight, or very sensitive, reduce exfoliating products and focus on moisture.
- Remember that these ideas are cosmetic and educational only, not medical advice.

## Why these suggestions?
The steps focus on gentle cleansing, steady hydration, and everyday cosmetic sun protection. This keeps the routine simple and easy to adjust while you observe how your skin feels over time.

## Citations
Used: {rule_str}
""".strip()

    return ChatCompletionResponse(
        id="chatcmpl-mock-1",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionChoiceMessage(role="assistant", content=content),
            )
        ],
    )


def _ollama_completion(req: ChatCompletionRequest) -> ChatCompletionResponse:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", req.model)

    # Ollama expects messages in a similar format, but we call its /api/chat.
    payload = {
        "model": model,
        "messages": [m.dict() for m in req.messages],
        "stream": False,
    }

    try:
        resp = requests.post(
            base_url.rstrip("/") + "/api/chat",
            json=payload,
            timeout=120,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama backend not reachable at {base_url}: {exc}",
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Ollama backend error: {resp.text[:500]}",
        )

    data = resp.json()
    msg = data.get("message") or {}
    content = msg.get("content") or ""

    return ChatCompletionResponse(
        id=f"chatcmpl-ollama-{int(time.time())}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionChoiceMessage(role="assistant", content=content),
            )
        ],
    )


def _openai_proxy_completion(req: ChatCompletionRequest) -> ChatCompletionResponse:
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set for openai backend.",
        )

    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = req.dict()

    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI backend not reachable at {base_url}: {exc}",
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"OpenAI backend error: {resp.text[:500]}",
        )

    # We simply return the backend's JSON to keep full compatibility.
    data = resp.json()
    # Ensure required fields for our client; if it's already present, FastAPI
    # will serialize as-is.
    return data  # type: ignore[return-value]


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Backends:
    - mock   (default) – deterministic skincare routine.
    - ollama – forwards to Ollama /api/chat.
    - openai – proxies to an OpenAI-compatible Chat Completions endpoint.
    """
    backend = get_backend()

    if backend == "mock":
        return _mock_completion(req)
    if backend == "ollama":
        return _ollama_completion(req)
    if backend == "openai":
        return _openai_proxy_completion(req)

    # Fallback to mock if unknown backend to keep the system usable.
    return _mock_completion(req)

