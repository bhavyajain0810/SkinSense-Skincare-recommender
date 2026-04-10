"""
Client for calling an OpenAI-compatible chat completions endpoint.
"""
import os
from typing import List
from openai import OpenAI

SYSTEM_MESSAGE = (
    "You are SkinSense, a friendly skincare routine assistant. "
    "You only provide cosmetic, non-medical suggestions, and you never diagnose or "
    "treat medical conditions. You must follow the user's prompt instructions exactly."
)


class LLMConfigurationError(Exception):
    pass


def call_llm(prompt: str) -> str:
    """
    Call the Groq API (OpenAI-compatible) and return the generated skincare routine.
    """
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    if not api_key:
        raise LLMConfigurationError(
            "Missing LLM_API_KEY. Please set it in your .env file."
        )

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        response = client.chat.completions.create(
            model=model,
            temperature=0.4,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Empty response from LLM")
        return content
    except LLMConfigurationError:
        raise
    except Exception as exc:
        raise RuntimeError(f"LLM API call failed: {exc}") from exc


def fallback_answer(rule_ids: List[str]) -> str:
    """
    Provide a deterministic, local-only markdown response if the LLM is not available.
    """
    rule_list = ", ".join(rule_ids) if rule_ids else "N/A"
    md = f"""
_⚠️ A local fallback template is being used because the live LLM endpoint is not available._

**Retrieved rule IDs:** {rule_list}

## 🌅 AM (Morning) Routine
- Rinse face with lukewarm water and a gentle, non-stripping cleanser.
- Pat skin dry and apply a hydrating serum.
- Follow with a comfortable moisturizer that feels good on your skin.
- Finish with a broad SPF 30+ sunscreen according to the label instructions.

## 🌙 PM (Night) Routine
- Gently cleanse to remove sunscreen, makeup, and daily buildup.
- Apply a hydrating serum or essence, focusing on areas that feel dry or tight.
- Seal in hydration with a moisturizer that does not feel heavy or irritating.

## 💡 Extra Tips
- Keep your routine simple and consistent instead of frequently changing products.
- If a product stings, burns, or makes your skin very uncomfortable, stop using it and return to a very basic routine.
- Introduce only one product at a time so you can clearly see how your skin responds.

## Why these suggestions?
This routine focuses on gentle cleansing, comfortable hydration, and everyday sun protection.
It stays away from strong or complicated product combinations so it can fit many skin types
and concerns while you learn what your skin enjoys.

## Citations
Used rules: {rule_list}
"""
    return md.strip()