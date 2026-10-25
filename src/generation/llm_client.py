"""Thin client for Groq's OpenAI-compatible chat-completions endpoint.

Only used when GROQ_API_KEY is set (see src/config.py). Isolated here so
the rest of the generation module does not need to know which provider
or SDK is behind `generate()`.
"""
import requests

from src.config import CFG


class LLMUnavailableError(Exception):
    pass


def chat_complete(system_prompt: str, user_prompt: str) -> str:
    if not CFG.use_llm:
        raise LLMUnavailableError("GROQ_API_KEY not set")

    url = f"{CFG.groq_api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {CFG.groq_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CFG.groq_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": CFG.llm_temperature,
        "max_tokens": CFG.llm_max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=CFG.llm_timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
