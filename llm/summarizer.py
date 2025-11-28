from __future__ import annotations

import os
from typing import List

from loguru import logger

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


def summarize_news(headlines: List[dict], model: str = "gpt-4.1-mini") -> str:
    if not OpenAI:
        return "OpenAI SDK not installed."
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY not set; skipping summary."
    client = OpenAI(api_key=api_key)
    text = "\n".join([f"- {h['title']}: {h.get('summary','')}" for h in headlines])
    prompt = (
        "You are a trading assistant. Summarize the BTC futures relevant news and sentiment in bullet points. "
        "Highlight risk events and macro tone. Keep it concise."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
            temperature=0.2,
            max_tokens=400,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        logger.error(f"OpenAI summarization failed: {exc}")
        return "LLM summarization failed."
