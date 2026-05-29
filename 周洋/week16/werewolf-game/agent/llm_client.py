"""LLM API client — calls Qwen (通义千问) via DashScope OpenAI-compatible endpoint."""

from openai import OpenAI
from configs.llm_config import LLM_CONFIG

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"],
        )
    return _client


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Synchronous LLM call. Returns the text content of the response."""
    client = _get_client()
    resp = client.chat.completions.create(
        model=LLM_CONFIG["model"],
        max_tokens=LLM_CONFIG["max_tokens"],
        temperature=LLM_CONFIG["temperature"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content or ""
