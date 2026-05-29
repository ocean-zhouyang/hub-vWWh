"""LLM configuration — reads from environment variables with sensible defaults.

Uses Alibaba Cloud DashScope API (通义千问) via OpenAI-compatible endpoint.
"""

import os

DASHSCOPE_API_KEY: str = os.environ.get("QWEN_API_KEY", "")

LLM_CONFIG = {
    "api_key": DASHSCOPE_API_KEY,
    "base_url": os.environ.get(
        "WEREWOLF_LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "model": os.environ.get("WEREWOLF_LLM_MODEL", "qwen-flash"),
    "max_tokens": int(os.environ.get("WEREWOLF_LLM_MAX_TOKENS", "1024")),
    "temperature": float(os.environ.get("WEREWOLF_LLM_TEMPERATURE", "0.7")),
}


def is_llm_available() -> bool:
    return bool(DASHSCOPE_API_KEY)
