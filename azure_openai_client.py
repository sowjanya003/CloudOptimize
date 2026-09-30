from __future__ import annotations
import os
from typing import Optional
from keyvault_client import get_secret

try:
    from openai import AsyncAzureOpenAI
    _OPENAI_IMPORTED = True
except ImportError:
    _OPENAI_IMPORTED = False

_client: Optional["AsyncAzureOpenAI"] = None

REQUIRED_ENV_VARS = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_GPT5_DEPLOYMENT",
)


def azure_gpt5_available() -> bool:
    return _OPENAI_IMPORTED and all(get_secret(v) for v in REQUIRED_ENV_VARS)


def _get_client() -> "AsyncAzureOpenAI":
    global _client
    if _client is None:
        if not _OPENAI_IMPORTED:
            raise RuntimeError(
                "openai package is not installed. Install with: pip install openai"
            )
        _client = AsyncAzureOpenAI(
             azure_endpoint=get_secret("AZURE_OPENAI_ENDPOINT", required=True),
            api_key=get_secret("AZURE_OPENAI_API_KEY", required=True),
            api_version=get_secret("AZURE_OPENAI_API_VERSION", default="2024-10-21"),
        )
    return _client

last_call_meta: dict = {}


async def azure_gpt5_caller(system_prompt: str, user_prompt: str) -> str:
    global last_call_meta
    client = _get_client()
    deployment = get_secret("AZURE_OPENAI_GPT5_DEPLOYMENT", required=True)  # deployment name, not the model name
    response = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    usage = getattr(response, "usage", None)
    last_call_meta = {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "stop_reason": (response.choices[0].finish_reason if response.choices else None),
        "deployment": deployment,
    }
    return response.choices[0].message.content or ""