"""LLM answer generation via OpenRouter with free model rotation."""

import json
import time
from typing import Iterator

import requests

import config

FREE_MODEL_POOL = [
    "openrouter/free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "google/gemma-3-27b-it:free",
    "deepseek/deepseek-r1-distill-llama-70b:free",
    "qwen/qwen3-8b:free",
]

user_model = config.OPENROUTER_MODEL
if user_model != "openrouter/free":
    if user_model in FREE_MODEL_POOL:
        FREE_MODEL_POOL.remove(user_model)
    FREE_MODEL_POOL.insert(1, user_model)

_cooldowns: dict[str, float] = {}
COOLDOWN_SECONDS = 60

SYSTEM_PROMPT = """You are an expert Ultralytics YOLO code assistant helping ML engineers understand the library by analyzing its source code.

When answering:
1. Give complete answers in easy to understand language.
2. Reference the specific classes, methods, and file paths shown in the retrieved code
3. Include short, runnable code examples where helpful
4. Use the exact parameter names and types shown in the source
5. If the retrieved context is insufficient, say so clearly
6. Format code with Python syntax highlighting"""


def _is_available(model: str) -> bool:
    return _cooldowns.get(model, 0) < time.time()


def _mark_rate_limited(model: str) -> None:
    _cooldowns[model] = time.time() + COOLDOWN_SECONDS


def get_model_status() -> dict[str, str]:
    """Return availability status for each model in the pool."""
    now = time.time()
    return {
        model: "available" if _cooldowns.get(model, 0) < now
               else str(int(_cooldowns[model] - now))
        for model in FREE_MODEL_POOL
    }


def _build_messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"## Retrieved Source Code\n\n{context}\n\n---\n\n## Question\n\n{question}"},
    ]


def _call_model(model: str, messages: list[dict]) -> tuple:
    """
    Make one streaming API call to the given model.

    Returns (True, response) on success, (None, None) on 429, (False, error) on other failures.
    """
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ultralytics-assistant",
        "X-Title": "Ultralytics Code Assistant",
    }
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": config.LLM_MAX_TOKENS,
    }

    try:
        response = requests.post(
            config.OPENROUTER_BASE_URL,
            json=body,
            headers=headers,
            stream=True,
            timeout=60,
        )
        if response.status_code == 429:
            _mark_rate_limited(model)
            return None, None
        response.raise_for_status()
        return True, response
    except requests.RequestException as e:
        return False, str(e)


def _stream_response(response) -> Iterator[str]:
    """Parse SSE lines and yield text tokens."""
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            return
        try:
            token = json.loads(payload)["choices"][0]["delta"].get("content", "")
            if token:
                yield token
        except (json.JSONDecodeError, KeyError, IndexError):
            continue


def stream_answer(question: str, context: str) -> Iterator[str]:
    """
    Stream an answer using the first available model in the pool.

    Tries each model once. On 429, moves to the next model without retrying the same one.
    """
    if not config.OPENROUTER_API_KEY:
        yield "**Configuration error**: `OPENROUTER_API_KEY` is not set in your `.env` file."
        return

    messages = _build_messages(question, context)
    tried_any = False

    for model in FREE_MODEL_POOL:
        if not _is_available(model):
            continue

        tried_any = True
        yield f"⚡ Using `{model.split('/')[-1]}`\n\n"

        success, result = _call_model(model, messages)

        if success is None:
            yield f"⚠️ `{model.split('/')[-1]}` rate-limited, trying next model…\n\n"
            continue

        if success is False:
            yield f"\n\n**API error**: {result}"
            return

        yield from _stream_response(result)
        return

    if not tried_any:
        soonest = min(FREE_MODEL_POOL, key=lambda m: _cooldowns.get(m, 0))
        wait = int(_cooldowns[soonest] - time.time()) + 1
        yield f"\n\n⏳ All models are rate-limited. Please wait **{wait} seconds** and try again."
    else:
        yield "\n\n⚠️ All models returned rate-limit errors. Please wait ~60 seconds and try again."


def get_full_answer(question: str, context: str) -> str:
    """Non-streaming version for CLI and tests."""
    return "".join(
        t for t in stream_answer(question, context)
        if not any(t.startswith(p) for p in ["⚡", "⏳", "⚠️"])
    )
