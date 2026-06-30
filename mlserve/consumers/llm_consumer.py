"""
llm_consumer.py — LLM consumer for 6kpro, wrapping the local vLLM OpenAI-compatible API.

Generic: any project that needs LLM generation without managing the HTTP client.

Topic   : nitrag.llm.request
Payload : {
    messages: [{role: str, content: str}, ...],
    model: str | null,         # default: Qwen/Qwen3-VL-8B-Instruct
    temperature: float,        # default 0.1
    max_tokens: int,           # default 2048
    stream: bool               # default false — streaming not supported via NSQ
  }
Result  : {
    content: str,
    model: str,
    usage: {prompt_tokens, completion_tokens, total_tokens}
  }

Env vars
--------
NSQ_NSQD_TCP       comma-separated nsqd TCP addresses (default: 10.9.0.36:4150)
NSQ_LOOKUPD_HTTP   comma-separated nsqlookupd HTTP addresses
VLLM_BASE_URL      vLLM base URL (default: http://localhost:8000/v1)
VLLM_API_KEY       API key for vLLM (default: dummy)
LLM_DEFAULT_MODEL  default model name (default: Qwen/Qwen3-VL-8B-Instruct)
LLM_TIMEOUT        request timeout seconds (default: 120)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from .base import BaseConsumer

log = logging.getLogger(__name__)

TOPIC = "nitrag.llm.request"
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "dummy")
DEFAULT_MODEL = os.environ.get("LLM_DEFAULT_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))


class LLMConsumer(BaseConsumer):
    def __init__(self, **kwargs) -> None:
        super().__init__(topic=TOPIC, channel="nitrag", max_in_flight=1, **kwargs)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, timeout=LLM_TIMEOUT)
            log.info("OpenAI client connected to %s", VLLM_BASE_URL)
        return self._client

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = payload.get("messages") or []
        model: str = payload.get("model") or DEFAULT_MODEL
        temperature: float = float(payload.get("temperature", 0.1))
        max_tokens: int = int(payload.get("max_tokens", 2048))

        if not messages:
            raise ValueError("payload.messages must be a non-empty list")

        client = self._get_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage

        return {
            "content": content,
            "model": response.model,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
        }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.info("llm_consumer starting — topic=%s, vllm=%s", TOPIC, VLLM_BASE_URL)
    LLMConsumer().run()


if __name__ == "__main__":
    main()
