"""
chunk_consumer.py — Text chunking consumer for mlcpu (CPU-only, no GPU).

Splits plain text into overlapping chunks suitable for RAG pipelines.
Generic: works for any project that needs text chunked by tokens.

Topic   : nitrag.chunk.request
Payload : {
    text: str,
    strategy: "fixed_tokens" | "sentence" | "paragraph",
    chunk_size: int,       # tokens (default 800)
    overlap: int,          # tokens (default 80)
    model: str             # tiktoken model for token counting (default "gpt-4o")
  }
Result  : {chunks: [{text, start_char, end_char, token_count}, ...], strategy: str}

Env vars
--------
NSQ_NSQD_TCP      comma-separated nsqd TCP addresses
NSQ_LOOKUPD_HTTP  comma-separated nsqlookupd HTTP addresses
CHUNK_SIZE        default chunk size in tokens (default: 800)
CHUNK_OVERLAP     default overlap in tokens (default: 80)
TIKTOKEN_MODEL    default tiktoken model (default: gpt-4o)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List

from .base import BaseConsumer

log = logging.getLogger(__name__)

TOPIC = "nitrag.chunk.request"
DEFAULT_CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
DEFAULT_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "80"))
DEFAULT_TIK_MODEL = os.environ.get("TIKTOKEN_MODEL", "gpt-4o")


class ChunkConsumer(BaseConsumer):
    def __init__(self, **kwargs) -> None:
        super().__init__(topic=TOPIC, channel="nitrag", max_in_flight=4, **kwargs)
        self._encoders: Dict[str, Any] = {}

    def _encoder(self, model: str):
        if model not in self._encoders:
            import tiktoken
            try:
                self._encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                self._encoders[model] = tiktoken.get_encoding("cl100k_base")
        return self._encoders[model]

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text: str = payload.get("text") or ""
        strategy: str = payload.get("strategy") or "fixed_tokens"
        chunk_size: int = int(payload.get("chunk_size") or DEFAULT_CHUNK_SIZE)
        overlap: int = int(payload.get("overlap") or DEFAULT_OVERLAP)
        model: str = payload.get("model") or DEFAULT_TIK_MODEL

        if not text.strip():
            return {"chunks": [], "strategy": strategy}

        if strategy == "sentence":
            chunks = self._sentence_chunks(text, chunk_size, overlap, model)
        elif strategy == "paragraph":
            chunks = self._paragraph_chunks(text, chunk_size, overlap, model)
        else:
            chunks = self._fixed_token_chunks(text, chunk_size, overlap, model)

        return {"chunks": chunks, "strategy": strategy, "total": len(chunks)}

    def _fixed_token_chunks(
        self, text: str, size: int, overlap: int, model: str
    ) -> List[Dict[str, Any]]:
        enc = self._encoder(model)
        tokens = enc.encode(text)
        step = max(1, size - overlap)
        chunks = []
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + size]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append({
                "text": chunk_text,
                "token_start": i,
                "token_end": i + len(chunk_tokens),
                "token_count": len(chunk_tokens),
            })
        return chunks

    def _sentence_chunks(
        self, text: str, size: int, overlap: int, model: str
    ) -> List[Dict[str, Any]]:
        enc = self._encoder(model)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return self._pack_units(sentences, size, overlap, enc)

    def _paragraph_chunks(
        self, text: str, size: int, overlap: int, model: str
    ) -> List[Dict[str, Any]]:
        enc = self._encoder(model)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return self._pack_units(paragraphs, size, overlap, enc)

    def _pack_units(self, units: List[str], size: int, overlap: int, enc) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        current_units: List[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = len(enc.encode(unit))
            if current_tokens + unit_tokens > size and current_units:
                chunk_text = " ".join(current_units)
                chunks.append({
                    "text": chunk_text,
                    "token_count": current_tokens,
                })
                # keep overlap: drop leading units until we're under overlap budget
                while current_units and current_tokens > overlap:
                    removed = current_units.pop(0)
                    current_tokens -= len(enc.encode(removed))
            current_units.append(unit)
            current_tokens += unit_tokens

        if current_units:
            chunks.append({"text": " ".join(current_units), "token_count": current_tokens})

        return chunks


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.info("chunk_consumer starting — topic=%s", TOPIC)
    ChunkConsumer().run()


if __name__ == "__main__":
    main()
