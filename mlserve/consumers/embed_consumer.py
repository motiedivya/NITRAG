"""
embed_consumer.py — Embedding consumer for mlcpu (CPU-only, no GPU).

Topic   : nitrag.embed.request
Payload : {texts: [str, ...], model: str, normalize: bool}
Result  : {embeddings: [[float, ...], ...], model: str, dim: int}

Env vars
--------
NSQ_NSQD_TCP       comma-separated nsqd TCP addresses (default: 10.9.0.36:4150)
NSQ_LOOKUPD_HTTP   comma-separated nsqlookupd HTTP addresses
EMBED_MODEL        default model (default: nomic-ai/nomic-embed-text-v1.5)
EMBED_BATCH_SIZE   batch size (default: 64)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from .base import BaseConsumer

log = logging.getLogger(__name__)

TOPIC = "nitrag.embed.request"
DEFAULT_MODEL = os.environ.get("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
DEFAULT_BATCH = int(os.environ.get("EMBED_BATCH_SIZE", "64"))


class EmbedConsumer(BaseConsumer):
    def __init__(self, **kwargs) -> None:
        super().__init__(topic=TOPIC, channel="nitrag", max_in_flight=1, **kwargs)
        self._models: Dict[str, Any] = {}

    def _get_model(self, model_name: str):
        if model_name not in self._models:
            log.info("Loading embedding model: %s", model_name)
            from fastembed import TextEmbedding
            self._models[model_name] = TextEmbedding(model_name=model_name)
            log.info("Model loaded: %s", model_name)
        return self._models[model_name]

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        texts: List[str] = payload.get("texts") or []
        model_name: str = payload.get("model") or DEFAULT_MODEL
        normalize: bool = payload.get("normalize", True)
        batch_size: int = int(payload.get("batch_size") or DEFAULT_BATCH)

        if not texts:
            return {"embeddings": [], "model": model_name, "dim": 0}

        model = self._get_model(model_name)
        embeddings = list(model.embed(texts, batch_size=batch_size))
        vectors = [e.tolist() for e in embeddings]

        return {
            "embeddings": vectors,
            "model": model_name,
            "dim": len(vectors[0]) if vectors else 0,
            "count": len(vectors),
        }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.info("embed_consumer starting — topic=%s", TOPIC)
    EmbedConsumer().run()


if __name__ == "__main__":
    main()
