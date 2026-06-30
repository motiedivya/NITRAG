"""
base.py — Generic NSQ consumer base class for mlserve.

Protocol
--------
Every consumer subscribes to a request topic and publishes to a per-request
reply topic carried inside the message:

  Request  (JSON): {request_id, reply_topic, payload: {...}}
  Response (JSON): {request_id, ok: bool, result: {...} | null, error: str | null}

Subclasses only implement `process(payload) -> dict`.
Retry, error handling, logging, and NSQ mechanics are handled here.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
import uuid
from typing import Any, Dict, Optional

import nsq

log = logging.getLogger(__name__)


class BaseConsumer:
    """
    Reusable NSQ consumer base.

    Parameters
    ----------
    topic       : NSQ topic to subscribe to
    channel     : NSQ channel name (default: "nitrag")
    nsqd_tcp    : list of nsqd TCP addresses, e.g. ["10.9.0.36:4150"]
    lookupd_http: list of nsqlookupd HTTP addresses (takes precedence over nsqd_tcp
                  if both provided — use lookupd in production)
    max_in_flight: max messages held simultaneously
    """

    def __init__(
        self,
        topic: str,
        channel: str = "nitrag",
        nsqd_tcp: Optional[list] = None,
        lookupd_http: Optional[list] = None,
        max_in_flight: int = 1,
    ) -> None:
        self.topic = topic
        self.channel = channel
        self.max_in_flight = max_in_flight

        self._nsqd_tcp = nsqd_tcp or _from_env("NSQ_NSQD_TCP", ["10.9.0.36:4150"])
        self._lookupd_http = lookupd_http or _from_env("NSQ_LOOKUPD_HTTP", [])

        self._writer: Optional[nsq.Writer] = None
        self._reader: Optional[nsq.Reader] = None

    # ── Subclass interface ────────────────────────────────────────────────────

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Override in subclass. Return result dict on success, raise on failure."""
        raise NotImplementedError

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info("Starting consumer topic=%s channel=%s", self.topic, self.channel)

        self._writer = nsq.Writer(nsqd_tcp_addresses=self._nsqd_tcp)

        reader_kwargs: dict = {
            "topic": self.topic,
            "channel": self.channel,
            "message_handler": self._handle,
            "max_in_flight": self.max_in_flight,
        }
        if self._lookupd_http:
            reader_kwargs["lookupd_http_addresses"] = self._lookupd_http
        else:
            reader_kwargs["nsqd_tcp_addresses"] = self._nsqd_tcp

        self._reader = nsq.Reader(**reader_kwargs)
        nsq.run()

    # ── Internal handler ──────────────────────────────────────────────────────

    def _handle(self, message: nsq.Message) -> None:
        try:
            body = json.loads(message.body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.error("Unparseable message: %s", exc)
            message.finish()
            return

        request_id: str = body.get("request_id") or str(uuid.uuid4())
        reply_topic: Optional[str] = body.get("reply_topic")
        payload: Dict[str, Any] = body.get("payload") or {}

        log.info("Received request_id=%s topic=%s", request_id, self.topic)

        try:
            result = self.process(payload)
            response = {"request_id": request_id, "ok": True, "result": result, "error": None}
        except Exception as exc:
            log.error("Processing failed request_id=%s: %s", request_id, exc)
            log.debug(traceback.format_exc())
            response = {"request_id": request_id, "ok": False, "result": None, "error": str(exc)}

        if reply_topic:
            message.enable_async()
            response_bytes = json.dumps(response).encode()
            self._writer.pub(
                reply_topic,
                response_bytes,
                callback=lambda _conn, _data, msg=message: msg.finish(),
            )
        else:
            message.finish()

        log.info("Finished request_id=%s ok=%s", request_id, response["ok"])


def _from_env(key: str, default: list) -> list:
    val = os.environ.get(key, "")
    return [v.strip() for v in val.split(",") if v.strip()] if val.strip() else default
