"""ASGI application entry point for model serving."""

from __future__ import annotations

import json


async def app(scope, receive, send) -> None:
    """Minimal ASGI app with a health endpoint."""
    if scope["type"] != "http":
        return

    path = scope.get("path", "/")
    status = 200 if path in {"/", "/health"} else 404
    payload = {"status": "ok"} if status == 200 else {"detail": "Not found"}
    body = json.dumps(payload).encode("utf-8")

    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})
