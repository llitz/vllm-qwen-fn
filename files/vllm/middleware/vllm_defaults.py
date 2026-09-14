import codecs
import json
import os
import sys

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

class FrequencyPenaltyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method != "POST" or request.url.path not in {
            "/v1/chat/completions",
            "/v1/completions",
        }:
            return await call_next(request)

        body = await request.body()

        try:
            data = json.loads(body.decode("utf-8"))

            if isinstance(data, dict):
                had_value = "frequency_penalty" in data
                data.setdefault("frequency_penalty", 1.0)

                body = json.dumps(data, separators=(",", ":")).encode("utf-8")

                #print(
                #    f"[frequency-penalty-middleware] "
                #    f"path={request.url.path} "
                #    f"had_value={had_value} "
                #    f"frequency_penalty={data.get('frequency_penalty')}",
                #    file=sys.stderr,
                #    flush=True,
                #)

                async def receive():
                    return {
                        "type": "http.request",
                        "body": body,
                        "more_body": False,
                    }

                request = Request(request.scope, receive)

                # Replace content-length so downstream sees the modified body length.
                headers = [
                    (k, v)
                    for k, v in request.scope.get("headers", [])
                    if k.lower() != b"content-length"
                ]
                headers.append((b"content-length", str(len(body)).encode()))
                request.scope["headers"] = headers

        except Exception as e:
            print(
                f"[frequency-penalty-middleware] failed: {e}",
                file=sys.stderr,
                flush=True,
            )

        return await call_next(request)

class PresencePenaltyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method != "POST" or request.url.path not in {
            "/v1/chat/completions",
            "/v1/completions",
        }:
            return await call_next(request)

        body = await request.body()

        try:
            data = json.loads(body.decode("utf-8"))

            if isinstance(data, dict):
                had_value = "presence_penalty" in data
                data.setdefault("presence_penalty", 1.0)

                body = json.dumps(data, separators=(",", ":")).encode("utf-8")

                #print(
                #    f"[presence-penalty-middleware] "
                #    f"path={request.url.path} "
                #    f"had_value={had_value} "
                #    f"presence_penalty={data.get('presence_penalty')}",
                #    file=sys.stderr,
                #    flush=True,
                #)

                async def receive():
                    return {
                        "type": "http.request",
                        "body": body,
                        "more_body": False,
                    }

                request = Request(request.scope, receive)

                # Replace content-length so downstream sees the modified body length.
                headers = [
                    (k, v)
                    for k, v in request.scope.get("headers", [])
                    if k.lower() != b"content-length"
                ]
                headers.append((b"content-length", str(len(body)).encode()))
                request.scope["headers"] = headers

        except Exception as e:
            print(
                f"[presence-penalty-middleware] failed: {e}",
                file=sys.stderr,
                flush=True,
            )

        return await call_next(request)
