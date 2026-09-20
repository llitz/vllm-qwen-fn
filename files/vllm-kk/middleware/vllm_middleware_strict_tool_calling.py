"""Default ``strict: true`` onto function tools in /v1/chat/completions requests
when the ``strict`` key is absent or explicitly null. Explicit ``true``/
``false`` values are never overridden.

Why
---
With ``tool_choice=auto`` and no strict tool, vLLM applies NO grammar constraint:
``structural_tag_registry.py:113`` only arms a structural tag when tool_choice is
``"required"``/a named tool, or when at least one tool definition carries
``strict: true`` (the ``auto`` + all-non-strict combination returns ``None``).
That leaves tool-call emission to unconstrained sampling.

Fix rationale
-------------
Setting ``strict: true`` arms the xgrammar ``qwen_xml`` structural tag, so the
entire tool-call emission — including JSON parameter contents — becomes
grammar-forced. Malformed emission becomes unrepresentable: raw (unescaped)
newlines inside JSON string values, wrong schema keys, missing required
parameters, and similar structural errors can no longer be sampled.

Schema-rejection note
---------------------
xgrammar's strict profile supports ``anyOf`` and the common JSON-Schema subset;
if a particular tool schema is rejected, the error surfaces as a loud request
error on the first tool turn. This middleware never touches schemas — it only
sets the ``strict`` flag.

Safety
------
Only POST /v1/chat/completions is intercepted. If the body is not valid JSON,
or nothing needs changing, the original request passes through untouched.
Payloads are never logged (they contain API keys and tools); a one-line stderr
note is printed on the first successful mutation per process only.
"""

import json
import sys

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_armed_note_printed = False


class StrictToolCallingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method != "POST" or request.url.path != "/v1/chat/completions":
            return await call_next(request)

        body = await request.body()

        try:
            data = json.loads(body.decode("utf-8"))

            if isinstance(data, dict):
                tools = data.get("tools")
                tool_choice = data.get("tool_choice")

                changed = False
                armed_count = 0

                if isinstance(tools, list) and tool_choice != "none":
                    for tool in tools:
                        if (
                            isinstance(tool, dict)
                            and tool.get("type") == "function"
                            and isinstance(tool.get("function"), dict)
                        ):
                            function = tool["function"]
                            armed_count += 1
                            if function.get("strict") is None:
                                function["strict"] = True
                                changed = True

                if changed:
                    body = json.dumps(data, separators=(",", ":")).encode("utf-8")

                    global _armed_note_printed
                    if not _armed_note_printed:
                        _armed_note_printed = True
                        print(
                            f"[tool-strict] armed grammar constraint on {armed_count} tools (once)",
                            file=sys.stderr,
                            flush=True,
                        )

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
                else:
                    # Nothing changed: replay the original bytes so the consumed
                    # stream does not starve downstream readers.
                    async def receive():
                        return {
                            "type": "http.request",
                            "body": body,
                            "more_body": False,
                        }

                    request = Request(request.scope, receive)

        except Exception:
            # Invalid JSON or unexpected shape: pass the original request through.
            pass

        return await call_next(request)
