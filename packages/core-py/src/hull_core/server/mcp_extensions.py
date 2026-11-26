"""MCP 2026-07-28 extension helpers: headers, files, tasks, and elicitation fallback."""

from __future__ import annotations

import base64
import json
import mimetypes
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Final, Generic, Literal, TypeVar

from fastmcp.exceptions import ToolError
from fastmcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from fastmcp.utilities.tasks import (
    DEFAULT_POLL_INTERVAL_MS,
    TASKS_EXTENSION_ID,
    TaskConfig,
    TaskMeta,
    TaskMode,
)
from mcp.shared.exceptions import MCPError

T = TypeVar("T")

MCP_2026_07_28_HEADERS: Final[tuple[str, ...]] = (
    "Mcp-Method",
    "Mcp-Name",
    "Mcp-Protocol-Version",
    "Mcp-Session-Id",
)

FILES_EXTENSION_ID: Final[str] = "io.modelcontextprotocol/files"


def build_file_payload(
    data: bytes | bytearray | str | Mapping[str, Any] | Sequence[Any],
    filename: str,
    *,
    mime_type: str | None = None,
    uri: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Build a JSON/text/binary-base64 payload compatible with ``ext-files`` + fallback."""
    if not filename or not filename.strip():
        raise ValueError("filename must be non-empty")

    resolved_uri = uri or f"file:///{filename.lstrip('/')}"

    if isinstance(data, (Mapping, list, tuple)):
        serializable = list(data) if isinstance(data, tuple) else dict(data) if isinstance(data, Mapping) else data
        text = json.dumps(serializable)
        resolved_mime = mime_type or "application/json"
        size_bytes = len(text.encode("utf-8"))
        resource: dict[str, Any] = {"uri": resolved_uri, "mimeType": resolved_mime, "text": text}
    elif isinstance(data, str):
        guessed = mimetypes.guess_type(filename)[0]
        resolved_mime = mime_type or guessed or "text/plain; charset=utf-8"
        size_bytes = len(data.encode("utf-8"))
        resource = {"uri": resolved_uri, "mimeType": resolved_mime, "text": data}
    elif isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
        guessed = mimetypes.guess_type(filename)[0]
        resolved_mime = mime_type or guessed or "application/octet-stream"
        size_bytes = len(raw)
        resource = {
            "uri": resolved_uri,
            "mimeType": resolved_mime,
            "blob": base64.b64encode(raw).decode("ascii"),
        }
    else:
        raise TypeError(f"Unsupported file payload type: {type(data).__name__}")

    payload: dict[str, Any] = {
        "extension": FILES_EXTENSION_ID,
        "filename": filename,
        "mimeType": resolved_mime,
        "sizeBytes": size_bytes,
        "resource": resource,
        "_meta": {
            FILES_EXTENSION_ID: {
                "filename": filename,
                "mimeType": resolved_mime,
                "sizeBytes": size_bytes,
                "uri": resolved_uri,
            }
        },
    }
    if description is not None:
        payload["description"] = description
    return payload


def build_task_config(
    mode: TaskMode = "optional",
    *,
    poll_interval_ms: int = DEFAULT_POLL_INTERVAL_MS,
) -> TaskConfig:
    """Create a validated :class:`TaskConfig` wrapping ``fastmcp.utilities.tasks``."""
    if mode not in ("forbidden", "optional", "required"):
        raise ValueError(f"Invalid task mode: {mode!r}")
    if poll_interval_ms <= 0:
        raise ValueError("poll_interval_ms must be positive")
    return TaskConfig(mode=mode, poll_interval=timedelta(milliseconds=poll_interval_ms))


@dataclass(frozen=True, slots=True)
class ElicitationOutcome(Generic[T]):
    """Structured outcome returned by :func:`elicit_with_fallback`."""

    status: Literal["accepted", "declined", "cancelled", "fallback"]
    data: T | None
    used_fallback: bool = False
    reason: str | None = None


async def elicit_with_fallback(
    ctx: Any,
    message: str,
    response_type: Any,
    *,
    default: T | None = None,
    response_title: str | None = None,
    response_description: str | None = None,
) -> ElicitationOutcome[Any]:
    """Invoke ``ctx.elicit(...)`` and fall back gracefully when unsupported or headless."""
    if ctx is None or not hasattr(ctx, "elicit"):
        return ElicitationOutcome(
            status="fallback",
            data=default,
            used_fallback=True,
            reason="no_elicitation_context",
        )

    try:
        result = await ctx.elicit(
            message,
            response_type,
            response_title=response_title,
            response_description=response_description,
        )
    except (ToolError, MCPError, RuntimeError, NotImplementedError) as exc:
        return ElicitationOutcome(
            status="fallback",
            data=default,
            used_fallback=True,
            reason=str(exc) or type(exc).__name__,
        )

    if isinstance(result, AcceptedElicitation):
        return ElicitationOutcome(status="accepted", data=result.data, used_fallback=False)
    if isinstance(result, DeclinedElicitation):
        return ElicitationOutcome(status="declined", data=None, used_fallback=False, reason="declined")
    if isinstance(result, CancelledElicitation):
        return ElicitationOutcome(status="cancelled", data=None, used_fallback=False, reason="cancelled")
    raise TypeError(f"Unexpected elicitation result: {type(result).__name__}")


__all__ = [
    "FILES_EXTENSION_ID",
    "MCP_2026_07_28_HEADERS",
    "TASKS_EXTENSION_ID",
    "ElicitationOutcome",
    "TaskConfig",
    "TaskMeta",
    "TaskMode",
    "build_file_payload",
    "build_task_config",
    "elicit_with_fallback",
]
