"""CLI entry point for hull-embedding-daemon."""

from __future__ import annotations

import argparse
import sys

from hull_embedding_daemon._extra import require_embedding_extra


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="hull-embedding-daemon",
        description="Shared ONNX/GGUF embedding server for the n24q02m MCP ecosystem",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9800,
        help="Bind port (default: 9800)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="uvicorn log level (default: info)",
    )
    args = parser.parse_args()

    try:
        require_embedding_extra()
    except ImportError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    import uvicorn

    uvicorn.run(
        "hull_embedding_daemon.api:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
