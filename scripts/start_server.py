"""Start the NITRAG FastAPI server.

Usage:
    uv run python scripts/start_server.py [--port 8000] [--reload]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="NITRAG Medical RAG server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not found. Install with: uv pip install 'uvicorn[standard]'")
        sys.exit(1)

    print(f"\n  NITRAG Medical RAG\n  http://{args.host}:{args.port}\n")
    uvicorn.run(
        "nitrag.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
