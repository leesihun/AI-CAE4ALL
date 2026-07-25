"""Compatibility entrypoint: the real implementation lives in studio_backend/.

Kept as a top-level module (not moved into the package) because
start_studio.py and START_STUDIO.bat both invoke it by this name.
"""

from __future__ import annotations

import sys

from studio_backend.http_handler import create_server, serve

__all__ = ["create_server", "serve"]

if __name__ == "__main__":
    selected_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    raise SystemExit(serve("127.0.0.1", selected_port))
