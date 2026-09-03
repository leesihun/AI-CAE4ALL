"""Compatibility entrypoint: the real implementation lives in studio_backend/.

Kept as a top-level module (not moved into the package) because
start_studio.py and START_STUDIO.bat both invoke it by this name.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running this file directly makes ``studio/`` (rather than the repository
# root) Python's first import location.  The backend intentionally reuses the
# top-level ``cae_suite`` package, so make the documented direct-launch command
# work independently of the caller's current working directory.
SUITE_ROOT = Path(__file__).resolve().parent.parent
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from studio_backend.http_handler import create_server, serve

__all__ = ["create_server", "serve"]

if __name__ == "__main__":
    selected_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    raise SystemExit(serve("127.0.0.1", selected_port))
