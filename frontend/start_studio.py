from __future__ import annotations

import os
import sys
import threading
import webbrowser

from studio_server import create_server


HOST = "127.0.0.1"
DEFAULT_PORT = 8080


def parse_port() -> int:
    if len(sys.argv) < 2:
        return DEFAULT_PORT
    try:
        port = int(sys.argv[1])
    except ValueError:
        raise SystemExit(f"Invalid port: {sys.argv[1]!r}. Use a number such as 8081.")
    if not 1 <= port <= 65535:
        raise SystemExit("Port must be between 1 and 65535.")
    return port


def open_studio(url: str) -> None:
    try:
        if os.name == "nt":
            os.startfile(url)  # type: ignore[attr-defined]
            return
        if webbrowser.open_new_tab(url):
            return
    except OSError as exc:
        print(f"\nCould not open the default browser automatically: {exc}", flush=True)

    print(f"\nOpen this address manually: {url}", flush=True)


def main() -> int:
    port = parse_port()
    try:
        server = create_server(HOST, port)
    except OSError as exc:
        print(f"\nAI-CAE4ALL Studio could not use port {port}: {exc}", flush=True)
        print(f"Try another port:\n    START_STUDIO.bat {port + 1}", flush=True)
        return 1

    url = f"http://{HOST}:{port}/index.html"
    print("\nAI-CAE4ALL Studio is ready.", flush=True)
    print(f"Opening: {url}", flush=True)
    print("Keep this window open. Press Ctrl+C to stop the server.\n", flush=True)

    threading.Timer(0.35, open_studio, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping AI-CAE4ALL Studio...", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
