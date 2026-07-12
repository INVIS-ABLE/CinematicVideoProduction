"""Native desktop shell for the local FastAPI studio."""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

from .config import StudioSettings


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _serve(port: int) -> None:
    import uvicorn
    uvicorn.run("local_studio.app:app", host="127.0.0.1", port=port, log_level="info")


def main() -> None:
    if "--worker" in sys.argv:
        index = sys.argv.index("--worker")
        from .worker import run
        run(Path(sys.argv[index + 1]), Path(sys.argv[index + 2]), Path(sys.argv[index + 3]), sys.argv[index + 4])
        return

    port = _free_port()
    token = StudioSettings.from_env().desktop_token()
    os.environ["CINESTUDIO_HOST"] = "127.0.0.1"
    os.environ["CINESTUDIO_PORT"] = str(port)
    os.environ["CINESTUDIO_AUTH_TOKEN"] = token
    thread = threading.Thread(target=_serve, args=(port,), daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/?token={token}"
    for _ in range(80):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.1):
                break
        except OSError:
            time.sleep(.1)

    try:
        import webview
        webview.create_window("Cinematic Studio Local", url, width=1500, height=980, min_size=(1024, 720))
        webview.start(private_mode=True, debug=False)
    except Exception:
        webbrowser.open(url)
        thread.join()


if __name__ == "__main__":
    main()
