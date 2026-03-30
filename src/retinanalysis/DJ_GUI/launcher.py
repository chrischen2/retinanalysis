"""Launcher for the DJ-GUI web application.

Starts the dj-server Flask backend and the Next.js frontend,
then opens the browser to the GUI.
"""

import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Path to the Next.js app bundled within this package
_NEXT_APP_DIR = Path(__file__).parent / "next-app"


def _check_npm_installed():
    """Return True if npm is available on PATH."""
    try:
        subprocess.run(["npm", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _check_node_modules():
    """Return True if node_modules exists (npm install has been run)."""
    return (_NEXT_APP_DIR / "node_modules").is_dir()


def _run_flask(host, port):
    """Start the Flask backend in the current thread."""
    from dj_server.app import app
    app.run(host=host, port=port, debug=False, use_reloader=False)


def launch(flask_port=5000, frontend_port=3000, frontend=True, open_browser=True):
    """Launch the DJ-GUI web application.

    Parameters
    ----------
    flask_port : int
        Port for the Flask API backend (default 5000).
    frontend_port : int
        Port for the Next.js frontend (default 3000).
    frontend : bool
        If True (default), also start the Next.js development server.
        If False, only start the Flask backend (API-only mode).
    open_browser : bool
        If True (default), open the browser once the server is ready.

    Returns
    -------
    dict
        Process handles: ``{"flask_thread": Thread, "next_process": Popen | None}``.
        Call ``next_process.terminate()`` to stop the frontend.

    Examples
    --------
    >>> import retinanalysis as ra
    >>> ra.DJ_GUI.launch()                    # full web app
    >>> ra.DJ_GUI.launch(frontend=False)      # API only
    """
    result = {"flask_thread": None, "next_process": None}

    # Start Flask backend in a daemon thread
    flask_thread = threading.Thread(
        target=_run_flask,
        args=("127.0.0.1", flask_port),
        daemon=True,
    )
    flask_thread.start()
    result["flask_thread"] = flask_thread
    print(f"Flask backend starting on http://127.0.0.1:{flask_port}")

    if frontend:
        if not _check_npm_installed():
            print(
                "ERROR: npm is not installed. Install Node.js first, or run "
                "with frontend=False for API-only mode."
            )
            return result

        if not _check_node_modules():
            print("Installing frontend dependencies (first time only)...")
            subprocess.run(
                ["npm", "install"],
                cwd=str(_NEXT_APP_DIR),
                check=True,
            )

        # Start Next.js dev server
        env = os.environ.copy()
        env["PORT"] = str(frontend_port)
        next_process = subprocess.Popen(
            ["npm", "run", "next-dev"],
            cwd=str(_NEXT_APP_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        result["next_process"] = next_process
        print(f"Next.js frontend starting on http://localhost:{frontend_port}")

        if open_browser:
            # Give servers a moment to start
            time.sleep(3)
            webbrowser.open(f"http://localhost:{frontend_port}")
    else:
        if open_browser:
            time.sleep(1)
            webbrowser.open(f"http://127.0.0.1:{flask_port}")

    url = f"http://localhost:{frontend_port}" if frontend else f"http://127.0.0.1:{flask_port}"
    print(f"\nDJ-GUI is running at {url}")
    print("Press Ctrl+C to stop.\n")

    return result
