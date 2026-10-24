"""Physics-violation video dataset generator (see the design doc, sections 4-9)."""
import os
import subprocess

__version__ = "0.1.0"


def _commit():
    """Git commit of the generator (doc 8.3); '-dirty' when the working tree has uncommitted changes."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
        sha = run("rev-parse", "--short", "HEAD")
        if not sha:
            return "unknown"
        return sha + ("-dirty" if run("status", "--porcelain", "--", "physgen") else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


__commit__ = _commit()
