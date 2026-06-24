"""
utils.py
========
A context manager that silences BOTH Python-level (sys.stderr) and C-level
(file descriptor 2) writes. FAISS prints its clustering notices from C++ straight
to fd 2, which Python's `warnings`/`logging` cannot intercept — this can.
"""
from __future__ import annotations

import contextlib
import os


@contextlib.contextmanager
def suppress_stderr():
    """Redirect fd 2 and sys.stderr to /dev/null for the duration of the block.

    Python exceptions still propagate (their tracebacks print after the block
    exits and stderr is restored), so real errors are never hidden.
    """
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    devnull_file = open(os.devnull, "w")
    try:
        os.dup2(devnull_fd, 2)
        with contextlib.redirect_stderr(devnull_file):
            yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)
        devnull_file.close()
