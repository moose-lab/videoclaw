"""JSON output mode for agent-friendly CLI responses.

When ``--json`` is passed globally, Rich terminal output is suppressed and
a structured JSON envelope is printed to stdout on command exit.

Envelope format::

    {"ok": true, "version": "0.1.0", "command": "video", "data": {...}, "error": null}

Thread safety: :class:`OutputContext` is a module-level singleton.  It is
**NOT** thread-safe and is designed exclusively for single-threaded CLI
invocations.  The ``json_mode`` flag must be set by the Typer callback
BEFORE any async operations begin.
"""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass
from typing import Any

from rich.console import Console

import videoclaw


@dataclass
class OutputContext:
    """Tracks JSON output state for the current CLI invocation."""

    json_mode: bool = False
    _command: str = ""
    _result: dict[str, Any] | None = None
    _error: str | None = None
    _exit_code: int = 0

    def set_result(self, data: dict[str, Any]) -> None:
        """Register the structured result for this command."""
        self._result = data

    def set_error(self, message: str, code: int = 1) -> None:
        """Register an error for this command."""
        self._error = message
        self._exit_code = code

    def emit(self) -> None:
        """Print JSON envelope to stdout if json_mode is active."""
        if not self.json_mode:
            return
        envelope = {
            "ok": self._error is None,
            "version": videoclaw.__version__,
            "command": self._command,
            "data": self._result,
            "error": self._error,
        }
        sys.stdout.write(json.dumps(envelope, ensure_ascii=False, default=str) + "\n")
        sys.stdout.flush()


# Module-level singleton -- safe because CLI commands run single-threaded.
_output_ctx = OutputContext()

# A "silent" console that discards all output (used in JSON mode).
_null_console = Console(file=io.StringIO(), no_color=True)


def get_output() -> OutputContext:
    """Return the global output context."""
    return _output_ctx


def get_console() -> Console:
    """Return the appropriate Rich console for the current output mode.

    In JSON mode returns a console that writes to /dev/null so that
    ``console.print()`` calls throughout the codebase become no-ops.
    In normal mode returns a real stderr/stdout console.
    """
    if _output_ctx.json_mode:
        return _null_console
    return _real_console


# The "real" console for normal (non-JSON) terminal output.
_real_console = Console()
