"""VideoClaw CLI package.

Importing this module registers all CLI commands on the Typer :data:`app`.

The entry point ``claw = "videoclaw.cli:app"`` in ``pyproject.toml``
resolves to the ``app`` instance exported here.
"""

from videoclaw.cli._app import app  # noqa: F401

# Import command modules to trigger Typer command registration.
import videoclaw.cli.generate  # noqa: F401
import videoclaw.cli.doctor  # noqa: F401
import videoclaw.cli.model  # noqa: F401
import videoclaw.cli.project  # noqa: F401
import videoclaw.cli.template  # noqa: F401
import videoclaw.cli.flow  # noqa: F401
import videoclaw.cli.drama  # noqa: F401
import videoclaw.cli.config_cmd  # noqa: F401
import videoclaw.cli.stage  # noqa: F401

# Version command (lightweight, lives here directly).
import typer
import videoclaw
from videoclaw.cli._output import get_console, get_output


@app.command()
def version() -> None:
    """Print the VideoClaw version."""
    console = get_console()
    out = get_output()
    out._command = "version"
    console.print(f"[bold cyan]VideoClaw[/bold cyan] v{videoclaw.__version__}")
    out.set_result({"version": videoclaw.__version__})
    out.emit()
