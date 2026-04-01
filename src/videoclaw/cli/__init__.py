"""VideoClaw CLI package.

Importing this module registers all CLI commands on the Typer :data:`app`.

The entry point ``claw = "videoclaw.cli:app"`` in ``pyproject.toml``
resolves to the ``app`` instance exported here.
"""

# Version command (lightweight, lives here directly).
import videoclaw
import videoclaw.cli.config_cmd
import videoclaw.cli.cost_cmd
import videoclaw.cli.doctor
import videoclaw.cli.drama
import videoclaw.cli.flow

# Import command modules to trigger Typer command registration.
import videoclaw.cli.generate
import videoclaw.cli.info
import videoclaw.cli.model
import videoclaw.cli.project
import videoclaw.cli.stage
import videoclaw.cli.template
from videoclaw.cli._app import app
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
