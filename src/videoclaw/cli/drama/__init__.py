"""``claw drama`` -- AI short drama series orchestration commands.

This package splits the drama CLI into logical submodules.  Importing
this module (e.g. ``import videoclaw.cli.drama``) triggers command
registration on the shared :data:`drama_app` Typer sub-app.
"""

import videoclaw.cli.drama._design
import videoclaw.cli.drama._export
import videoclaw.cli.drama._generate
import videoclaw.cli.drama._plan
import videoclaw.cli.drama._quality
import videoclaw.cli.drama._setup
import videoclaw.cli.drama._status  # noqa: F401 — side-effect import for command registration
