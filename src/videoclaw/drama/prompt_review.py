"""Interactive CLI breakpoint for reviewing and editing enhanced prompts."""

from __future__ import annotations
import os, subprocess, tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from videoclaw.drama.models import DramaScene, DramaSeries, Episode

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt


class PromptReviewer:
    """Interactive CLI prompt reviewer for drama scenes.

    Parameters
    ----------
    enabled: When False, review is skipped (all auto-accepted).
    console: Rich console for output.
    """

    def __init__(self, enabled: bool = True, console: Console | None = None):
        self.enabled = enabled
        self.console = console or Console()

    def review_episode(self, episode: Episode, series: DramaSeries) -> list[DramaScene]:
        """Review all scenes interactively.

        Returns list of confirmed scenes (with possibly edited prompts).
        Skipped scenes excluded.
        """
        if not self.enabled:
            return list(episode.scenes)

        # Show header panel with controls
        self.console.print(Panel(
            f"[bold]Episode {episode.number}: {episode.title}[/bold]\n"
            f"[dim]{len(episode.scenes)} scenes to review[/dim]\n\n"
            f"[bold]Controls:[/bold]\n"
            f"  [cyan]a[/cyan] = accept    [cyan]e[/cyan] = edit ($EDITOR)\n"
            f"  [cyan]s[/cyan] = skip      [cyan]A[/cyan] = accept all remaining",
            title="[bold yellow]Prompt Review[/bold yellow]",
            border_style="yellow",
        ))

        confirmed: list[DramaScene] = []
        accept_all = False

        for idx, scene in enumerate(episode.scenes):
            if accept_all:
                confirmed.append(scene)
                continue

            self._display_scene(scene, idx, len(episode.scenes))

            choice = Prompt.ask(
                "[bold][e]dit / [a]ccept / [s]kip / accept [A]ll[/bold]",
                choices=["e", "a", "s", "A"],
                default="a",
                console=self.console,
            )

            if choice == "A":
                accept_all = True
                confirmed.append(scene)
                self.console.print("[green]Accepting all remaining scenes.[/green]")
            elif choice == "a":
                confirmed.append(scene)
            elif choice == "e":
                edited = self._edit_prompt(scene)
                if edited is not None:
                    scene.enhanced_visual_prompt = edited
                confirmed.append(scene)
            elif choice == "s":
                self.console.print(f"[yellow]Skipped {scene.scene_id}[/yellow]")

        self.console.print(
            f"\n[bold green]Review complete: {len(confirmed)}/{len(episode.scenes)} scenes confirmed[/bold green]"
        )
        return confirmed

    def _display_scene(self, scene: DramaScene, idx: int, total: int):
        """Display a single scene's enhanced prompt."""
        meta = (
            f"[dim]Duration: {scene.duration_seconds}s | "
            f"Scale: {scene.shot_scale.value if scene.shot_scale else 'n/a'} | "
            f"Camera: {scene.camera_movement}[/dim]"
        )
        chars = ""
        if scene.characters_present:
            chars = f"\n[dim]Characters: {', '.join(scene.characters_present)}[/dim]"
        dlg = ""
        if scene.dialogue:
            dlg_trunc = scene.dialogue[:80] + ("\u2026" if len(scene.dialogue) > 80 else "")
            dlg = f'\n[dim]Dialogue: "{dlg_trunc}"[/dim]'

        self.console.print(Panel(
            f"{meta}{chars}{dlg}\n\n"
            f"[green]{scene.effective_prompt}[/green]",
            title=f"[bold cyan]Shot {idx+1}/{total}: {scene.scene_id}[/bold cyan]",
            border_style="cyan",
        ))

    def _edit_prompt(self, scene: DramaScene) -> str | None:
        """Open prompt in $EDITOR. Returns edited text or None if unchanged."""
        editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vim"))
        prompt_text = scene.effective_prompt

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt",
            prefix=f"prompt_{scene.scene_id}_",
            delete=False,
        ) as f:
            f.write(prompt_text)
            tmp_path = f.name

        try:
            subprocess.run([editor, tmp_path], check=True)
            with open(tmp_path) as f:
                edited = f.read().strip()
            if edited and edited != prompt_text:
                self.console.print("[green]Prompt updated.[/green]")
                return edited
            self.console.print("[dim]No changes.[/dim]")
            return None
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.console.print("[red]Editor failed. Keeping original prompt.[/red]")
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
