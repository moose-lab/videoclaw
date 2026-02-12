"""Local filesystem storage backend."""

from __future__ import annotations

import shutil
from pathlib import Path

from videoclaw.config import get_config


class LocalStorage:
    """Manages project assets on the local filesystem.

    Layout::

        {projects_dir}/{project_id}/assets/{filename}
        {projects_dir}/{project_id}/output/{filename}
    """

    def __init__(self, projects_dir: Path | None = None) -> None:
        self.projects_dir = projects_dir or get_config().projects_dir

    def _project_dir(self, project_id: str) -> Path:
        return self.projects_dir / project_id

    def assets_dir(self, project_id: str) -> Path:
        d = self._project_dir(project_id) / "assets"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def output_dir(self, project_id: str) -> Path:
        d = self._project_dir(project_id) / "output"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_asset(self, project_id: str, filename: str, data: bytes) -> Path:
        path = self.assets_dir(project_id) / filename
        path.write_bytes(data)
        return path

    def load_asset(self, project_id: str, filename: str) -> bytes:
        return (self.assets_dir(project_id) / filename).read_bytes()

    def save_output(self, project_id: str, filename: str, data: bytes) -> Path:
        path = self.output_dir(project_id) / filename
        path.write_bytes(data)
        return path

    def list_assets(self, project_id: str) -> list[str]:
        d = self.assets_dir(project_id)
        return [f.name for f in d.iterdir() if f.is_file()]

    def delete_project(self, project_id: str) -> None:
        d = self._project_dir(project_id)
        if d.exists():
            shutil.rmtree(d)
