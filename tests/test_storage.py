"""Tests for local storage."""

from pathlib import Path

from videoclaw.storage.local import LocalStorage


def test_save_and_load_asset(tmp_path: Path):
    store = LocalStorage(projects_dir=tmp_path)
    store.save_asset("proj1", "shot_001.mp4", b"fake video data")
    data = store.load_asset("proj1", "shot_001.mp4")
    assert data == b"fake video data"


def test_list_assets(tmp_path: Path):
    store = LocalStorage(projects_dir=tmp_path)
    store.save_asset("proj1", "a.mp4", b"a")
    store.save_asset("proj1", "b.mp4", b"b")
    assets = store.list_assets("proj1")
    assert set(assets) == {"a.mp4", "b.mp4"}


def test_delete_project(tmp_path: Path):
    store = LocalStorage(projects_dir=tmp_path)
    store.save_asset("proj1", "x.mp4", b"x")
    store.delete_project("proj1")
    assert not (tmp_path / "proj1").exists()
