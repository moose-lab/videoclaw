"""Tests for FastAPI server endpoints."""

import pytest
from fastapi.testclient import TestClient

from videoclaw.server.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOCLAW_PROJECTS_DIR", str(tmp_path))
    from videoclaw.config import get_config
    get_config.cache_clear()
    app = create_app()
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_and_list_project(client):
    r = client.post("/api/projects/", json={"prompt": "test video"})
    assert r.status_code == 200
    data = r.json()
    assert data["prompt"] == "test video"
    pid = data["project_id"]

    r = client.get("/api/projects/")
    assert r.status_code == 200
    assert any(p["project_id"] == pid for p in r.json())


def test_get_project(client):
    r = client.post("/api/projects/", json={"prompt": "hello"})
    pid = r.json()["project_id"]

    r = client.get(f"/api/projects/{pid}")
    assert r.status_code == 200
    assert r.json()["prompt"] == "hello"


def test_get_nonexistent_project(client):
    r = client.get("/api/projects/nope")
    assert r.status_code == 404


def test_delete_project(client):
    r = client.post("/api/projects/", json={"prompt": "delete me"})
    pid = r.json()["project_id"]

    r = client.delete(f"/api/projects/{pid}")
    assert r.status_code == 200
    assert r.json()["deleted"] == pid
