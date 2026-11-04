"""Web UI served by the backend, and projects made before the agentseg rename."""
import shutil

from fastapi.testclient import TestClient
from PIL import Image

import backend.main as main
from core.store import DB_FILENAME, ProjectStore, find_project_db


def _images(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    Image.new("RGB", (8, 8)).save(img_dir / "0.png")
    return img_dir


def test_new_project_uses_agentseg_db(tmp_path):
    proj = ProjectStore.create_project(str(tmp_path / "p"), str(_images(tmp_path)), "p")
    assert proj.db_path.endswith(DB_FILENAME)


def test_legacy_db_opens_and_moved_project_heals_paths(tmp_path):
    ProjectStore.create_project(str(tmp_path / "p"), str(_images(tmp_path)), "p")
    shutil.move(tmp_path / "p" / DB_FILENAME, tmp_path / "p" / "autoseg.db")
    shutil.move(tmp_path / "p", tmp_path / "moved")

    proj = ProjectStore.load_project(str(tmp_path / "moved"))
    assert proj.root_dir == str(tmp_path / "moved")
    assert proj.db_path == find_project_db(str(tmp_path / "moved")) == str(tmp_path / "moved" / "autoseg.db")


def test_backend_serves_built_ui(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<div id="root"></div>')
    (dist / "assets" / "app-123.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("no")
    monkeypatch.setattr(main, "FRONTEND_DIST", dist)
    c = TestClient(main.app)

    assert 'id="root"' in c.get("/").text
    assert 'id="root"' in c.get("/some/client/route").text
    r = c.get("/assets/app-123.js")
    assert r.text == "console.log(1)" and "immutable" in r.headers["cache-control"]
    assert "no" != c.get("/../secret.txt").text
    assert c.get("/api/nope").status_code == 404
    assert c.get("/api/health").json()["ui"] is True


def test_backend_without_ui_build(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "FRONTEND_DIST", tmp_path / "missing")
    c = TestClient(main.app)
    assert c.get("/").status_code == 404
    assert c.get("/api/health").status_code == 200
