"""A delayed browser prefetch must not cache bytes from another project's DB."""
from fastapi.testclient import TestClient
from PIL import Image

from core.store import ProjectStore


def test_optimized_images_are_scoped_to_the_project_path(tmp_path, monkeypatch):
    import backend.api.projects as projects_api
    from backend.main import app

    projects = []
    stores = []
    for name, size in [("old", (20, 12)), ("new", (12, 8))]:
        images = tmp_path / name / "images"
        images.mkdir(parents=True)
        Image.new("RGB", size, (20, 30, 40)).save(images / "0.jpg")
        project = ProjectStore.create_project(str(tmp_path / name / "project"), str(images), name)
        projects.append(project)
        stores.append(ProjectStore(project.db_path))
    assert projects[0].id == projects[1].id
    try:
        with TestClient(app) as client:
            monkeypatch.setattr(projects_api, "_current_project", projects[0])
            monkeypatch.setattr(projects_api, "_current_store", stores[0])
            old = client.get("/api/images/1/optimized", params={"project_root_dir": projects[0].root_dir})
            assert old.status_code == 200
            assert old.headers["X-Original-Width"] == "20"

            monkeypatch.setattr(projects_api, "_current_project", projects[1])
            monkeypatch.setattr(projects_api, "_current_store", stores[1])
            stale = client.get("/api/images/1/optimized", params={"project_root_dir": projects[0].root_dir})
            assert stale.status_code == 409
            assert "max-age" not in stale.headers.get("Cache-Control", "")
            fresh = client.get("/api/images/1/optimized", params={"project_root_dir": projects[1].root_dir})
            assert fresh.status_code == 200
            assert fresh.headers["X-Original-Width"] == "12"
            # Preserve compatibility with older callers without a project key.
            assert client.get("/api/images/1/optimized").status_code == 200
    finally:
        for store in stores:
            store.close()
