"""Contract tests for the shared CLI/web activity feed."""
from fastapi.testclient import TestClient
from PIL import Image

from core.store import ProjectStore


def test_activity_events_and_artifacts(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (12, 8), (20, 30, 40)).save(image_dir / "0.png")
    project = ProjectStore.create_project(str(tmp_path / "project"), str(image_dir), "activity")
    store = ProjectStore(project.db_path)

    import backend.api.projects as projects_api
    from backend.main import app

    projects_api._current_project = project
    projects_api._current_store = store
    artifact = tmp_path / "preview.png"
    artifact.write_bytes(b"png-test")
    try:
        with TestClient(app) as client:
            created = client.post("/api/activity/events", json={
                "type": "mask.proposed",
                "source": "cli",
                "action": "segment",
                "run_id": "run-1",
                "data": {"frame": 3, "score": 0.91},
            })
            assert created.status_code == 200
            event = created.json()
            assert event["type"] == "mask.proposed"
            assert event["frame"] == 3
            assert event["project_root_dir"] == project.root_dir

            listed = client.get("/api/activity/events", params={"after": 0})
            assert listed.status_code == 200
            assert listed.json()["events"][-1]["id"] == event["id"]
            assert listed.json()["events"][-1]["project_root_dir"] == project.root_dir

            registered = client.post("/api/activity/artifacts", json={
                "path": str(artifact), "kind": "frame_overlay",
            })
            assert registered.status_code == 200
            fetched = client.get(registered.json()["url"])
            assert fetched.status_code == 200
            assert fetched.content == b"png-test"
    finally:
        projects_api._current_project = None
        projects_api._current_store = None
        store.close()
