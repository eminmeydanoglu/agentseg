"""`agentseg` CLI against the real FastAPI app with the GPU-free fake SAM client."""
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import cli.main as cli
from core.store import ProjectStore
from tests.fake_sam3 import FakeSam3Client


def test_frame_specs_and_ranges():
    assert cli._frames("3,10-12, 5") == [3, 5, 10, 11, 12]
    assert cli._frames("12-10") == [10, 11, 12]
    assert cli._ranges([1, 2, 3, 7, 9, 10]) == "1-3,7,9-10"
    with pytest.raises(cli.CliError):
        cli._frames("a-b")
    with pytest.raises(cli.CliError):
        cli._box("10,10,5,20")


@pytest.fixture
def run(tmp_path, monkeypatch, capsys):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(6):
        Image.new("RGB", (32, 24), (10, 20, 30)).save(image_dir / f"{i}.png")
    project = ProjectStore.create_project(str(tmp_path / "project"), str(image_dir), "video")
    store = ProjectStore(project.db_path)
    store.close()
    store._conn = sqlite3.connect(project.db_path, check_same_thread=False)
    store._conn.row_factory = sqlite3.Row

    import backend.api.projects as projects_api
    import backend.api.video_tracks as video_api
    from backend.main import app

    projects_api._current_project, projects_api._current_store = project, store
    fake = FakeSam3Client(6, 24, 32, dx=0)
    monkeypatch.setattr(video_api, "_client", lambda: fake)
    monkeypatch.setenv("VIDEO_HEALTH_DINO", "0")
    monkeypatch.setattr(cli, "HOME", tmp_path / ".agentseg")
    video_api._ENGINES.clear(); video_api._JOBS.clear()
    client = TestClient(app)

    class TestApi(cli.Api):
        def __init__(self, base):
            self.base, self.http = base, client

    monkeypatch.setattr(cli, "Api", TestApi)

    def _run(*argv):
        code = cli.main(list(argv))
        return code, json.loads(capsys.readouterr().out)

    try:
        yield _run
    finally:
        projects_api._current_project = projects_api._current_store = None
        for engine in video_api._ENGINES.values():
            engine.close()
        video_api._ENGINES.clear()


def test_label_flow(run, tmp_path):
    code, seg = run("segment", "0", "--box", "2,3,10,12", "--pos", "5,6")
    assert code == 0 and seg["preview_id"] and seg["area"] > 0
    assert (tmp_path / ".agentseg" / "views").exists() and seg["image"].endswith(".png")

    code, made = run("track", "create", "--label", "board", "--preview", seg["preview_id"])
    assert code == 0
    tid = str(made["track_id"])

    code, prop = run("track", "propagate", tid)
    assert code == 0 and prop["frames_written"] == 5 and isinstance(prop["issues"], list)

    code, res = run("track", "absent", tid, "4-5", "--no-propagate")
    assert code == 0 and res["absent"] == "4-5"

    code, res = run("track", "refine", tid, "2", "--box", "2,3,10,12")
    assert code == 0 and res["keyframe"]["frame"] == 2 and "propagate" in res

    code, res = run("track", "keyframes", tid)
    assert res["keyframes"] == [0, 2]
    code, res = run("track", "list")
    assert res["tracks"][0]["absent"] == "4-5"

    code, res = run("sheet", "--track", tid, "--every", "2")
    assert code == 0 and res["frames"] == [0, 2, 4] and "legend" in res
    code, res = run("view", "1", "--track", tid, "--grid")
    assert code == 0

    log = (tmp_path / ".agentseg" / "cli_log.jsonl").read_text().splitlines()
    assert json.loads(log[0])["argv"][0] == "segment"


def test_errors_are_json(run):
    code, res = run("track", "refine", "1", "999", "--box", "1,1,5,5")
    assert code != 0 and res["ok"] is False and "error" in res
    code, res = run("segment", "0", "--box", "1,2,3")
    assert code == 2 and "--box" in res["error"]
