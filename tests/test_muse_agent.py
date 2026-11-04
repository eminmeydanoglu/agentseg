"""Muse Code agent launcher and its `muse exec --json` -> activity-event mapper.

The fixtures under tests/fixtures/muse are real `muse exec --json` output (Muse
Code 1.4.14) recorded against a local fake model endpoint that scripted the
tool calls, plus the matching records from Muse's session log.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.muse_events import MuseEventMapper, SessionLogTail, parse_agentseg_commands, parse_view_path
from core.envfile import read_env_file
from core.store import ProjectStore

FIX = Path(__file__).parent / "fixtures" / "muse"


def _records(name):
    return [json.loads(line) for line in (FIX / name).read_text().splitlines() if line.strip()]


def _session_root(tmp_path, exec_name, session_name):
    sid = _records(exec_name)[0]["stream"]["id"]
    d = tmp_path / "sessions" / "2026" / "11" / "03" / sid
    d.mkdir(parents=True)
    shutil.copy(FIX / session_name, d / "session.jsonl")
    return tmp_path / "sessions"


def _replay(exec_name, session_root=None):
    mapper = MuseEventMapper(SessionLogTail(session_root) if session_root else None)
    events = []
    for record in _records(exec_name):
        events += mapper.feed(record)
    return events + mapper.flush(), mapper


def test_agentseg_run_maps_to_tool_events(tmp_path):
    events, mapper = _replay("exec_agentseg.jsonl", _session_root(tmp_path, "exec_agentseg.jsonl",
                                                                  "session_agentseg.jsonl"))
    types = [e["type"] for e in events]
    assert types == ["agent.thinking", "agent.message", "agent.tool.started", "agent.tool.completed",
                     "agent.thinking", "agent.tool.started", "agent.tool.completed",
                     "agent.thinking", "agent.tool.started", "agent.tool.completed",
                     "agent.thinking", "agent.message", "agent.finished"]

    seg_start, seg_done = events[2], events[3]
    assert seg_start["status"] == "running" and seg_start["tool"] == "bash"
    assert seg_start["call_id"] == seg_done["call_id"]
    assert seg_start["agentseg"] == [{
        "op": "segment",
        "argv": ["segment", "12", "--box", "10,20,200,180", "--pos", "105,98", "--neg", "30,170"],
        "args": {"frame": 12, "box": [10.0, 20.0, 200.0, 180.0], "pos": [[105.0, 98.0]],
                 "neg": [[30.0, 170.0]], "no_grid": False}}]
    assert seg_start["description"] == "Preview a mask on frame 12"
    assert seg_done["ok"] is True and seg_done["exit_code"] == 0
    assert seg_done["agentseg_result"]["preview_id"] == "3f9c2a1b7d4e5f60"
    assert seg_done["agentseg_result"]["bbox"] == [11.0, 22.0, 198.0, 176.0]

    look = events[5]
    assert look["tool"] == "read_file"
    assert look["view"] == {"kind": "mask_preview", "frame": 12, "preview_id": "3f9c2a1b7d4e5f60",
                            "path": ".agentseg/views/preview_00012_3f9c2a1b7d4e5f60.png"}

    prop = events[9]
    assert prop["agentseg"][0]["op"] == "track.propagate"
    assert prop["agentseg_result"]["issue_count"] == 1
    assert prop["agentseg_result"]["issues"][0]["worst_frame"] == 397

    done = events[-1]
    assert done["status"] == "completed"
    assert done["text"] == "Track 1 propagated; one issue at frames 393-401."
    assert done["model"] == "muse-spark-1.3" and done["usage"]["calls"] == 4
    assert mapper.terminal == "completed"
    # Nothing the UI uses to move the canvas leaks to the top level of agent events.
    assert not any("frame" in e or "bbox" in e for e in events)


def test_without_session_log_args_arrive_with_the_result():
    events, _ = _replay("exec_agentseg.jsonl")
    started = [e for e in events if e["type"] == "agent.tool.started"]
    assert started[0]["message"] == "Running a command" and "args" not in started[0]
    done = [e for e in events if e["type"] == "agent.tool.completed"]
    assert done[0]["agentseg"][0]["op"] == "segment"       # recovered from bash's result JSON
    assert done[1]["tool"] == "read_file" and "view" not in done[1]


def test_failed_tool_streamed_text_and_skill(tmp_path):
    events, _ = _replay("exec_tools.jsonl", _session_root(tmp_path, "exec_tools.jsonl", "session_tools.jsonl"))
    messages = [e["text"] for e in events if e["type"] == "agent.message"]
    assert messages == ["Looking at frame 42 to pick a box.", "The track id was wrong; stopping here."]
    done = {e["tool"]: e for e in events if e["type"] == "agent.tool.completed"}
    assert done["read_skill"]["skill"] == "sxclaudexskills"
    assert done["read_file"]["view"]["frame"] == 42 and done["read_file"]["view"]["track_id"] == 1
    assert done["bash"]["ok"] is False and done["bash"]["status"] == "failed"
    assert done["bash"]["exit_code"] == 1
    assert done["bash"]["output"] == '{"ok": false, "error": "track 9 not found"}\n'


def test_rejected_key_becomes_agent_failed():
    events, mapper = _replay("exec_auth_failed.jsonl")
    assert events[-1]["type"] == "agent.failed" and events[-1]["status"] == "error"
    assert "META_API_KEY was rejected" in events[-1]["error"]
    assert mapper.terminal == "failed"


def test_parse_agentseg_commands():
    cmd = ("cd /repo && .venv/bin/agentseg track refine 1 120 --box 5,6,70,80 --neg 9,9 --no-propagate "
           "| head -c 300; uv run agentseg track absent 1 300-302,310")
    calls = parse_agentseg_commands(cmd)
    assert [c["op"] for c in calls] == ["track.refine", "track.absent"]
    assert calls[0]["args"]["box"] == [5.0, 6.0, 70.0, 80.0] and calls[0]["args"]["no_propagate"] is True
    assert calls[1]["args"]["frame_list"] == [300, 301, 302, 310]
    assert parse_agentseg_commands("python -m cli.main track keyframes 2 --rm 40")[0]["args"] == {
        "track": 2, "rm": 40, "no_propagate": False}
    assert parse_agentseg_commands(".venv/bin/agentseg status")[0]["op"] == "status"
    assert parse_agentseg_commands("agentseg nonsense --x")[0]["parse_error"] is True
    assert parse_agentseg_commands("ls -la && cat x.json") == []
    assert parse_view_path("/x/.agentseg/views/sheet_t1_every22.png") == {
        "kind": "contact_sheet", "tag": "every22", "path": "/x/.agentseg/views/sheet_t1_every22.png", "track_id": 1}


def test_read_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nexport META_API_KEY='abc def'\nEMPTY=\nOTHER = \"x\"\nnot a line\n")
    assert read_env_file(env) == {"META_API_KEY": "abc def", "EMPTY": "", "OTHER": "x"}
    assert read_env_file(tmp_path / "missing") == {}


# ---------------------------------------------------------------- launcher

@pytest.fixture
def launcher(tmp_path, monkeypatch):
    import backend.api.agent as agent_api
    import backend.api.projects as projects_api
    from backend.main import app

    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setattr(agent_api, "REPO", tmp_path / "repo")
    (tmp_path / "repo").mkdir()
    monkeypatch.setenv("AGENTSEG_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    agent_api._RUNS.clear()

    seen = tmp_path / "fake_muse_seen.json"
    fake = tmp_path / "muse"
    fake.write_text(f"""#!{sys.executable}
import json, os, sys, time
args = sys.argv[1:]
prompt = open(args[args.index("--prompt-file") + 1]).read()
json.dump({{"argv": args, "cwd": os.getcwd(), "key_set": bool(os.environ.get("META_API_KEY")),
           "backend": os.environ.get("AGENTSEG_BACKEND"), "prompt": prompt}}, open({str(seen)!r}, "w"))
for line in open({str(FIX / 'exec_agentseg.jsonl')!r}):
    sys.stdout.write(line); sys.stdout.flush()
""")
    fake.chmod(0o755)
    monkeypatch.setenv("AGENTSEG_MUSE_BIN", str(fake))

    image_dir = tmp_path / "images"; image_dir.mkdir()
    Image.new("RGB", (12, 8)).save(image_dir / "0.png")
    project = ProjectStore.create_project(str(tmp_path / "project"), str(image_dir), "agent")
    store = ProjectStore(project.db_path)

    def open_project():
        projects_api._current_project, projects_api._current_store = project, store

    try:
        with TestClient(app) as client:
            yield client, agent_api, seen, open_project
    finally:
        projects_api._current_project = projects_api._current_store = None
        store.close()
        agent_api._RUNS.clear()


def _wait_done(client, run_id, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        run = client.get(f"/api/agent/runs/{run_id}").json()
        if run["status"] not in ("starting", "running"):
            return run
        time.sleep(0.05)
    raise AssertionError("agent run did not finish")


def test_launcher_requires_api_key(launcher):
    client, *_ = launcher
    res = client.post("/api/agent/runs", json={"instruction": "label the board"})
    assert res.status_code == 400
    assert "META_API_KEY is not set" in res.json()["detail"]
    assert client.get("/api/agent/config").json()["api_key_set"] is False


def test_launcher_runs_muse_and_publishes_agent_events(launcher, monkeypatch):
    client, agent_api, seen, open_project = launcher
    # The key is only read from .env; its value must reach muse but never the API responses.
    (agent_api.REPO / ".env").write_text("META_API_KEY=test-placeholder\n")
    open_project()
    res = client.post("/api/agent/runs", json={"instruction": "label the board", "images": "/data/seq1"})
    assert res.status_code == 200, res.text
    run = _wait_done(client, res.json()["id"])
    assert run["status"] == "finished" and run["exit_code"] == 0
    assert run["final_text"] == "Track 1 propagated; one issue at frames 393-401."
    assert "test-placeholder" not in json.dumps(run)

    info = json.loads(seen.read_text())
    assert info["key_set"] is True and info["cwd"] == str(agent_api.REPO)
    assert info["argv"][:4] == ["exec", "--json", "--model", "muse-spark-1.3-contributor"]
    assert "--yolo" in info["argv"]
    assert "agentseg-label" in info["prompt"] and "/data/seq1" in info["prompt"]
    assert info["backend"]

    events = client.get("/api/activity/events", params={"after": 0, "limit": 500}).json()["events"]
    agent = [e for e in events if e["source"] == "agent"]
    assert agent[0]["type"] == "agent.started" and agent[0]["status"] == "running"
    assert agent[-1]["type"] == "agent.finished"
    assert all(e["run_id"] == run["id"] and e["agent_run_id"] == run["id"] for e in agent)
    seg = next(e for e in agent if e["type"] == "agent.tool.completed" and e["action"] == "segment")
    assert seg["agentseg_result"]["preview_id"] == "3f9c2a1b7d4e5f60"
    assert Path(run["transcript"]).read_text() == (FIX / "exec_agentseg.jsonl").read_text()


def test_events_wait_for_a_project(launcher, monkeypatch):
    client, agent_api, _seen, open_project = launcher
    run = agent_api.AgentRun("r1", "x", None, "muse-spark-1.3", None, "http://b", 60)
    run.emit("agent.started", status="running", message="started")
    run.emit("agent.message", message="hi", text="hi")
    assert run.snapshot()["pending_events"] == 2
    open_project()
    run.emit("agent.finished", status="completed", message="done")
    assert run.snapshot()["pending_events"] == 0
    events = client.get("/api/activity/events", params={"after": 0}).json()["events"]
    assert [e["type"] for e in events if e["source"] == "agent"] == [
        "agent.started", "agent.message", "agent.finished"]


def test_commentary_from_the_session_log_becomes_agent_messages(tmp_path):
    """`muse exec --json` streams only the final answer; the short sentences the
    model writes next to its tool calls live in the session log (phase "commentary")."""
    root = _session_root(tmp_path, "exec_agentseg.jsonl", "session_agentseg.jsonl")
    session = next(root.glob("*/*/*/*/session.jsonl"))
    said = {"schema_version": 1, "stream": {"kind": "session", "id": session.parent.name},
            "payload_type": "runtime.session",
            "payload": {"event": {"kind": "assistant_message_committed", "message_id": "m-1",
                                  "phase": "commentary",
                                  "text": "Frame 13 shows the whole board, so I will prompt there."}}}
    session.write_text(json.dumps(said) + "\n" + session.read_text())

    events, _ = _replay("exec_agentseg.jsonl", root)
    commentary = [e for e in events if e.get("phase") == "commentary"]
    assert [e["text"] for e in commentary] == ["Frame 13 shows the whole board, so I will prompt there."]
    first_tool = next(i for i, e in enumerate(events) if e["type"] == "agent.tool.started")
    assert events.index(commentary[0]) < first_tool


def test_recording_endpoints(launcher, tmp_path, monkeypatch):
    client, agent_api, _seen, open_project = launcher
    open_project()
    monkeypatch.setattr(agent_api, "_home", lambda: tmp_path)
    run = agent_api.AgentRun("rec1", "label it", None, "muse-spark-1.3", None, "http://b", 60,
                             agentseg_home=str(tmp_path))
    run.emit("agent.started", status="running", message="started")
    run.emit("agent.message", message="I will prompt frame 3", text="I will prompt frame 3")
    run.finished_at = run.started_at + 1
    assert run.save_recording() is not None
    listed = client.get("/api/agent/recordings").json()["recordings"]
    assert [r["id"] for r in listed] == ["rec1"]
    rec = client.get("/api/agent/recordings/rec1").json()
    assert [e["type"] for e in rec["events"]] == ["agent.started", "agent.message"]
    assert client.get("/api/agent/recordings/nope").status_code == 404
