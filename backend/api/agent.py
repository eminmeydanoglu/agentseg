"""Launch the labeling agent (Muse Code, headless) and mirror it into the activity feed.

POST /api/agent/runs starts ``muse exec --json`` in the repo root with the
agentseg-label skill available. A reader thread turns Muse's JSONL records
into ``agent.*`` activity events (see backend/muse_events.py), so the web UI
shows what the agent is doing next to the ``source="cli"`` events that each
`agentseg` call publishes itself.

Settings (env):
  META_API_KEY            Muse's own API-key variable; required. Read from the
                          backend's environment, else from the repo's .env.
  AGENTSEG_AGENT_MODEL    default ``muse-spark-1.3-contributor``
  AGENTSEG_AGENT_EFFORT   Muse reasoning effort (default: Muse's own, ``high``)
  AGENTSEG_MUSE_BIN       path to the ``muse`` launcher (default: PATH, ~/.local/bin/muse)
  AGENTSEG_AGENT_TIMEOUT  seconds before a run is stopped (default 2700)
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.muse_events import MuseEventMapper, SessionLogTail
from core.envfile import read_env_file

router = APIRouter()

REPO = Path(__file__).resolve().parents[2]
API_KEY_ENV = "META_API_KEY"
DEFAULT_MODEL = "muse-spark-1.3-contributor"
SKILL = "agentseg-label"
_PENDING_LIMIT = 500

_LOCK = threading.RLock()
_RUNS: dict[str, "AgentRun"] = {}


def _home() -> Path:
    return Path(os.getenv("AGENTSEG_HOME") or REPO / ".agentseg")


def muse_binary() -> Optional[str]:
    configured = os.getenv("AGENTSEG_MUSE_BIN")
    if configured:
        return configured if os.access(configured, os.X_OK) else None
    found = shutil.which("muse")
    if found:
        return found
    fallback = Path.home() / ".local" / "bin" / "muse"
    return str(fallback) if os.access(fallback, os.X_OK) else None


def resolve_api_key() -> Optional[str]:
    """META_API_KEY from the environment, else from the repo's gitignored .env."""
    value = os.environ.get(API_KEY_ENV, "").strip()
    if value:
        return value
    return read_env_file(REPO / ".env").get(API_KEY_ENV, "").strip() or None


def missing_key_message() -> str:
    return (f"{API_KEY_ENV} is not set, so Muse Code cannot run. Set it in the shell that starts the backend "
            f"(`read -rs {API_KEY_ENV} && export {API_KEY_ENV}`, then `agentseg down && agentseg up`) "
            f"or add a `{API_KEY_ENV}=...` line to {REPO / '.env'} (gitignored).")


def build_prompt(instruction: str, images: Optional[str]) -> str:
    lines = [f"Use the `{SKILL}` skill (open it with read_skill first) and follow its workflow.",
             "Run the CLI from the repo root as `.venv/bin/agentseg`. This is a headless run: nobody can "
             "answer questions, so make reasonable choices, state them, and finish the job.",
             "A person is watching your work live in the agentseg web UI, which shows your text replies "
             "as your thoughts. Before each step, write one short plain sentence saying what you will do "
             "and why (for example: \"Frame 23 shows the whole board up close, so I'll prompt there.\"). "
             "No markdown, no lists, no file paths, no command lines in these sentences; keep each under "
             "25 words. Frame numbers in these sentences are 1-based, as the UI shows them. Narrate the "
             "labeling, not the plumbing: say nothing about reading the skill, checking services, opening "
             "the project or listing labels; start talking when you start looking at the video.",
             "", f"Task: {instruction.strip()}"]
    if images:
        lines.append(f"Image folder / project: {images}")
    return "\n".join(lines) + "\n"


def build_muse_command(muse: str, prompt_file: Path, model: str, effort: Optional[str]) -> list[str]:
    cmd = [muse, "exec", "--json", "--model", model,
           # Same trust level the old `claude -p --dangerously-skip-permissions` trial had:
           # no approval prompts, no sandbox (the CLI needs localhost + the GPU server), and
           # a trusted workspace so the repo's skills load.
           "--yolo", "--disable-request-user-input",
           "--prompt-file", str(prompt_file)]
    if effort:
        cmd += ["--reasoning-effort", effort]
    return cmd


class AgentRun:
    def __init__(self, run_id: str, instruction: str, images: Optional[str], model: str,
                 effort: Optional[str], backend_url: str, timeout_s: float,
                 agentseg_home: Optional[str] = None):
        self.agentseg_home = agentseg_home
        self.id, self.instruction, self.images = run_id, instruction, images
        self.model, self.effort, self.backend_url, self.timeout_s = model, effort, backend_url, timeout_s
        self.dir = Path(agentseg_home or _home()) / "agent_runs" / run_id
        self.transcript = self.dir / "muse.jsonl"
        self.stderr_log = self.dir / "muse.stderr.log"
        self.status = "starting"
        self.error: Optional[str] = None
        self.exit_code: Optional[int] = None
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.session_id: Optional[str] = None
        self.final_text: Optional[str] = None
        self.usage: Optional[dict] = None
        self.proc: Optional[subprocess.Popen] = None
        self._stop_requested = False
        self._pending: list[tuple[str, dict]] = []
        # Every agent.* event this run emitted, with its own timestamp. Recordings
        # merge these in, so a run that switches projects (or starts before one is
        # open) still has its start, thoughts and end in the project it worked in.
        self.own_events: list[dict] = []
        self.mapper = MuseEventMapper(SessionLogTail(), on_session=self._on_session)

    # ---------------------------------------------------------------- lifecycle

    def start(self, muse: str, api_key: str) -> "AgentRun":
        self.dir.mkdir(parents=True, exist_ok=True)
        prompt_file = self.dir / "prompt.md"
        prompt_file.write_text(build_prompt(self.instruction, self.images))
        cmd = build_muse_command(muse, prompt_file, self.model, self.effort)
        env = {**os.environ, API_KEY_ENV: api_key, "AGENTSEG_BACKEND": self.backend_url,
               "AGENTSEG_AGENT_RUN": self.id, "PYTHONUNBUFFERED": "1"}
        if self.agentseg_home:
            env["AGENTSEG_HOME"] = self.agentseg_home
        with open(self.stderr_log, "ab") as err:
            self.proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=err, start_new_session=True)
        self.status = "running"
        self.emit("agent.started", status="running", message="Muse Code started labeling",
                  runner="muse", model=self.model, effort=self.effort, instruction=self.instruction,
                  images=self.images, pid=self.proc.pid)
        threading.Thread(target=self._read, daemon=True, name=f"agent-{self.id}").start()
        threading.Thread(target=self._watchdog, daemon=True, name=f"agent-{self.id}-timeout").start()
        return self

    def stop(self, reason: str = "stopped") -> None:
        self._stop_requested = True
        self.error = self.error or reason
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def _watchdog(self) -> None:
        end = self.started_at + self.timeout_s
        while self.proc and self.proc.poll() is None:
            if time.time() > end:
                self.stop(f"timed out after {self.timeout_s:.0f}s")
                return
            time.sleep(2)

    def _on_session(self, session_id: str) -> None:
        self.session_id = session_id

    def _read(self) -> None:
        assert self.proc and self.proc.stdout
        with open(self.transcript, "ab") as raw:
            for line in self.proc.stdout:
                raw.write(line); raw.flush()
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    try:
                        self._publish_all(self.mapper.feed(record))
                    except Exception:  # an odd record must not stop the run or wedge its status
                        continue
        self.exit_code = self.proc.wait()
        self._publish_all(self.mapper.flush())
        self.final_text, self.usage = self.mapper.final_text or None, self.mapper.usage
        terminal = self.mapper.terminal
        if self._stop_requested and terminal != "completed":
            self.status = "stopped"
            if terminal is None:
                self.emit("agent.finished", status="stopped", message=f"Agent run {self.error}",
                          exit_code=self.exit_code, usage=self.usage)
        elif terminal == "completed":
            self.status = "finished"
        elif terminal == "cancelled":
            self.status = "stopped"
        else:
            self.status = "error"
            if terminal is None:  # muse died before reporting a terminal record
                tail = _tail(self.stderr_log)
                self.error = f"muse exited with code {self.exit_code}" + (f": {tail[-1]}" if tail else "")
                self.emit("agent.failed", status="error", message=self.error[:240], error=self.error,
                          exit_code=self.exit_code, stderr_tail=tail)
            else:
                self.error = self.error or "agent run failed"
        self.finished_at = time.time()
        self.save_recording()

    def window(self) -> tuple[float, float]:
        return self.started_at - 1.0, (self.finished_at or time.time()) + 2.0

    def events(self) -> list[dict]:
        """The open project's activity in the run's time window, plus this run's
        own agent events that were published elsewhere (or not at all)."""
        from backend.api.activity import events_between

        try:
            events = events_between(*self.window(), skip_types=_RECORDING_SKIP)
        except HTTPException:
            events = []
        published = {e.get("agent_event_id") for e in events if e.get("agent_run_id") == self.id}
        with _LOCK:
            missing = [dict(e) for e in self.own_events if e["agent_event_id"] not in published]
        return sorted(events + missing, key=lambda e: (e.get("ts") or 0))

    def save_recording(self) -> Optional[Path]:
        """Write the run's activity events (every source) to the backend's
        agent_runs dir, so the web UI can replay it (GET /api/agent/recordings/{id})."""
        return write_recording(self.id, self.snapshot(), *self.window(), self.events())

    # ---------------------------------------------------------------- events

    def _publish_all(self, events: list[dict]) -> None:
        for ev in events:
            ev = dict(ev)
            event_type = ev.pop("type")
            calls = ev.get("agentseg") or []
            ev.setdefault("action", calls[0].get("op") if calls else ev.get("tool") or "agent")
            if event_type == "agent.failed":
                self.error = ev.get("error") or self.error
            self.emit(event_type, **ev)

    def emit(self, event_type: str, status: Optional[str] = None, message: Optional[str] = None, **data) -> None:
        """Publish to the activity feed. The feed needs an open project; events
        from before the agent opens one are kept and published once it has."""
        from backend.api.activity import publish

        data.setdefault("agent_run_id", self.id)
        with _LOCK:
            data["agent_event_id"] = len(self.own_events) + 1
            if len(self.own_events) < 20000:
                self.own_events.append({"id": f"a{self.id}-{data['agent_event_id']}",
                                        "seq": f"a{self.id}-{data['agent_event_id']}",
                                        "run_id": self.id, "source": "agent", "type": event_type,
                                        "action": data.get("action", "agent"), "status": status,
                                        "message": message, "ts": time.time(),
                                        **{k: v for k, v in data.items() if k != "action" and v is not None}})
        item = (event_type, {"status": status, "message": message, **data})
        with _LOCK:
            queue = self._pending + [item]
            self._pending = []
            for i, (etype, payload) in enumerate(queue):
                payload = dict(payload)
                try:
                    publish(etype, source="agent", action=payload.pop("action", "agent"),
                            status=payload.pop("status"), message=payload.pop("message"),
                            run_id=self.id, **payload)
                except HTTPException:
                    self._pending = queue[i:][-_PENDING_LIMIT:]
                    return
                except Exception:
                    continue  # observability must never take the run down

    def snapshot(self) -> dict:
        return {"id": self.id, "runner": "muse", "status": self.status, "model": self.model,
                "effort": self.effort, "instruction": self.instruction, "images": self.images,
                "pid": self.proc.pid if self.proc else None, "exit_code": self.exit_code,
                "error": self.error, "session_id": self.session_id, "final_text": self.final_text,
                "usage": self.usage, "started_at": self.started_at, "finished_at": self.finished_at,
                "transcript": str(self.transcript), "stderr_log": str(self.stderr_log),
                "pending_events": len(self._pending)}

    @property
    def active(self) -> bool:
        return self.status in ("starting", "running")


def _tail(path: Path, n: int = 8) -> list[str]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return [line for line in lines if line.strip()][-n:]


# Per-frame job echoes duplicate `track.progress` / `track.frames`; a replay does not need them.
_RECORDING_SKIP = ("video_job.frame",)


def _current_root() -> Optional[str]:
    from backend.api.projects import get_project

    try:
        return get_project().root_dir
    except HTTPException:
        return None


def write_recording(run_id: str, meta: dict, t0: float, t1: float, events: list[dict]) -> Optional[Path]:
    """Recordings always live in the backend's own agent_runs dir (whatever
    AGENTSEG_HOME the agent's CLI used), tagged with the project they belong to."""
    root = _current_root()
    if root is None or not events:
        return None
    run_dir = _runs_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "recording.json"
    path.write_text(json.dumps({"run": {**meta, "project_root_dir": root}, "t0": t0, "t1": t1, "events": events},
                               separators=(",", ":"), default=str))
    return path


def _runs_root() -> Path:
    return _home() / "agent_runs"


@router.get("/recordings")
def list_recordings():
    out = []
    for path in sorted(_runs_root().glob("*/recording.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        run = data.get("run") or {}
        out.append({"id": path.parent.name, "instruction": run.get("instruction"), "model": run.get("model"),
                    "status": run.get("status"), "started_at": run.get("started_at"),
                    "finished_at": run.get("finished_at"), "events": len(data.get("events") or [])})
    return {"recordings": sorted(out, key=lambda r: r.get("started_at") or 0)}


@router.get("/recordings/{run_id}")
def get_recording(run_id: str):
    if not run_id.isalnum():
        raise HTTPException(400, "bad run id")
    path = _runs_root() / run_id / "recording.json"
    if not path.exists():
        raise HTTPException(404, "recording not found")
    return json.loads(path.read_text())


@router.post("/recordings/{run_id}")
def record_run(run_id: str, t0: Optional[float] = None, t1: Optional[float] = None):
    """(Re)write a run's recording from the activity log; for runs that ended
    before recordings existed, or to widen the time window."""
    from backend.api.activity import events_between

    if not run_id.isalnum():
        raise HTTPException(400, "bad run id")
    with _LOCK:
        run = _RUNS.get(run_id)
    if run is not None and t0 is None and t1 is None:
        path = run.save_recording()
    else:
        old = _runs_root() / run_id / "recording.json"
        meta = json.loads(old.read_text()).get("run", {}) if old.exists() else {"id": run_id}
        if run is not None:
            meta = run.snapshot()
        if meta.get("project_root_dir") and meta["project_root_dir"] != _current_root():
            raise HTTPException(409, f"recording {run_id} belongs to {meta['project_root_dir']}; open that project first")
        start = t0 if t0 is not None else (run.window()[0] if run else None)
        end = t1 if t1 is not None else (run.window()[1] if run else None)
        if start is None or end is None:
            raise HTTPException(400, "unknown run: give t0 and t1")
        try:
            events = events_between(start, end, skip_types=_RECORDING_SKIP)
        except HTTPException:
            events = []
        path = write_recording(run_id, meta, start, end, events)
    if path is None:
        raise HTTPException(409, "no project open or no events in that window")
    return {"path": str(path), "events": len(json.loads(path.read_text())["events"])}


class StartAgentRequest(BaseModel):
    instruction: str = Field(min_length=1, max_length=4000)
    images: Optional[str] = Field(default=None, description="image folder or project dir for the agent")
    model: Optional[str] = None
    effort: Optional[str] = Field(default=None, pattern="^(none|minimal|low|medium|high|xhigh|max|ultra)$")
    timeout_s: Optional[float] = Field(default=None, gt=0)
    agentseg_home: Optional[str] = Field(default=None, description="AGENTSEG_HOME for the agent's CLI calls")


@router.post("/runs")
def start_run(req: StartAgentRequest, request: Request):
    with _LOCK:
        active = [r for r in _RUNS.values() if r.active]
        if active:
            raise HTTPException(409, f"agent run {active[0].id} is still running; stop it first")
        muse = muse_binary()
        if not muse:
            raise HTTPException(503, "Muse Code (`muse`) not found; install it or set AGENTSEG_MUSE_BIN")
        key = resolve_api_key()
        if not key:
            raise HTTPException(400, missing_key_message())
        backend_url = os.getenv("AGENTSEG_BACKEND") or str(request.base_url).rstrip("/")
        run = AgentRun(uuid.uuid4().hex[:12], req.instruction, req.images,
                       req.model or os.getenv("AGENTSEG_AGENT_MODEL") or DEFAULT_MODEL,
                       req.effort or os.getenv("AGENTSEG_AGENT_EFFORT") or None, backend_url,
                       req.timeout_s or float(os.getenv("AGENTSEG_AGENT_TIMEOUT") or 2700),
                       req.agentseg_home)
        _RUNS[run.id] = run
        try:
            run.start(muse, key)
        except OSError as exc:
            run.status, run.error = "error", f"could not start muse: {exc}"
            raise HTTPException(500, run.error)
        return run.snapshot()


@router.get("/runs")
def list_runs():
    with _LOCK:
        return {"runs": [r.snapshot() for r in sorted(_RUNS.values(), key=lambda r: r.started_at)]}


def _get(run_id: str) -> AgentRun:
    with _LOCK:
        if run_id == "latest" and _RUNS:
            return max(_RUNS.values(), key=lambda r: r.started_at)
        run = _RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "agent run not found")
    return run


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    return _get(run_id).snapshot()


@router.get("/runs/{run_id}/events")
def get_run_events(run_id: str):
    """Everything published since the run started, so a page opened mid-run can catch up."""
    run = _get(run_id)
    return {"run": run.snapshot(), "events": run.events()}


@router.post("/runs/{run_id}/stop")
def stop_run(run_id: str):
    run = _get(run_id)
    if run.active:
        threading.Thread(target=run.stop, args=("stopped by user",), daemon=True).start()
    return run.snapshot()


@router.get("/config")
def agent_config():
    """What a run would use, without revealing the key."""
    return {"runner": "muse", "muse": muse_binary(), "api_key_env": API_KEY_ENV,
            "api_key_set": resolve_api_key() is not None,
            "model": os.getenv("AGENTSEG_AGENT_MODEL") or DEFAULT_MODEL,
            "effort": os.getenv("AGENTSEG_AGENT_EFFORT") or None, "skill": SKILL, "cwd": str(REPO)}
