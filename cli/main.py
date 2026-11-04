"""`agentseg` CLI: a thin client over the backend's video tracking API.

`agentseg up` starts everything that is down (sam_server, the backend and the
web UI it serves) and prints the UI URL; `agentseg down` stops it again.

Every command prints exactly one JSON object on stdout. Images are written as
PNG files (default `.agentseg/views/`) and their paths are returned in the JSON.
Errors exit non-zero and print `{"ok": false, "error": "..."}`. Long jobs print
progress on stderr. Every invocation is appended to `.agentseg/cli_log.jsonl`.

Coordinates are always pixels, boxes are `x0,y0,x1,y1`, frames are 0-based
order indices of the project's images.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
HOME = Path(os.getenv("AGENTSEG_HOME") or REPO / ".agentseg")
DEFAULT_BACKEND = "http://localhost:5172"   # the port the frontend dev server proxies to
DB_FILENAMES = ("agentseg.db", "autoseg.db")  # autoseg.db: projects made before the rename


def _default_backend() -> str:
    """Use the UI backend when it is already running.

    The web UI is configured to proxy to port 5172, while standalone CLI
    sessions historically used port 8000.  Prefer the UI backend so CLI
    commands operate on the same in-memory current project; retain 8000 as a
    fallback for standalone use.
    """
    configured = os.getenv("AGENTSEG_BACKEND")
    if configured:
        return configured
    for candidate in (DEFAULT_BACKEND, "http://localhost:8000"):
        try:
            if httpx.get(f"{candidate}/api/health", timeout=0.5).status_code < 400:
                return candidate
        except httpx.HTTPError:
            continue
    return DEFAULT_BACKEND


SAM_URL = os.getenv("SAM_SERVER_URL", "http://localhost:8770")


class CliError(Exception):
    def __init__(self, message: str, code: int = 1, **extra):
        super().__init__(message)
        self.code, self.extra = code, extra


# ---------------------------------------------------------------- HTTP helpers

class Api:
    def __init__(self, base: str, run_id: str | None = None, command: str | None = None):
        self.base = base.rstrip("/")
        self.http = httpx.Client(base_url=self.base, timeout=600.0)
        self.run_id = run_id
        self.command = command

    def _call(self, method: str, path: str, **kw):
        headers = dict(kw.pop("headers", {}) or {})
        run_id = getattr(self, "run_id", None)
        command = getattr(self, "command", None)
        if run_id:
            headers.update({"X-AgentSeg-Run-ID": run_id, "X-AgentSeg-Source": "cli"})
            if command:
                headers["X-AgentSeg-Command"] = command[:240]
        if headers:
            kw["headers"] = headers
        for attempt in range(3):
            try:
                r = self.http.request(method, path, **kw)
                break
            except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectError) as exc:
                # a pooled connection the server just closed (keep-alive race) or a busy moment: retry
                if attempt == 2:
                    raise CliError(f"backend not reachable at {self.base} ({exc.__class__.__name__}); "
                                   "run `agentseg status`")
                time.sleep(1.0 + 2 * attempt)
            except httpx.TransportError as exc:
                raise CliError(f"backend not reachable at {self.base} ({exc.__class__.__name__}); run `agentseg up`")
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise CliError(f"{method} {path} -> {r.status_code}: {detail}", status=r.status_code)
        return r

    def get(self, path, **kw):
        return self._call("GET", path, **kw).json()

    def post(self, path, body=None):
        return self._call("POST", path, json=body or {}).json()

    def delete(self, path):
        return self._call("DELETE", path).json()

    def png(self, path, params, out: Path) -> Path:
        r = self._call("GET", path, params=params)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(r.content)
        return out.resolve()

    def activity(self, event_type: str, *, action: str | None = None,
                 status: str | None = None, message: str | None = None,
                 **data):
        """Best-effort publish; observability must never break a CLI command."""
        try:
            return self.post("/api/activity/events", {
                "type": event_type, "source": "cli", "action": action,
                "status": status, "run_id": self.run_id, "message": message,
                "data": {k: v for k, v in data.items() if v is not None},
            })
        except (CliError, httpx.HTTPError):
            return None

    def artifact(self, path: Path, *, kind: str, **metadata):
        try:
            return self.post("/api/activity/artifacts", {
                "path": str(path), "kind": kind, "mime_type": "image/png",
                "metadata": metadata,
            })
        except (CliError, httpx.HTTPError):
            return None


def _activity(api, *args, **kwargs):
    fn = getattr(api, "activity", None)
    if not fn:
        return None
    try:
        return fn(*args, **kwargs)
    except (AttributeError, TypeError, CliError, httpx.HTTPError):
        return None


def _artifact(api, *args, **kwargs):
    fn = getattr(api, "artifact", None)
    if not fn:
        return None
    try:
        return fn(*args, **kwargs)
    except (AttributeError, TypeError, CliError, httpx.HTTPError):
        return None


def _alive(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2.0).status_code < 400
    except httpx.HTTPError:
        return False


def _sam_ready() -> dict | None:
    try:
        r = httpx.get(f"{SAM_URL}/health", timeout=2.0)
        return r.json() if r.status_code < 400 else None
    except (httpx.HTTPError, ValueError):
        return None


# ---------------------------------------------------------------- parsing

def _floats(s: str, n: int, what: str) -> list[float]:
    try:
        v = [float(x) for x in s.split(",")]
    except ValueError:
        v = []
    if len(v) != n:
        raise CliError(f"{what} must be {n} comma-separated numbers, got {s!r}", 2)
    return v


def _box(s: str) -> list[float]:
    b = _floats(s, 4, "--box")
    if b[2] <= b[0] or b[3] <= b[1]:
        raise CliError("--box must be x0,y0,x1,y1 with x1>x0 and y1>y0", 2)
    return b


def _points(items) -> list[list[float]]:
    return [_floats(p, 2, "point") for p in (items or [])]


def _frames(spec: str) -> list[int]:
    """'3,10-14,20' -> [3,10,11,12,13,14,20]."""
    out: set[int] = set()
    try:
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                a, b = (int(x) for x in part.split("-", 1))
                out.update(range(min(a, b), max(a, b) + 1))
            elif part:
                out.add(int(part))
    except ValueError:
        raise CliError(f"bad frame spec {spec!r}; use e.g. 3,10-14", 2)
    if not out:
        raise CliError("empty frame spec", 2)
    return sorted(out)


def _compact_issue(i: dict) -> dict:
    out = {"id": i.get("id"), "start": i.get("start_frame", i.get("start")),
           "end": i.get("end_frame", i.get("end"))}
    keep = ("worst_frame", "kind", "n_frames", "n_lost", "reasons", "review")
    return {**out, **{k: i[k] for k in keep if k in i}}


# ---------------------------------------------------------------- commands

def cmd_status(api: Api, a) -> dict:
    out = {"backend": {"url": api.base, "up": _alive(f"{api.base}/api/health")},
           "sam_server": {"url": SAM_URL, "health": _sam_ready()}}
    if out["backend"]["up"]:
        proj = api.get("/api/projects/current")
        out["project"] = proj
        if proj:
            out["tracks"] = [_track_summary(t) for t in api.get("/api/video/tracks")]
    if not out["backend"]["up"] or not out["sam_server"]["health"]:
        out["hint"] = "run `agentseg up` to start the missing services"
    return out


def _spawn(cmd: list[str], cwd: Path, log: Path, env: dict | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as fh:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True, env={**os.environ, **(env or {})})
    return p.pid


def _wait(check, timeout: float, what: str, pid: int | None = None, log: Path | None = None):
    end = time.time() + timeout
    while time.time() < end:
        if check():
            return
        if pid is not None and not _pid_alive(pid):
            raise CliError(f"{what} exited during startup; see {log}", log=str(log), tail=_tail(log))
        time.sleep(2)
    raise CliError(f"{what} did not become ready within {timeout:.0f}s; see {log}", log=str(log), tail=_tail(log))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a zombie child counts as dead
        return not Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].startswith("Z")
    except (OSError, IndexError):
        return True


def _tail(log: Path | None, n: int = 15) -> list[str]:
    try:
        return log.read_text(errors="replace").splitlines()[-n:] if log else []
    except OSError:
        return []


def _port(url: str) -> int:
    return int(url.rstrip("/").rsplit(":", 1)[-1])


def _tool(name: str, *fallbacks: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for f in fallbacks:
        f = os.path.expanduser(f)
        if os.access(f, os.X_OK):
            return f
    return None


def _listeners(port: int) -> list[int]:
    """PIDs listening on a local TCP port."""
    try:
        out = subprocess.run(["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout
        return sorted({int(x) for x in out.split()})
    except (OSError, ValueError):
        return []


def _stop_port(port: int, what: str, timeout: float = 20) -> list[int]:
    """SIGTERM the process group listening on `port`, SIGKILL if it hangs."""
    pids = _listeners(port)
    for pid in pids:
        try:
            pgid = os.getpgid(pid)
            if pgid == os.getpgid(0):  # never signal our own process group
                raise PermissionError
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
    end = time.time() + timeout
    while pids and _listeners(port) and time.time() < end:
        time.sleep(0.5)
    for pid in _listeners(port):
        print(f"{what} did not stop, killing pid {pid}", file=sys.stderr)
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return pids


UI_INPUTS = ("src", "public", "index.html", "package.json", "bun.lock", "vite.config.ts",
             "tsconfig.json", "tsconfig.app.json", "tsconfig.node.json")


def _ui_stale(frontend: Path) -> str | None:
    """Why the built UI needs a rebuild, or None if it is up to date."""
    index = frontend / "dist" / "index.html"
    if not index.exists():
        return "not built"
    built = index.stat().st_mtime
    for name in UI_INPUTS:
        root = frontend / name
        files = [root] if root.is_file() else (root.rglob("*") if root.is_dir() else [])
        for f in files:
            if f.is_file() and f.stat().st_mtime > built:
                return f"{f.relative_to(frontend)} changed"
    return None


def _build_ui(force: bool) -> dict | str:
    frontend = REPO / "frontend"
    reason = "--rebuild" if force else _ui_stale(frontend)
    if reason is None:
        return "up to date"
    bun = _tool("bun", "~/.bun/bin/bun")
    if not bun:
        raise CliError("bun is needed to build the web UI (https://bun.sh), or pass --no-ui")
    log = (HOME / "frontend_build.log").resolve()
    log.parent.mkdir(parents=True, exist_ok=True)
    steps = ([[bun, "install", "--frozen-lockfile"]] if not (frontend / "node_modules").is_dir() else []) + \
            [[bun, "run", "build"]]
    print(f"building web UI ({reason}, log {log}) ...", file=sys.stderr)
    with open(log, "ab") as fh:
        for cmd in steps:
            r = subprocess.run(cmd, cwd=frontend, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            if r.returncode:
                raise CliError(f"`{' '.join(cmd[1:])}` failed in frontend/; see {log}", log=str(log), tail=_tail(log))
    return {"built": reason, "log": str(log)}


def _start_sam(a) -> dict | str:
    if _sam_ready():
        return "already running"
    if a.no_sam:
        return "down (skipped)"
    if _listeners(_port(SAM_URL)):
        # Something holds the port but is not healthy yet: most likely a sam_server still loading.
        print(f"waiting for the process on {SAM_URL} to become ready ...", file=sys.stderr)
        _wait(lambda: (_sam_ready() or {}).get("ready"), 600, "sam_server", log=HOME / "sam_server.log")
        return "already running"
    if not (REPO / "third_party/sam3/pyproject.toml").exists():
        raise CliError("third_party/sam3 is empty; run `git submodule update --init --recursive`")
    uv = _tool("uv", "~/.local/bin/uv", "~/.cargo/bin/uv")
    if not uv:
        raise CliError("uv is needed to run sam_server (https://docs.astral.sh/uv/)")
    log = (HOME / "sam_server.log").resolve()
    attrs = '{"batched_grounding_batch_size": 4, "postprocess_batch_size": 4}'
    cmd = [uv, "run", "sam-server", "--port", str(_port(SAM_URL)), "--models", "sam3,image",
           "--video-model-attrs", attrs]
    # A local checkpoint skips the Hugging Face download (facebook/sam3, needs `hf auth login`).
    ckpt = REPO / "models/sam3/sam3.pt"
    if ckpt.exists():
        cmd += ["--sam3-checkpoint", str(ckpt)]
    pid = _spawn(cmd, REPO / "sam_server", log, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    print(f"starting sam_server (pid {pid}, log {log}); loading SAM 3 takes 1-2 min ...", file=sys.stderr)
    _wait(lambda: (_sam_ready() or {}).get("ready"), 600, "sam_server", pid, log)
    return {"started": pid, "log": str(log)}


def _start_backend(api: Api, a) -> dict | str:
    if _alive(f"{api.base}/api/health"):
        return "already running"
    log = (HOME / "backend.log").resolve()
    pid = _spawn([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", a.host,
                  "--port", str(_port(api.base)), "--timeout-keep-alive", "120"], REPO, log,
                 {"PYTHONPATH": str(REPO), "SAM_SERVER_URL": SAM_URL, "VIDEO_CLOSE_STALE_SESSIONS": "1"})
    print(f"starting backend (pid {pid}, log {log}) ...", file=sys.stderr)
    _wait(lambda: _alive(f"{api.base}/api/health"), 120, "backend", pid, log)
    return {"started": pid, "log": str(log)}


def cmd_up(api: Api, a) -> dict:
    out = {"sam_server": _start_sam(a)}
    if not a.no_ui:
        out["web_ui_build"] = _build_ui(a.rebuild)
    out["backend"] = _start_backend(api, a)
    if not a.no_ui:
        ui = f"{api.base}/"
        try:
            r = httpx.get(ui, timeout=5.0)
            served = r.status_code == 200 and "<div id=\"root\">" in r.text
        except httpx.HTTPError:
            served = False
        if not served:
            raise CliError(f"backend at {api.base} does not serve the web UI; an old backend may be running, "
                           "try `agentseg down && agentseg up`", **out)
        out["ui"] = ui
        print(f"web UI: {ui}", file=sys.stderr)
    out["agent_api_key"] = _api_key_source(api)
    return out


def _api_key_source(api: Api) -> str:
    """Whether the backend can give the agent a META_API_KEY; never the value."""
    from core.envfile import read_env_file
    try:  # the backend's view counts: an already-running backend keeps its old environment
        if api.get("/api/agent/config").get("api_key_set"):
            return "set"
        return ("missing in the running backend: `read -rs META_API_KEY && export META_API_KEY`, "
                "then `agentseg down && agentseg up`, or add it to .env")
    except (CliError, httpx.HTTPError, ValueError, AttributeError):
        pass
    if os.environ.get("META_API_KEY", "").strip():
        return "set (environment)"
    if read_env_file(REPO / ".env").get("META_API_KEY", "").strip():
        return "set (.env)"
    return "missing: `read -rs META_API_KEY && export META_API_KEY` before `agentseg up`, or add it to .env"


def cmd_down(api: Api, a) -> dict:
    if os.environ.get("AGENTSEG_AGENT_RUN"):
        raise CliError("refusing `down` inside an agent run: the backend hosts this run and stopping it ends "
                       "the run. If sam_server failed (e.g. CUDA out of memory from another GPU job), wait "
                       "30-60 s and retry the failed command; if it keeps failing, stop and report it", 2)
    out = {"backend": {"stopped": _stop_port(_port(api.base), "backend")}}
    if a.all:
        # Give the backend's sam3 sessions back before the server goes; harmless if it is already down.
        out["sam_server"] = {"stopped": _stop_port(_port(SAM_URL), "sam_server", timeout=60)}
    else:
        out["sam_server"] = "left running (use --all to stop it too)"
    return out


def _project_info(api: Api, proj: dict) -> dict:
    first = api.get("/api/images/by-index/0") if proj["image_count"] else None
    return {**proj, "frames": proj["image_count"],
            "last_frame": proj["image_count"] - 1,
            "image_size": [first["width"], first["height"]] if first else None}


def _is_project(path: Path) -> bool:
    return any((path / name).exists() for name in DB_FILENAMES)


def cmd_project_open(api: Api, a) -> dict:
    path = Path(a.path).expanduser().resolve()
    if not path.is_dir():
        raise CliError(f"not a directory: {path}", 2)
    if _is_project(path):
        proj, created = api.post("/api/projects/open", {"project_dir": str(path)}), False
    else:
        name = a.name or path.name if path.name != "images" else (a.name or path.parent.name)
        pdir = Path(a.project_dir).expanduser().resolve() if a.project_dir else (HOME / "projects" / name).resolve()
        if _is_project(pdir):
            proj, created = api.post("/api/projects/open", {"project_dir": str(pdir)}), False
        else:
            proj, created = api.post("/api/projects", {"project_dir": str(pdir), "image_dir": str(path),
                                                      "name": name}), True
    return {"created": created, **_project_info(api, proj)}


def cmd_labels(api: Api, a) -> dict:
    if a.add:
        existing = {l["name"] for l in api.get("/api/labels")}
        label = api.post("/api/labels", {"name": a.add})
        if a.add not in existing:
            _activity(api, "label.created", action="add_label", status="completed",
                      label=label.get("name"), label_id=label.get("id"), color=label.get("color"),
                      message=f"Added label {label.get('name')}")
    return {"labels": api.get("/api/labels")}


def _views() -> Path:
    return HOME / "views"


def _zoom(api: Api, frame: int, out_dir, track=None, preview=None, box=None) -> Path:
    params, name = {"frame": frame}, f"zoom_{frame:05d}"
    if track is not None:
        params["track_id"] = track; name += f"_t{track}"
    if preview:
        params["preview_id"] = preview; name += f"_{preview}"
    if box:
        params["box"] = ",".join(str(round(v)) for v in box)
    return api.png("/api/video/render/zoom", params, Path(out_dir or _views()) / f"{name}.png")


ZOOM_LEGEND = ("close-up, no fill: [keyframe = what the object looks like] [this frame] [this frame with the "
               "mask outline in magenta, prompt box in cyan]. Same object? (color, body shape, details vs the "
               "keyframe) Whole object? (outline hugs every edge: both ends, wheels/limbs, roof; nothing of "
               "the object outside it, nothing else inside it)")


def cmd_view(api: Api, a) -> dict:
    _activity(api, "frame.started", action="inspect_frame", status="running",
                  frame=a.frame, track_id=a.track, message=f"Inspecting frame {a.frame + 1}")
    if a.zoom:
        if a.track is None and not a.preview:
            raise CliError("--zoom needs --track T or --preview P", 2)
        path = _zoom(api, a.frame, a.out, a.track, a.preview)
        artifact = _artifact(api, path, kind="zoom", frame=a.frame, track_id=a.track)
        _activity(api, "artifact.created", action="inspect_frame", status="completed", frame=a.frame,
                  track_id=a.track, artifact=artifact, message=f"Close-up of frame {a.frame + 1}")
        return {"frame": a.frame, "image": str(path), **({"artifact": artifact} if artifact else {}),
                "legend": ZOOM_LEGEND}
    params = {"frame": a.frame, "grid": a.grid}
    name = f"frame_{a.frame:05d}"
    if a.track is not None:
        params["track_id"] = a.track; name += f"_t{a.track}"
    if a.preview:
        params["preview_id"] = a.preview; name += "_preview"
    if a.grid:
        name += "_grid"
    path = api.png("/api/video/render/frame", params, Path(a.out or _views()) / f"{name}.png")
    artifact = _artifact(api, path, kind="frame_overlay", frame=a.frame, track_id=a.track)
    _activity(api, "artifact.created", action="inspect_frame", status="completed", frame=a.frame,
                  track_id=a.track, artifact=artifact, message=f"Frame {a.frame + 1} is ready to inspect")
    return {"frame": a.frame, "image": str(path),
            **({"artifact": artifact} if artifact else {}),
            "legend": "track masks tinted, yellow box = keyframe, cyan box = propagated, "
                      "yellow fill = preview; grid lines every 50 px labelled in pixels"}


def cmd_sheet(api: Api, a) -> dict:
    n = api.get("/api/projects/current") or {}
    count = n.get("image_count", 0)
    params, tag = {}, ""
    if a.frames:
        frames = _frames(a.frames)
        params["frames"] = ",".join(map(str, frames)); tag = f"f{frames[0]}-{frames[-1]}_{len(frames)}"
    else:
        every = a.every or max(1, math.ceil(count / 24))
        params["every"] = every; frames = list(range(0, count, every)); tag = f"every{every}"
    if a.track is not None:
        params["track_id"] = a.track; tag = f"t{a.track}_{tag}"
    path = api.png("/api/video/render/sheet", params, Path(a.out or _views()) / f"sheet_{tag}.png")
    artifact = _artifact(api, path, kind="contact_sheet", track_id=a.track, frames=frames)
    _activity(api, "artifact.created", action="inspect_sequence", status="completed", track_id=a.track,
                  artifact=artifact, frames=frames, message=f"Contact sheet ready ({len(frames)} frames)")
    out = {"image": str(path), "frames": frames}
    if artifact:
        out["artifact"] = artifact
    if a.track is not None:
        out["legend"] = ("thumbnail border: yellow=keyframe, green=ok, orange=review (health flagged), "
                         "grey=absent, red=no mask")
    return out


def cmd_segment(api: Api, a) -> dict:
    box = _box(a.box)
    _activity(api, "segment.started", action="segment", status="running", frame=a.frame,
                  bbox=box, pos=_points(a.pos), neg=_points(a.neg),
                  message=f"Creating a mask proposal on frame {a.frame + 1}")
    body = {"frame": a.frame, "box": box, "pos": _points(a.pos), "neg": _points(a.neg)}
    res = api.post("/api/video/segment", body)
    params = {"frame": a.frame, "preview_id": res["preview_id"], "grid": not a.no_grid}
    path = api.png("/api/video/render/frame", params,
                   Path(a.out or _views()) / f"preview_{a.frame:05d}_{res['preview_id']}.png")
    artifact = _artifact(api, path, kind="mask_preview", frame=a.frame, preview_id=res["preview_id"])
    zoom = _zoom(api, a.frame, a.out, a.track, res["preview_id"], box)
    _activity(api, "mask.proposed", action="segment", status="completed", frame=a.frame,
                  preview_id=res["preview_id"], bbox=res.get("bbox"), score=res.get("score"),
                  render_url=(f"/api/video/render/frame?frame={a.frame}"
                              f"&preview_id={res['preview_id']}"),
                  artifact=artifact, message="Mask proposal is ready for inspection")
    check = res.get("check") or {}
    nxt = ("look at BOTH images (the zoom shows the edges). Accept only if the outline covers the whole "
           "object; then use the preview_id with `track create` or `track refine --preview`. Otherwise re-run "
           "segment: box around ALL of the object, --pos on the parts the mask misses, --neg on what leaks in")
    if check.get("warning"):
        nxt = "WARNING: " + check["warning"] + ". " + nxt
    return {**res, "image": str(path), "zoom": str(zoom),
            **({"artifact": artifact} if artifact else {}),
            "zoom_legend": ZOOM_LEGEND, "next": nxt}


def _track_summary(t: dict) -> dict:
    job = t.get("job") or {}
    return {"id": t["id"], "label_id": t["label_id"], "keyframes": t["keyframes"],
            "absent": _ranges(t.get("absent") or []), "coverage": t["coverage"],
            "issue_count": t["issue_count"], "job_status": job.get("status")}


def _ranges(frames: list[int]) -> str:
    frames = sorted(frames); out = []; i = 0
    while i < len(frames):
        j = i
        while j + 1 < len(frames) and frames[j + 1] == frames[j] + 1:
            j += 1
        out.append(str(frames[i]) if i == j else f"{frames[i]}-{frames[j]}"); i = j + 1
    return ",".join(out)


def _get_track(api: Api, tid: int) -> dict:
    for t in api.get("/api/video/tracks"):
        if t["id"] == tid:
            return t
    raise CliError(f"track {tid} not found")


def cmd_track_create(api: Api, a) -> dict:
    _activity(api, "track.started", action="create_track", status="running",
                  message=f"Creating track for {a.label}")
    res = api.post("/api/video/tracks", {"preview_id": a.preview, "label": a.label, "name": a.name})
    _activity(api, "track.created", action="create_track", status="completed", track_id=res.get("track_id"),
                  frame=(res.get("keyframe") or {}).get("frame"), label=a.label, label_id=res.get("label_id"),
                  preview_id=a.preview, message=f"Track {res.get('track_id')} created")
    return {**res, "next": f"run `agentseg track propagate {res['track_id']}`"}


MAX_LOST = 60  # empty frames in a row before a tracking direction stops and hands back


def _propagate(api: Api, tid: int, full=False, start=None, end=None, wait=True, max_lost=MAX_LOST) -> dict:
    _activity(api, "track.started", action="propagate", status="running", track_id=tid,
                  message=f"Propagating track {tid}")
    body = {"full": full, "max_lost": max_lost}
    if start is not None: body["start"] = start
    if end is not None: body["end"] = end
    job = api.post(f"/api/video/tracks/{tid}/propagate", body)
    if not wait:
        return {"job": job}
    t0, seq, frames = time.time(), 0, 0
    lo, hi = job.get("frame_range") or (0, 0)
    total = hi - lo + 1
    batch: list[list] = []
    last_batch = time.time()

    def flush_batch(final=False):
        nonlocal batch, last_batch
        if batch and (final or time.time() - last_batch >= 0.25):
            _activity(api, "track.frames", action="propagate", status="running", track_id=tid,
                      frames=batch, progress={"done": frames, "total": total},
                      message=f"Propagating track {tid}")
            batch, last_batch = [], time.time()

    while True:
        res = api.get(f"/api/video/jobs/{job['id']}/events", params={"after": seq, "wait": 15})
        for ev in res["events"]:
            seq = max(seq, ev["seq"])
            if ev["type"] == "frame":
                frames += 1
                batch.append([ev.get("frame"), bool(ev.get("visible", True))])
                if frames == 1 or frames == total or ev.get("render_url"):
                    _activity(api, "track.progress", action="propagate", status="running", track_id=tid,
                              frame=ev.get("frame"), bbox=ev.get("bbox"),
                              visible=ev.get("visible", True), render_url=ev.get("render_url"),
                              progress={"done": frames, "total": total},
                              message=f"Propagating track {tid}: frame {int(ev.get('frame', 0)) + 1}")
                if frames % 50 == 0:
                    print(f"  propagated {frames}/{total} frames ({time.time() - t0:.0f}s)", file=sys.stderr)
            elif ev["type"] == "lost":
                print(f"  lost the object from frame {ev.get('lost_from')} (tracking from keyframe "
                      f"{ev.get('keyframe')} {ev.get('direction')}); stopped at {ev.get('stopped_at')}",
                      file=sys.stderr)
                _activity(api, "track.lost", action="propagate", status="running", track_id=tid,
                          frame=ev.get("lost_from"), keyframe=ev.get("keyframe"), direction=ev.get("direction"),
                          stopped_at=ev.get("stopped_at"), untracked=ev.get("untracked"),
                          message=f"Lost the object at frame {int(ev.get('lost_from') or 0) + 1}; stopped tracking")
            elif ev["type"] == "error":
                raise CliError(f"propagation failed: {ev.get('message')}")
        flush_batch()
        if res["status"] in ("finished", "stopped", "error") and not res["events"]:
            break
    flush_batch(final=True)
    job = api.get(f"/api/video/jobs/{job['id']}")
    if job["status"] != "finished":
        raise CliError(f"propagation ended with status {job['status']}: {job.get('error')}")
    issues = [_compact_issue(i) for i in api.get(f"/api/video/tracks/{tid}/issues")]
    lost = job.get("lost") or []
    _activity(api, "track.finished", action="propagate", status="completed", track_id=tid,
                  progress={"done": job["counts"]["frames"], "total": total},
                  issues=issues, lost=lost, message=f"Track {tid} propagation finished")
    if lost:
        nxt = ("tracking LOST the object and stopped (see `lost`; `untracked` frames have no mask yet). "
               "For each: `agentseg sheet --frames <lost_from-5>-<lost_from+40>` (no --track) to see where "
               "the object really is. Object gone (occlusion, cut, out of frame): `track absent` that range "
               "and keyframe it where it comes back. Object still visible: the tracker dropped it, add a "
               "keyframe on a clear frame just after lost_from. Then propagate again.")
    elif issues:
        nxt = "inspect each issue's worst_frame with `agentseg view <frame> --track %d --zoom`" % tid
    else:
        nxt = "no issues: check `agentseg sheet --track %d` then export" % tid
    return {"track_id": tid, "frame_range": [lo, hi], "frames_written": job["counts"]["frames"],
            "seconds": round(time.time() - t0, 1), **({"lost": lost} if lost else {}), "issues": issues,
            "next": nxt}


def cmd_track_propagate(api: Api, a) -> dict:
    return _propagate(api, a.track, a.full, a.start, a.end, not a.no_wait, a.max_lost)


def cmd_track_issues(api: Api, a) -> dict:
    raw = api.get(f"/api/video/tracks/{a.track}/issues")
    _activity(api, "issue.review", action="review_issues", status="completed", track_id=a.track,
                  count=len(raw), message=f"Reviewed {len(raw)} issue(s) for track {a.track}")
    return {"track_id": a.track, "issues": raw if a.verbose else [_compact_issue(i) for i in raw]}


def cmd_track_review(api: Api, a) -> dict:
    ids = None if a.issues == "all" else [int(x) for x in a.issues.split(",") if x.strip()]
    res = api.post(f"/api/video/tracks/{a.track}/issues/review", {"issue_ids": ids, "by": "agent", "note": a.note})
    _activity(api, "issue.reviewed", action="review_issues", status="completed", track_id=a.track,
              issue_ids=res["reviewed"], ranges=res["ranges"], note=a.note,
              message=f"Checked issue(s) {', '.join(map(str, res['reviewed'])) or '-'}: masks are right")
    return {**res, "next": "export when open_issues is 0" if not res["open_issues"] else
            "inspect the remaining issues"}


def cmd_track_refine(api: Api, a) -> dict:
    _activity(api, "mask.refine_started", action="refine", status="running", track_id=a.track,
                  frame=a.frame, preview_id=a.preview, bbox=_box(a.box) if a.box else None,
                  pos=_points(a.pos), neg=_points(a.neg),
                  message=f"Refining track {a.track} on frame {a.frame + 1}")
    body = {"frame": a.frame}
    if a.preview:
        body["preview_id"] = a.preview
    elif a.box:
        body.update(box=_box(a.box), pos=_points(a.pos), neg=_points(a.neg))
    else:
        raise CliError("give --preview P or --box x0,y0,x1,y1", 2)
    res = api.post(f"/api/video/tracks/{a.track}/keyframes", body)
    _activity(api, "keyframe.added", action="refine", status="completed", track_id=a.track,
              frame=a.frame, preview_id=a.preview, message=f"Keyframe on frame {a.frame + 1}")
    out = {"keyframe": res}
    if not a.no_propagate:
        out["propagate"] = _propagate(api, a.track)
    return out


def cmd_track_keyframes(api: Api, a) -> dict:
    out = {}
    if a.rm is not None:
        out["removed"] = api.delete(f"/api/video/tracks/{a.track}/keyframes/{a.rm}")
        _activity(api, "keyframe.removed", action="remove_keyframe", status="completed", track_id=a.track,
                  frame=a.rm, message=f"Removed keyframe on frame {a.rm + 1}")
        if not a.no_propagate:
            out["propagate"] = _propagate(api, a.track)
    t = _get_track(api, a.track)
    out["keyframes"] = t["keyframes"]
    return out


def cmd_track_absent(api: Api, a) -> dict:
    frames = _frames(a.frames)
    res = api.post(f"/api/video/tracks/{a.track}/absent", {"frames": frames})
    _activity(api, "track.absent", action="mark_absent", status="completed", track_id=a.track,
                  frames=frames, message=f"Marked {len(frames)} frame(s) as absent")
    out = {"track_id": a.track, "absent": _ranges(res["frames"])}
    if not a.no_propagate:
        out["propagate"] = _propagate(api, a.track)
    return out


def cmd_track_list(api: Api, a) -> dict:
    return {"tracks": [_track_summary(t) for t in api.get("/api/video/tracks")]}


def cmd_export(api: Api, a) -> dict:
    proj = api.get("/api/projects/current")
    if not proj:
        raise CliError("no project open; run `agentseg project open <dir>`")
    out_dir = Path(a.out or HOME / "export" / f"{proj['name']}_{a.format}").expanduser().resolve()
    common = {"output_dir": str(out_dir), "train_split": 1.0, "approved_only": a.approved_only,
              "include_negative": True, "labels_colocate": False, "labels_only": a.labels_only}
    if a.format == "yolo-seg":
        res = api.post("/api/export/yolo", common)
    else:
        res = api.post("/api/export/bbox", {**common, "format": a.format})
    _activity(api, "export.completed", action="export", status="completed", format=a.format,
                  output_dir=str(out_dir), message=f"Exported {a.format} labels")
    return {"format": a.format, "output_dir": str(out_dir), **res}


AGENT_ACTIVE = ("starting", "running")


def _print_agent_event(ev: dict) -> None:
    kind = ev.get("type", "")
    if kind == "agent.message":
        print(f"[agent] {ev.get('text') or ev.get('message')}", file=sys.stderr)
    elif kind in ("agent.tool.started", "agent.failed", "agent.finished", "agent.status") or (
            kind == "agent.tool.completed" and not ev.get("ok")):
        print(f"[agent] {kind.split('.', 1)[1]}: {ev.get('message')}", file=sys.stderr)


def cmd_agent_run(api: Api, a) -> dict:
    body = {"instruction": a.instruction, "model": a.model, "effort": a.effort, "timeout_s": a.timeout,
            "images": str(Path(a.images).expanduser().resolve()) if a.images else None,
            # keep the agent's CLI state (views, cli_log.jsonl) where this shell's would go
            "agentseg_home": str(HOME.resolve()) if os.getenv("AGENTSEG_HOME") else None}
    run = api.post("/api/agent/runs", body)
    print(f"agent run {run['id']} started (muse pid {run.get('pid')}, model {run.get('model')}); "
          f"transcript {run.get('transcript')}", file=sys.stderr)
    if a.no_wait:
        return {"run": run, "next": f"watch the web UI, or `agentseg agent status {run['id']}`"}
    cursor, printed = None, set()
    try:
        while run["status"] in AGENT_ACTIVE:
            time.sleep(2)
            try:  # the activity feed needs an open project; the agent opens one itself
                res = api.get("/api/activity/events", params={"after": cursor or 0, "limit": 500})
                for ev in res["events"]:
                    if ev.get("run_id") == run["id"] and ev["id"] not in printed:
                        printed.add(ev["id"]); _print_agent_event(ev)
                cursor = res.get("last_seq", cursor)
            except CliError:
                cursor = None
            run = api.get(f"/api/agent/runs/{run['id']}")
    except KeyboardInterrupt:
        api.post(f"/api/agent/runs/{run['id']}/stop")
        raise
    if run["status"] == "error":
        raise CliError(f"agent run failed: {run.get('error')}", run=run)
    return {"run": run}


def cmd_agent_status(api: Api, a) -> dict:
    if a.run:
        return {"run": api.get(f"/api/agent/runs/{a.run}")}
    cfg = api.get("/api/agent/config")
    return {"config": cfg, "runs": api.get("/api/agent/runs")["runs"]}


def cmd_agent_stop(api: Api, a) -> dict:
    return {"run": api.post(f"/api/agent/runs/{a.run}/stop")}


# ---------------------------------------------------------------- argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentseg", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default=None,
                   help=f"backend URL (default: env AGENTSEG_BACKEND, else {DEFAULT_BACKEND} or :8000 if that one is up)")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    s = sub.add_parser("status", help="backend, sam_server and project state")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("up", help="start whatever is down (sam_server, backend, web UI) and print the UI URL")
    s.add_argument("--no-sam", action="store_true", help="do not start sam_server")
    s.add_argument("--no-ui", action="store_true", help="do not build or check the web UI")
    s.add_argument("--rebuild", action="store_true", help="rebuild the web UI even if it looks up to date")
    s.add_argument("--host", default=os.getenv("AGENTSEG_HOST", "127.0.0.1"),
                   help="interface the backend binds to (default 127.0.0.1; 0.0.0.0 exposes it on the network)")
    s.set_defaults(fn=cmd_up)

    s = sub.add_parser("down", help="stop the backend (and sam_server with --all)")
    s.add_argument("--all", action="store_true", help="also stop sam_server (frees the GPU)")
    s.set_defaults(fn=cmd_down)

    s = sub.add_parser("project", help="open or create a project")
    ps = s.add_subparsers(dest="sub", required=True, metavar="SUB")
    o = ps.add_parser("open", help="open a project dir (has agentseg.db) or create one from an image folder")
    o.add_argument("path", help="project dir or image folder")
    o.add_argument("--name", help="project name when creating (default: folder name)")
    o.add_argument("--project-dir", help="where to create the project (default .agentseg/projects/<name>)")
    o.set_defaults(fn=cmd_project_open)

    s = sub.add_parser("labels", help="list labels; --add NAME creates one")
    s.add_argument("--add", metavar="NAME")
    s.set_defaults(fn=cmd_labels)

    s = sub.add_parser("view", help="render one frame with overlays to a PNG")
    s.add_argument("frame", type=int)
    s.add_argument("--track", type=int, help="only this track's masks")
    s.add_argument("--grid", action="store_true", help="draw a 50 px pixel-coordinate grid")
    s.add_argument("--preview", help="overlay a segment preview")
    s.add_argument("--zoom", action="store_true",
                   help="close-up, outline only, next to the track's nearest keyframe (needs --track or --preview)")
    s.add_argument("--out", help="output folder (default .agentseg/views)")
    s.set_defaults(fn=cmd_view)

    s = sub.add_parser("sheet", help="contact sheet of many frames (status-coloured with --track)")
    s.add_argument("--track", type=int)
    s.add_argument("--every", type=int, help="every Nth frame (default: about 24 thumbnails)")
    s.add_argument("--frames", help="explicit frames, e.g. 100-110,140")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_sheet)

    s = sub.add_parser("segment", help="preview a mask from a box (+ optional clicks); nothing is saved")
    s.add_argument("frame", type=int)
    s.add_argument("--box", required=True, help="x0,y0,x1,y1 in pixels")
    s.add_argument("--pos", action="append", metavar="X,Y", help="positive click (repeatable)")
    s.add_argument("--neg", action="append", metavar="X,Y", help="negative click (repeatable)")
    s.add_argument("--no-grid", action="store_true", help="render the preview without the grid")
    s.add_argument("--track", type=int, help="track this keyframe is for: the zoom shows its keyframe for comparison")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_segment)

    s = sub.add_parser("track", help="track commands")
    ts = s.add_subparsers(dest="sub", required=True, metavar="SUB")

    t = ts.add_parser("create", help="create a track from an approved preview (first keyframe)")
    t.add_argument("--label", required=True, help="label name (created if missing)")
    t.add_argument("--preview", required=True)
    t.add_argument("--name")
    t.set_defaults(fn=cmd_track_create)

    t = ts.add_parser("propagate", help="propagate keyframes through the video, then list issues")
    t.add_argument("track", type=int)
    t.add_argument("--full", action="store_true", help="regenerate every frame, not just the dirty range")
    t.add_argument("--start", type=int); t.add_argument("--end", type=int)
    t.add_argument("--no-wait", action="store_true", help="return right after starting the job")
    t.add_argument("--max-lost", type=int, default=MAX_LOST,
                   help=f"stop a direction after N empty frames in a row (default {MAX_LOST}, 0 = never)")
    t.set_defaults(fn=cmd_track_propagate)

    t = ts.add_parser("issues", help="list health issues of a track")
    t.add_argument("track", type=int)
    t.add_argument("--verbose", action="store_true", help="include raw signals")
    t.set_defaults(fn=cmd_track_issues)

    t = ts.add_parser("review", help="mark issues you checked as fine (their frames become approved)")
    t.add_argument("track", type=int)
    t.add_argument("issues", help="issue ids, e.g. 2,5, or 'all' for every open issue")
    t.add_argument("--note", help="one line why the masks are right")
    t.set_defaults(fn=cmd_track_review)

    t = ts.add_parser("refine", help="add/replace a keyframe at FRAME, then re-propagate")
    t.add_argument("track", type=int); t.add_argument("frame", type=int)
    t.add_argument("--preview", help="preview id from `segment` on the same frame")
    t.add_argument("--box", help="x0,y0,x1,y1 (segments directly, no preview check)")
    t.add_argument("--pos", action="append", metavar="X,Y"); t.add_argument("--neg", action="append", metavar="X,Y")
    t.add_argument("--no-propagate", action="store_true", help="batch several edits, propagate later")
    t.set_defaults(fn=cmd_track_refine)

    t = ts.add_parser("keyframes", help="list keyframes; --rm FRAME deletes one and re-propagates")
    t.add_argument("track", type=int)
    t.add_argument("--rm", type=int, metavar="FRAME")
    t.add_argument("--no-propagate", action="store_true")
    t.set_defaults(fn=cmd_track_keyframes)

    t = ts.add_parser("absent", help="mark frames where the object is not visible, e.g. 120-139,150")
    t.add_argument("track", type=int); t.add_argument("frames")
    t.add_argument("--no-propagate", action="store_true")
    t.set_defaults(fn=cmd_track_absent)

    t = ts.add_parser("list", help="tracks with keyframes, absent ranges, coverage, issue count")
    t.set_defaults(fn=cmd_track_list)

    s = sub.add_parser("agent", help="run the labeling agent (Muse Code, headless) through the backend")
    ags = s.add_subparsers(dest="sub", required=True, metavar="SUB")
    g = ags.add_parser("run", help="start a labeling run; streams what the agent does until it ends")
    g.add_argument("instruction", help='what to label, e.g. "label the torpedo"')
    g.add_argument("--images", help="image folder or project dir to label")
    g.add_argument("--model", help="Muse model id (default muse-spark-1.3-contributor, env AGENTSEG_AGENT_MODEL)")
    g.add_argument("--effort", choices=["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"])
    g.add_argument("--timeout", type=float, help="stop the run after this many seconds (default 2700)")
    g.add_argument("--no-wait", action="store_true", help="return right after the run starts")
    g.set_defaults(fn=cmd_agent_run)
    g = ags.add_parser("status", help="agent settings and runs, or one run")
    g.add_argument("run", nargs="?", help="run id or `latest`")
    g.set_defaults(fn=cmd_agent_status)
    g = ags.add_parser("stop", help="stop a run")
    g.add_argument("run", nargs="?", default="latest", help="run id (default: latest)")
    g.set_defaults(fn=cmd_agent_stop)

    s = sub.add_parser("export", help="export labels")
    s.add_argument("--format", choices=["yolo-seg", "yolo-detect", "coco"], default="yolo-seg")
    s.add_argument("--out", help="output folder (default .agentseg/export/<project>_<format>)")
    s.add_argument("--approved-only", action="store_true", help="skip frames health flagged for review")
    s.add_argument("--labels-only", action="store_true", help="do not copy images")
    s.set_defaults(fn=cmd_export)
    return p


def _log(argv, code, seconds):
    try:
        HOME.mkdir(parents=True, exist_ok=True)
        with open(HOME / "cli_log.jsonl", "a") as f:
            f.write(json.dumps({"ts": time.time(), "argv": argv, "exit": code, "seconds": round(seconds, 2)}) + "\n")
    except OSError:
        pass


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    t0 = time.time()
    parser = build_parser()
    try:
        a = parser.parse_args(argv)
    except SystemExit as e:
        _log(argv, e.code, time.time() - t0)
        raise
    code = 0
    run_id = uuid.uuid4().hex[:12]
    command = "agentseg " + " ".join(argv)
    api = Api(a.backend or _default_backend())
    # Keep compatibility with lightweight Api doubles used by integrations and
    # tests while real Api instances retain correlation metadata.
    if hasattr(api, "run_id"):
        api.run_id, api.command = run_id, command
    _activity(api, "command.started", action=a.cmd, status="running", message=f"Started {command}")
    try:
        result = a.fn(api, a)
        print(json.dumps({"ok": True, **result}, indent=1))
        _activity(api, "command.completed", action=a.cmd, status="completed", message=f"Completed {command}")
    except CliError as e:
        code = e.code
        print(json.dumps({"ok": False, "error": str(e), **e.extra}))
        _activity(api, "command.failed", action=a.cmd, status="error", message=str(e), error=str(e))
    except KeyboardInterrupt:
        code = 130
        print(json.dumps({"ok": False, "error": "interrupted"}))
        _activity(api, "command.failed", action=a.cmd, status="error", message="Command interrupted")
    _log(argv, code, time.time() - t0)
    return code


if __name__ == "__main__":
    sys.exit(main())
