"""Turn Muse Code's headless output (`muse exec --json`) into activity events.

`muse exec --json` prints one event-log record per line:
``{"payload_type": "...", "payload": {...}, "stream": {"kind": "session", "id": ...}, ...}``.
The records used here:

* ``run.output.delta``            assistant text, streamed in chunks
* ``task.lifecycle.*``            per-task lifecycle; tool tasks have
  ``task_kind = "tool.<name>"``, model calls ``task_kind = "model.<provider>.response"``.
  ``scheduled`` carries ``idempotency_key = "tool:<call_id>"``; ``output`` carries
  the tool's result text; ``completed`` / ``failed`` end the task.
* ``run.usage`` / ``run.terminal.{completed,failed,cancelled}``  end of the run

The stdout stream has no tool arguments until the tool finishes. Muse's own
session log (``~/.local/share/muse/sessions/Y/M/D/<session>/session.jsonl``)
records them when the model commits the call (``runtime.session`` records with
``event.kind = "assistant_tool_calls_committed"``), so the mapper reads new lines
from that file just before it reports a tool start. Without the file, tool
starts carry only the tool name and the arguments arrive with the result.

Every event this module produces is a dict ``{"type", "status", "message", ...data}``
that the launcher publishes with ``source="agent"`` (see backend/api/agent.py).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

OUTPUT_LIMIT = 4000        # characters of tool output kept on an event
MESSAGE_LIMIT = 240        # characters in the one-line `message` summary

_SHELL_OPS = {"&&", "||", ";", "|", "&", "\n", "(", ")"}
_VIEW_RE = {
    "frame": re.compile(r"^frame_(?P<frame>\d+)(?:_t(?P<track>\d+))?(?P<preview>_preview)?(?P<grid>_grid)?\.png$"),
    "preview": re.compile(r"^preview_(?P<frame>\d+)_(?P<pid>[0-9a-f]+)\.png$"),
    "sheet": re.compile(r"^sheet_(?:t(?P<track>\d+)_)?(?P<tag>.+)\.png$"),
    "zoom": re.compile(r"^zoom_(?P<frame>\d+)(?:_t(?P<track>\d+))?(?:_(?P<pid>[0-9a-f]{16}))?\.png$"),
}


def _short(text: Any, limit: int = MESSAGE_LIMIT) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _clip(text: Optional[str], limit: int = OUTPUT_LIMIT) -> Optional[str]:
    if text is None or len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} more characters]"


def muse_sessions_root() -> Path:
    base = os.getenv("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "muse" / "sessions"


# ------------------------------------------------------------ agentseg commands

def _floats(value: Any) -> Any:
    try:
        return [float(x) for x in str(value).split(",")]
    except ValueError:
        return value


def parse_agentseg_commands(command: str) -> list[dict]:
    """Every `agentseg …` invocation inside a shell command line, parsed with the
    CLI's own argparse definition.

    Returns ``[{"op": "segment", "argv": [...], "args": {...}}]``; `op` is the
    command path joined by dots (``track.propagate``, ``project.open``). Boxes
    become ``[x0, y0, x1, y1]`` and points ``[[x, y], ...]``.
    """
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return []
    segments, current = [], []
    for tok in tokens:
        if tok in _SHELL_OPS or set(tok) <= set(";&|()"):
            segments.append(current); current = []
        else:
            current.append(tok)
    segments.append(current)

    out = []
    for seg in segments:
        for i, tok in enumerate(seg):
            name = os.path.basename(tok)
            if name in ("agentseg", "autoseg"):
                argv = seg[i + 1:]
                break
            if tok == "cli.main" and i > 0 and seg[i - 1] == "-m":
                argv = seg[i + 1:]
                break
        else:
            continue
        out.append(_parse_agentseg_argv(argv))
    return out


def _parse_agentseg_argv(argv: list[str]) -> dict:
    from cli.main import build_parser  # lazy: keeps this module importable on its own

    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            ns, _unknown = build_parser().parse_known_args(argv)
    except SystemExit:  # argparse reports bad argv (or --help) by exiting
        return {"op": argv[0] if argv else None, "argv": argv, "parse_error": True}
    args = {k: v for k, v in vars(ns).items() if k not in ("fn", "cmd", "sub", "backend") and v is not None}
    if "box" in args:
        args["box"] = _floats(args["box"])
    for key in ("pos", "neg"):
        if key in args:
            args[key] = [_floats(p) for p in args[key]]
    if isinstance(args.get("frames"), str):
        from cli.main import CliError, _frames
        try:
            args["frame_list"] = _frames(args["frames"])
        except CliError:
            pass
    op = ns.cmd + (f".{ns.sub}" if getattr(ns, "sub", None) else "")
    return {"op": op, "argv": argv, "args": args}


def parse_view_path(path: str) -> Optional[dict]:
    """Recognise PNGs written by `agentseg view/segment/sheet` (frame overlays,
    mask previews, contact sheets) so the UI knows what the agent looked at."""
    name = os.path.basename(path or "")
    m = _VIEW_RE["preview"].match(name)
    if m:
        return {"kind": "mask_preview", "frame": int(m["frame"]), "preview_id": m["pid"], "path": path}
    m = _VIEW_RE["frame"].match(name)
    if m:
        view = {"kind": "frame_overlay", "frame": int(m["frame"]), "grid": bool(m["grid"]), "path": path}
        if m["track"]:
            view["track_id"] = int(m["track"])
        if m["preview"]:
            view["kind"] = "mask_preview"
        return view
    m = _VIEW_RE["zoom"].match(name)
    if m:
        view = {"kind": "zoom", "frame": int(m["frame"]), "path": path}
        if m["track"]:
            view["track_id"] = int(m["track"])
        if m["pid"]:
            view["preview_id"] = m["pid"]
        return view
    m = _VIEW_RE["sheet"].match(name)
    if m:
        view = {"kind": "contact_sheet", "tag": m["tag"], "path": path}
        if m["track"]:
            view["track_id"] = int(m["track"])
        return view
    if name.lower().endswith((".png", ".jpg", ".jpeg")):
        return {"kind": "image", "path": path}
    return None


# ------------------------------------------------------------ session log tail

class SessionLogTail:
    """Reads Muse's session.jsonl incrementally to learn tool-call arguments."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or muse_sessions_root()
        self.session_id: Optional[str] = None
        self.path: Optional[Path] = None
        self._pos = 0
        self._partial = ""

    def records(self) -> Iterable[dict]:
        if self.path is None and self.session_id:
            hits = sorted(self.root.glob(f"*/*/*/{self.session_id}/session.jsonl"))
            self.path = hits[0] if hits else None
        if self.path is None:
            return []
        try:
            with open(self.path, "r", errors="replace") as fh:
                fh.seek(self._pos)
                chunk = fh.read()
                self._pos = fh.tell()
        except OSError:
            return []
        data = self._partial + chunk
        lines = data.split("\n")
        self._partial = lines.pop()  # incomplete last line, if any
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out


# ------------------------------------------------------------ the mapper

class MuseEventMapper:
    """Stateful: feed() one Muse record at a time, get back activity events."""

    def __init__(self, session_log: Optional[SessionLogTail] = None,
                 on_session: Optional[Callable[[str], None]] = None):
        self.session_log = session_log
        self.on_session = on_session
        self.session_id: Optional[str] = None
        self.model: Optional[str] = None
        self.terminal: Optional[str] = None     # completed / failed / cancelled once seen
        self.final_text: str = ""
        self.last_message: str = ""
        self.usage: Optional[dict] = None
        self._text: list[str] = []               # pending assistant text deltas
        self._tasks: dict[str, dict] = {}        # task_id -> {"tool", "call_id", "output": [...]}
        self._args: dict[str, dict] = {}         # call_id -> parsed args (from session log)
        self._model_steps = 0
        self._said: set = set()                   # commentary message ids already published

    # -- public

    def feed(self, record: dict) -> list[dict]:
        ptype = record.get("payload_type") or ""
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        out: list[dict] = []
        if ptype != "run.output.delta":
            out += self._flush_text()
        stream = record.get("stream") or {}
        if self.session_id is None and stream.get("kind") == "session" and stream.get("id"):
            self.session_id = stream["id"]
            if self.session_log is not None:
                self.session_log.session_id = self.session_id
            if self.on_session:
                self.on_session(self.session_id)

        if ptype == "run.output.delta":
            self._text.append(str(payload.get("text") or ""))
        elif ptype == "run.model.configured":
            self.model = payload.get("model_id") or self.model
        elif ptype.startswith("task.lifecycle."):
            out += self._task(ptype.rsplit(".", 1)[1], payload)
        elif ptype == "run.usage":
            self.usage = _usage(payload)
        elif ptype.startswith("run.terminal."):
            out += self._pump_session_log() + self._terminal(payload)
        return out

    def flush(self) -> list[dict]:
        return self._pump_session_log() + self._flush_text()

    # -- internals

    def _flush_text(self) -> list[dict]:
        if not self._text:
            return []
        text = "".join(self._text).strip()
        self._text = []
        if not text:
            return []
        self.last_message = text
        return [{"type": "agent.message", "status": None, "message": _short(text), "text": text}]

    def _task(self, phase: str, payload: dict) -> list[dict]:
        event = payload.get("event") if isinstance(payload.get("event"), dict) else {}
        task_id = payload.get("task_id") or event.get("task_id")
        if not task_id:
            return []
        if phase == "proposed":
            kind = str(event.get("task_kind") or "")
            if kind.startswith("tool."):
                self._tasks[task_id] = {"tool": kind[len("tool."):], "call_id": None, "output": []}
            elif kind.startswith("model.") and kind.endswith(".response"):
                self._tasks[task_id] = {"model": True}
            return []
        task = self._tasks.get(task_id)
        if task is None:
            return []
        if task.get("model"):
            if phase == "started":
                self._model_steps += 1
                return [{"type": "agent.thinking", "status": "running", "message": "Thinking",
                         "step": self._model_steps}]
            if phase == "status" and (event.get("details") or {}).get("phase") == "retry_scheduled":
                return [{"type": "agent.status", "status": None,
                         "message": _short(event.get("message") or "Retrying the model request")}]
            if phase in ("completed", "failed", "cancelled", "timed_out"):
                self._tasks.pop(task_id, None)
            return []

        if phase == "scheduled":
            key = str(event.get("idempotency_key") or "")
            if key.startswith("tool:"):
                task["call_id"] = key[len("tool:"):]
            return []
        if phase == "started":
            said = self._pump_session_log()
            args = self._args.get(task.get("call_id") or "")
            task["args"] = args
            ev = {"type": "agent.tool.started", "status": "running", "tool": task["tool"],
                  "call_id": task.get("call_id"), "task_id": task_id}
            ev.update(_tool_facts(task["tool"], args))
            ev["message"] = _tool_message(ev, started=True)
            return said + [ev]
        if phase == "output":
            task["output"].append(str(event.get("chunk") or ""))
            return []
        if phase in ("completed", "failed", "cancelled", "timed_out"):
            self._tasks.pop(task_id, None)
            said: list[dict] = []
            if not task.get("args"):  # the session log may have been written after the start
                said = self._pump_session_log()
                task["args"] = self._args.get(task.get("call_id") or "")
            return said + [self._tool_done(task, task_id, phase, event)]
        return []

    def _tool_done(self, task: dict, task_id: str, phase: str, event: dict) -> dict:
        text = "".join(task["output"])
        ok = phase == "completed"
        ev: dict[str, Any] = {"type": "agent.tool.completed", "status": "completed" if ok else "failed",
                              "ok": ok, "tool": task["tool"], "call_id": task.get("call_id"),
                              "task_id": task_id}
        if not ok:
            ev["error"] = _short(event.get("reason") or phase, 1000)
        args = task.get("args")
        result: Any = None
        if task["tool"] == "bash":
            try:
                result = json.loads(text)
            except ValueError:
                result = None
            if isinstance(result, dict):
                if not args and result.get("command"):
                    args = {"command": result.get("command"), "description": result.get("description")}
                ev["exit_code"] = result.get("exit_code")
                text = str(result.get("output") or "")
                if result.get("exit_code") not in (None, 0):
                    ev["ok"] = False; ev["status"] = "failed"
                    ev.setdefault("error", _short(result.get("terminal_reason") or "non-zero exit", 1000))
        ev.update(_tool_facts(task["tool"], args))
        ev["output"] = _clip(text)
        if ev.get("agentseg"):
            parsed = _json_object(text)
            if parsed is not None:
                ev["agentseg_result"] = _compact_cli_result(parsed)
        ev["message"] = _tool_message(ev, started=False)
        return ev

    def _terminal(self, payload: dict) -> list[dict]:
        terminal = str(payload.get("terminal") or "completed")
        self.terminal = terminal
        self.final_text = self.last_message or str(payload.get("text") or "")
        base = {"usage": self.usage, "model": self.model, "session_id": self.session_id}
        if terminal == "completed":
            return [{"type": "agent.finished", "status": "completed",
                     "message": "Agent finished", "text": _clip(self.final_text), **base}]
        if terminal == "cancelled":
            return [{"type": "agent.finished", "status": "stopped",
                     "message": "Agent run cancelled", "text": _clip(self.final_text), **base}]
        reason = payload.get("reason") or "run failed"
        return [{"type": "agent.failed", "status": "error", "message": _short(f"Agent failed: {reason}"),
                 "error": str(reason), **base}]

    def _pump_session_log(self) -> list[dict]:
        """Learn tool arguments, and return the agent's commentary lines.

        `muse exec --json` only streams the final answer; the short sentences
        the model writes next to its tool calls ("I will scan the overview
        sheet ...") are committed to the session log with phase "commentary"."""
        if self.session_log is None:
            return []
        said: list[dict] = []
        for rec in self.session_log.records():
            payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
            ev = payload.get("event") if isinstance(payload.get("event"), dict) else {}
            if ev.get("kind") == "assistant_message_committed" and ev.get("phase") == "commentary":
                text = str(ev.get("text") or "").strip()
                if text and ev.get("message_id") not in self._said:
                    self._said.add(ev.get("message_id"))
                    said.append({"type": "agent.message", "status": None, "message": _short(text),
                                 "text": text, "phase": "commentary"})
                continue
            if ev.get("kind") != "assistant_tool_calls_committed":
                continue
            for call in ev.get("tool_calls") or []:
                if not isinstance(call, dict) or not call.get("call_id"):
                    continue
                raw = call.get("args")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except ValueError:
                    args = {"raw_args": raw}
                self._args[call["call_id"]] = args if isinstance(args, dict) else {"raw_args": raw}
        return said


def _usage(payload: dict) -> dict:
    models = payload.get("models") or []
    tot = {k: sum(int(m.get(k) or 0) for m in models)
           for k in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens")}
    return {**tot, "calls": payload.get("calls"), "failed_calls": payload.get("failed_calls")}


def _tool_facts(tool: str, args: Optional[dict]) -> dict:
    if not args:
        return {}
    facts: dict[str, Any] = {"args": args}
    if tool == "bash":
        cmd = str(args.get("command") or "")
        facts["command"] = cmd
        if args.get("description"):
            facts["description"] = args["description"]
        calls = parse_agentseg_commands(cmd) if "agentseg" in cmd or "cli.main" in cmd else []
        if calls:
            facts["agentseg"] = calls
    elif tool == "read_file":
        path = str(args.get("path") or "")
        facts["path"] = path
        view = parse_view_path(path)
        if view:
            facts["view"] = view
    elif tool == "read_skill":
        facts["skill"] = args.get("name")
    elif args.get("path"):
        facts["path"] = args["path"]
    return facts


_TOOL_VERBS = {"bash": "Running a command", "read_file": "Reading a file", "write_file": "Writing a file",
               "edit_file": "Editing a file", "search": "Searching the repo", "read_skill": "Reading a skill",
               "write_todos": "Updating the plan"}


def _tool_message(ev: dict, started: bool) -> str:
    tool = ev.get("tool")
    if ev.get("agentseg"):
        what = "agentseg " + " ".join(ev["agentseg"][0]["argv"])
    elif ev.get("description"):
        what = ev["description"]
    elif ev.get("command"):
        what = ev["command"]
    elif ev.get("view"):
        v = ev["view"]
        what = f"Looking at {v['kind'].replace('_', ' ')}" + (f" (frame {v['frame'] + 1})" if "frame" in v else "")
    elif ev.get("path"):
        what = f"{tool} {ev['path']}"
    elif ev.get("skill"):
        what = f"Reading skill {ev['skill']}"
    else:
        what = _TOOL_VERBS.get(str(tool), str(tool))
    if not started and not ev.get("ok"):
        what = f"Failed: {what}"
    return _short(what)


def _json_object(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text.startswith("{"):
        start = text.find("\n{")
        if start < 0:
            return None
        text = text[start + 1:]
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _compact_cli_result(res: dict) -> dict:
    """The parts of an `agentseg` JSON reply worth showing; never the full payload."""
    keep = ("ok", "error", "next", "image", "preview_id", "frame", "bbox", "score", "area",
            "track_id", "frame_range", "frames_written", "seconds", "absent", "format", "output_dir",
            "created", "frames", "ui")
    out = {k: res[k] for k in keep if k in res and not (k == "frames" and isinstance(res[k], list))}
    if isinstance(res.get("frames"), int):
        out["frames"] = res["frames"]
    if isinstance(res.get("issues"), list):
        out["issue_count"] = len(res["issues"])
        out["issues"] = res["issues"][:20]
    if isinstance(res.get("keyframe"), dict):
        out["keyframe"] = res["keyframe"]
    if isinstance(res.get("propagate"), dict):
        out["propagate"] = _compact_cli_result(res["propagate"])
    if isinstance(res.get("artifact"), dict):
        out["artifact"] = {k: res["artifact"].get(k) for k in ("url", "kind")}
    return out
