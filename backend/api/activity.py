"""Live activity and artifact feed shared by the CLI and the web editor.

The feed is deliberately small and transport agnostic.  Clients publish
semantic events (``segment.started``, ``track.progress`` and so on), while
the browser consumes the same persisted history through HTTP or a live SSE
stream.  Keeping this at the API boundary means the CLI and the frontend can
evolve independently.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.api.projects import get_project, get_store

router = APIRouter()

_LOCK = threading.RLock()
_SUBSCRIBERS: dict[int, set[queue.Queue]] = {}
_ARTIFACTS: dict[str, tuple[int, str, str]] = {}


def _ensure_tables(conn) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS activity_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            run_id TEXT,
            source TEXT NOT NULL DEFAULT 'system',
            type TEXT NOT NULL,
            action TEXT,
            status TEXT,
            message TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_activity_project_id ON activity_events(project_id, id)"
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS activity_artifacts (
            id TEXT PRIMARY KEY,
            project_id INTEGER NOT NULL,
            path TEXT NOT NULL,
            mime_type TEXT NOT NULL DEFAULT 'application/octet-stream',
            kind TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        )"""
    )
    conn.commit()


def _project_context(project_id: Optional[int] = None):
    project, store = get_project(), get_store()
    if project_id is not None and project.id != project_id:
        raise HTTPException(status_code=409, detail="Activity belongs to another project")
    with _LOCK:
        _ensure_tables(store.conn)
    return project, store


def publish(
    event_type: str,
    *,
    source: str = "system",
    action: Optional[str] = None,
    status: Optional[str] = None,
    run_id: Optional[str] = None,
    message: Optional[str] = None,
    project_id: Optional[int] = None,
    **data: Any,
) -> dict:
    """Persist and broadcast one activity event.

    This helper is safe to call from worker threads.  It intentionally accepts
    arbitrary JSON data so existing job event payloads can be forwarded without
    a second translation layer.
    """
    project, store = _project_context(project_id)
    now = time.time()
    payload = {k: v for k, v in data.items() if v is not None}
    # IDs are scoped to a project DB and commonly repeat across CLI projects.
    # Let an already-open editor reject events from a different DB immediately.
    payload["project_root_dir"] = project.root_dir
    with _LOCK:
        cur = store.conn.execute(
            """INSERT INTO activity_events
               (project_id, run_id, source, type, action, status, message, payload_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (project.id, run_id, source, event_type, action, status, message,
             json.dumps(payload, separators=(",", ":"), default=str), now),
        )
        try:
            store.conn.commit()
        except sqlite3.OperationalError as exc:
            # the connection is shared across request threads; another thread's commit already
            # committed this insert
            if "no transaction is active" not in str(exc):
                raise
        event_id = int(cur.lastrowid)
    event = {
        "id": event_id,
        "seq": event_id,
        "project_id": project.id,
        "run_id": run_id,
        "source": source,
        "type": event_type,
        "action": action,
        "status": status,
        "message": message,
        "ts": now,
        **payload,
    }
    with _LOCK:
        subscribers = list(_SUBSCRIBERS.get(project.id, ()))
    for subscriber in subscribers:
        try:
            subscriber.put_nowait(event)
        except queue.Full:
            # A slow browser can replay from its cursor; never block a worker.
            try:
                subscriber.get_nowait()
                subscriber.put_nowait(event)
            except queue.Empty:
                pass
    return event


def _rows(store, project_id: int, after: int, limit: int, *, latest: bool = False) -> list[dict]:
    order = "DESC" if latest else "ASC"
    with _LOCK:
        rows = store.conn.execute(
            """SELECT id, project_id, run_id, source, type, action, status, message,
                      payload_json, created_at
               FROM activity_events WHERE project_id = ? AND id > ?
               ORDER BY id {order} LIMIT ?""".format(order=order), (project_id, after, limit)
        ).fetchall()
    out = []
    for row in rows:
        try:
            payload = json.loads(row[8] or "{}")
        except (TypeError, ValueError):
            payload = {}
        out.append({"id": row[0], "seq": row[0], "project_id": row[1], "run_id": row[2],
                    "source": row[3], "type": row[4], "action": row[5], "status": row[6],
                    "message": row[7], "ts": row[9], **payload})
    return list(reversed(out)) if latest else out


def events_between(t0: float, t1: float, *, skip_types: tuple[str, ...] = ()) -> list[dict]:
    """Every event of the open project published between two timestamps (a recorded run)."""
    project, store = _project_context()
    with _LOCK:
        rows = store.conn.execute(
            """SELECT id, project_id, run_id, source, type, action, status, message,
                      payload_json, created_at
               FROM activity_events WHERE project_id = ? AND created_at >= ? AND created_at <= ?
               ORDER BY id ASC""", (project.id, t0, t1)
        ).fetchall()
    out = []
    for row in rows:
        if row[4] in skip_types:
            continue
        try:
            payload = json.loads(row[8] or "{}")
        except (TypeError, ValueError):
            payload = {}
        out.append({"id": row[0], "seq": row[0], "project_id": row[1], "run_id": row[2],
                    "source": row[3], "type": row[4], "action": row[5], "status": row[6],
                    "message": row[7], "ts": row[9], **payload})
    return out


class ActivityEventRequest(BaseModel):
    type: str = Field(min_length=1, max_length=120)
    source: str = Field(default="cli", max_length=40)
    action: Optional[str] = Field(default=None, max_length=120)
    status: Optional[str] = Field(default=None, max_length=40)
    run_id: Optional[str] = Field(default=None, max_length=120)
    message: Optional[str] = Field(default=None, max_length=1000)
    project_id: Optional[int] = None
    data: dict[str, Any] = Field(default_factory=dict)


class ArtifactRequest(BaseModel):
    path: str
    kind: str = Field(default="image", max_length=80)
    mime_type: str = Field(default="image/png", max_length=120)
    project_id: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.post("/events")
def create_event(req: ActivityEventRequest):
    return publish(req.type, source=req.source, action=req.action, status=req.status,
                   run_id=req.run_id, message=req.message, project_id=req.project_id, **req.data)


@router.get("/events")
def list_events(after: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500),
                project_id: Optional[int] = None):
    project, store = _project_context(project_id)
    events = _rows(store, project.id, after, limit, latest=(after == 0))
    with _LOCK:
        row = store.conn.execute("SELECT COALESCE(MAX(id), ?) FROM activity_events WHERE project_id = ?",
                                 (after, project.id)).fetchone()
    return {"events": events, "last_seq": int(row[0]) if row else after}


@router.get("/stream")
async def stream_events(after: int = Query(0, ge=0), project_id: Optional[int] = None):
    project, store = _project_context(project_id)
    subscriber: queue.Queue = queue.Queue(maxsize=256)
    with _LOCK:
        _SUBSCRIBERS.setdefault(project.id, set()).add(subscriber)

    async def generate():
        cursor = after
        try:
            # Subscribe before replaying so events cannot be missed on reload.
            for event in _rows(store, project.id, cursor, 500):
                cursor = event["id"]
                yield f"id: {cursor}\nevent: activity\ndata: {json.dumps(event)}\n\n"
            while True:
                try:
                    event = await asyncio.to_thread(subscriber.get, True, 15.0)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if event["id"] <= cursor:
                    continue
                cursor = event["id"]
                yield f"id: {cursor}\nevent: activity\ndata: {json.dumps(event)}\n\n"
        finally:
            with _LOCK:
                listeners = _SUBSCRIBERS.get(project.id)
                if listeners:
                    listeners.discard(subscriber)
                    if not listeners:
                        _SUBSCRIBERS.pop(project.id, None)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/artifacts")
def register_artifact(req: ArtifactRequest):
    project, store = _project_context(req.project_id)
    path = Path(req.path).expanduser().resolve()
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    artifact_id = uuid.uuid4().hex
    now = time.time()
    with _LOCK:
        store.conn.execute(
            """INSERT INTO activity_artifacts
               (id, project_id, path, mime_type, kind, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (artifact_id, project.id, str(path), req.mime_type, req.kind,
             json.dumps(req.metadata, default=str), now),
        )
        store.conn.commit()
    with _LOCK:
        _ARTIFACTS[artifact_id] = (project.id, str(path), req.mime_type)
    return {"artifact_id": artifact_id, "url": f"/api/activity/artifacts/{artifact_id}",
            "kind": req.kind, "mime_type": req.mime_type, "metadata": req.metadata}


@router.get("/artifacts/{artifact_id}")
def get_artifact(artifact_id: str):
    project, store = _project_context()
    with _LOCK:
        row = store.conn.execute(
            "SELECT path, mime_type FROM activity_artifacts WHERE id = ? AND project_id = ?",
            (artifact_id, project.id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    path = Path(row[0])
    if not path.is_file():
        raise HTTPException(status_code=410, detail="Artifact file is no longer available")
    return FileResponse(path, media_type=row[1])
