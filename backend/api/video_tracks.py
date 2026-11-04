"""SAM3 video tracking API (WP4).

The router deliberately keeps the HTTP surface small: previews live in a bounded
in-process cache, while keyframes, absent frames, propagated annotations and
health issues are persisted in the project database.
"""
from __future__ import annotations

import io
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from backend.api.projects import get_project, get_store
from backend.api.activity import publish as publish_activity
from core.masks import drop_specks, mask_area, mask_to_bbox, mask_to_rle, rle_to_mask
from core.polygons import mask_to_yolo_polygon
from ml.sam3_video import FrameSet, Sam3Client, TrackEngine
from ml.sam3_video.health import HealthConfig, assess

router = APIRouter()


@dataclass
class Preview:
    id: str
    project_id: int
    frame: int
    mask: np.ndarray
    score: float
    bbox: list[float]
    created: float = field(default_factory=time.time)


class VideoTrackJob:
    """Background TrackEngine run; clients poll its events."""
    def __init__(self, project_id: int, track_id: int, engine: TrackEngine,
                 store_db: str, frame_range: Optional[tuple[int, int]] = None,
                 full: bool = False, max_lost: int = 0):
        self.id = uuid.uuid4().hex[:12]
        self.project_id, self.track_id = project_id, track_id
        self.engine, self.store_db = engine, store_db
        self.frame_range, self.full, self.max_lost = frame_range, full, max_lost
        self.lost: list[dict] = []  # segments cut short because the tracker lost the object
        self.status = "pending"
        self.error: Optional[str] = None
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.counts = {"frames": 0, "issues": 0}
        self._events: list[dict] = []
        self._seq = 0
        self._cond = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self.run, daemon=True, name=f"video-track-{self.id}")

    @property
    def is_active(self):
        return self.status not in {"finished", "stopped", "error"}

    @property
    def last_seq(self):
        return self._seq

    def emit(self, typ: str, **data):
        with self._cond:
            self._seq += 1
            ev = {"seq": self._seq, "type": typ, "job_id": self.id, "track_id": self.track_id,
                  "ts": time.time(), **data}
            self._events.append(ev)
            self._cond.notify_all()
        # Mirror video-job events into the shared activity feed so browser
        # sessions can observe work started by either the CLI or the editor.
        # A frame event is also the visual hand-off to the editor.  Keep every
        # frame in the activity stream so the canvas can paint masks as soon as
        # SAM produces them.  The payload stays compact (bbox + render URL),
        # while the native job endpoint remains available for detailed polling.
        should_publish = True
        if should_publish:
            try:
                publish_activity(
                    f"video_job.{typ}", source="backend", action="propagate",
                    status=("running" if typ not in {"finished", "stopped", "error"} else
                            ("error" if typ == "error" else "completed")),
                    run_id=self.id, project_id=self.project_id, track_id=self.track_id,
                    **data,
                )
            except Exception:
                pass
        return ev

    def events_after(self, after=0):
        with self._cond:
            return [e for e in self._events if e["seq"] > after]

    def wait_events(self, after=0, timeout=15.0):
        end = time.time() + timeout
        with self._cond:
            while True:
                out = [e for e in self._events if e["seq"] > after]
                if out or not self.is_active or time.time() >= end:
                    return out
                self._cond.wait(max(0.01, end - time.time()))

    def snapshot(self):
        return {"id": self.id, "project_id": self.project_id, "track_id": self.track_id,
                "status": self.status, "frame_range": self.frame_range,
                "full": self.full, "counts": dict(self.counts), "error": self.error, "lost": self.lost,
                "last_seq": self.last_seq, "created_at": self.started_at,
                "finished_at": self.finished_at}

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop = True

    def _restore_unchanged(self, store, lo: int, hi: int, written: set[int]) -> None:
        """The range was cleared up front, but the engine only recomputes segments that changed:
        write back its still-valid masks for the frames it skipped (an untouched keyframe's
        segments inside a wide dirty range)."""
        t = self.engine.track(self.track_id)
        track = store.get_track(self.track_id)
        for frame in range(lo, hi + 1):
            if frame in written or frame in t.keyframes:
                continue
            m = t.masks.get(frame)
            if m is None or not m.any():
                continue
            image = store.get_image_by_index(self.project_id, frame)
            if image is None or track is None:
                continue
            store.create_annotation(
                image.id, track.label_id, mask_to_bbox(m), source="propagated", status="approved",
                confidence=None, mask_rle=mask_to_rle(m),
                polygon_norm=mask_to_yolo_polygon(m, image.width, image.height), track_id=self.track_id)
            self.counts["frames"] += 1

    def run(self):
        from core.store import ProjectStore
        store = ProjectStore(self.store_db)
        try:
            self.status = "running"
            lo, hi = self.frame_range or (0, self.engine.num_frames - 1)
            store.delete_propagated_annotations(self.track_id, lo, hi)
            store.clear_issues(self.track_id)
            self.emit("started", start_frame=lo, end_frame=hi)
            written: set[int] = set()

            def on_event(ev):
                if self._stop:
                    return
                if ev.kind == "frame" and ev.mask is not None:
                    frame = int(ev.frame)
                    if frame < lo or frame > hi:
                        return
                    image = store.get_image_by_index(self.project_id, frame)
                    track = store.get_track(self.track_id)
                    if image is None or track is None:
                        return
                    written.add(frame)
                    m = np.asarray(ev.mask, dtype=bool)
                    if not m.any():  # tracker says "not here": no annotation, export writes nothing
                        self.emit("frame", frame=frame, visible=False,
                                  progress=self.counts["frames"])
                        return
                    ann = store.create_annotation(
                        image.id, track.label_id, mask_to_bbox(m), source="propagated",
                        status="approved", confidence=None, mask_rle=mask_to_rle(m),
                        polygon_norm=mask_to_yolo_polygon(m, image.width, image.height),
                        track_id=self.track_id,
                    )
                    self.counts["frames"] += 1
                    self.emit("frame", frame=frame, annotation_id=ann.id,
                              bbox=mask_to_bbox(m), visible=True,
                              render_url=(f"/api/video/render/frame?frame={frame}"
                                          f"&track_id={self.track_id}"),
                              progress=self.counts["frames"])
                elif ev.kind == "lost":
                    self.lost.append(ev.info)
                    self.emit("lost", **ev.info)
                elif ev.kind in {"start", "segment"}:
                    self.emit(ev.kind, frame=getattr(ev, "frame", None),
                              done=ev.done, total=ev.total)

            self.engine.run(track_ids=[self.track_id], full=self.full,
                            frame_range=self.frame_range, on_event=on_event,
                            should_stop=lambda: self._stop, max_lost=self.max_lost)
            if self._stop:
                self.status = "stopped"
                self.emit("stopped")
                return
            self._restore_unchanged(store, lo, hi, written)

            embed = _embedder(self.engine)
            health, issues = assess(self.engine, self.track_id,
                                    HealthConfig() if embed is not None else HealthConfig.geometry_only(), embed)
            health_by_frame = {h.frame: h for h in health}
            # A frame is approved only when health says ok; keyframe/absent are
            # deliberately excluded from the propagated table.
            for ann, image in store.list_track_annotations(self.track_id):
                if ann.source != "propagated" or not (lo <= image.order_index <= hi):
                    continue
                h = health_by_frame.get(image.order_index)
                if h is not None:
                    store.update_annotation(ann.id, status="approved" if h.status == "ok" else "needs_review")
            for issue in issues:
                d = issue.to_dict()
                store.create_issue(self.track_id, d, self.id)
                self.counts["issues"] += 1
                self.emit("issue", issue=d)
            self.status = "finished"
            self.finished_at = time.time()
            self.emit("finished", counts=dict(self.counts))
        except Exception as exc:  # pragma: no cover - exercised by integration failures
            self.error = str(exc)
            self.status = "error"
            self.finished_at = time.time()
            self.emit("error", message=self.error)
        finally:
            store.close()


_PREVIEWS: OrderedDict[str, Preview] = OrderedDict()
_PREVIEW_LIMIT = int(os.getenv("VIDEO_PREVIEW_LIMIT", "64"))
_PREVIEW_TTL = float(os.getenv("VIDEO_PREVIEW_TTL", "1800"))
_ENGINES: dict[str, TrackEngine] = {}
_ENGINE_LOCK = threading.RLock()
_JOBS: dict[str, VideoTrackJob] = {}
_JOBS_LOCK = threading.RLock()


def close_engine_for_db(db_path: str) -> None:
    """Release the SAM session when a project is closed or replaced."""
    with _ENGINE_LOCK:
        eng = _ENGINES.pop(db_path, None)
        if eng is not None:
            eng.close()


def close_stale_sessions() -> None:
    """Close sam3 sessions left behind by a previous backend process.

    Only enabled when this backend is the sole client of the sam_server
    (``agentseg up`` sets VIDEO_CLOSE_STALE_SESSIONS=1); a killed backend
    otherwise leaks several GB of VRAM until the server-side expiry.
    """
    try:
        c = _client()
        stale = (c.info().get("video_sessions") or {}).get(c.model, [])
        for sid in stale:
            c.close_session(sid)
        if stale:
            print(f"Closed {len(stale)} stale sam3 session(s)")
    except Exception as exc:
        print(f"Warning: could not close stale sam3 sessions: {exc}")


def _embedder(engine: TrackEngine):
    """DINO similarity for health checks (tuned default); VIDEO_HEALTH_DINO=0 falls back to geometry only."""
    if os.getenv("VIDEO_HEALTH_DINO", "1") == "0":
        return None
    emb = getattr(engine, "_video_embedder", None)
    if emb is None:
        try:
            from ml.sam3_video.health import DinoEmbedder
            emb = DinoEmbedder(engine.frames.paths)
        except Exception as exc:  # pragma: no cover - missing weights / GPU
            print(f"Warning: DINO health disabled: {exc}")
            return None
        engine._video_embedder = emb
    return emb


def _client() -> Sam3Client:
    return Sam3Client(os.getenv("SAM_SERVER_URL", "http://localhost:8770"))


def _engine() -> TrackEngine:
    project, store = get_project(), get_store()
    with _ENGINE_LOCK:
        key = project.db_path
        eng = _ENGINES.get(key)
        if eng is not None and eng.num_frames == store.get_image_count(project.id):
            return eng
        if eng is not None:
            eng.close()
        images = store.list_images(project.id)
        frames = FrameSet([i.path for i in images], root=str(Path(project.root_dir) / ".agentseg" / "frames"))
        # decoded frames live in CPU RAM: saves most of SAM's VRAM on a shared GPU
        eng = TrackEngine(_client(), frames, offload_video_to_cpu=os.getenv("VIDEO_OFFLOAD_TO_CPU", "1") == "1")
        _ENGINES[key] = eng
        return eng


def _track_loaded(engine: TrackEngine, track_id: int):
    if track_id in engine.tracks:
        return
    store = get_store()
    track = store.get_track(track_id)
    if track is None:
        raise HTTPException(404, "Track not found")
    engine.add_track(track_id)
    for ann, image in store.list_track_annotations(track_id):
        if ann.source == "propagated" or ann.mask_rle is None:
            continue
        try:
            engine.add_keyframe(track_id, image.order_index, rle_to_mask(ann.mask_rle).astype(bool), source=ann.source)
        except (ValueError, IndexError):
            continue
    absent = store.list_track_absences(track_id)
    if absent:
        engine.mark_absent(track_id, absent)


def _keyframe_dirty_range(engine: TrackEngine, track_id: int, frame: int) -> tuple[int, int]:
    """Frames owned by a newly inserted keyframe (inclusive)."""
    t = engine.track(track_id)
    positives = sorted(t.positive_keyframes())
    pos = positives.index(frame) if frame in positives else 0
    lo = ((positives[pos - 1] + frame) // 2 + 1) if pos else 0
    hi = ((frame + positives[pos + 1]) // 2) if pos + 1 < len(positives) else engine.num_frames - 1
    return lo, hi


def _mark_dirty(engine: TrackEngine, lo: int, hi: int) -> None:
    """Widen the pending regeneration range so batched edits are all re-propagated."""
    cur = getattr(engine, "_video_dirty_range", None)
    engine._video_dirty_range = (lo, hi) if cur is None else (min(cur[0], lo), max(cur[1], hi))


def _previews_dir() -> Path:
    home = os.getenv("AGENTSEG_HOME")
    return Path(home or Path(__file__).resolve().parents[2] / ".agentseg") / "previews"


def _save_preview_mask(p: Preview) -> None:
    """Keep preview masks after the in-memory cache drops them, so a recorded
    agent run can show its mask proposals again (see GET /previews/{id})."""
    try:
        d = _previews_dir(); d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(p.mask.astype(np.uint8) * 255).save(d / f"{p.id}.png", optimize=True)
    except OSError:
        pass


def _get_preview(pid: str) -> Preview:
    p = _PREVIEWS.get(pid)
    if p is None or time.time() - p.created > _PREVIEW_TTL:
        _PREVIEWS.pop(pid, None)
        raise HTTPException(404, "Preview not found or expired")
    _PREVIEWS.move_to_end(pid)
    return p


class SegmentRequest(BaseModel):
    frame: int
    box: list[float] = Field(min_length=4, max_length=4)
    pos: list[list[float]] = Field(default_factory=list)
    neg: list[list[float]] = Field(default_factory=list)


class CreateTrackRequest(BaseModel):
    preview_id: str
    label_id: Optional[int] = None
    label: Optional[str] = None
    name: Optional[str] = None


class PropagateRequest(BaseModel):
    full: bool = False
    max_lost: int = 0  # stop a direction after this many empty frames in a row (0 = never)
    start: Optional[int] = None
    end: Optional[int] = None


class KeyframeRequest(BaseModel):
    frame: int
    preview_id: Optional[str] = None
    box: Optional[list[float]] = None
    pos: list[list[float]] = Field(default_factory=list)
    neg: list[list[float]] = Field(default_factory=list)


class AbsentRequest(BaseModel):
    frames: Optional[list[int]] = None
    start: Optional[int] = None
    end: Optional[int] = None


def _image_for_frame(frame: int):
    image = get_store().get_image_by_index(get_project().id, frame)
    if image is None:
        raise HTTPException(404, f"Frame {frame} not found")
    return image


def _keep_preview(frame: int, mask: np.ndarray, score: float) -> Preview:
    p = Preview(uuid.uuid4().hex[:16], get_project().id, frame, mask, float(score), mask_to_bbox(mask))
    _PREVIEWS[p.id] = p
    _save_preview_mask(p)
    while len(_PREVIEWS) > _PREVIEW_LIMIT:
        _PREVIEWS.popitem(last=False)
    return p


def _coverage_check(client, image, req: SegmentRequest, mask: np.ndarray) -> dict:
    """Does the mask look like the whole object? A mask that touches the prompt box, or that
    grows when the box is padded, usually means the box cut the object off."""
    x0, y0, x1, y1 = req.box
    area = int(mask.sum())
    out: dict = {"area": area}
    if not area:
        out["warning"] = "empty mask"
        return out
    bx0, by0, bx1, by1 = mask_to_bbox(mask)
    touches = [n for n, hit in (("left", bx0 <= x0 + 3), ("top", by0 <= y0 + 3),
                                ("right", bx1 >= x1 - 3), ("bottom", by1 >= y1 - 3)) if hit]
    out["touches_box"] = touches
    pw, ph = 0.2 * (x1 - x0), 0.2 * (y1 - y0)
    h, w = mask.shape
    padded = [max(0, x0 - pw), max(0, y0 - ph), min(w - 1, x1 + pw), min(h - 1, y1 + ph)]
    try:
        m2, s2 = client.segment(image.path, box=padded, points=req.pos + req.neg,
                                point_labels=[1] * len(req.pos) + [0] * len(req.neg))
    except Exception:
        return out
    m2 = drop_specks(m2)
    a2 = int(m2.sum())
    keeps = (m2 & mask).sum() / area  # the padded mask still holds the original one
    out.update(padded_box=[round(v) for v in padded], padded_area=a2)
    if a2 > 1.15 * area and keeps > 0.85:
        p2 = _keep_preview(req.frame, m2, s2)
        out["padded_preview_id"] = p2.id
        out["warning"] = (f"the mask grows {a2 / area:.2f}x with a 20% larger box (bbox {mask_to_bbox(m2)}): "
                          "your box probably cuts the object off. Look at the zoom; if the bigger mask is "
                          "the whole object use padded_preview_id, else redraw the box around all of it")
    elif touches:
        out["warning"] = (f"the mask touches the box's {'/'.join(touches)} edge: check in the zoom that "
                          "the object does not continue past it")
    return out


@router.post("/segment")
def segment_preview(req: SegmentRequest):
    image = _image_for_frame(req.frame)
    client = _client()
    try:
        mask, score = client.segment(image.path, box=req.box,
                                     points=req.pos + req.neg,
                                     point_labels=[1] * len(req.pos) + [0] * len(req.neg))
        mask = drop_specks(mask)  # stray islands would stretch the bbox (and the label) past the object
        check = _coverage_check(client, image, req, mask)
    except Exception as exc:
        raise HTTPException(502, f"sam_server segmentation failed: {exc}")
    finally:
        close = getattr(client, "close", None)
        if close:
            close()
    p = _keep_preview(req.frame, mask, score)
    try:
        publish_activity("mask.proposed", source="backend", action="segment", status="completed",
                         project_id=p.project_id, frame=p.frame, preview_id=p.id,
                         bbox=p.bbox, score=p.score,
                         render_url=(f"/api/video/render/frame?frame={p.frame}"
                                     f"&preview_id={p.id}"),
                         message="Mask proposal is ready")
    except Exception:
        pass
    return {"preview_id": p.id, "frame": p.frame, "bbox": p.bbox,
            "area": mask_area(p.mask), "score": p.score, "check": check}


@router.post("/tracks")
def create_video_track(req: CreateTrackRequest):
    p = _get_preview(req.preview_id)
    project, store = get_project(), get_store()
    if p.project_id != project.id:
        raise HTTPException(400, "Preview belongs to another project")
    image = _image_for_frame(p.frame)
    label = store.get_label_by_id(req.label_id) if req.label_id else None
    if label is None and req.label:
        label = store.upsert_label(project.id, req.label)
    if label is None:
        raise HTTPException(400, "label_id or label is required")
    track = store.create_track(project.id, label.id, name=req.name)
    ann = store.create_annotation(image.id, label.id, p.bbox, source="sam", status="approved",
                                  mask_rle=mask_to_rle(p.mask),
                                  polygon_norm=mask_to_yolo_polygon(p.mask, image.width, image.height), track_id=track.id)
    eng = _engine(); _track_loaded(eng, track.id)
    if track.id not in eng.tracks:
        eng.add_track(track.id)
    eng.add_keyframe(track.id, p.frame, p.mask, source="sam")
    _mark_dirty(eng, 0, eng.num_frames - 1)
    try:
        publish_activity("track.created", source="backend", action="create_track", status="completed",
                         project_id=project.id, track_id=track.id, frame=p.frame,
                         message=f"Track {track.id} created")
    except Exception:
        pass
    return {"track_id": track.id, "keyframe": {"frame": p.frame, "annotation_id": ann.id,
            "bbox": p.bbox, "area": mask_area(p.mask)}, "label_id": label.id}


@router.post("/tracks/{track_id}/propagate")
def start_propagate(track_id: int, req: PropagateRequest = PropagateRequest()):
    project, store = get_project(), get_store()
    if store.get_track(track_id) is None:
        raise HTTPException(404, "Track not found")
    with _JOBS_LOCK:
        if any(j.is_active and j.project_id == project.id for j in _JOBS.values()):
            raise HTTPException(409, "A video tracking job is already running for this project")
        eng = _engine(); _track_loaded(eng, track_id)
        if not eng.track(track_id).positive_keyframes():
            raise HTTPException(400, "Track has no keyframes")
        dirty = getattr(eng, "_video_dirty_range", None)
        lo = req.start if req.start is not None else (0 if req.full or dirty is None else dirty[0])
        hi = req.end if req.end is not None else (eng.num_frames - 1 if req.full or dirty is None else dirty[1])
        if lo < 0 or hi >= eng.num_frames or lo > hi:
            raise HTTPException(400, "Invalid frame range")
        job = VideoTrackJob(project.id, track_id, eng, store.db_path,
                            frame_range=(lo, hi), full=req.full, max_lost=max(0, req.max_lost))
        _JOBS[job.id] = job
        eng._video_dirty_range = None
        job.start()
    return job.snapshot()


def _job_or_404(job_id: str) -> VideoTrackJob:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Video tracking job not found")
    return job


@router.get("/jobs/{job_id}")
def video_job_status(job_id: str):
    return _job_or_404(job_id).snapshot()


@router.get("/jobs/{job_id}/events")
def video_job_events(job_id: str, after: int = 0, wait: float = 0.0):
    job = _job_or_404(job_id)
    events = job.wait_events(after, min(wait, 30)) if wait > 0 else job.events_after(after)
    return {"events": events, "status": job.status, "last_seq": job.last_seq}


@router.get("/tracks/{track_id}/issues")
def track_issues(track_id: int):
    if get_store().get_track(track_id) is None:
        raise HTTPException(404, "Track not found")
    return get_store().list_issues(track_id)


@router.post("/tracks/{track_id}/keyframes")
def add_video_keyframe(track_id: int, req: KeyframeRequest):
    eng = _engine(); _track_loaded(eng, track_id)
    image = _image_for_frame(req.frame)
    if req.preview_id:
        p = _get_preview(req.preview_id)
        if p.frame != req.frame:
            raise HTTPException(400, "Preview frame does not match keyframe frame")
        mask, score = p.mask, p.score
    elif req.box:
        c = _client()
        try:
            mask, score = c.segment(image.path, box=req.box, points=req.pos + req.neg,
                                    point_labels=[1] * len(req.pos) + [0] * len(req.neg))
        finally:
            close = getattr(c, "close", None)
            if close:
                close()
        mask = drop_specks(mask)
    else:
        raise HTTPException(400, "preview_id or box is required")
    store = get_store(); track = store.get_track(track_id)
    store.delete_propagated_annotations(track_id, req.frame, req.frame)
    # Replace an existing non-propagated annotation at this frame.
    for ann, img in store.list_track_annotations(track_id):
        if img.order_index == req.frame and ann.source != "propagated":
            store.delete_annotation(ann.id)
    ann = store.create_annotation(image.id, track.label_id, mask_to_bbox(mask), source="sam", status="approved",
                                  confidence=float(score), mask_rle=mask_to_rle(mask),
                                  polygon_norm=mask_to_yolo_polygon(mask, image.width, image.height), track_id=track_id)
    eng.add_keyframe(track_id, req.frame, mask, source="sam")
    _mark_dirty(eng, *_keyframe_dirty_range(eng, track_id, req.frame))
    return {"track_id": track_id, "frame": req.frame, "annotation_id": ann.id, "score": float(score),
            "bbox": mask_to_bbox(mask)}


@router.delete("/tracks/{track_id}/keyframes/{frame}")
def remove_video_keyframe(track_id: int, frame: int):
    eng = _engine(); _track_loaded(eng, track_id)
    if frame not in eng.track(track_id).keyframes:
        raise HTTPException(404, "Keyframe not found")
    eng.remove_keyframe(track_id, frame)
    _mark_dirty(eng, 0, eng.num_frames - 1)
    store = get_store()
    for ann, img in store.list_track_annotations(track_id):
        if img.order_index == frame and ann.source != "propagated":
            store.delete_annotation(ann.id)
    return {"track_id": track_id, "frame": frame, "removed": True}


class IssueReviewRequest(BaseModel):
    issue_ids: Optional[list[int]] = None  # None = all open issues
    by: str = "agent"
    note: Optional[str] = None


@router.post("/tracks/{track_id}/issues/review")
def review_track_issues(track_id: int, req: IssueReviewRequest):
    """The reviewer looked at these issues and the masks are right: approve their
    propagated frames so they stop showing as needs_review."""
    store = get_store()
    known = {i["id"] for i in store.list_issues(track_id)}
    unknown = sorted(set(req.issue_ids or []) - known)
    if unknown:
        raise HTTPException(404, f"unknown issue id(s) for track {track_id}: {unknown}")
    issues = store.review_issues(track_id, req.issue_ids, req.by, req.note)
    approved = 0
    for ann, image in store.list_track_annotations(track_id):
        if ann.source == "propagated" and ann.status == "needs_review" and any(
                i["start_frame"] <= image.order_index <= i["end_frame"] for i in issues):
            store.update_annotation(ann.id, status="approved")
            approved += 1
    return {"track_id": track_id, "reviewed": [i["id"] for i in issues],
            "ranges": [[i["start_frame"], i["end_frame"]] for i in issues], "frames_approved": approved,
            "open_issues": sum(1 for i in store.list_issues(track_id) if not i.get("review"))}


@router.post("/tracks/{track_id}/absent")
def mark_track_absent(track_id: int, req: AbsentRequest):
    eng = _engine(); _track_loaded(eng, track_id)
    frames = req.frames or (list(range(req.start, req.end + 1)) if req.start is not None and req.end is not None else [])
    if not frames:
        raise HTTPException(400, "frames or start/end is required")
    try:
        eng.mark_absent(track_id, frames)
    except (ValueError, IndexError) as exc:
        raise HTTPException(400, str(exc))
    store = get_store(); store.add_track_absences(track_id, frames)
    fs = sorted(set(frames)); run_start = fs[0]
    for prev, cur in zip(fs, fs[1:] + [None]):
        if cur != prev + 1:  # only the listed frames, not everything between them
            store.delete_propagated_annotations(track_id, run_start, prev)
            run_start = cur
    _mark_dirty(eng, min(frames), max(frames))
    return {"track_id": track_id, "frames": sorted(set(frames)), "status": "absent"}


@router.get("/tracks")
def list_video_tracks():
    project, store = get_project(), get_store()
    out = []
    for track in store.list_tracks(project.id):
        anns = store.list_track_annotations(track.id)
        anchors = [img.order_index for ann, img in anns if ann.source != "propagated"]
        propagated = [img.order_index for ann, img in anns if ann.source == "propagated"]
        job = next((j for j in reversed(list(_JOBS.values())) if j.track_id == track.id), None)
        out.append({"id": track.id, "label_id": track.label_id, "status": track.status,
                    "keyframes": sorted(anchors), "absent": store.list_track_absences(track.id),
                    "coverage": {"first": min(propagated) if propagated else None,
                                 "last": max(propagated) if propagated else None,
                                 "count": len(propagated)},
                    "issue_count": sum(1 for i in store.list_issues(track.id) if not i.get("review")),
                    "job": job.snapshot() if job else None})
    return out


def _mask_layer(m: np.ndarray, rgb: tuple[int, int, int]) -> Image.Image:
    inner = m.copy()
    inner[1:, :] &= m[:-1, :]; inner[:-1, :] &= m[1:, :]; inner[:, 1:] &= m[:, :-1]; inner[:, :-1] &= m[:, 1:]
    edge = m & ~inner
    edge = edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)
    a = np.zeros((m.shape[0], m.shape[1], 4), np.uint8)
    a[m] = (*rgb, 90)
    a[edge & m] = (*rgb, 255)
    return Image.fromarray(a, "RGBA")


def _overlay(path: str, track_id: Optional[int], frame: int, grid: bool, preview_id: Optional[str]):
    base = Image.open(path).convert("RGBA")
    store = get_store(); anns = store.list_annotations(_image_for_frame(frame).id)
    if track_id is not None:
        anns = [a for a in anns if a.track_id == track_id]
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    # High-contrast against water/sky blues; a solid outline makes scope errors obvious.
    colors = [(255, 0, 200), (255, 120, 0), (120, 255, 0)]
    for i, ann in enumerate(anns):
        if not ann.mask_rle:
            continue
        overlay = Image.alpha_composite(overlay, _mask_layer(rle_to_mask(ann.mask_rle).astype(bool),
                                                             colors[i % len(colors)]))
    if preview_id:
        overlay = Image.alpha_composite(overlay, _mask_layer(_get_preview(preview_id).mask, (255, 220, 20)))
    out = Image.alpha_composite(base, overlay)
    d = ImageDraw.Draw(out)
    d.rectangle((4, out.height - 28, 110, out.height - 4), fill=(0, 0, 0, 170))
    d.text((8, out.height - 24), f"frame {frame}", fill="white")
    for ann in anns:
        if ann.bbox_xyxy:
            b = ann.bbox_xyxy; col = "yellow" if ann.source != "propagated" else "cyan"
            d.rectangle(b, outline=col, width=2)
    if grid:
        for x in range(0, out.width, 50):
            d.line((x, 0, x, out.height), fill=(255, 255, 255, 100), width=1); d.text((x + 2, 2), str(x), fill="white")
        for y in range(0, out.height, 50):
            d.line((0, y, out.width, y), fill=(255, 255, 255, 100), width=1); d.text((2, y + 2), str(y), fill="white")
    buf = io.BytesIO(); out.convert("RGB").save(buf, format="PNG"); return buf.getvalue()


@router.get("/render/frame")
def render_frame(frame: int, track_id: Optional[int] = None, grid: bool = False, preview_id: Optional[str] = None):
    image = _image_for_frame(frame)
    return Response(_overlay(image.path, track_id, frame, grid, preview_id), media_type="image/png")


def _crop_box(boxes: list, w: int, h: int, pad: float = 0.35, min_side: int = 96) -> tuple[int, int, int, int]:
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw = max((x1 - x0) * (1 + 2 * pad), min_side) / 2; hh = max((y1 - y0) * (1 + 2 * pad), min_side) / 2
    return (int(max(0, cx - hw)), int(max(0, cy - hh)), int(min(w, cx + hw)), int(min(h, cy + hh)))


def _outline(m: np.ndarray, rgb: tuple[int, int, int]) -> Image.Image:
    a = np.asarray(_mask_layer(m, rgb)).copy()
    a[..., 3] = np.where(a[..., 3] == 255, 255, 0)  # edge only: the object's own colors stay visible
    return Image.fromarray(a, "RGBA")


def _panel(path: str, crop, mask: Optional[np.ndarray], title: str, box=None, height: int = 300) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    if mask is not None:
        im = Image.alpha_composite(im, _outline(mask, (255, 0, 200)))
    d = ImageDraw.Draw(im)
    if box is not None:
        d.rectangle(box, outline=(0, 230, 255), width=2)
    im = im.crop(crop).convert("RGB")
    im = im.resize((max(1, round(im.width * height / im.height)), height), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 8 + 7 * len(title), 18), fill=(0, 0, 0))
    d.text((4, 3), title, fill="white")
    return im


@router.get("/render/zoom")
def render_zoom(frame: int, track_id: Optional[int] = None, preview_id: Optional[str] = None,
                box: Optional[str] = None):
    """Close-up without fill: [nearest keyframe of the track] [frame plain] [frame + outline].
    The keyframe panel answers "same object?", the outline panel "whole object?"."""
    image = _image_for_frame(frame)
    store = get_store()
    prompt = [float(v) for v in box.split(",")] if box else None
    mask, ref = None, None
    if preview_id:
        p = _PREVIEWS.get(preview_id)
        path = _previews_dir() / f"{preview_id}.png"
        if p is not None:
            mask = p.mask
        elif preview_id.isalnum() and path.exists():  # expired from memory (e.g. a replay): from disk
            mask = np.array(Image.open(path)) > 0
        else:
            raise HTTPException(404, "Preview not found")
    elif track_id is not None:
        anns = [a for a in store.list_annotations(image.id) if a.track_id == track_id and a.mask_rle]
        if anns:
            mask = rle_to_mask(anns[0].mask_rle).astype(bool)
    if track_id is not None:
        kfs = [(ann, img) for ann, img in store.list_track_annotations(track_id)
               if ann.source != "propagated" and ann.mask_rle and img.order_index != frame]
        if kfs:
            ref = min(kfs, key=lambda k: abs(k[1].order_index - frame))
    boxes = [b for b in (mask_to_bbox(mask) if mask is not None and mask.any() else None, prompt) if b]
    title = f"frame {frame} mask outline"
    if not boxes and track_id is not None:
        # no mask here (lost, absent, untracked): look where the track last had the object
        near = [(img.order_index, ann.bbox_xyxy) for ann, img in store.list_track_annotations(track_id)
                if ann.bbox_xyxy and img.order_index != frame]
        if near:
            f2, b2 = min(near, key=lambda n: abs(n[0] - frame))
            boxes = [b2]
            title = f"frame {frame}: NO MASK (crop at frame {f2}'s object)"
    if not boxes:
        boxes = [[0, 0, image.width, image.height]]
        title = f"frame {frame}: NO MASK"
    crop = _crop_box(boxes, image.width, image.height, pad=0.6 if mask is None or not mask.any() else 0.35)
    panels = []
    if ref is not None:
        kann, kimg = ref
        km = rle_to_mask(kann.mask_rle).astype(bool)
        panels.append(_panel(kimg.path, _crop_box([mask_to_bbox(km)], kimg.width, kimg.height), None,
                             f"keyframe {kimg.order_index} (reference)"))
    panels.append(_panel(image.path, crop, None, f"frame {frame}"))
    panels.append(_panel(image.path, crop, mask, title, prompt))
    out = Image.new("RGB", (sum(p.width for p in panels) + 6 * (len(panels) - 1), panels[0].height), "#222")
    x = 0
    for p in panels:
        out.paste(p, (x, 0)); x += p.width + 6
    buf = io.BytesIO(); out.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


_SHEET_COLORS = {"keyframe": (255, 220, 20), "ok": (60, 200, 90), "review": (255, 140, 0),
                 "absent": (150, 150, 150), "empty": (230, 50, 50)}


def _frame_statuses(track_id: int) -> dict[int, str]:
    """keyframe | ok | review | absent per frame; frames without a mask are missing."""
    store = get_store()
    out = {f: "absent" for f in store.list_track_absences(track_id)}
    for ann, img in store.list_track_annotations(track_id):
        if ann.source != "propagated":
            out[img.order_index] = "keyframe"
        elif img.order_index not in out:
            out[img.order_index] = "ok" if ann.status == "approved" else "review"
    return out


@router.get("/previews/{preview_id}")
def get_preview(preview_id: str):
    """A mask proposal as RLE, from memory or from disk once it has expired."""
    p = _PREVIEWS.get(preview_id)
    if p is not None:
        mask, frame, score, bbox = p.mask, p.frame, p.score, p.bbox
    else:
        if not preview_id.isalnum():
            raise HTTPException(400, "bad preview id")
        path = _previews_dir() / f"{preview_id}.png"
        if not path.exists():
            raise HTTPException(404, "Preview not found")
        mask = np.array(Image.open(path)) > 0
        frame, score, bbox = None, None, mask_to_bbox(mask)
    return {"preview_id": preview_id, "frame": frame, "score": score, "bbox": bbox,
            "mask_rle": mask_to_rle(mask)}


@router.get("/tracks/{track_id}/frames")
def track_frame_statuses(track_id: int):
    """Per-frame health of a track for the timeline: keyframe | ok | review | absent | null."""
    store = get_store()
    if store.get_track(track_id) is None:
        raise HTTPException(404, "Track not found")
    n = store.get_image_count(get_project().id)
    statuses = _frame_statuses(track_id)
    return {"track_id": track_id, "n": n, "statuses": [statuses.get(f) for f in range(n)]}


@router.get("/render/sheet")
def render_sheet(track_id: Optional[int] = None, every: int = Query(10, ge=1), frames: Optional[str] = None,
                 grid: bool = False):
    n = get_store().get_image_count(get_project().id)
    selected = sorted({int(x) for x in frames.split(",")}) if frames else list(range(0, n, every))
    statuses = _frame_statuses(track_id) if track_id is not None else {}
    thumbs = []
    for f in selected:
        if 0 <= f < n:
            im = Image.open(io.BytesIO(_overlay(_image_for_frame(f).path, track_id, f, grid, None))).convert("RGB")
            im.thumbnail((320, 240)); thumbs.append((f, im.copy()))
    cols = 4; cell_w, cell_h = 320, 270
    sheet = Image.new("RGB", (cols * cell_w, max(1, (len(thumbs) + cols - 1) // cols) * cell_h), "#222")
    d = ImageDraw.Draw(sheet)
    for i, (f, im) in enumerate(thumbs):
        x, y = (i % cols) * cell_w, (i // cols) * cell_h
        sheet.paste(im, (x, y + 22))
        label = f"frame {f}"
        if track_id is not None:
            st = statuses.get(f, "empty")
            col = _SHEET_COLORS[st]
            d.rectangle((x, y + 22, x + im.width - 1, y + 22 + im.height - 1), outline=col, width=4)
            label += f"  {st}"
            d.text((x + 5, y + 4), label, fill=col)
        else:
            d.text((x + 5, y + 4), label, fill="white")
    buf = io.BytesIO(); sheet.save(buf, format="PNG"); return Response(buf.getvalue(), media_type="image/png")
