"""TrackEngine: keyframe-driven SAM 3 video tracking over sam_server.

Model
-----
* A *track* is one object. Its source of truth is ``keyframes``: frame -> Keyframe, where
  a keyframe holds an approved mask (``source`` = "manual", "agent", ...) or is *absent*
  (the object is not visible there). Everything else (``masks``) is derived and can be
  regenerated from the keyframes at any time; the SAM session is only a cache.
* Every track is one ``obj_id`` in a single sam3 video session (one session per engine).
  Keyframe masks enter SAM via ``add_mask``.
* Canonical propagation: each frame is tracked from its *nearest* positive keyframe.
  Between consecutive keyframes a < b the split is m = (a + b) // 2: a tracks forward over
  a+1..m, b tracks backward over m+1..b-1; the first keyframe also tracks backward to 0 and
  the last one forward to the end. Per keyframe the forward half runs before the backward
  half (the backward pass then starts with a fresh, consistent memory on its other side).
  A *segment* = (keyframe, direction, range).
* Incremental edits:
  - ``add_keyframe``: only the new keyframe's two segments run (neighbours' segments shrink
    and keep their frames). Frames elsewhere are not recomputed even though the new
    conditioning frame would slightly change them (accepted approximation; ``propagate(
    full=True)`` regenerates everything).
  - ``remove_keyframe``: SAM cannot forget a conditioning frame, so the track is rebuilt on a
    fresh obj_id (``remove_object`` + re-adding the remaining keyframes, or a session reset
    with ``remove_mode="session"``) and fully re-propagated. The result is the same as never
    having added the keyframe.
  - ``mark_absent``: absent frames are forced empty; with ``absent_to_sam`` the first and last
    frame of each absent run are also given to SAM as empty-mask conditioning frames.
* Progress: ``propagate()`` is a generator of ``PropagationEvent``; closing it cancels the
  running server-side propagation (the interrupted segment stays pending). ``run()`` drains
  it with an optional callback.

Pure Python + numpy: no web framework imports; masks at the API boundary are bool (H, W).
Not thread-safe: drive one engine from one thread (e.g. a backend job worker).
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable, Iterator, Optional, Sequence

import numpy as np

from ml.sam3_video.client import Sam3Client, SamServerError
from ml.sam3_video.frames import FrameSet

TrackId = Hashable

FORWARD = "forward"
BACKWARD = "backward"


@dataclass
class Keyframe:
    frame: int
    mask: Optional[np.ndarray]  # bool (H, W); None = object absent on this frame
    source: str = "manual"
    meta: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)

    @property
    def absent(self) -> bool:
        return self.mask is None


@dataclass(frozen=True)
class Segment:
    """Frames ``kf -> end`` (inclusive, ``kf`` itself excluded) tracked from keyframe ``kf``."""

    kf: int
    direction: str
    end: int

    @property
    def frames(self) -> range:
        if self.direction == FORWARD:
            return range(self.kf + 1, self.end + 1)
        return range(self.kf - 1, self.end - 1, -1)

    def __len__(self) -> int:
        return abs(self.end - self.kf)

    def covers(self, other: "Segment") -> bool:
        """Same keyframe and direction, and ``other``'s range is a prefix of ours."""
        if (self.kf, self.direction) != (other.kf, other.direction):
            return False
        return other.end <= self.end if self.direction == FORWARD else other.end >= self.end


@dataclass
class PropagationEvent:
    kind: str  # "start" | "segment" | "frame" | "lost" | "done"
    track_id: Optional[TrackId] = None
    frame: Optional[int] = None
    mask: Optional[np.ndarray] = None  # "frame": the track's new mask on that frame
    segment: Optional[Segment] = None
    done: int = 0  # frames processed so far in this propagate() call
    total: int = 0  # frames planned for this call
    info: dict = field(default_factory=dict)


@dataclass
class Track:
    track_id: TrackId
    obj_id: int
    keyframes: dict[int, Keyframe] = field(default_factory=dict)
    masks: dict[int, np.ndarray] = field(default_factory=dict)  # derived per-frame result
    origin: dict[int, int] = field(default_factory=dict)  # frame -> keyframe it was tracked from
    meta: dict = field(default_factory=dict)
    # cache bookkeeping
    _done: dict = field(default_factory=dict)  # (kf, direction) -> Segment last computed
    _sam: dict = field(default_factory=dict)  # frame -> id(Keyframe) currently in SAM for obj_id
    _hidden: dict = field(default_factory=dict)  # frame -> (mask, origin) under a keyframe
    _rebuild: bool = False

    def positive_keyframes(self) -> list[int]:
        return sorted(f for f, k in self.keyframes.items() if not k.absent)

    def absent_frames(self) -> list[int]:
        return sorted(f for f, k in self.keyframes.items() if k.absent)


def plan_segments(positive: Sequence[int], num_frames: int) -> list[Segment]:
    """Canonical segments for sorted positive keyframes (empty ranges dropped)."""
    segs = []
    kfs = sorted(positive)
    for i, k in enumerate(kfs):
        prev_k = kfs[i - 1] if i > 0 else None
        next_k = kfs[i + 1] if i + 1 < len(kfs) else None
        fwd_end = (k + next_k) // 2 if next_k is not None else num_frames - 1
        bwd_end = (prev_k + k) // 2 + 1 if prev_k is not None else 0
        if fwd_end > k:
            segs.append(Segment(k, FORWARD, fwd_end))
        if bwd_end < k:
            segs.append(Segment(k, BACKWARD, bwd_end))
    return segs


def absent_runs(frames: Iterable[int]) -> list[tuple[int, int]]:
    """Sorted frames -> inclusive (start, end) runs of consecutive frames."""
    runs = []
    for _, grp in itertools.groupby(enumerate(sorted(set(frames))), key=lambda t: t[1] - t[0]):
        g = [f for _, f in grp]
        runs.append((g[0], g[-1]))
    return runs


class TrackEngine:
    def __init__(self, client: Sam3Client, frames: FrameSet, *, absent_to_sam: bool = False,
                 remove_mode: str = "object", offload_video_to_cpu: bool = False,
                 window: Optional[int] = None):
        if remove_mode not in ("object", "session"):
            raise ValueError(remove_mode)
        if window is not None:
            # Interface reserved for long sequences (overlapping windows, last mask carried
            # as keyframe). Deferred to WP8.
            raise NotImplementedError("windowed propagation is not implemented yet")
        self.client = client
        self.frames = frames
        self.absent_to_sam = absent_to_sam
        self.remove_mode = remove_mode
        self.offload_video_to_cpu = offload_video_to_cpu
        self.tracks: dict[TrackId, Track] = {}
        self.session_id: Optional[str] = None
        self._next_obj_id = 1
        self._next_track = 1
        self.stats = {"sam_frames": 0, "sam_seconds": 0.0, "rebuilds": 0, "segments": 0}

    # ── basics ──

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def image_size(self) -> tuple[int, int]:
        """(width, height)."""
        return self.frames.size

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        """Close the SAM session (keyframes and masks stay; the next propagate reopens one)."""
        if self.session_id is not None:
            sid, self.session_id = self.session_id, None
            try:
                self.client.close_session(sid)
            except Exception:
                pass
        for t in self.tracks.values():
            t._sam.clear()
            t._rebuild = True

    def segment(self, frame: int, box: Optional[Sequence[float]] = None,
                pos: Sequence[Sequence[float]] = (), neg: Sequence[Sequence[float]] = ()) -> tuple[np.ndarray, float]:
        """Image-head mask for ``frame`` from a pixel xyxy box and positive/negative points."""
        pts = [*pos, *neg]
        labels = [1] * len(pos) + [0] * len(neg)
        return self.client.segment(self.frames.path(frame), box=box, points=pts or None,
                                   point_labels=labels or None)

    # ── tracks and keyframes ──

    def add_track(self, track_id: Optional[TrackId] = None, **meta) -> TrackId:
        if track_id is None:
            while self._next_track in self.tracks:
                self._next_track += 1
            track_id = self._next_track
        if track_id in self.tracks:
            raise KeyError(f"track {track_id!r} exists")
        self.tracks[track_id] = Track(track_id, self._new_obj_id(), meta=dict(meta))
        return track_id

    def remove_track(self, track_id: TrackId) -> None:
        t = self._track(track_id)
        if self.session_id is not None and t._sam:
            self.client.remove_object(self.session_id, t.obj_id)
        del self.tracks[track_id]

    def track(self, track_id: TrackId) -> Track:
        return self._track(track_id)

    def keyframes(self, track_id: TrackId) -> dict[int, Keyframe]:
        return dict(sorted(self._track(track_id).keyframes.items()))

    def add_keyframe(self, track_id: TrackId, frame: int, mask: np.ndarray, source: str = "manual",
                     **meta) -> Keyframe:
        """Add (or replace) an approved mask on ``frame``. Run ``propagate()`` to apply."""
        t = self._track(track_id)
        self._check_frame(frame)
        mask = self._check_mask(mask)
        if not mask.any():
            raise ValueError("empty keyframe mask; use mark_absent() for frames without the object")
        old = t.keyframes.get(frame)
        if old is None:
            self._hide(t, frame)
        elif old.absent and frame in t._sam:
            t._rebuild = True  # an empty conditioning mask is in SAM there: replace it cleanly
        t.keyframes[frame] = Keyframe(frame, mask, source, dict(meta))
        t.masks[frame] = mask
        t.origin[frame] = frame
        self._invalidate_keyframe(t, frame)
        return t.keyframes[frame]

    def remove_keyframe(self, track_id: TrackId, frame: int) -> None:
        """Drop a keyframe (positive or absent). Run ``propagate()`` to apply.

        A keyframe that already reached SAM forces a rebuild of the track (fresh obj_id, full
        re-propagation): the result equals never having added it.
        """
        t = self._track(track_id)
        t.keyframes.pop(frame)
        self._invalidate_keyframe(t, frame)
        if frame in t._sam:
            t._rebuild = True
        self._unhide(t, frame)

    def mark_absent(self, track_id: TrackId, frames: Iterable[int], source: str = "manual") -> list[int]:
        """Declare the object not visible on ``frames`` (replaces keyframes there). Returns them."""
        t = self._track(track_id)
        frames = sorted({int(f) for f in frames})
        for f in frames:
            self._check_frame(f)
        for f in frames:
            old = t.keyframes.get(f)
            if old is None:
                self._hide(t, f)
            elif not old.absent:
                if f in t._sam:
                    t._rebuild = True
                self._invalidate_keyframe(t, f)
            t.keyframes[f] = Keyframe(f, None, source)
            t.masks[f] = self._empty()
            t.origin[f] = f
        if self.absent_to_sam:
            for a, b in absent_runs(t.absent_frames()):
                for f in {a, b}:
                    owner = self._owner_segment(t, f)
                    if f not in t._sam and owner is not None:
                        # the segment tracking through f must see the new conditioning frame
                        t._done.pop((owner.kf, owner.direction), None)
        return frames

    def clear_absent(self, track_id: TrackId, frames: Iterable[int]) -> None:
        t = self._track(track_id)
        for f in frames:
            if f in t.keyframes and t.keyframes[f].absent:
                self.remove_keyframe(track_id, f)

    # ── results ──

    def mask(self, track_id: TrackId, frame: int) -> Optional[np.ndarray]:
        """Current mask (bool H, W) or None if the frame has not been tracked yet."""
        return self._track(track_id).masks.get(frame)

    def masks(self, track_id: TrackId) -> dict[int, np.ndarray]:
        return dict(sorted(self._track(track_id).masks.items()))

    def frame_kind(self, track_id: TrackId, frame: int) -> str:
        """"keyframe" | "absent" | "propagated" | "missing"."""
        t = self._track(track_id)
        kf = t.keyframes.get(frame)
        if kf is not None:
            return "absent" if kf.absent else "keyframe"
        return "propagated" if frame in t.masks else "missing"

    def pending(self, track_id: Optional[TrackId] = None) -> dict[TrackId, list[Segment]]:
        """Segments the next ``propagate()`` would run, per track."""
        ids = [track_id] if track_id is not None else list(self.tracks)
        return {tid: self._pending_segments(self.tracks[tid]) for tid in ids}

    # ── propagation ──

    def propagate(self, track_ids: Optional[Iterable[TrackId]] = None, *, full: bool = False,
                  frame_range: Optional[tuple[int, int]] = None,
                  max_lost: int = 0) -> Iterator[PropagationEvent]:
        """Bring the given tracks (default: all) up to date with their keyframes.

        full: rebuild from scratch (fresh SAM object, every segment).
        frame_range: (lo, hi) inclusive; segments are clipped to it (clipped segments stay
        pending for the frames outside).
        max_lost: stop a segment after this many empty masks in a row (frames marked absent
        do not count) and yield a "lost" event; its untracked rest stays pending. 0 = never.
        """
        ids = list(self.tracks) if track_ids is None else [self._track(i).track_id for i in track_ids]
        if full:
            for tid in ids:
                self.tracks[tid]._rebuild = True
        self._ensure_session()
        ids = self._sync(ids)
        work: list[tuple[Track, Segment, Segment]] = []  # (track, canonical, clipped)
        for tid in ids:
            t = self.tracks[tid]
            for seg in self._pending_segments(t):
                run = self._clip(seg, frame_range)
                if run is not None:
                    work.append((t, seg, run))
        total = sum(len(r) for _, _, r in work)
        yield PropagationEvent("start", total=total, info={"segments": len(work), "tracks": ids})
        done = 0
        t0 = time.time()
        for t, seg, run in work:
            yield PropagationEvent("segment", t.track_id, segment=run, done=done, total=total)
            seg_t0 = time.time()
            stream = self.client.propagate(self.session_id, run.kf, run.direction, max_frames=len(run))
            absent = set(t.absent_frames()) if max_lost else set()
            empty_from, empty_n, cut = None, 0, None
            try:
                for out in stream:
                    f = out.frame_index
                    if f == run.kf:
                        continue
                    if f not in run.frames:
                        break
                    m = out.mask(t.obj_id)
                    if m is None:
                        m = self._empty()
                    if f in t.keyframes:
                        t._hidden[f] = (m, seg.kf)  # keyframes are never overwritten
                        continue
                    t.masks[f] = m
                    t.origin[f] = seg.kf
                    done += 1
                    yield PropagationEvent("frame", t.track_id, frame=f, mask=m, segment=run, done=done,
                                           total=total)
                    if max_lost and f not in absent:
                        if m.any():
                            empty_from, empty_n = None, 0
                        else:
                            empty_from = f if empty_from is None else empty_from
                            empty_n += 1
                            if empty_n >= max_lost:
                                cut = f
                                break
            finally:
                stream.close()  # cancels server-side if interrupted
            if cut is not None:
                # drop stale masks past the cut: those frames are untracked, not "lost"
                rest = [f for f in run.frames if (f > cut if run.direction == FORWARD else f < cut)]
                for f in rest:
                    t.masks.pop(f, None); t.origin.pop(f, None)
                done += len(rest)
                yield PropagationEvent("lost", t.track_id, frame=empty_from, segment=run, done=done,
                                       total=total, info={"keyframe": run.kf, "direction": run.direction,
                                                          "lost_from": empty_from, "stopped_at": cut,
                                                          "untracked": [rest[0], rest[-1]] if rest else None})
                run = Segment(run.kf, run.direction, cut)
            self.stats["sam_frames"] += len(run)
            self.stats["sam_seconds"] += time.time() - seg_t0
            self.stats["segments"] += 1
            # a clipped run records only its computed prefix; a later full run redoes it from kf
            t._done[(seg.kf, seg.direction)] = run
        yield PropagationEvent("done", done=done, total=total, info={"seconds": round(time.time() - t0, 3)})

    def run(self, track_ids: Optional[Iterable[TrackId]] = None, *, full: bool = False,
            frame_range: Optional[tuple[int, int]] = None,
            on_event: Optional[Callable[[PropagationEvent], None]] = None,
            should_stop: Optional[Callable[[], bool]] = None, max_lost: int = 0) -> dict:
        """Drain ``propagate()``; returns {"frames", "seconds", "stopped"}."""
        t0 = time.time()
        gen = self.propagate(track_ids, full=full, frame_range=frame_range, max_lost=max_lost)
        n, stopped = 0, False
        try:
            for ev in gen:
                if on_event is not None:
                    on_event(ev)
                if ev.kind == "frame":
                    n += 1
                if should_stop is not None and should_stop():
                    stopped = True
                    break
        finally:
            gen.close()
        return {"frames": n, "seconds": round(time.time() - t0, 3), "stopped": stopped}

    # ── internals ──

    def _track(self, track_id: TrackId) -> Track:
        try:
            return self.tracks[track_id]
        except KeyError:
            raise KeyError(f"unknown track {track_id!r}") from None

    def _new_obj_id(self) -> int:
        oid = self._next_obj_id
        self._next_obj_id += 1
        return oid

    def _empty(self) -> np.ndarray:
        w, h = self.image_size
        return np.zeros((h, w), dtype=bool)

    def _check_frame(self, frame: int) -> None:
        if not 0 <= frame < self.num_frames:
            raise IndexError(f"frame {frame} outside 0..{self.num_frames - 1}")

    def _check_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask)
        if mask.dtype != bool:
            mask = mask > 0
        w, h = self.image_size
        if mask.shape != (h, w):
            raise ValueError(f"mask shape {mask.shape} != frame size {(h, w)}")
        return mask

    def _ensure_session(self) -> None:
        if self.session_id is None:
            self.session_id = self.client.start_session(self.frames.folder, self.offload_video_to_cpu)
            for t in self.tracks.values():
                t._sam.clear()
                t._rebuild = True

    def _sam_keyframes(self, t: Track) -> dict[int, Keyframe]:
        """Keyframes that belong in SAM for this track."""
        want = {f: k for f, k in t.keyframes.items() if not k.absent}
        if self.absent_to_sam:
            for a, b in absent_runs(t.absent_frames()):
                want[a] = t.keyframes[a]
                want[b] = t.keyframes[b]
        return want

    def _sync(self, ids: Sequence[TrackId]) -> list[TrackId]:
        """Make the SAM session hold exactly each track's keyframes; returns the track ids
        that must be propagated (a session reset adds every track)."""
        ids = list(ids)
        rebuild = [self.tracks[i] for i in ids if self.tracks[i]._rebuild]
        used = {id(t) for t in rebuild if t._sam}
        if self.remove_mode == "session" and used:
            # reset drops every object: all tracks are rebuilt (also those not requested)
            self.client.reset_session(self.session_id)
            rebuild = list(self.tracks.values())
            ids = list(self.tracks)
            used = {id(t) for t in rebuild if t._sam}
            for t in rebuild:
                t._sam.clear()
        for t in rebuild:
            if t._sam:
                try:
                    self.client.remove_object(self.session_id, t.obj_id)
                except SamServerError:
                    pass
            if id(t) in used:
                # never reuse an obj_id: nothing keyed on the old object can leak into the new one
                t.obj_id = self._new_obj_id()
            t._sam.clear()
            t._done.clear()
            t._rebuild = False
            self.stats["rebuilds"] += 1
            # derived frames are recomputed; frames no segment will reach become missing
            keep = set(t.keyframes)
            for f in [f for f in t.masks if f not in keep]:
                del t.masks[f]
                t.origin.pop(f, None)
            t._hidden.clear()
        for i in ids:
            t = self.tracks[i]
            want = self._sam_keyframes(t)
            for f, kf in sorted(want.items()):
                if t._sam.get(f) != id(kf):
                    mask = self._empty() if kf.absent else kf.mask
                    self.client.add_mask(self.session_id, f, t.obj_id, mask)
                    t._sam[f] = id(kf)
        return ids

    def _pending_segments(self, t: Track) -> list[Segment]:
        if t._rebuild:
            return plan_segments(t.positive_keyframes(), self.num_frames)
        out = []
        for seg in plan_segments(t.positive_keyframes(), self.num_frames):
            done = t._done.get((seg.kf, seg.direction))
            if done is None or not done.covers(seg):
                out.append(seg)
        return out

    def _owner_segment(self, t: Track, frame: int) -> Optional[Segment]:
        for seg in plan_segments(t.positive_keyframes(), self.num_frames):
            if frame in seg.frames:
                return seg
        return None

    @staticmethod
    def _invalidate_keyframe(t: Track, frame: int) -> None:
        t._done.pop((frame, FORWARD), None)
        t._done.pop((frame, BACKWARD), None)

    @staticmethod
    def _hide(t: Track, frame: int) -> None:
        """Keep the derived result underneath a new keyframe (restored if it is removed)."""
        if frame in t.masks:
            t._hidden[frame] = (t.masks[frame], t.origin.get(frame))

    @staticmethod
    def _unhide(t: Track, frame: int) -> None:
        hidden = t._hidden.pop(frame, None)
        if hidden is None or t._rebuild:
            t.masks.pop(frame, None)
            t.origin.pop(frame, None)
        else:
            t.masks[frame], t.origin[frame] = hidden

    @staticmethod
    def _clip(seg: Segment, frame_range: Optional[tuple[int, int]]) -> Optional[Segment]:
        if frame_range is None:
            return seg
        lo, hi = frame_range
        if not lo <= seg.kf <= hi:
            return None
        end = min(seg.end, hi) if seg.direction == FORWARD else max(seg.end, lo)
        return Segment(seg.kf, seg.direction, end) if end != seg.kf else None
