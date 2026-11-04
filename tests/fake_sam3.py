"""In-memory stand-in for ml.sam3_video.client.Sam3Client (no GPU, no server).

Tracking model: a propagation pass started on frame s (which must hold a conditioning
mask for the object) outputs, on frame f, the conditioning mask of s shifted right by
|f - s| * ``dx`` px plus ``len(cond)`` px. The ``len(cond)`` term makes every output depend
on all conditioning frames the object holds (like SAM's memory bank), so a stale
conditioning frame left behind after a removal shows up as a different mask. Empty
conditioning masks (absent) yield empty outputs from that start frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ml.sam3_video.client import FrameOutput


def box_mask(h, w, box):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    m = np.zeros((h, w), dtype=bool)
    m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    return m


def shift(mask, dx):
    out = np.zeros_like(mask)
    if dx >= mask.shape[1]:
        return out
    out[:, dx:] = mask[:, : mask.shape[1] - dx]
    return out


@dataclass
class FakeObject:
    cond: dict = field(default_factory=dict)  # frame -> mask


class FakeSam3Client:
    def __init__(self, num_frames: int, h: int, w: int, dx: int = 1):
        self.num_frames, self.h, self.w, self.dx = num_frames, h, w, dx
        self.calls: list[tuple] = []
        self.sessions: dict[str, dict[int, FakeObject]] = {}
        self.cancelled = 0
        self._n = 0

    # video
    def start_session(self, resource_path, offload_video_to_cpu=False):
        self._n += 1
        sid = f"s{self._n}"
        self.sessions[sid] = {}
        self.calls.append(("start_session", resource_path))
        return sid

    def close_session(self, sid):
        self.calls.append(("close_session", sid))
        self.sessions.pop(sid, None)

    def reset_session(self, sid):
        self.calls.append(("reset_session", sid))
        self.sessions[sid].clear()

    def add_mask(self, sid, frame_index, obj_id, mask):
        self.calls.append(("add_mask", frame_index, obj_id, bool(np.asarray(mask).any())))
        self.sessions[sid].setdefault(obj_id, FakeObject()).cond[frame_index] = np.asarray(mask, bool).copy()
        return FrameOutput(frame_index, {obj_id: np.asarray(mask, bool)})

    def remove_object(self, sid, obj_id, is_user_action=False):
        self.calls.append(("remove_object", obj_id))
        del self.sessions[sid][obj_id]

    def cancel_propagation(self, sid):
        self.cancelled += 1

    def propagate(self, sid, start_frame, direction, max_frames: Optional[int] = None,
                  output_prob_thresh=None, force_tracker_propagation=True):
        self.calls.append(("propagate", start_frame, direction, max_frames))
        n = self.num_frames if max_frames is None else max_frames
        if direction == "forward":
            order = range(start_frame, min(start_frame + n, self.num_frames - 1) + 1)
        else:
            order = range(start_frame, max(start_frame - n, 0) - 1, -1)
        objs = self.sessions[sid]
        finished = False
        try:
            for f in order:
                out = {}
                for oid, o in objs.items():
                    if f in o.cond:
                        out[oid] = o.cond[f]
                    elif start_frame in o.cond:
                        out[oid] = shift(o.cond[start_frame], abs(f - start_frame) * self.dx + len(o.cond))
                    else:
                        out[oid] = np.zeros((self.h, self.w), bool)
                yield FrameOutput(f, {k: v for k, v in out.items() if v.any()})
            finished = True
        finally:
            if not finished:
                self.cancel_propagation(sid)

    # image
    def segment(self, image_path, box=None, points=None, point_labels=None):
        self.calls.append(("segment", image_path, tuple(box) if box is not None else None))
        return box_mask(self.h, self.w, box), 0.9

    def propagate_calls(self):
        return [c[1:] for c in self.calls if c[0] == "propagate"]
