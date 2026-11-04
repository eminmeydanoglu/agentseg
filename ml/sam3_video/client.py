"""HTTP client for sam_server (SAM 3 video head + image head).

Thin wrapper over the endpoints documented in ``sam_server/README.md``:

* ``POST /video/{model}/request``         start/close/reset session, add_mask, remove_object,
                                          cancel_propagation
* ``POST /video/{model}/stream_request``  propagate_in_video (NDJSON, one line per frame)
* ``POST /image/set_image``, ``POST /image/call/predict_inst``, ``DELETE /image/states/{id}``

Masks cross this boundary as numpy bool arrays (H, W); RLE encoding is internal.
Request shapes follow the upstream sam3 video predictor API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import httpx
import numpy as np

from ml.sam3_video.codec import decode, encode

DEFAULT_URL = "http://localhost:8770"
DEFAULT_MODEL = "sam3"


class SamServerError(RuntimeError):
    """sam_server returned an error (HTTP status != 200 or an error line in a stream)."""


@dataclass
class FrameOutput:
    """Masks of all tracked objects on one frame (objects missing from the output are empty)."""

    frame_index: int
    masks: dict[int, np.ndarray] = field(default_factory=dict)  # obj_id -> bool (H, W)

    def mask(self, obj_id: int) -> Optional[np.ndarray]:
        return self.masks.get(obj_id)


def parse_outputs(frame_index: int, outputs: dict) -> FrameOutput:
    """Decode an upstream ``outputs`` dict (``out_obj_ids``, ``out_binary_masks``, ...)."""
    ids = np.asarray(decode(outputs.get("out_obj_ids", [])), dtype=np.int64).ravel()
    masks = {}
    if ids.size:
        planes = np.asarray(decode(outputs["out_binary_masks"]))
        planes = planes.reshape(-1, *planes.shape[-2:])
        for k, oid in enumerate(ids.tolist()):
            if k < len(planes):
                m = planes[k]
                masks[int(oid)] = m if m.dtype == bool else m > 0
    return FrameOutput(int(frame_index), masks)


class Sam3Client:
    """Synchronous client. ``http`` may be any object with httpx.Client's post/get/delete/stream."""

    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL, timeout: float = 1800.0,
                 http: Optional[httpx.Client] = None):
        self.url = url
        self.model = model
        self.http = http if http is not None else httpx.Client(base_url=url, timeout=timeout)

    # ── server ──

    def info(self) -> dict:
        r = self.http.get("/info")
        self._check(r, "info")
        return r.json()

    def close(self) -> None:
        self.http.close()

    # ── video ──

    def request(self, body: dict) -> dict:
        r = self.http.post(f"/video/{self.model}/request", json=encode(body))
        self._check(r, body.get("type", "request"))
        return r.json()

    def start_session(self, resource_path: str, offload_video_to_cpu: bool = False) -> str:
        r = self.request({"type": "start_session", "resource_path": resource_path,
                          "offload_video_to_cpu": offload_video_to_cpu})
        return r["session_id"]

    def close_session(self, session_id: str) -> None:
        try:
            self.cancel_propagation(session_id)
        except Exception:
            pass
        self.request({"type": "close_session", "session_id": session_id, "clear_cache_threshold": 0})

    def reset_session(self, session_id: str) -> None:
        self.request({"type": "reset_session", "session_id": session_id})

    def add_mask(self, session_id: str, frame_index: int, obj_id: int, mask: np.ndarray) -> FrameOutput:
        mask = np.ascontiguousarray(np.asarray(mask, dtype=bool))
        if mask.ndim != 2:
            raise ValueError(f"mask must be (H, W), got {mask.shape}")
        r = self.request({"type": "add_mask", "session_id": session_id, "frame_index": int(frame_index),
                          "obj_id": int(obj_id), "mask": mask})
        return parse_outputs(r.get("frame_index", frame_index), r.get("outputs", {}))

    def remove_object(self, session_id: str, obj_id: int, is_user_action: bool = False) -> None:
        # is_user_action=False: programmatic pruning, keeps upstream's action history clean
        self.request({"type": "remove_object", "session_id": session_id, "obj_id": int(obj_id),
                      "is_user_action": is_user_action})

    def cancel_propagation(self, session_id: str) -> None:
        self.request({"type": "cancel_propagation", "session_id": session_id})

    def propagate(self, session_id: str, start_frame: int, direction: str,
                  max_frames: Optional[int] = None, output_prob_thresh: Optional[float] = None,
                  force_tracker_propagation: bool = True) -> Iterator[FrameOutput]:
        """Yield one FrameOutput per processed frame, in processing order.

        ``direction`` is "forward" or "backward"; ``max_frames`` frames are tracked *beyond*
        ``start_frame`` (upstream ``max_frame_num_to_track``). ``force_tracker_propagation``
        makes upstream always run the tracker on every object instead of consulting its action
        history (which turns the third propagate call into a cache fetch). Closing the generator
        early cancels the server-side propagation.
        """
        if direction not in ("forward", "backward"):
            raise ValueError(direction)
        body = {"type": "propagate_in_video", "session_id": session_id,
                "propagation_direction": direction, "start_frame_index": int(start_frame),
                "evict_cached_frame_outputs": True,
                "force_tracker_propagation": force_tracker_propagation}
        if max_frames is not None:
            body["max_frame_num_to_track"] = int(max_frames)
        if output_prob_thresh is not None:
            body["output_prob_thresh"] = output_prob_thresh
        finished = False
        with self.http.stream("POST", f"/video/{self.model}/stream_request", json=body) as resp:
            if resp.status_code != 200:
                raise SamServerError(f"propagate -> {resp.status_code}: {resp.read()[:500]!r}")
            try:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    item = json.loads(line)
                    if "error" in item:
                        raise SamServerError(f"propagate error: {item}")
                    yield parse_outputs(item["frame_index"], item.get("outputs", {}))
                finished = True
            finally:
                if not finished:
                    try:
                        self.cancel_propagation(session_id)
                    except Exception:
                        pass

    # ── image ──

    def segment(self, image_path: str, box: Optional[Sequence[float]] = None,
                points: Optional[Sequence[Sequence[float]]] = None,
                point_labels: Optional[Sequence[int]] = None) -> tuple[np.ndarray, float]:
        """Image-head mask from a pixel xyxy box and/or pixel (x, y) points (label 1 = pos, 0 = neg).

        Returns (bool mask (H, W), predicted quality score).
        """
        if box is None and not points:
            raise ValueError("need a box or at least one point")
        kwargs: dict = {"multimask_output": False}
        if box is not None:
            kwargs["box"] = [[float(v) for v in box]]
        if points:
            if point_labels is None or len(point_labels) != len(points):
                raise ValueError("point_labels must match points")
            kwargs["point_coords"] = [[float(x), float(y)] for x, y in points]
            kwargs["point_labels"] = [int(v) for v in point_labels]
        r = self.http.post("/image/set_image", json={"image_path": image_path})
        self._check(r, "set_image")
        sid = r.json()["state_id"]
        try:
            r = self.http.post("/image/call/predict_inst", json={"state_id": sid, "kwargs": kwargs})
            self._check(r, "predict_inst")
            masks, scores, _ = decode(r.json()["result"])
        finally:
            try:
                self.http.delete(f"/image/states/{sid}")
            except Exception:
                pass
        masks = np.asarray(masks)
        m = masks.reshape(-1, *masks.shape[-2:])[0]
        return (m if m.dtype == bool else m > 0), float(np.asarray(scores).ravel()[0])

    @staticmethod
    def _check(r, what: str) -> None:
        if r.status_code != 200:
            raise SamServerError(f"{what} -> {r.status_code}: {r.text[:500]}")
