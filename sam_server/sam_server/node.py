"""Model holder + dispatch. No AgentSeg logic: everything is passed to upstream sam3.

All model work runs on ONE worker thread (``self.executor``), so requests are
serialized. Models are also built on that thread, because upstream predictors
enter a thread-local bf16 autocast context in ``__init__``.
"""

from __future__ import annotations

import base64
import inspect
import io
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator, Optional

from .codec import decode, encode
from .config import ServerConfig

logger = logging.getLogger("sam_server")

PUBLIC_IMAGE_FIELDS = ("boxes", "scores", "masks")
# Image methods reached through /image/set_image instead of /image/call
_IMAGE_SETTERS = {"set_image", "set_image_batch"}


class NotFound(KeyError):
    pass


class _PresenceHolder:
    def __init__(self):
        self.value = None


def wrap_forward_grounding(model, holder: _PresenceHolder) -> None:
    """Capture the presence score the grounding head computes (added, never changed)."""
    orig = model.forward_grounding

    def forward_grounding(*args, **kwargs):
        out = orig(*args, **kwargs)
        p = out.get("presence_logit_dec") if isinstance(out, dict) else None
        if p is not None:
            holder.value = p.sigmoid()
        return out

    model.forward_grounding = forward_grounding


def _filter_init_state_kwargs(model) -> None:
    """Upstream compat shim (sam3 f1647e8): Sam3BasePredictor.start_session always
    passes ``offload_state_to_cpu`` to ``model.init_state``, but the 3.1 multiplex
    model's ``init_state`` does not accept it, so start_session raises TypeError.
    Drop kwargs the model's init_state does not take (no-op for models that do)."""
    orig = model.init_state
    params = inspect.signature(orig).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return

    def init_state(*args, **kwargs):
        dropped = [k for k in kwargs if k not in params]
        if dropped:
            logger.debug("init_state: dropping unsupported kwargs %s", dropped)
        return orig(*args, **{k: v for k, v in kwargs.items() if k in params})

    model.init_state = init_state


def _presence_to_json(p):
    from .codec import tensor_to_numpy

    arr = tensor_to_numpy(p).reshape(-1)
    # single image -> plain float; batch -> one value per image
    return float(arr[0]) if arr.size == 1 else encode(arr)


class SamNode:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam-gpu")
        self.video: dict[str, Any] = {}
        self.image_model = None
        self.processor = None
        self.image_states: dict[str, dict] = {}
        self._presence = _PresenceHolder()
        # session_id -> set of threading.Event for running streams (cancel bypass)
        self._streams: dict[str, set[threading.Event]] = {}
        self._streams_lock = threading.Lock()
        self.load_times: dict[str, float] = {}
        self.ready = False

    # ── loading (call on the worker thread) ───────────────────────────
    def load(self) -> None:
        import torch
        from sam3.model_builder import build_sam3_image_model, build_sam3_predictor

        cfg = self.cfg
        if cfg.device.startswith("cuda:"):
            torch.cuda.set_device(torch.device(cfg.device))
        if "sam3.1" in cfg.models:
            t = time.time()
            self.video["sam3.1"] = build_sam3_predictor(
                version="sam3.1",
                checkpoint_path=cfg.sam31_checkpoint,
                bpe_path=cfg.bpe_path,
                compile=cfg.compile,
                warm_up=cfg.warm_up,
                max_num_objects=cfg.max_num_objects,
                multiplex_count=cfg.multiplex_count,
                use_fa3=cfg.use_fa3,
                use_rope_real=cfg.use_rope_real,
                async_loading_frames=cfg.async_loading_frames,
                session_expiration_sec=cfg.session_expiration_sec,
                default_output_prob_thresh=cfg.default_output_prob_thresh,
            )
            self.load_times["sam3.1"] = time.time() - t
        if "sam3" in cfg.models:
            t = time.time()
            pred = build_sam3_predictor(
                version="sam3",
                checkpoint_path=cfg.sam3_checkpoint,
                bpe_path=cfg.bpe_path,
                compile=cfg.compile,
                async_loading_frames=cfg.async_loading_frames,
            )
            pred.default_output_prob_thresh = cfg.default_output_prob_thresh
            self.video["sam3"] = pred
            self.load_times["sam3"] = time.time() - t
        if "image" in cfg.models:
            from sam3.model.sam3_image_processor import Sam3Processor

            t = time.time()
            self.image_model = build_sam3_image_model(
                bpe_path=cfg.bpe_path,
                device=cfg.device,
                checkpoint_path=cfg.sam3_checkpoint,
                enable_inst_interactivity=cfg.image_inst_interactivity,
                compile=cfg.compile,
            )
            self.processor = Sam3Processor(
                self.image_model,
                resolution=cfg.image_resolution,
                device=cfg.device,
                confidence_threshold=cfg.image_confidence_threshold,
            )
            self.load_times["image"] = time.time() - t
        for pred in self.video.values():
            _filter_init_state_kwargs(pred.model)
            for k, v in cfg.video_model_attrs.items():
                if hasattr(pred.model, k):
                    setattr(pred.model, k, v)
        self._attach_image_hooks()
        self.ready = True

    def _attach_image_hooks(self) -> None:
        if self.image_model is not None:
            wrap_forward_grounding(self.image_model, self._presence)

    # ── video ─────────────────────────────────────────────────────────
    def get_video(self, model: str):
        pred = self.video.get(model)
        if pred is None:
            raise NotFound(f"video model {model!r} not loaded (loaded: {sorted(self.video)})")
        return pred

    def video_request(self, model: str, body: dict) -> Any:
        """Worker-thread: predictor.handle_request(body)."""
        return encode(self.get_video(model).handle_request(decode(body)))

    def video_cancel(self, model: str, body: dict) -> Any:
        """Caller-thread (bypasses the queue): stop running streams of the session,
        then forward cancel_propagation to upstream."""
        pred = self.get_video(model)
        sid = body.get("session_id")
        with self._streams_lock:
            for ev in self._streams.get(sid, ()):
                ev.set()
        import torch

        with torch.inference_mode():
            return encode(pred.handle_request(decode(body)))

    def register_stream(self, session_id) -> threading.Event:
        ev = threading.Event()
        with self._streams_lock:
            self._streams.setdefault(session_id, set()).add(ev)
        return ev

    def unregister_stream(self, session_id, ev: threading.Event) -> None:
        with self._streams_lock:
            s = self._streams.get(session_id)
            if s is not None:
                s.discard(ev)
                if not s:
                    del self._streams[session_id]

    def video_stream(self, model: str, body: dict, cancel: threading.Event,
                     emit: Callable[[str], None]) -> None:
        """Worker-thread: iterate handle_stream_request, emit one JSON line per item.
        Holds the worker until the generator ends or ``cancel`` is set."""
        if cancel.is_set():
            return
        gen: Optional[Iterator] = None
        try:
            gen = self.get_video(model).handle_stream_request(decode(body))
            for item in gen:
                emit(json.dumps(encode(item)))
                if cancel.is_set():
                    emit(json.dumps({"cancelled": True}))
                    break
        except Exception as e:  # report mid-stream errors in-band
            logger.exception("stream_request failed")
            emit(json.dumps({"error": type(e).__name__, "detail": str(e)}))
        finally:
            if gen is not None:
                gen.close()

    # ── image ─────────────────────────────────────────────────────────
    def _require_image(self):
        if self.processor is None:
            raise NotFound("image model not loaded")

    @staticmethod
    def _load_pil(path: Optional[str], b64: Optional[str]):
        from PIL import Image

        if path is not None:
            img = Image.open(path)
        elif b64 is not None:
            img = Image.open(io.BytesIO(base64.b64decode(b64)))
        else:
            raise ValueError("need image_path or image_b64")
        return img.convert("RGB")

    def image_set(self, body: dict) -> dict:
        """Worker-thread. {image_path|image_b64} -> set_image;
        {image_paths|images_b64: [...]} -> set_image_batch."""
        self._require_image()
        if "image_paths" in body or "images_b64" in body:
            paths = body.get("image_paths") or [None] * len(body["images_b64"])
            b64s = body.get("images_b64") or [None] * len(paths)
            images = [self._load_pil(p, b) for p, b in zip(paths, b64s)]
            state = self.processor.set_image_batch(images)
            info = {"original_heights": state["original_heights"],
                    "original_widths": state["original_widths"]}
        else:
            image = self._load_pil(body.get("image_path"), body.get("image_b64"))
            state = self.processor.set_image(image)
            info = {"original_height": state["original_height"],
                    "original_width": state["original_width"]}
        state_id = body.get("state_id") or str(uuid.uuid4())
        self.image_states[state_id] = {"state": state, "presence": None,
                                       "created": time.time()}
        return {"state_id": state_id, **info}

    def resolve_image_method(self, method: str):
        """Return (callable, kwarg name for the state). Sam3Processor first, then model
        methods that take ``inference_state`` (predict_inst, predict_inst_batch, ...)."""
        self._require_image()
        if method.startswith("_") or method in _IMAGE_SETTERS:
            raise NotFound(f"image method {method!r} not callable via /image/call")
        fn = getattr(self.processor, method, None)
        if callable(fn):
            params = inspect.signature(fn).parameters
            return fn, ("state" if "state" in params else None)
        fn = getattr(self.image_model, method, None)
        if callable(fn) and "inference_state" in inspect.signature(fn).parameters:
            return fn, "inference_state"
        raise NotFound(f"unknown image method {method!r}")

    def image_call(self, method: str, body: dict) -> dict:
        """Worker-thread. Calls the upstream method with the server-held state."""
        import torch

        fn, state_kw = self.resolve_image_method(method)
        rec = self.image_states.get(body.get("state_id"))
        if state_kw is not None and rec is None:
            raise NotFound(f"unknown state_id {body.get('state_id')!r}")
        kwargs = decode(body.get("kwargs") or {})
        args = decode(body.get("args") or [])
        if state_kw is not None:
            kwargs[state_kw] = rec["state"]
        self._presence.value = None
        with torch.inference_mode():
            out = fn(*args, **kwargs)
        if rec is not None:
            if self._presence.value is not None:
                rec["presence"] = self._presence.value
            if "scores" not in rec["state"]:
                rec["presence"] = None
        if rec is not None and out is rec["state"]:
            res = {k: encode(out[k]) for k in PUBLIC_IMAGE_FIELDS if k in out}
            if rec["presence"] is not None:
                res["presence"] = _presence_to_json(rec["presence"])
            return {"result": res}
        return {"result": encode(out)}

    def image_delete(self, state_id: str) -> dict:
        if self.image_states.pop(state_id, None) is None:
            raise NotFound(f"unknown state_id {state_id!r}")
        return {"is_success": True}

    # ── info ──────────────────────────────────────────────────────────
    def info(self) -> dict:
        import sam3

        out = {
            "ready": self.ready,
            "config": self.cfg.to_dict(),
            "models": {
                **{k: type(v).__name__ for k, v in self.video.items()},
                **({"image": type(self.image_model).__name__} if self.image_model else {}),
            },
            "load_times_sec": self.load_times,
            "sam3_version": getattr(sam3, "__version__", None),
            "sam3_commit": _sam3_commit(),
            "video_sessions": {k: list(getattr(v, "_all_inference_states", {}).keys())
                               for k, v in self.video.items()},
            "image_states": list(self.image_states),
            "running_streams": {k: len(v) for k, v in self._streams.items()},
        }
        try:
            import torch

            out["torch"] = torch.__version__
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                out["gpu"] = {
                    "name": torch.cuda.get_device_name(),
                    "free_bytes": free, "total_bytes": total,
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                    "max_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
        except Exception as e:  # pragma: no cover
            out["gpu_error"] = str(e)
        return out


def _sam3_commit() -> Optional[str]:
    import os
    import subprocess

    import sam3

    try:
        d = os.path.dirname(os.path.dirname(sam3.__file__))
        return subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None
