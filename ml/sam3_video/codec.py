"""Lossless JSON codec for SAM inputs/outputs (copy of ``sam_server/sam_server/codec.py``).

Kept as a small copy so the agentseg venv does not depend on the sam_server package
(separate venv, different torch build). ``tests/test_sam3_video_engine.py`` checks that
the two copies stay wire-compatible.

Wire formats (all JSON-safe):

* bool arrays with ndim >= 2 (masks), shape (..., H, W):
    {"__rle__": [<COCO RLE {"size": [H, W], "counts": str}>, ...], "shape": [...]}
  one RLE per HxW plane, leading dims flattened in C order.
* other arrays / tensors:
    {"__ndarray__": <base64 of C-contiguous bytes>, "dtype": "<numpy dtype>", "shape": [...]}
  1-D numeric arrays with 0 < size <= SMALL_LIST_MAX are sent as plain lists.
  torch.bfloat16 is widened to float32 (exact).
* dicts with any non-str key (e.g. obj_id -> mask):
    {"__dict__": [[key, value], ...]}
* bytes: {"__bytes__": <base64>}
* tuples -> lists, numpy scalars -> Python scalars.

decode() inverts all tagged forms (into numpy arrays); plain lists stay lists, which is
what upstream accepts for points/boxes.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

try:  # torch is optional for the codec (unit tests run without it)
    import torch
except ImportError:  # pragma: no cover
    torch = None

SMALL_LIST_MAX = 64


def _encode_rle(arr: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils

    shape = list(arr.shape)
    h, w = shape[-2], shape[-1]
    planes = arr.reshape(-1, h, w)
    rles = []
    if planes.shape[0]:
        # pycocotools wants (H, W, N) Fortran-ordered uint8
        fortran = np.asfortranarray(planes.transpose(1, 2, 0).astype(np.uint8))
        for r in mask_utils.encode(fortran):
            rles.append({"size": [int(r["size"][0]), int(r["size"][1])],
                         "counts": r["counts"].decode("ascii")})
    return {"__rle__": rles, "shape": shape}


def _decode_rle(obj: dict) -> np.ndarray:
    from pycocotools import mask as mask_utils

    shape = list(obj["shape"])
    h, w = shape[-2], shape[-1]
    rles = obj["__rle__"]
    if not rles:
        return np.zeros(shape, dtype=bool)
    enc = [{"size": r["size"], "counts": r["counts"].encode("ascii")} for r in rles]
    planes = mask_utils.decode(enc)  # (H, W, N)
    if planes.ndim == 2:
        planes = planes[:, :, None]
    return planes.transpose(2, 0, 1).reshape(shape).astype(bool)


def _encode_array(arr: np.ndarray) -> Any:
    if arr.dtype == bool and arr.ndim >= 2:
        return _encode_rle(arr)
    if arr.dtype == object:
        return [encode(x) for x in arr.tolist()]
    if arr.ndim == 1 and 0 < arr.size <= SMALL_LIST_MAX and arr.dtype.kind in "biuf":
        return arr.tolist()
    arr = np.ascontiguousarray(arr)
    return {
        "__ndarray__": base64.b64encode(arr.tobytes()).decode("ascii"),
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
    }


def _decode_array(obj: dict) -> np.ndarray:
    data = base64.b64decode(obj["__ndarray__"])
    return np.frombuffer(data, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()


def tensor_to_numpy(t) -> np.ndarray:
    t = t.detach()
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.cpu().numpy()


def encode(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if torch is not None and isinstance(obj, torch.Tensor):
        return _encode_array(tensor_to_numpy(obj))
    if isinstance(obj, np.ndarray):
        return _encode_array(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj):
            return {k: encode(v) for k, v in obj.items()}
        return {"__dict__": [[encode(k), encode(v)] for k, v in obj.items()]}
    if isinstance(obj, (list, tuple)):
        return [encode(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return [encode(x) for x in obj]
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": base64.b64encode(bytes(obj)).decode("ascii")}
    raise TypeError(f"cannot encode object of type {type(obj).__name__}")


def decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return _decode_array(obj)
        if "__rle__" in obj:
            return _decode_rle(obj)
        if "__dict__" in obj:
            return {_hashable(decode(k)): decode(v) for k, v in obj["__dict__"]}
        if "__bytes__" in obj:
            return base64.b64decode(obj["__bytes__"])
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(x) for x in obj]
    return obj


def _hashable(k):
    return tuple(k) if isinstance(k, list) else k
