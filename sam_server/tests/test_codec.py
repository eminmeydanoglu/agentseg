import json

import numpy as np
import pytest
import torch

from sam_server.codec import decode, encode


def rt(x):
    return decode(json.loads(json.dumps(encode(x))))


@pytest.mark.parametrize("shape", [(5, 7), (3, 5, 7), (2, 3, 4, 6), (0, 5, 7)])
def test_bool_masks_rle_roundtrip(shape):
    m = np.random.rand(*shape) > 0.5
    enc = encode(m)
    assert "__rle__" in enc and enc["shape"] == list(shape)
    out = rt(m)
    assert out.dtype == bool and out.shape == m.shape and (out == m).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int64, np.uint8, np.int32])
def test_ndarray_roundtrip(dtype):
    a = (np.random.rand(4, 3, 2) * 100).astype(dtype)
    out = rt(a)
    assert out.dtype == a.dtype and out.shape == a.shape and np.array_equal(out, a)


def test_small_1d_is_list_and_empty_keeps_shape():
    assert encode(np.array([1, 2, 3], dtype=np.int64)) == [1, 2, 3]
    e = rt(np.zeros((0, 4), dtype=np.float32))
    assert e.shape == (0, 4) and e.dtype == np.float32


def test_torch_tensors():
    t = torch.randn(2, 3, dtype=torch.bfloat16)
    out = rt(t)
    assert out.dtype == np.float32 and torch.equal(torch.from_numpy(out), t.float())
    m = torch.rand(2, 8, 9) > 0.5
    assert (rt(m) == m.numpy()).all()


def test_nonstr_keys_and_nesting():
    x = {"frame_index": 3, "outputs": {1: np.ones((4, 4), bool), 2: np.zeros((4, 4), bool)},
         "t": (1, 2.5), "s": np.float32(0.25), "b": b"\x00\x01"}
    out = rt(x)
    assert set(out["outputs"]) == {1, 2}
    assert out["outputs"][1].all() and not out["outputs"][2].any()
    assert out["t"] == [1, 2.5] and out["s"] == 0.25 and out["b"] == b"\x00\x01"
