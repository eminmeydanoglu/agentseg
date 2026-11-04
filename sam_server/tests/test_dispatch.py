"""Dispatch tests with fake upstream objects (no GPU, no checkpoints)."""

import base64
import io
import json
import threading
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from sam_server.app import create_app
from sam_server.codec import decode, encode
from sam_server.config import ServerConfig
from sam_server.node import SamNode, wrap_forward_grounding


class FakeVideoPredictor:
    def __init__(self):
        self._all_inference_states = {}
        self.requests = []
        self.cancelled = threading.Event()
        self.closed_gen = False

    def handle_request(self, request):
        self.requests.append(request)
        t = request["type"]
        if t == "start_session":
            self._all_inference_states["s1"] = {}
            return {"session_id": "s1"}
        if t == "add_mask":
            m = request["mask"]
            return {"frame_index": request["frame_index"],
                    "outputs": {"mask_type": type(m).__name__, "mask_sum": int(m.sum()),
                                "out_binary_masks": m[None]}}
        if t == "cancel_propagation":
            self.cancelled.set()
            return {"is_success": True}
        raise RuntimeError(f"invalid request type: {t}")

    def handle_stream_request(self, request):
        n = request.get("n", 3)
        try:
            for i in range(n):
                if request.get("slow"):
                    time.sleep(0.05)
                yield {"frame_index": i,
                       "outputs": {"out_obj_ids": np.array([1]),
                                   "out_binary_masks": np.ones((1, 4, 5), bool)}}
        finally:
            self.closed_gen = True


class FakeImageModel:
    inst_interactive_predictor = object()

    def forward_grounding(self, **kw):
        return {"presence_logit_dec": torch.tensor([[0.0]])}

    def predict_inst(self, inference_state, **kwargs):
        mi = kwargs.get("mask_input")
        return (np.ones((1, 2, 3), np.float32), np.array([0.9], np.float32),
                None if mi is None else mi.shape)

    def to(self, *a):  # not callable via /image/call (no inference_state)
        raise AssertionError


class FakeProcessor:
    def __init__(self, model):
        self.model = model
        self.confidence_threshold = 0.5

    def set_image(self, image, state=None):
        w, h = image.size
        return {"original_height": h, "original_width": w, "backbone_out": {"big": 1}}

    def set_text_prompt(self, prompt, state):
        self.model.forward_grounding()
        state["boxes"] = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        state["scores"] = torch.tensor([0.4])
        state["masks"] = torch.zeros(1, 1, state["original_height"], state["original_width"],
                                     dtype=torch.bool)
        state["masks_logits"] = torch.zeros(1)
        state["prompt"] = prompt
        return state

    def reset_all_prompts(self, state):
        for k in ("boxes", "scores", "masks", "masks_logits"):
            state.pop(k, None)

    def set_confidence_threshold(self, threshold, state=None):
        self.confidence_threshold = threshold
        return state


@pytest.fixture
def setup():
    node = SamNode(ServerConfig(models=[]))
    vp = FakeVideoPredictor()
    node.video["sam3.1"] = vp
    node.image_model = FakeImageModel()
    node.processor = FakeProcessor(node.image_model)
    node._attach_image_hooks()
    node.ready = True
    with TestClient(create_app(node, load_on_startup=False)) as c:
        yield c, node, vp


def _png_b64(w=6, h=4):
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_health_info(setup):
    c, _, _ = setup
    assert c.get("/health").json()["ready"] is True
    info = c.get("/info").json()
    assert info["models"]["sam3.1"] == "FakeVideoPredictor"


def test_video_request_passthrough_and_decode(setup):
    c, _, vp = setup
    r = c.post("/video/sam3.1/request", json={"type": "start_session", "resource_path": "/x"})
    assert r.json() == {"session_id": "s1"}
    assert vp.requests[-1] == {"type": "start_session", "resource_path": "/x"}
    mask = np.zeros((4, 5), bool)
    mask[1:3, 2:4] = True
    r = c.post("/video/sam3.1/request", json={"type": "add_mask", "session_id": "s1",
                                              "frame_index": 0, "obj_id": 1,
                                              "mask": encode(mask)})
    out = decode(r.json())
    assert out["outputs"]["mask_type"] == "ndarray" and out["outputs"]["mask_sum"] == 4
    assert (out["outputs"]["out_binary_masks"][0] == mask).all()


def test_video_errors(setup):
    c, _, _ = setup
    assert c.post("/video/sam3/request", json={"type": "x"}).status_code == 404
    r = c.post("/video/sam3.1/request", json={"type": "bogus"})
    assert r.status_code == 500 and "invalid request type" in r.json()["detail"]


def test_stream_ndjson(setup):
    c, _, vp = setup
    r = c.post("/video/sam3.1/stream_request",
               json={"type": "propagate_in_video", "session_id": "s1", "n": 4})
    lines = [decode(json.loads(l)) for l in r.text.strip().split("\n")]
    assert [l["frame_index"] for l in lines] == [0, 1, 2, 3]
    assert lines[0]["outputs"]["out_binary_masks"].shape == (1, 4, 5)
    assert vp.closed_gen


def test_cancel_bypasses_queue_and_stops_stream(setup):
    c, node, vp = setup
    res = {}

    def consume():
        r = c.post("/video/sam3.1/stream_request",
                   json={"type": "propagate_in_video", "session_id": "s1", "n": 1000,
                         "slow": True})
        res["lines"] = r.text.strip().split("\n")

    t = threading.Thread(target=consume)
    t.start()
    for _ in range(100):
        if node._streams.get("s1"):
            break
        time.sleep(0.01)
    time.sleep(0.1)
    r = c.post("/video/sam3.1/request", json={"type": "cancel_propagation", "session_id": "s1"})
    assert r.json() == {"is_success": True} and vp.cancelled.is_set()
    t.join(timeout=10)
    assert not t.is_alive()
    assert len(res["lines"]) < 1000 and json.loads(res["lines"][-1]) == {"cancelled": True}
    assert vp.closed_gen and not node._streams


def test_image_flow(setup):
    c, node, _ = setup
    r = c.post("/image/set_image", json={"image_b64": _png_b64()})
    sid = r.json()["state_id"]
    assert r.json()["original_height"] == 4 and r.json()["original_width"] == 6
    r = c.post("/image/call/set_text_prompt", json={"state_id": sid, "kwargs": {"prompt": "cat"}})
    res = decode(r.json()["result"])
    assert set(res) == {"boxes", "scores", "masks", "presence"}
    assert isinstance(r.json()["result"]["presence"], float) and res["presence"] == 0.5 and res["masks"].shape == (1, 1, 4, 6)
    assert node.image_states[sid]["state"]["prompt"] == "cat"
    # model method with inference_state, encoded input
    r = c.post("/image/call/predict_inst",
               json={"state_id": sid, "kwargs": {"point_coords": [[1, 2]], "point_labels": [1],
                                                  "mask_input": encode(np.zeros((1, 256, 256), np.float32))}})
    masks, scores, mi_shape = decode(r.json()["result"])
    assert masks.shape == (1, 2, 3) and mi_shape == [1, 256, 256]
    r = c.post("/image/call/reset_all_prompts", json={"state_id": sid})
    assert r.json() == {"result": None} and node.image_states[sid]["presence"] is None
    r = c.post("/image/call/set_confidence_threshold", json={"state_id": sid, "kwargs": {"threshold": 0.3}})
    assert r.json() == {"result": {}} and node.processor.confidence_threshold == 0.3
    assert c.post("/image/call/to", json={"state_id": sid}).status_code == 404
    assert c.post("/image/call/_forward_grounding", json={"state_id": sid}).status_code == 404
    assert c.post("/image/call/set_image", json={"state_id": sid}).status_code == 404
    assert c.post("/image/call/set_text_prompt", json={"state_id": "nope", "kwargs": {"prompt": "x"}}).status_code == 404
    assert c.delete(f"/image/states/{sid}").json() == {"is_success": True}
    assert c.delete(f"/image/states/{sid}").status_code == 404


def test_config_cli(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"port": 9000, "use_fa3": True}))
    cfg = ServerConfig.from_argv(["--config", str(p), "--models", "sam3.1,sam3",
                                  "--use-fa3", "false", "--max-num-objects", "32"])
    assert cfg.port == 9000 and cfg.use_fa3 is False and cfg.models == ["sam3.1", "sam3"]
    assert cfg.max_num_objects == 32
