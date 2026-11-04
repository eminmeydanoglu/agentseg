"""ml.sam3_video: frames, client wire format and TrackEngine (fake sam_server, no GPU)."""

import importlib.util
import json
import os

import httpx
import numpy as np
import pytest
from PIL import Image

from ml.sam3_video import codec
from ml.sam3_video.client import Sam3Client
from ml.sam3_video.engine import BACKWARD, FORWARD, Segment, TrackEngine, absent_runs, plan_segments
from ml.sam3_video.frames import FrameSet, build_symlink_folder, list_images
from tests.fake_sam3 import FakeSam3Client, box_mask

N, H, W = 30, 20, 64
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def frames(tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(N):
        Image.new("RGB", (W, H), (i, i, i)).save(img_dir / f"frame_{i:03d}.png")
    return FrameSet.from_dir(str(img_dir), root=str(tmp_path / "work"))


@pytest.fixture
def fake():
    return FakeSam3Client(N, H, W)


def kf_mask(x0=2):
    return box_mask(H, W, (x0, 5, x0 + 4, 10))


def engine(fake, frames, **kw):
    return TrackEngine(fake, frames, **kw)


# ── planning ──

def test_plan_segments_cover_every_frame_once():
    for kfs in ([0], [29], [12], [0, 29], [3, 10, 11, 25]):
        segs = plan_segments(kfs, N)
        covered = [f for s in segs for f in s.frames]
        assert sorted(covered + kfs) == list(range(N)), kfs
    assert plan_segments([0], N) == [Segment(0, FORWARD, N - 1)]
    assert plan_segments([10], N) == [Segment(10, FORWARD, N - 1), Segment(10, BACKWARD, 0)]
    # midpoint split between 4 and 10: 4 owns 5..7, 10 owns 8..9
    assert plan_segments([4, 10], N)[:2] == [Segment(4, FORWARD, 7), Segment(4, BACKWARD, 0)]
    assert Segment(10, BACKWARD, 8) in plan_segments([4, 10], N)


def test_absent_runs():
    assert absent_runs([5, 3, 4, 9, 10, 20]) == [(3, 5), (9, 10), (20, 20)]


# ── frames ──

def test_frameset_symlink_folder(frames):
    folder = frames.folder
    names = sorted(os.listdir(folder), key=lambda f: int(f.split(".")[0]))
    assert names == [f"{i}.jpg" for i in range(N)]
    assert os.readlink(os.path.join(folder, "7.jpg")) == frames.path(7)
    assert frames.size == (W, H)
    assert frames.index_of(frames.path(4)) == 4
    # reused when unchanged, rebuilt when the list changes
    assert build_symlink_folder(frames.paths, folder) == folder
    build_symlink_folder(frames.paths[:3], folder)
    assert len(os.listdir(folder)) == 3
    frames.cleanup()


def test_list_images_numeric_order(tmp_path):
    for i in (10, 2, 1):
        Image.new("RGB", (4, 4)).save(tmp_path / f"{i}.jpg")
    assert [os.path.basename(p) for p in list_images(str(tmp_path))] == ["1.jpg", "2.jpg", "10.jpg"]


# ── codec / client wire format ──

def _server_codec():
    spec = importlib.util.spec_from_file_location("sam_server_codec", os.path.join(ROOT, "sam_server/sam_server/codec.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_codec_matches_sam_server_copy():
    srv = _server_codec()
    m = np.zeros((2, 7, 9), bool)
    m[0, 1:4, 2:5] = True
    obj = {"a": m, "b": np.arange(5, dtype=np.float32), "c": np.zeros((3, 70)), "d": {1: "x"}}
    assert json.dumps(codec.encode(obj), sort_keys=True) == json.dumps(srv.encode(obj), sort_keys=True)
    back = codec.decode(srv.encode(obj))
    assert np.array_equal(back["a"], m) and back["d"] == {1: "x"}


def test_client_requests_and_stream():
    seen = []
    mask = np.zeros((4, 6), bool)
    mask[1:3, 2:4] = True

    def handler(req: httpx.Request):
        body = json.loads(req.content) if req.content else None
        seen.append((req.method, req.url.path, body))
        if req.url.path == "/video/sam3/request":
            if body["type"] == "start_session":
                return httpx.Response(200, json={"session_id": "S"})
            if body["type"] == "add_mask":
                assert np.array_equal(codec.decode(body["mask"]), mask)
                return httpx.Response(200, json={"frame_index": body["frame_index"], "outputs": {}})
            return httpx.Response(200, json={"is_success": True})
        if req.url.path == "/video/sam3/stream_request":
            lines = []
            for f in (3, 4):
                out = {"out_obj_ids": [7], "out_probs": [0.9], "out_binary_masks": mask[None] if f == 3 else mask[None] & False}
                lines.append(json.dumps({"frame_index": f, "outputs": codec.encode(out)}))
            return httpx.Response(200, content=("\n".join(lines) + "\n").encode())
        if req.url.path == "/image/set_image":
            return httpx.Response(200, json={"state_id": "I"})
        if req.url.path == "/image/call/predict_inst":
            res = codec.encode([mask[None], np.array([0.8], np.float32), np.zeros((1, 2, 2), np.float32)])
            return httpx.Response(200, json={"result": res})
        if req.method == "DELETE":
            return httpx.Response(200, json={})
        return httpx.Response(404)

    c = Sam3Client(http=httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler)))
    assert c.start_session("/tmp/f") == "S"
    c.add_mask("S", 3, 7, mask)
    outs = list(c.propagate("S", 3, "forward", max_frames=1))
    assert [o.frame_index for o in outs] == [3, 4]
    assert np.array_equal(outs[0].mask(7), mask) and not outs[1].mask(7).any()
    prop = next(b for m_, p, b in seen if p.endswith("stream_request"))
    assert prop["force_tracker_propagation"] is True and prop["max_frame_num_to_track"] == 1
    assert prop["propagation_direction"] == "forward" and prop["start_frame_index"] == 3
    m, score = c.segment("/img.jpg", box=[1, 2, 3, 4], points=[[2, 2], [5, 1]], point_labels=[1, 0])
    assert np.array_equal(m, mask) and score == pytest.approx(0.8)
    call = next(b for m_, p, b in seen if p.endswith("predict_inst"))
    assert call["kwargs"]["box"] == [[1.0, 2.0, 3.0, 4.0]] and call["kwargs"]["point_labels"] == [1, 0]
    assert any(m_ == "DELETE" and p == "/image/states/I" for m_, p, _ in seen)


# ── engine ──

def test_single_keyframe_forward(fake, frames):
    e = engine(fake, frames)
    t = e.add_track()
    e.add_keyframe(t, 0, kf_mask())
    events = list(e.propagate())
    assert events[0].kind == "start" and events[0].total == N - 1 and events[-1].kind == "done"
    assert sum(ev.kind == "frame" for ev in events) == N - 1
    assert fake.propagate_calls() == [(0, FORWARD, N - 1)]
    m = e.masks(t)
    assert len(m) == N and np.array_equal(m[0], kf_mask())
    # fake: shift = |f - s| + n_cond
    assert np.array_equal(m[5], np.roll(kf_mask(), 5 + 1, axis=1))
    assert e.track(t).origin[17] == 0 and e.frame_kind(t, 17) == "propagated"
    # nothing pending -> no SAM work
    assert e.pending(t) == {t: []}
    list(e.propagate())
    assert len(fake.propagate_calls()) == 1


def test_middle_keyframe_runs_both_directions(fake, frames):
    e = engine(fake, frames)
    t = e.add_track()
    e.add_keyframe(t, 12, kf_mask())
    e.run()
    assert fake.propagate_calls() == [(12, FORWARD, N - 1 - 12), (12, BACKWARD, 12)]
    assert sorted(e.masks(t)) == list(range(N))


def test_add_keyframe_is_incremental(fake, frames):
    e = engine(fake, frames)
    t = e.add_track()
    e.add_keyframe(t, 0, kf_mask())
    e.run()
    before = e.masks(t)
    e.add_keyframe(t, 20, kf_mask(30), source="agent")
    assert e.pending(t)[t] == [Segment(20, FORWARD, N - 1), Segment(20, BACKWARD, 11)]
    e.run()
    calls = fake.propagate_calls()
    assert calls[1:] == [(20, FORWARD, N - 1 - 20), (20, BACKWARD, 20 - 11)]
    after = e.masks(t)
    origin = e.track(t).origin
    for f in range(1, 11):  # still owned by kf 0, untouched
        assert np.array_equal(after[f], before[f]) and origin[f] == 0
    for f in range(11, N):
        assert origin[f] == 20
    assert np.array_equal(after[20], kf_mask(30))
    assert e.keyframes(t)[20].source == "agent"


def _fresh_run(frames, kfs):
    fake = FakeSam3Client(N, H, W)
    e = TrackEngine(fake, frames)
    t = e.add_track()
    for f, x in kfs:
        e.add_keyframe(t, f, kf_mask(x))
    e.run()
    return e.masks(t)


@pytest.mark.parametrize("mode", ["object", "session"])
def test_remove_keyframe_equals_never_added(fake, frames, mode):
    e = engine(fake, frames, remove_mode=mode)
    t = e.add_track()
    e.add_keyframe(t, 0, kf_mask())
    e.add_keyframe(t, 25, kf_mask(10))
    e.run()
    e.add_keyframe(t, 12, kf_mask(40))
    e.run()
    old_obj = e.track(t).obj_id
    e.remove_keyframe(t, 12)
    e.run()
    got = e.masks(t)
    want = _fresh_run(frames, [(0, 2), (25, 10)])
    assert sorted(got) == sorted(want)
    for f in want:
        assert np.array_equal(got[f], want[f]), f
    assert e.track(t).obj_id != old_obj  # rebuilt on a fresh object
    assert ("remove_object", old_obj) in fake.calls if mode == "object" else ("reset_session", "s1") in fake.calls


def test_remove_unsynced_keyframe_restores_without_rebuild(fake, frames):
    e = engine(fake, frames)
    t = e.add_track()
    e.add_keyframe(t, 0, kf_mask())
    e.run()
    before = e.masks(t)
    e.add_keyframe(t, 9, kf_mask(30))
    e.remove_keyframe(t, 9)
    assert e.pending(t)[t] == []
    e.run()
    assert len(fake.propagate_calls()) == 1
    after = e.masks(t)
    assert all(np.array_equal(after[f], before[f]) for f in before)


def test_mark_absent_forces_empty_and_clear_restores(fake, frames):
    e = engine(fake, frames)
    t = e.add_track()
    e.add_keyframe(t, 0, kf_mask())
    e.run()
    prev = e.mask(t, 7).copy()
    assert e.mark_absent(t, range(5, 9)) == [5, 6, 7, 8]
    assert all(not e.mask(t, f).any() for f in range(5, 9))
    assert e.frame_kind(t, 6) == "absent"
    e.run(full=True)  # a full rebuild still keeps them empty
    assert all(not e.mask(t, f).any() for f in range(5, 9))
    assert not any(c[0] == "add_mask" and not c[3] for c in fake.calls)  # nothing empty sent to SAM
    e.clear_absent(t, [7])
    assert np.array_equal(e.mask(t, 7), prev)
    assert e.frame_kind(t, 7) == "propagated"


def test_absent_to_sam_sends_run_edges(fake, frames):
    e = engine(fake, frames, absent_to_sam=True)
    t = e.add_track()
    e.add_keyframe(t, 0, kf_mask())
    e.mark_absent(t, list(range(5, 9)) + [20])
    e.run()
    empty_adds = sorted(c[1] for c in fake.calls if c[0] == "add_mask" and not c[3])
    assert empty_adds == [5, 8, 20]
    assert all(not e.mask(t, f).any() for f in (5, 6, 7, 8, 20))


def test_keyframe_over_absent_frame(fake, frames):
    e = engine(fake, frames, absent_to_sam=True)
    t = e.add_track()
    e.add_keyframe(t, 0, kf_mask())
    e.mark_absent(t, [10])
    e.run()
    e.add_keyframe(t, 10, kf_mask(20))
    e.run()
    assert e.track(t)._rebuild is False and e.frame_kind(t, 10) == "keyframe"
    assert all(c[1] != 10 or c[3] for c in fake.calls[-40:] if c[0] == "add_mask" and c[2] == e.track(t).obj_id)


def test_multi_track_obj_ids_and_independence(fake, frames):
    e = engine(fake, frames)
    a = e.add_track("a")
    b = e.add_track("b")
    e.add_keyframe(a, 0, kf_mask(2))
    e.add_keyframe(b, 10, kf_mask(30))
    e.run()
    assert e.track(a).obj_id != e.track(b).obj_id
    assert len(fake.sessions) == 1
    single = _fresh_run(frames, [(0, 2)])
    for f in range(N):
        assert np.array_equal(e.mask(a, f), single[f])
    assert e.track(b).origin[3] == 10
    # editing b does not touch a
    n_calls = len(fake.propagate_calls())
    e.add_keyframe(b, 20, kf_mask(5))
    e.run(["b"])
    assert all(c[0] in (20,) for c in fake.propagate_calls()[n_calls:])
    e.remove_track("a")
    assert ("remove_object", 1) in fake.calls


def test_closing_generator_cancels_and_keeps_segment_pending(fake, frames):
    e = engine(fake, frames)
    t = e.add_track()
    e.add_keyframe(t, 0, kf_mask())
    gen = e.propagate()
    for ev in gen:
        if ev.kind == "frame" and ev.frame == 5:
            break
    gen.close()
    assert fake.cancelled == 1
    assert e.pending(t)[t] == [Segment(0, FORWARD, N - 1)]
    stop = {"n": 0}

    def should_stop():
        stop["n"] += 1
        return stop["n"] > 3

    r = e.run(should_stop=should_stop)
    assert r["stopped"] and fake.cancelled == 2
    r = e.run()
    assert r["frames"] == N - 1 and e.pending(t)[t] == []


def test_frame_range_clips_segments(fake, frames):
    e = engine(fake, frames)
    t = e.add_track()
    e.add_keyframe(t, 10, kf_mask())
    e.run(frame_range=(5, 15))
    assert fake.propagate_calls() == [(10, FORWARD, 5), (10, BACKWARD, 5)]
    assert sorted(e.masks(t)) == list(range(5, 16))
    assert len(e.pending(t)[t]) == 2
    e.run()
    assert sorted(e.masks(t)) == list(range(N))


def test_validation(fake, frames):
    e = engine(fake, frames)
    t = e.add_track()
    with pytest.raises(ValueError):
        e.add_keyframe(t, 0, np.zeros((H, W), bool))
    with pytest.raises(ValueError):
        e.add_keyframe(t, 0, np.ones((H + 1, W), bool))
    with pytest.raises(IndexError):
        e.add_keyframe(t, N, kf_mask())
    with pytest.raises(KeyError):
        e.add_track(t)
    with pytest.raises(NotImplementedError):
        TrackEngine(fake, frames, window=100)


def test_close_then_propagate_rebuilds(fake, frames):
    e = engine(fake, frames)
    t = e.add_track()
    e.add_keyframe(t, 0, kf_mask())
    e.run()
    e.close()
    assert ("close_session", "s1") in fake.calls
    e.run()
    assert len(fake.sessions) == 1 and "s2" in fake.sessions
    assert len(e.masks(t)) == N


def test_segment_helper_maps_pos_neg(fake, frames):
    e = engine(fake, frames)
    m, s = e.segment(3, box=(1, 1, 5, 5), pos=[(2, 2)], neg=[(4, 4)])
    assert m.shape == (H, W) and s == 0.9
    assert fake.calls[-1] == ("segment", frames.path(3), (1, 1, 5, 5))


def test_max_lost_stops_a_direction_and_leaves_the_rest_pending(fake, frames):
    # the fake slides the mask right by 1 px per frame, so it leaves the frame after ~W frames
    e = engine(fake, frames)
    e.add_track(1)
    e.add_keyframe(1, 0, box_mask(H, W, (W - 6, 5, W - 2, 10)))
    events = list(e.propagate(max_lost=3))
    lost = [ev for ev in events if ev.kind == "lost"]
    assert len(lost) == 1
    info = lost[0].info
    assert info["direction"] == "forward" and info["stopped_at"] == info["lost_from"] + 2
    assert all(f <= info["stopped_at"] for f in e.track(1).masks)
    assert any(s.kf == 0 for s in e._pending_segments(e.track(1)))  # the cut rest is tracked again next time
