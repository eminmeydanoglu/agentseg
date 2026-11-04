"""ml.sam3_video.health on synthetic tracks (no GPU)."""


import numpy as np

from ml.sam3_video.engine import TrackEngine
from ml.sam3_video.health import (
    ABSENT,
    KEYFRAME,
    LOST,
    OK,
    REVIEW,
    HealthConfig,
    assess,
    assess_track,
    group_issues,
    mask_geom,
)
from tests.fake_sam3 import FakeSam3Client, box_mask

H, W, N = 100, 200, 40


def rect(x0, y0, x1, y1):
    return box_mask(H, W, (x0, y0, x1, y1))


def steady_track(n=N, kf=0):
    """A 20x20 square moving 1 px/frame, tracked forward from ``kf``."""
    masks = {f: rect(50 + f, 40, 70 + f, 60) for f in range(n)}
    origin = {f: kf for f in range(n)}
    return masks, {kf: masks[kf]}, origin


def statuses(health):
    return {h.frame: h.status for h in health}


def test_mask_geom_components():
    m = rect(10, 10, 30, 30) | rect(60, 10, 70, 20)
    g = mask_geom(m, min_component_px=30)
    assert g.area == 500 and g.n_components == 2 and abs(g.largest_frac - 0.8) < 1e-6
    assert g.box == (10, 10, 70, 30)
    assert mask_geom(np.zeros((H, W), bool)).area == 0


def test_steady_track_is_ok():
    masks, kfs, origin = steady_track()
    hs = assess_track(masks, kfs, origin, (W, H))
    st = statuses(hs)
    assert st[0] == KEYFRAME
    assert all(st[f] == OK for f in range(1, N)), [h.to_dict() for h in hs if h.status != OK][:3]
    assert group_issues(hs, "t") == []


def test_loss_scope_jump_fragment_are_flagged():
    masks, kfs, origin = steady_track()
    for f in range(10, 15):  # lost
        masks[f] = np.zeros((H, W), bool)
    masks[20] = rect(50 + 20, 40, 70 + 20, 95)  # grows onto the "stand"
    masks[21] = rect(50 + 21, 40, 70 + 21, 95)
    masks[30] = rect(150, 40, 170, 60)  # jump
    masks[35] = rect(85, 40, 95, 60) | rect(120, 40, 130, 60)  # split in two
    cfg = HealthConfig(max_center_jump=0.5, max_aspect_log=0.4)
    hs = assess_track(masks, kfs, origin, (W, H), cfg)
    st = statuses(hs)
    assert all(st[f] == LOST for f in range(10, 15))
    assert st[20] == REVIEW and st[30] == REVIEW and st[35] == REVIEW
    issues = group_issues(hs, "t", cfg)
    kinds = {i.kind for i in issues}
    lost = next(i for i in issues if i.kind == "lost")
    assert (lost.start, lost.end, lost.n_lost) == (10, 14, 5)
    assert lost.worst_frame == 12  # middle of the empty run
    assert "scope" in kinds
    assert any("jump" in i.reasons for i in issues) and any("fragment" in i.reasons for i in issues)
    scope = next(i for i in issues if i.kind == "scope")
    assert scope.start <= 20 <= scope.end and scope.worst_frame in (20, 21)


def test_gaps_merge_and_absent_breaks_issues():
    masks, kfs, origin = steady_track()
    for f in (10, 11, 14, 15):
        masks[f] = np.zeros((H, W), bool)
    issues = group_issues(assess_track(masks, kfs, origin, (W, H)), "t", HealthConfig(merge_gap=3))
    assert [(i.start, i.end, i.n_lost) for i in issues] == [(10, 15, 4)]
    issues = group_issues(assess_track(masks, kfs, origin, (W, H)), "t", HealthConfig(merge_gap=0))
    assert sum(i.kind == "lost" for i in issues) == 2
    # declared absent frames are neither issues nor merged across
    kfs2 = {**kfs, 12: None, 13: None}
    masks[12] = masks[13] = np.zeros((H, W), bool)
    hs = assess_track(masks, kfs2, origin, (W, H))
    assert statuses(hs)[12] == ABSENT
    lost = [i for i in group_issues(hs, "t", HealthConfig(merge_gap=5)) if i.kind == "lost"]
    assert [(i.start, i.end) for i in lost] == [(10, 11), (14, 15)]


def test_border_excuses_shrink():
    masks, kfs, origin = steady_track()
    for f in range(20, N):  # leaving through the right edge, area shrinking
        x0 = 50 + f * 4
        masks[f] = rect(min(x0, W - 4), 40, W, 60)
    hs = assess_track(masks, kfs, origin, (W, H))
    for h in hs:
        if h.frame >= 20 and h.signals.get("touches_border"):
            assert not any(r in ("step_shrink", "kf_shrink", "kf_area_tiny") for r in h.reasons), h.to_dict()


def test_edge_sliver_is_exit_issue():
    masks, kfs, origin = steady_track()
    for f in range(30, N):  # only a thin sliver left at the right border
        masks[f] = rect(W - 3, 40, W, 60)
    issues = group_issues(assess_track(masks, kfs, origin, (W, H)), "t")
    assert [(i.kind, i.start, i.end) for i in issues] == [("exit", 30, N - 1)]


def test_window_area_flags_sudden_shrink_but_not_slow_zoom():
    masks, kfs, origin = steady_track()
    zoom = {f: rect(50, 40, 70 + f, 60 + f // 2) for f in range(N)}  # grows slowly ~4x
    hs = assess_track(zoom, {0: zoom[0]}, origin, (W, H))
    assert all(h.status in (OK, KEYFRAME) for h in hs)
    for f in range(20, 23):  # sudden partial occlusion, recovered by 23
        masks[f] = rect(50 + f, 40, 56 + f, 50)
    st = statuses(assess_track(masks, kfs, origin, (W, H)))
    assert all(st[f] == REVIEW for f in range(20, 23)) and st[25] == OK


def test_geometry_only_preset():
    assert HealthConfig.geometry_only().kf_area_min == 0.3
    assert HealthConfig.geometry_only(merge_gap=1).merge_gap == 1


def test_backward_segment_uses_previous_frame_towards_keyframe():
    masks, _, _ = steady_track()
    origin = {f: N - 1 for f in range(N)}
    hs = assess_track(masks, {N - 1: masks[N - 1]}, origin, (W, H))
    assert all(h.status in (OK, KEYFRAME) for h in hs)
    assert hs[5].signals["step_area"] == 1.0


def test_dino_embed_hook():
    masks, kfs, origin = steady_track()
    calls = []

    def embed(f, m):
        calls.append(f)
        v = np.array([1.0, 0.0]) if f < 25 else np.array([0.0, 1.0])
        return v

    hs = assess_track(masks, kfs, origin, (W, H), HealthConfig(dino_every=1), embed=embed)
    st = statuses(hs)
    assert st[10] == OK and st[30] == LOST
    assert "dino_lost" in hs[30].reasons
    sparse = []
    assess_track(masks, kfs, origin, (W, H), HealthConfig(dino_every=5), embed=lambda f, m: sparse.append(f) or np.ones(2) / 2 ** .5)
    assert len(sparse) < len(calls) / 3


def test_assess_engine_and_csv(tmp_path):
    from PIL import Image

    from ml.sam3_video.frames import FrameSet

    d = tmp_path / "im"
    d.mkdir()
    for i in range(N):
        Image.new("RGB", (W, H)).save(d / f"{i}.jpg")
    fake = FakeSam3Client(N, H, W, dx=1)
    e = TrackEngine(fake, FrameSet.from_dir(str(d), root=str(tmp_path / "w")))
    t = e.add_track()
    e.add_keyframe(t, 0, rect(20, 40, 40, 60))
    e.run()
    health, issues = assess(e, t)
    assert len(health) == N and issues == []
