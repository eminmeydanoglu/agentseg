"""Tracking health: cheap per-frame signals -> ok | review | lost -> issues.

SAM 3's tracker never stops on its own and fails silently (drift, scope creep onto the
stand, loss, occlusion), and its ``out_probs`` is ~constant per object, so health is judged
here from the masks alone (plus optional DINO similarity):

=================  ==========================================================================
signal             computation
=================  ==========================================================================
empty / empty_run  mask empty; length of the run of empty frames containing this frame
step_area          area / area of the previous frame in tracking direction (towards the
                   frame's keyframe), relative to an EMA area trend
                   (log-space EMA of the mask area)
kf_area            area / area of the keyframe this frame was tracked from
win_area           area / median area of the previous ``area_window`` frames (slow zoom ok)
edge               touches the border while small vs the keyframe: leaving / partly out
jump               centre shift vs previous frame / previous bbox diagonal
components         # connected components >= ``min_component_px``; largest component's share
aspect / fill      |log| bbox aspect change vs keyframe; (area / bbox area) / keyframe's
touches_border     bbox touches the image edge (info; excuses an area drop)
dino_sim           (optional) cosine similarity of the masked DINO descriptor to the keyframe's
=================  ==========================================================================

An **issue** is a run of review/lost frames (gaps <= ``merge_gap`` merged) with one
suggested ``worst_frame``. Absent keyframes are excluded (the object is declared away).

Pure numpy/cv2; DINO is injected as an ``embed(frame, mask) -> vector`` callable
(``DinoEmbedder`` wraps ``ml.embed``).
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable, Hashable, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np

OK = "ok"
REVIEW = "review"
LOST = "lost"
ABSENT = "absent"  # absent keyframe (declared not visible)
KEYFRAME = "keyframe"

# reason -> issue kind
REASON_KIND = {
    "empty": "lost",
    "kf_area_tiny": "lost",
    "dino_lost": "lost",
    "step_grow": "scope",
    "kf_grow": "scope",
    "aspect": "scope",
    "fill": "scope",
    "step_shrink": "drift",
    "kf_shrink": "drift",
    "dino_low": "drift",
    "win_shrink": "drift",
    "win_grow": "scope",
    "jump": "jump",
    "fragment": "fragment",
    "edge": "exit",
}
KIND_PRIORITY = ("lost", "exit", "scope", "jump", "fragment", "drift")


@dataclass
class HealthConfig:
    """Thresholds. Defaults tuned on the torpedo sequence + its stress variants, assuming the DINO signal is on; use
    ``HealthConfig.geometry_only()`` when no embedder is available.

    Findings behind the defaults: legit scale change vs the keyframe spans 0.12x..2.9x and
    the (duplicated-frame) footage jumps up to ~1.9 box diagonals between frames, so the
    keyframe-area and jump checks are effectively off; sudden changes are judged against the
    median of the last frames (``win_area``) and the occlusion / wrong-object case by DINO.
    """

    # lost
    lost_kf_area_ratio: float = 0.1  # area below this x keyframe area -> lost (unless at border)
    # per-step area change relative to the EMA trend
    step_area_min: float = 0.5
    step_area_max: float = 2.0
    trend_alpha: float = 0.3
    trend_clamp: tuple = (0.8, 1.25)
    # vs median of the last ``area_window`` frames in tracking direction
    area_window: int = 10
    win_area_min: float = 0.4
    win_area_max: float = 3.0
    # touching the border with area below this x keyframe area -> "edge" (kind "exit")
    edge_kf_area: float = 0.3
    # vs keyframe
    kf_area_min: float = 0.05
    kf_area_max: float = 10.0
    max_aspect_log: float = 0.7  # |log(aspect / aspect_kf)|
    min_fill_ratio: float = 0.6  # fill / fill_kf
    # motion / shape
    max_center_jump: float = 3.0  # / previous bbox diagonal
    min_component_px: int = 30
    min_largest_frac: float = 0.9
    border_margin: int = 2
    # DINO (only used when an embedder is given)
    dino_review: float = 0.6
    dino_lost: float = 0.35
    dino_every: int = 3  # compute every k-th frame of a segment, plus on suspicious frames
    # grouping
    merge_gap: int = 3
    min_issue_frames: int = 1

    @classmethod
    def geometry_only(cls, **kw) -> "HealthConfig":
        """Best no-DINO setting from the WP3 sweep: catches occlusion through a keyframe-area
        drop (kf_area_min 0.3), at the cost of long low-precision issues on far-away stretches."""
        return cls(**{"kf_area_min": 0.3, **kw})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FrameHealth:
    frame: int
    status: str  # ok | review | lost | absent | keyframe
    reasons: list[str] = field(default_factory=list)
    severity: float = 0.0
    signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"frame": self.frame, "status": self.status, "reasons": list(self.reasons),
                "severity": round(self.severity, 4), "signals": _round(self.signals)}


@dataclass
class Issue:
    track_id: Hashable
    start: int
    end: int
    worst_frame: int
    kind: str  # lost | exit | scope | jump | fragment | drift
    n_frames: int
    n_lost: int
    reasons: dict = field(default_factory=dict)  # reason -> frame count
    signals: dict = field(default_factory=dict)  # at worst_frame

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signals"] = _round(self.signals)
        return d

    def __contains__(self, frame: int) -> bool:
        return self.start <= frame <= self.end


def _round(d: Mapping) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating)):
            out[k] = None if not np.isfinite(v) else round(float(v), 4)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.bool_):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


# ── per-mask geometry ──

@dataclass
class MaskGeom:
    area: int
    box: Optional[tuple]  # x0, y0, x1, y1 (pixels, exclusive end)
    n_components: int
    largest_frac: float

    @property
    def center(self):
        x0, y0, x1, y1 = self.box
        return (x0 + x1) / 2, (y0 + y1) / 2

    @property
    def diag(self) -> float:
        x0, y0, x1, y1 = self.box
        return float(math.hypot(x1 - x0, y1 - y0))

    @property
    def aspect(self) -> float:
        x0, y0, x1, y1 = self.box
        return (x1 - x0) / max(1.0, y1 - y0)

    @property
    def fill(self) -> float:
        x0, y0, x1, y1 = self.box
        return self.area / max(1.0, (x1 - x0) * (y1 - y0))


def mask_geom(mask: Optional[np.ndarray], min_component_px: int = 30) -> MaskGeom:
    if mask is None or not mask.any():
        return MaskGeom(0, None, 0, 0.0)
    ys, xs = np.nonzero(mask)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    x0, y0, x1, y1 = box
    crop = mask[y0:y1, x0:x1].astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([crop.sum()])
    area = int(areas.sum())
    big = int((areas >= min_component_px).sum())
    return MaskGeom(area, box, max(1, big), float(areas.max() / area) if area else 0.0)


# ── assessment ──

def assess_track(masks: Mapping[int, np.ndarray], keyframes: Mapping[int, Optional[np.ndarray]],
                 origin: Mapping[int, int], image_size: tuple[int, int],
                 config: Optional[HealthConfig] = None,
                 embed: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
                 frames: Optional[Iterable[int]] = None) -> list[FrameHealth]:
    """Per-frame health of one track.

    masks: frame -> bool mask (derived result incl. keyframes); keyframes: frame -> mask or
    None (absent); origin: frame -> keyframe frame it was tracked from; image_size (w, h).
    embed: optional DINO descriptor function (unit vectors). frames: subset to report
    (default: every frame in ``masks``).
    """
    cfg = config or HealthConfig()
    w, h = image_size
    geoms: dict[int, MaskGeom] = {}

    def geom(f):
        g = geoms.get(f)
        if g is None:
            g = geoms[f] = mask_geom(masks.get(f), cfg.min_component_px)
        return g

    kf_desc: dict[int, np.ndarray] = {}
    wanted = sorted(masks) if frames is None else sorted(frames)
    empty_run = _empty_runs(masks, keyframes)

    # walk every segment outward from its keyframe so the area trend is causal
    by_seg: dict[tuple[int, int], list[int]] = {}
    for f in sorted(masks):
        if f in keyframes:
            continue
        k = origin.get(f)
        if k is None:
            continue
        by_seg.setdefault((k, 1 if f > k else -1), []).append(f)
    out: dict[int, FrameHealth] = {}
    for f, km in keyframes.items():
        out[f] = FrameHealth(f, ABSENT if km is None else KEYFRAME, signals={"area": geom(f).area})
    for (k, step), fs in by_seg.items():
        fs.sort(key=lambda f: abs(f - k))
        kg = geom(k)
        trend = 1.0
        since_dino = cfg.dino_every
        window: list[int] = [kg.area] if kg.area else []
        for f in fs:
            g = geom(f)
            prev = f - step
            pg = geom(prev) if prev in masks else MaskGeom(0, None, 0, 0.0)
            s: dict = {"origin": k, "area": g.area, "empty_run": empty_run.get(f, 0)}
            reasons: list[str] = []
            sev = 0.0
            if g.area == 0:
                reasons.append("empty")
                sev += 10.0 + min(empty_run.get(f, 1), 50) / 50.0
            else:
                border = (g.box[0] <= cfg.border_margin or g.box[1] <= cfg.border_margin
                          or g.box[2] >= w - cfg.border_margin or g.box[3] >= h - cfg.border_margin)
                s["touches_border"] = border
                s["n_components"] = g.n_components
                s["largest_frac"] = g.largest_frac
                if window:
                    wr = g.area / float(np.median(window[-cfg.area_window:]))
                    s["win_area"] = wr
                    if wr < cfg.win_area_min and not border:
                        reasons.append("win_shrink")
                        sev += _excess(cfg.win_area_min / wr)
                    elif wr > cfg.win_area_max:
                        reasons.append("win_grow")
                        sev += _excess(wr / cfg.win_area_max)
                if kg.area:
                    r = g.area / kg.area
                    if border and r < cfg.edge_kf_area:
                        reasons.append("edge")
                        sev += 1.0 + _excess(cfg.edge_kf_area / r)
                    s["kf_area"] = r
                    s["aspect_log"] = abs(math.log(g.aspect / kg.aspect))
                    s["fill_ratio"] = g.fill / kg.fill if kg.fill else 1.0
                    if r < cfg.lost_kf_area_ratio and not border:
                        reasons.append("kf_area_tiny")
                        sev += 5.0
                    elif r < cfg.kf_area_min and not border:
                        reasons.append("kf_shrink")
                        sev += _excess(cfg.kf_area_min / r)
                    elif r > cfg.kf_area_max:
                        reasons.append("kf_grow")
                        sev += _excess(r / cfg.kf_area_max)
                    if s["aspect_log"] > cfg.max_aspect_log and not border:
                        reasons.append("aspect")
                        sev += s["aspect_log"] - cfg.max_aspect_log
                    if s["fill_ratio"] < cfg.min_fill_ratio:
                        reasons.append("fill")
                        sev += _excess(cfg.min_fill_ratio / s["fill_ratio"])
                if pg.area:
                    ratio = g.area / pg.area
                    rel = ratio / trend
                    s["step_area"] = ratio
                    s["step_rel"] = rel
                    s["trend"] = trend
                    jump = math.hypot(g.center[0] - pg.center[0], g.center[1] - pg.center[1]) / max(1.0, pg.diag)
                    s["jump"] = jump
                    if rel < cfg.step_area_min and not border:
                        reasons.append("step_shrink")
                        sev += _excess(cfg.step_area_min / rel)
                    elif rel > cfg.step_area_max:
                        reasons.append("step_grow")
                        sev += _excess(rel / cfg.step_area_max)
                    else:
                        trend = _update_trend(trend, ratio, cfg)
                    if jump > cfg.max_center_jump:
                        reasons.append("jump")
                        sev += jump / cfg.max_center_jump - 1.0
                if g.n_components > 1 and g.largest_frac < cfg.min_largest_frac:
                    reasons.append("fragment")
                    sev += (cfg.min_largest_frac - g.largest_frac) * 5
                if embed is not None:
                    since_dino += 1
                    if reasons or since_dino >= cfg.dino_every:
                        since_dino = 0
                        if k not in kf_desc:
                            kf_desc[k] = embed(k, masks[k])
                        sim = float(np.dot(embed(f, masks[f]), kf_desc[k]))
                        s["dino_sim"] = sim
                        if sim < cfg.dino_lost:
                            reasons.append("dino_lost")
                            sev += 5.0
                        elif sim < cfg.dino_review:
                            reasons.append("dino_low")
                            sev += (cfg.dino_review - sim) * 10
            if g.area:
                window.append(g.area)
            lost = any(REASON_KIND[r] == "lost" for r in reasons)
            status = LOST if lost else REVIEW if reasons else OK
            out[f] = FrameHealth(f, status, reasons, sev, s)
    return [out[f] for f in wanted if f in out]


def _excess(ratio_over_threshold: float) -> float:
    return math.log(max(ratio_over_threshold, 1.0)) * 4


def _update_trend(trend: float, ratio: float, cfg: HealthConfig) -> float:
    # log-space EMA of the mask area
    if ratio <= 0:
        return trend
    v = math.exp((1 - cfg.trend_alpha) * math.log(max(trend, 1e-6)) + cfg.trend_alpha * math.log(ratio))
    return min(cfg.trend_clamp[1], max(cfg.trend_clamp[0], v))


def _empty_runs(masks: Mapping[int, np.ndarray], keyframes: Mapping) -> dict[int, int]:
    """frame -> length of the run of consecutive empty (non-absent) frames containing it."""
    out: dict[int, int] = {}
    run: list[int] = []
    for f in sorted(masks):
        empty = f not in keyframes and not masks[f].any()
        if empty and (not run or run[-1] == f - 1):
            run.append(f)
            continue
        for g in run:
            out[g] = len(run)
        run = [f] if empty else []
    for g in run:
        out[g] = len(run)
    return out


def group_issues(health: Sequence[FrameHealth], track_id: Hashable = None,
                 config: Optional[HealthConfig] = None) -> list[Issue]:
    """Runs of review/lost frames (gaps <= merge_gap ok frames merged) -> issues."""
    cfg = config or HealthConfig()
    bad = [fh for fh in sorted(health, key=lambda x: x.frame) if fh.status in (REVIEW, LOST)]
    groups: list[list[FrameHealth]] = []
    for fh in bad:
        if groups and fh.frame - groups[-1][-1].frame <= cfg.merge_gap + 1 and not _blocked(health, groups[-1][-1].frame, fh.frame):
            groups[-1].append(fh)
        else:
            groups.append([fh])
    issues = []
    for g in groups:
        if len(g) < cfg.min_issue_frames:
            continue
        reasons = Counter(r for fh in g for r in fh.reasons)
        kinds = Counter(REASON_KIND[r] for fh in g for r in set(fh.reasons))
        n_lost = sum(fh.status == LOST for fh in g)
        kind = "lost" if n_lost else max(kinds, key=lambda k: (kinds[k], -KIND_PRIORITY.index(k)))
        worst = _worst(g)
        issues.append(Issue(track_id, g[0].frame, g[-1].frame, worst.frame, kind, len(g), n_lost,
                            dict(reasons.most_common()), {"severity": worst.severity,
                                                          "reasons": ",".join(worst.reasons), **worst.signals}))
    return issues


def _blocked(health: Sequence[FrameHealth], a: int, b: int) -> bool:
    """Do not merge across a keyframe / absent frame."""
    return any(a < fh.frame < b and fh.status in (KEYFRAME, ABSENT) for fh in health)


def _worst(group: Sequence[FrameHealth]) -> FrameHealth:
    top = max(fh.severity for fh in group)
    tied = [fh for fh in group if fh.severity >= top - 1e-6]
    mid = (tied[0].frame + tied[-1].frame) / 2
    return min(tied, key=lambda fh: abs(fh.frame - mid))


def assess(engine, track_id: Hashable, config: Optional[HealthConfig] = None,
           embed: Optional[Callable[[int, np.ndarray], np.ndarray]] = None) -> tuple[list[FrameHealth], list[Issue]]:
    """Health + issues of one ``TrackEngine`` track."""
    t = engine.track(track_id)
    kfs = {f: k.mask for f, k in t.keyframes.items()}
    health = assess_track(t.masks, kfs, t.origin, engine.image_size, config, embed)
    return health, group_issues(health, track_id, config)


CSV_FIELDS = ["frame", "status", "severity", "reasons", "origin", "area", "empty_run", "kf_area", "win_area", "step_area",
              "step_rel", "trend", "jump", "n_components", "largest_frac", "aspect_log", "fill_ratio",
              "touches_border", "dino_sim"]


class DinoEmbedder:
    """Masked DINOv3 descriptor per frame via ``ml.embed.EmbedService`` (loads on first use).

    Callable as ``embed(frame, mask) -> unit vector``; caches per (frame, mask bytes hash).
    ``seconds`` / ``calls`` accumulate timing for cost measurement.
    """

    def __init__(self, paths: Sequence[str], model: str = "vitb16", device: str = "cuda"):
        from ml.embed import weights_path

        if not weights_path(model).exists():  # fail here so the caller falls back to geometry only
            raise FileNotFoundError(f"DINOv3 weights not found: {weights_path(model)}")
        self.paths = list(paths)
        self.model = model
        self.device = device
        self._svc = None
        self._cache: dict = {}
        self.seconds = 0.0
        self.calls = 0
        self.load_seconds = 0.0

    def _service(self):
        if self._svc is None:
            t0 = time.time()
            from ml.embed import EmbedService

            self._svc = EmbedService(device=self.device, model_name=self.model)
            self._svc.load_model()
            self.load_seconds = time.time() - t0
        return self._svc

    def __call__(self, frame: int, mask: np.ndarray) -> np.ndarray:
        key = (frame, hash(np.packbits(mask).tobytes()))
        if key in self._cache:
            return self._cache[key]
        svc = self._service()
        t0 = time.time()
        ys, xs = np.nonzero(mask)
        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        d = svc.get_object_descriptor(self.paths[frame], box, mask=mask.astype(np.uint8),
                                      use_cache=False)  # per-image GPU feature cache would grow unbounded
        self.seconds += time.time() - t0
        self.calls += 1
        self._cache[key] = d
        return d
