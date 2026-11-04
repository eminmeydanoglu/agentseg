# ml/sam3_video

Keyframe-driven SAM 3 video tracking over `sam_server`. Pure Python + numpy/httpx;
no web framework imports. Masks at every public boundary are `bool` arrays `(H, W)`.

Needs a running `sam_server` with `--models sam3,image` (recipe in `sam_server/README.md`).
One engine holds one sam3 video session (~4 GB VRAM for 526 frames at 800x448); call
`close()` when done.

```python
from ml.sam3_video import Sam3Client, FrameSet, TrackEngine
from ml.sam3_video.health import HealthConfig, DinoEmbedder, assess

client = Sam3Client("http://localhost:8770")          # model="sam3"
frames = FrameSet(ordered_image_paths, root=".agentseg/frames")   # builds <i>.jpg symlinks
with TrackEngine(client, frames) as eng:
    t = eng.add_track("torpedo")
    mask, score = eng.segment(0, box=[x0, y0, x1, y1], pos=[(x, y)], neg=[(x, y)])
    eng.add_keyframe(t, 0, mask, source="agent")
    for ev in eng.propagate():          # or eng.run(on_event=cb, should_stop=fn)
        ...                             # ev.kind: start | segment | frame | done
    health, issues = assess(eng, t, HealthConfig(), embed=DinoEmbedder(frames.paths))
    # refine at an issue's worst frame, then propagate() again (incremental)
    eng.add_keyframe(t, issues[0].worst_frame, new_mask)
    eng.mark_absent(t, range(120, 140))
    eng.remove_keyframe(t, 300)         # rebuilds the track: same as never added
```

## Semantics

- **Keyframes are the source of truth** (`Track.keyframes`: frame -> `Keyframe(mask | None)`);
  `Track.masks` / `Track.origin` are derived and regenerable. The SAM session is a cache.
- **Segments**: every frame is tracked from its nearest positive keyframe; between keyframes
  a < b the split is `(a + b) // 2`. Forward half first, then backward half.
- `add_keyframe` (incremental): runs only the new keyframe's two segments.
- `remove_keyframe`: if the keyframe reached SAM, the track is rebuilt on a fresh obj_id
  (`remove_object` + re-add, or `remove_mode="session"` = `reset_session` for all tracks)
  and fully re-propagated. Verified on torpedo: identical to never adding it.
- `mark_absent`: frames forced empty, never overwritten, excluded from issues.
  `absent_to_sam=True` also feeds the first/last frame of each absent run to SAM as an
  empty conditioning mask (implemented and unit-tested, not evaluated on the GPU yet).
- Several tracks share one session (one obj_id each); upstream propagates all objects in
  every call (`force_tracker_propagation`), the engine records only the requested tracks.
- `propagate(frame_range=(lo, hi))` clips segments; the remainder stays pending.
- Closing the `propagate()` generator cancels server-side propagation; the interrupted
  segment stays pending.
- Windowing for long sequences: `TrackEngine(window=...)` is reserved (raises
  NotImplementedError).

## Health

`assess_track(masks, keyframes, origin, (w, h), config, embed)` -> `FrameHealth` per frame
(`ok | review | lost | keyframe | absent`, reasons, severity, signals);
`group_issues(health, track_id, config)` -> `Issue(start, end, worst_frame, kind, ...)`
with `kind in lost | exit | scope | jump | fragment | drift`.
Defaults are tuned with DINO on; `HealthConfig.geometry_only()` is the no-DINO preset.
DINO (`DinoEmbedder`, vitb16) costs ~50 ms/frame; `dino_every=3` evaluates every 3rd
frame plus every suspicious frame.

Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra dev python -m pytest tests/test_sam3_video_*.py`
(fake sam_server in `tests/fake_sam3.py`).

## Backend video API

`agentseg up` starts the backend and `sam_server --models sam3,image` (at `SAM_SERVER_URL`,
default `http://localhost:8770`). The backend keeps one `TrackEngine` session per open project.

All routes below are under `/api/video` and use the currently open project. Errors use FastAPI's JSON shape `{ "detail": "..." }`.

* `POST /segment` `{ "frame": 0, "box": [x0,y0,x1,y1], "pos": [[x,y]], "neg": [] }` returns `{ "preview_id": "...", "frame": 0, "bbox": [...], "area": 1234, "score": 0.9 }`. A preview is server-side and expires after 30 minutes.
* `POST /tracks` `{ "preview_id": "...", "label": "torpedo" }` returns `{ "track_id": 1, "keyframe": {"frame": 0, "annotation_id": 4, "bbox": [...], "area": 1234}, "label_id": 1 }`.
* `POST /tracks/{id}/propagate` `{ "full": false, "start": 0, "end": 525 }` returns a job snapshot `{ "id": "...", "status": "pending", "track_id": 1, ... }`.
* `GET /jobs/{job_id}` and `GET /jobs/{job_id}/events?after=0&wait=15` provide status and polling events (`started`, `frame`, `issue`, `finished`, `error`).
* `GET /tracks/{id}/issues` returns persisted health issues; `GET /tracks` returns keyframes, absent frames, coverage, issue count, and latest job state.
* `POST /tracks/{id}/keyframes` accepts `{ "frame": 20, "preview_id": "..." }` or `{ "frame": 20, "box": [...] , "pos": [], "neg": [] }`; `DELETE /tracks/{id}/keyframes/{frame}` removes an anchor and regenerates on the next propagate.
* `POST /tracks/{id}/absent` accepts `{ "frames": [120,121] }` or `{ "start": 120, "end": 139 }`.
* `GET /render/frame?frame=20&track_id=1&grid=true` and `GET /render/sheet?track_id=1&every=10` return PNG overlays. A 50 px coordinate grid can be requested for frame renders.

`POST /api/ml/segment` (the web UI's box tool) uses the same sam_server image head.
