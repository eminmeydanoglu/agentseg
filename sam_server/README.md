# sam_server

SAM 3 / 3.1 inference node: keeps the models on the GPU and exposes the upstream
`sam3` API over HTTP, unchanged (issue #6). No AgentSeg logic lives here.

## Run

```bash
git submodule update --init third_party/sam3
cd sam_server
uv sync                      # separate venv, torch cu128
hf auth login                # needs access to facebook/sam3 and facebook/sam3.1
uv run sam-server --port 8765 --models sam3.1,image   # add ",sam3" for the SAM 3 video predictor (add_mask)
```

Startup options (CLI flags or `--config file.json` with the same field names, see
`sam_server/config.py`): `models`, `device`, `use_fa3` (keep `false` on RTX 5090),
`max_num_objects`, `multiplex_count`, `compile`, `warm_up`, `session_expiration_sec`,
`default_output_prob_thresh`, `sam31_checkpoint`, `sam3_checkpoint`, `image_*`.

## API

| Endpoint | Upstream call |
|---|---|
| `GET /health`, `GET /info` | status; models, config, sam3 commit, VRAM, sessions/states |
| `POST /video/{sam3.1\|sam3}/request` | `predictor.handle_request(body)` |
| `POST /video/{model}/stream_request` | `predictor.handle_stream_request(body)` → NDJSON, one line per item |
| `POST /image/set_image` | `{image_path\|image_b64}` → `Sam3Processor.set_image` (`{image_paths\|images_b64}` → `set_image_batch`) → `{state_id, ...}` |
| `POST /image/call/{method}` | `{state_id, kwargs}` → `Sam3Processor.<method>(..., state=)` or `Sam3Image.<method>(inference_state=, ...)` (e.g. `predict_inst`) → `{result}` |
| `DELETE /image/states/{state_id}` | drop a server-held image state |

- Bodies/args are passed verbatim (same names and coordinate conventions as upstream).
- Encoding (`sam_server/codec.py`, accepted on input too): bool masks →
  `{"__rle__": [COCO RLE...], "shape"}`; arrays → `{"__ndarray__", "dtype", "shape"}`
  (small 1-D arrays → lists); non-str dict keys → `{"__dict__": [[k, v], ...]}`.
- Image state results return `boxes`, `scores`, `masks` plus extra `presence`
  (presence score the grounding head multiplies into `scores`).
- One GPU worker serializes all requests; a stream holds it until done.
  `cancel_propagation` skips the queue and stops the running stream for that session.

Shared GPU / low VRAM: upstream hardcodes 16-frame batched grounding for sam3.1
(peak > 20 GB). Lower it with
`--video-model-attrs '{"batched_grounding_batch_size": 4, "postprocess_batch_size": 4}'`
and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Smoke (RTX 5090, shared with another 9 GB job; sam3.1 + image, batch 4, 640x480):
image set_image 170 ms, set_text_prompt 180 ms, predict_inst 440 ms;
video propagate 200 frames at 9.7 FPS over HTTP, torch peak 15.9 GiB.

Tests: `uv run pytest` (no GPU). Smoke: `uv run python scripts/smoke.py --frames DIR --image IMG`.

## SAM3 video tracking recipe

From this directory, launch the shared SAM3 video and image heads with the low-VRAM batch settings `agentseg up` uses:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run sam-server --port 8770 --models sam3,image --sam3-checkpoint ../models/sam3/sam3.pt --video-model-attrs '{"batched_grounding_batch_size": 4, "postprocess_batch_size": 4}'
```
