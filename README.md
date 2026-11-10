https://github.com/user-attachments/assets/fb0eea35-f1d4-40ef-aed4-acd7993e6c02

[Watch on YouTube](https://youtu.be/ZMFX1F-xzEI)

# AgentSeg

AgentSeg labels objects in sequences of video frames. A Muse Code agent uses the
`agentseg` CLI to prompt SAM 3, track objects across frames, review suspected
tracking errors, and refine tracks with additional keyframes. You can watch and
replay agent runs in the web UI or label individual frames by hand. AgentSeg
exports YOLO segmentation, YOLO detection, and COCO annotations.

## Requirements

- Python 3.12 (managed by [uv](https://docs.astral.sh/uv/)), [Bun](https://bun.sh)
  for the web UI, and `lsof` for stopping local services with `agentseg down`.
- **Local SAM 3:** Linux, a CUDA 12.8 compatible NVIDIA driver and GPU (plan
  for around 16 GB of VRAM for video tracking), and either access to the
  gated [facebook/sam3](https://huggingface.co/facebook/sam3) model or a local checkpoint.
- **Or a remote SAM 3 API:** A reachable server that implements the
  [`sam_server` HTTP API](sam_server/README.md#api) for SAM 3 image segmentation
  and video tracking. AgentSeg itself does not need an NVIDIA GPU or SAM 3 weights.
- **For automated labeling:** Muse Code (`muse`) and a `META_API_KEY`.
  Manual labeling from the web UI or CLI does not require Muse Code.

## Setup

```bash
git clone https://github.com/eminmeydanoglu/agentseg.git
cd agentseg
uv sync --locked --python 3.12
```

### Local SAM 3

```bash
git submodule update --init third_party/sam3
(cd sam_server && uv sync --locked --python 3.12)
```

Put a local checkpoint at `models/sam3/sam3.pt`, or request access to
[facebook/sam3](https://huggingface.co/facebook/sam3) and run
`(cd sam_server && uv run hf auth login)`. The server downloads the weights on
first launch if no local checkpoint is present.

From the repository root:

```bash
uv run agentseg up
```

This starts `sam_server` on port 8770. The first load takes 1–2 minutes.

### Remote SAM 3 API

Skip the local SAM 3 setup. Set `SAM_SERVER_URL` to the base URL of a compatible
server and start AgentSeg without launching a local SAM 3 server:

```bash
export SAM_SERVER_URL=http://sam-host:8770
uv run agentseg up --no-sam
```

Mount the source images and project directory at the same absolute paths on
both machines. AgentSeg sends paths instead of uploading image files. For video
tracking, SAM 3 also needs to read the symlinks AgentSeg creates under the
project's `.agentseg/frames/` directory and resolve them to the source images.

If the backend is already running, use `uv run agentseg down` before starting it
with a different `SAM_SERVER_URL`.

### Open the UI

Both startup commands build the web UI and start the backend. Open
<http://localhost:5172>. Use `uv run agentseg status` to inspect the services
and `uv run agentseg down` to stop the backend. With local SAM 3, add `--all`
to stop its server too. Logs are in `.agentseg/` by default.

### Optional: appearance checks with DINOv3

To check object appearance as well as mask geometry, run
`git submodule update --init third_party/dinov3` and place the
[DINOv3 ViT-B/16 weights](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
at `models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`. Without them,
track health checks use mask geometry alone.

## Label and export

Open an existing AgentSeg project or use a directory of images. For a video
file, extract the frames first. For example, with FFmpeg installed:

```bash
mkdir -p frames
ffmpeg -i input.mp4 frames/%06d.jpg
```

**Label with the agent.** Install Muse Code where the backend runs. Copy
`.env.example` to `.env` if needed, and set `META_API_KEY` there. Then start a run:

```bash
uv run agentseg agent run "label the torpedo" --images /path/to/frames
```

The run streams into the web UI. Transcripts go to `.agentseg/agent_runs/<id>/`, and
`http://localhost:5172/?replay=<id>` plays a recorded run back. Use
`uv run agentseg agent status` to find run IDs and `uv run agentseg agent stop`
to stop the latest run.

**Label from the CLI.** The commands the agent uses also work by hand:
`project open`, `segment`, `track create`, `track propagate`, `track issues`,
and `track refine`. Use `uv run agentseg --help` for the full list and see the
[labeling workflow](.agent/skills/agentseg-label/SKILL.md) for examples.

**Label in the UI.** Create or open a project, pick a label, draw a box (B) and press
S for a SAM mask. In refine mode (R), click to add positive points or right-click to
add negative ones, then press S again.

**Export labels.** Use Export in the web UI, or run
`uv run agentseg export --format yolo-seg` for YOLO polygons. The CLI also
supports `--format yolo-detect` and `--format coco` for bounding boxes. To include
available segmentation polygons in COCO JSON, enable that option in the web UI.

## Development

Run these commands from the repository root. To develop the UI against the
backend, start AgentSeg with the SAM 3 setup above, then run the Vite server:

```bash
(cd frontend && bun install --frozen-lockfile && bun run dev)
```

Open <http://localhost:5173> for hot reload. Vite forwards `/api` requests to
the backend on port 5172.

Run the backend and frontend tests independently:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked --extra dev python -m pytest
(cd frontend && bun install --frozen-lockfile && bun run test)
```

## License

MIT for this repository. SAM 3 ([SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE))
and DINOv3 ([DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md))
come with their own licenses, and you download their weights under those terms.
