https://github.com/user-attachments/assets/fb0eea35-f1d4-40ef-aed4-acd7993e6c02

[Watch on YouTube](https://youtu.be/ZMFX1F-xzEI)

# AgentSeg

AgentSeg labels objects in videos and image sequences with a coding agent. The agent
drives SAM 3 through the `agentseg` CLI: it draws the first mask, tracks the object
through every frame, checks the track for problems (lost object, drift, jumps), fixes
them with extra keyframes, and exports YOLO / COCO labels. The web UI shows each step
live, plays runs back, and has a box → SAM tool so you can label by hand or help the agent.

## Requirements

- [uv](https://docs.astral.sh/uv/) and [bun](https://bun.sh)
- **SAM 3**, either locally or over HTTP:
  - **Local:** Linux with an NVIDIA GPU (about 16 GB VRAM for video tracking) and
    either Hugging Face access to [facebook/sam3](https://huggingface.co/facebook/sam3)
    or a local SAM 3 checkpoint.
  - **Remote API:** A reachable endpoint that implements the [`sam_server` HTTP API](sam_server/README.md)
    for SAM 3 image segmentation and video tracking. The machine running AgentSeg
    does not need an NVIDIA GPU or SAM 3 weights.
- For the agent: the Muse Code CLI (`muse`) and a `META_API_KEY`

## Setup

```bash
git clone https://github.com/eminmeydanoglu/agentseg.git
cd agentseg
uv sync                          # backend + CLI venv
```

For local SAM 3, initialize its submodule and set up its separate environment:

```bash
git submodule update --init third_party/sam3
(cd sam_server && uv sync && uv run hf auth login)
```

Skip Hugging Face login if you already have the local checkpoint below. If you
use a remote SAM 3 API, skip this server setup entirely.

Optional model files in `models/`:

- `models/sam3/sam3.pt`: a local SAM 3 checkpoint, used instead of the Hugging Face download.
- `models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`: DINOv3 ViT-B/16
  ([request access](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)).
  It improves the track health checks; without it they use mask geometry only.
  To use it, also run `git submodule update --init third_party/dinov3`.

## Run

**With local SAM 3:**

```bash
uv run agentseg up
```

This starts `sam_server` on port 8770. The first load takes 1–2 minutes.

**With a remote SAM 3 API:** Set `SAM_SERVER_URL` to the server's base URL and
skip launching a local SAM 3 server:

```bash
SAM_SERVER_URL=http://sam-host:8770 uv run agentseg up --no-sam
```

AgentSeg sends image and frame directory paths to the API. Make the files
available on both machines at the same paths (for example, via a shared mount).

Both modes build the web UI and start the backend. Open http://localhost:5172.
`agentseg status` shows what is running, `agentseg down` stops the backend, and
`agentseg down --all` also stops a locally started `sam_server`. Logs are in `.agentseg/`.

**Label with the agent.** Give the agent its key, then start a run:

```bash
cp .env.example .env             # put META_API_KEY=... in it
uv run agentseg agent run "label the torpedo" --images /path/to/frames
```

The run streams into the web UI. Transcripts go to `.agentseg/agent_runs/<id>/`, and
`http://localhost:5172/?replay=<id>` plays a run back. `agentseg agent status` and
`agentseg agent stop` manage runs.

**Label from the CLI.** The commands the agent uses also work by hand:
`agentseg project open`, `segment`, `track create | propagate | issues | refine`,
`view`, `sheet`, `export`. Run `agentseg --help` for the full list. The agent's
workflow is in [.agent/skills/agentseg-label/SKILL.md](.agent/skills/agentseg-label/SKILL.md).

**Label in the UI.** Create or open a project, pick a label, draw a box (B) and press
S for a SAM mask. In refine mode (R), click to add positive points or right-click to
add negative ones, then press S again.

## Development

```bash
cd frontend && bun run dev       # hot reload on :5173, proxies /api to :5172
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra dev python -m pytest
cd frontend && bun run test
```

## License

MIT for this repository. SAM 3 ([SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE))
and DINOv3 ([DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md))
come with their own licenses, and you download their weights under those terms.
