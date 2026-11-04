# AgentSeg

AgentSeg labels objects in videos and image sequences with a coding agent. The agent
drives SAM 3 through the `agentseg` CLI: it draws the first mask, tracks the object
through every frame, checks the track for problems (lost object, drift, jumps), fixes
them with extra keyframes, and exports YOLO / COCO labels. The web UI shows each step
live, plays runs back, and has a box → SAM tool so you can label by hand or help the agent.

## Demo

https://github.com/user-attachments/assets/a79194ce-0ec5-490c-a6d3-dc302c9da3ac

## Requirements

- Linux with an NVIDIA GPU (about 16 GB VRAM for SAM 3 video)
- [uv](https://docs.astral.sh/uv/) and [bun](https://bun.sh)
- Hugging Face access to [facebook/sam3](https://huggingface.co/facebook/sam3)
- For the agent: the Muse Code CLI (`muse`) and a `META_API_KEY`

## Setup

```bash
git clone --recurse-submodules https://github.com/talhabw/autoseg.git
cd autoseg
uv sync                          # backend + CLI venv
cd sam_server && uv sync         # SAM 3 server venv (separate torch build)
uv run hf auth login             # SAM 3 weights download on the first start
cd ..
```

Optional model files in `models/`:

- `models/sam3/sam3.pt`: a local SAM 3 checkpoint, used instead of the Hugging Face download.
- `models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`: DINOv3 ViT-B/16
  ([request access](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)).
  It improves the track health checks; without it they use mask geometry only.

## Run

```bash
uv run agentseg up
```

This starts `sam_server` (:8770, the first load takes 1-2 min), builds the web UI and
starts the backend. Open http://localhost:5172. `agentseg status` shows what is
running, `agentseg down` stops the backend and `agentseg down --all` also stops
`sam_server`. Logs are in `.agentseg/`.

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
