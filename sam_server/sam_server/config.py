"""Startup config for build-time-only options (JSON file and/or CLI flags)."""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Optional

VIDEO_MODELS = ("sam3.1", "sam3")


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    device: str = "cuda"
    # Which models to load: any of "sam3.1" (video, multiplex), "sam3" (video), "image"
    models: list[str] = field(default_factory=lambda: ["sam3.1", "image"])
    # Video (passed to upstream builders)
    use_fa3: bool = False  # must stay False on Blackwell (RTX 5090): FA3 is Hopper-only
    use_rope_real: bool = True
    max_num_objects: int = 16
    multiplex_count: int = 16
    compile: bool = False
    warm_up: bool = False
    session_expiration_sec: int = 1200
    default_output_prob_thresh: float = 0.5
    async_loading_frames: bool = True
    sam31_checkpoint: Optional[str] = None  # None -> download facebook/sam3.1
    sam3_checkpoint: Optional[str] = None  # None -> download facebook/sam3 (image + sam3 video)
    bpe_path: Optional[str] = None
    # Post-build attribute overrides on predictor.model, e.g. upstream hardcodes
    # {"batched_grounding_batch_size": 16, "postprocess_batch_size": 16} for sam3.1;
    # lower them to cut peak VRAM on a shared GPU. CLI: --video-model-attrs '{"...": 4}'
    video_model_attrs: dict = field(default_factory=dict)
    # Image
    image_inst_interactivity: bool = True  # needed for predict_inst / predict_inst_batch
    image_confidence_threshold: float = 0.5
    image_resolution: int = 1008

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_argv(cls, argv=None) -> "ServerConfig":
        p = argparse.ArgumentParser("sam-server")
        p.add_argument("--config", help="JSON file with ServerConfig fields")
        for f in dataclasses.fields(cls):
            flag = "--" + f.name.replace("_", "-")
            if f.type in ("bool", bool):
                p.add_argument(flag, dest=f.name, default=None,
                               type=lambda s: s.lower() in ("1", "true", "yes", "on"))
            elif f.name == "models":
                p.add_argument(flag, dest=f.name, default=None,
                               type=lambda s: [m.strip() for m in s.split(",") if m.strip()])
            elif f.name == "video_model_attrs":
                p.add_argument(flag, dest=f.name, default=None, type=json.loads)
            elif f.type in ("int", int):
                p.add_argument(flag, dest=f.name, default=None, type=int)
            elif f.type in ("float", float):
                p.add_argument(flag, dest=f.name, default=None, type=float)
            else:
                p.add_argument(flag, dest=f.name, default=None)
        ns = vars(p.parse_args(argv))
        values = {}
        if ns.pop("config"):
            with open(p.parse_args(argv).config) as fh:
                values.update(json.load(fh))
        values.update({k: v for k, v in ns.items() if v is not None})
        cfg = cls(**values)
        bad = [m for m in cfg.models if m not in VIDEO_MODELS + ("image",)]
        if bad:
            raise SystemExit(f"unknown models {bad}; choose from {VIDEO_MODELS + ('image',)}")
        return cfg
