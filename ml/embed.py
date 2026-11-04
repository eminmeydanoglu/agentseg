"""DINOv3 object descriptors for track health checks (``ml.sam3_video.health.DinoEmbedder``).

Weights are loaded from ``models/`` through ``torch.hub`` with the code in
``third_party/dinov3``. Without weights the health checks use geometry only.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = _REPO / "models"
_DINOV3_REPO = _REPO / "third_party" / "dinov3"

DINOV3_MODELS = {
    "vitb16": ("dinov3_vitb16", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    "vitl16": ("dinov3_vitl16", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
    "vith16": ("dinov3_vith16plus", "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"),
}


def weights_path(model_name: str = "vitb16") -> Path:
    return MODELS_DIR / DINOV3_MODELS[model_name][1]


class EmbedService:
    """Masked, L2-normalised DINOv3 patch-feature descriptor of an object."""

    patch_size = 16
    img_size = 518  # 32 x 32 patches of 16 px

    def __init__(self, device: str = "cuda", model_name: str = "vitb16"):
        if model_name not in DINOV3_MODELS:
            raise ValueError(f"Unknown DINOv3 model {model_name!r}; choose from {list(DINOV3_MODELS)}")
        self.device = device
        self.model_name = model_name
        self.model = None
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((self.img_size, self.img_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def load_model(self) -> None:
        hub_name, _ = DINOV3_MODELS[self.model_name]
        weights = weights_path(self.model_name)
        if not weights.exists():
            raise FileNotFoundError(f"DINOv3 weights not found: {weights}")
        logger.info("Loading DINOv3 %s from %s", self.model_name, weights)
        self.model = torch.hub.load(str(_DINOV3_REPO), hub_name, source="local", weights=str(weights))
        self.model = self.model.to(self.device).eval()

    def get_object_descriptor(self, image: Union[np.ndarray, Image.Image, str], bbox_xyxy: list[float],
                              mask: Optional[np.ndarray] = None, use_cache: bool = False) -> np.ndarray:
        """Average the patch features inside the box (and mask, when given); returns (embed_dim,)."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        img_w, img_h = image.size

        with torch.inference_mode():
            x = self.transform(image).unsqueeze(0).to(self.device)
            features = self.model.forward_features(x)["x_norm_patchtokens"]  # (1, patches, dim)

        n = self.img_size // self.patch_size
        feat_map = features.reshape(1, n, n, -1).permute(0, 3, 1, 2)  # (1, dim, n, n)

        x1, y1, x2, y2 = bbox_xyxy
        fx1 = max(0, min(int(x1 / img_w * n), n - 1))
        fy1 = max(0, min(int(y1 / img_h * n), n - 1))
        fx2 = max(fx1 + 1, min(int(np.ceil(x2 / img_w * n)), n))
        fy2 = max(fy1 + 1, min(int(np.ceil(y2 / img_h * n)), n))
        region = feat_map[:, :, fy1:fy2, fx1:fx2]

        descriptor = region.mean(dim=(2, 3))
        if mask is not None and mask.sum() > 0:
            m = torch.from_numpy(mask).float()[None, None].to(self.device)
            m = F.interpolate(m, size=(n, n), mode="nearest")[:, :, fy1:fy2, fx1:fx2]
            if m.sum() > 0:  # thin objects can vanish at patch resolution; keep the box average then
                descriptor = (region * m).sum(dim=(2, 3)) / m.sum()

        return F.normalize(descriptor, dim=-1).squeeze(0).cpu().numpy()
