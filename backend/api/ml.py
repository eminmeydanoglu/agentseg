"""Box (+ click) segmentation for the web UI, served by sam_server's SAM 3 image head."""

import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.api.projects import get_store
from core.masks import drop_specks, mask_to_bbox, mask_to_rle
from core.polygons import mask_to_yolo_polygon
from ml.sam3_video import Sam3Client

router = APIRouter()


class SegmentRequest(BaseModel):
    image_id: int
    bbox: list[float]  # [x1, y1, x2, y2] in pixels
    pos_points: Optional[list[list[float]]] = None
    neg_points: Optional[list[list[float]]] = None


class SegmentResponse(BaseModel):
    mask_rle: dict
    polygon: list[float]
    score: float
    bbox: list[float]


@router.post("/segment", response_model=SegmentResponse)
def segment(request: SegmentRequest):
    """Mask from a box and optional positive/negative clicks."""
    image = get_store().get_image_by_id(request.image_id)
    if image is None:
        raise HTTPException(404, f"Image {request.image_id} not found")
    pos, neg = request.pos_points or [], request.neg_points or []
    client = Sam3Client(os.getenv("SAM_SERVER_URL", "http://localhost:8770"))
    try:
        mask, score = client.segment(image.path, box=request.bbox, points=pos + neg,
                                     point_labels=[1] * len(pos) + [0] * len(neg))
    except Exception as exc:
        raise HTTPException(502, f"sam_server segmentation failed: {exc}")
    finally:
        client.close()
    mask = drop_specks(mask)
    if not mask.any():
        raise HTTPException(422, "SAM returned an empty mask")
    return SegmentResponse(mask_rle=mask_to_rle(mask),
                           polygon=mask_to_yolo_polygon(mask, image.width, image.height),
                           score=score, bbox=[float(v) for v in mask_to_bbox(mask)])
