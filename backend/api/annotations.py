"""
Annotations API endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.api.projects import get_store

router = APIRouter()


class AnnotationResponse(BaseModel):
    id: int
    image_id: int
    label_id: int
    bbox: Optional[list[float]] = None
    polygon: Optional[list[float]] = None
    mask_rle: Optional[dict] = None
    source: str
    confidence: Optional[float] = None
    status: str
    track_id: Optional[int] = None
    is_anchor: bool = False


# Sources that continue an existing object rather than introduce a new one.
# ("propagated" is used by find-instances for *new* objects, so it gets a track.)
TRACKED_SOURCES = {"tracked"}


class CreateAnnotationRequest(BaseModel):
    image_id: int
    label_id: int
    bbox: list[float]  # [x1, y1, x2, y2]
    source: str = "manual"
    status: str = "approved"
    confidence: Optional[float] = None
    mask_rle: Optional[dict] = None
    polygon: Optional[list[float]] = None
    # Object identity. If omitted, a new track is created automatically unless
    # the source is 'tracked' or new_track=False.
    track_id: Optional[int] = None
    new_track: Optional[bool] = None


class UpdateAnnotationRequest(BaseModel):
    label_id: Optional[int] = None
    bbox: Optional[list[float]] = None
    polygon: Optional[list[float]] = None
    mask_rle: Optional[dict] = None
    status: Optional[str] = None
    confidence: Optional[float] = None
    track_id: Optional[int] = None  # reassign to another track (e.g. fix an ID switch)
    clear_track: bool = False  # set track_id = NULL


def annotation_to_response(ann) -> AnnotationResponse:
    """Convert Annotation to AnnotationResponse."""
    return AnnotationResponse(
        id=ann.id,
        image_id=ann.image_id,
        label_id=ann.label_id,
        bbox=ann.bbox_xyxy,
        polygon=ann.polygon_norm,
        mask_rle=ann.mask_rle,
        source=ann.source,
        confidence=ann.confidence,
        status=ann.status,
        track_id=ann.track_id,
        is_anchor=ann.is_anchor,
    )


@router.get("", response_model=list[AnnotationResponse])
async def list_annotations(image_id: int):
    """List annotations for an image."""
    store = get_store()
    annotations = store.list_annotations(image_id)
    return [annotation_to_response(ann) for ann in annotations]


@router.get("/{annotation_id}", response_model=AnnotationResponse)
async def get_annotation(annotation_id: int):
    """Get annotation by ID."""
    store = get_store()
    ann = store.get_annotation_by_id(annotation_id)

    if ann is None:
        raise HTTPException(status_code=404, detail="Annotation not found")

    return annotation_to_response(ann)


@router.post("", response_model=AnnotationResponse)
async def create_annotation(request: CreateAnnotationRequest):
    """Create a new annotation."""
    store = get_store()

    try:
        track_id = request.track_id
        if track_id is not None:
            if store.get_track(track_id) is None:
                raise HTTPException(status_code=404, detail=f"Track {track_id} not found")
        else:
            want_new = (
                request.new_track
                if request.new_track is not None
                else request.source not in TRACKED_SOURCES
            )
            if want_new:
                image = store.get_image_by_id(request.image_id)
                if image is None:
                    raise HTTPException(status_code=404, detail="Image not found")
                track_id = store.create_track(image.project_id, request.label_id).id

        ann = store.create_annotation(
            image_id=request.image_id,
            label_id=request.label_id,
            bbox_xyxy=request.bbox,
            source=request.source,
            status=request.status,
            confidence=request.confidence,
            mask_rle=request.mask_rle,
            polygon_norm=request.polygon,
            track_id=track_id,
        )
        return annotation_to_response(ann)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{annotation_id}", response_model=AnnotationResponse)
async def update_annotation(annotation_id: int, request: UpdateAnnotationRequest):
    """Update an annotation."""
    store = get_store()

    # Build update fields - use field names that store.update_annotation expects
    update_fields = {}
    if request.label_id is not None:
        update_fields["label_id"] = request.label_id
    if request.bbox is not None:
        update_fields["bbox_xyxy"] = request.bbox
    if request.polygon is not None:
        update_fields["polygon_norm"] = request.polygon
    if request.mask_rle is not None:
        update_fields["mask_rle"] = request.mask_rle
    if request.status is not None:
        update_fields["status"] = request.status
    if request.confidence is not None:
        update_fields["confidence"] = request.confidence
    if request.clear_track:
        update_fields["track_id"] = None
    elif request.track_id is not None:
        if store.get_track(request.track_id) is None:
            raise HTTPException(status_code=404, detail="Track not found")
        update_fields["track_id"] = request.track_id

    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    try:
        ann = store.update_annotation(annotation_id, **update_fields)
        return annotation_to_response(ann)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{annotation_id}")
async def delete_annotation(annotation_id: int):
    """Delete an annotation."""
    store = get_store()

    try:
        store.delete_annotation(annotation_id)
        return {"status": "deleted", "id": annotation_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/after-index/{project_id}")
async def delete_annotations_after_index(project_id: int, after_index: int):
    """
    Delete all annotations for images after a given index.

    Args:
        project_id: Project ID
        after_index: Delete annotations for images with order_index > this value.
                     Example: after_index=799 deletes annotations from image 801 onward (0-based).
    """
    store = get_store()

    try:
        deleted_count = store.delete_annotations_after_index(project_id, after_index)
        return {"status": "deleted", "count": deleted_count, "after_index": after_index}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class MissingAnnotationsResponse(BaseModel):
    """Response for images missing annotations for a specific label."""

    image_indices: list[int]
    total_missing: int


@router.get("/missing/{project_id}", response_model=MissingAnnotationsResponse)
async def find_images_missing_annotations(
    project_id: int,
    label_id: Optional[int] = None,
):
    """
    Find images that are missing annotations.

    If label_id is provided, finds images missing that specific label.
    If label_id is None, finds images with no annotations at all.

    Returns image indices sorted by order_index.
    """
    store = get_store()

    # Get all images in project
    images = store.list_images(project_id)
    images_sorted = sorted(images, key=lambda img: img.order_index)

    # Batch query all annotations for the project (single query instead of N+1)
    annotations_by_image = store.list_annotations_for_project(project_id)

    missing_indices = []

    for img in images_sorted:
        annotations = annotations_by_image.get(img.id, [])

        if label_id is not None:
            # Check if this specific label is missing
            has_label = any(ann.label_id == label_id for ann in annotations)
            if not has_label:
                missing_indices.append(img.order_index)
        else:
            # Check if image has no annotations at all
            if len(annotations) == 0:
                missing_indices.append(img.order_index)

    return MissingAnnotationsResponse(
        image_indices=missing_indices,
        total_missing=len(missing_indices),
    )
