"""
Tests for mask utilities
"""

import numpy as np

from core.masks import (
    mask_to_rle,
    rle_to_mask,
    mask_to_bbox,
    mask_area,
    mask_iou,
)


def test_simple_rle_roundtrip():
    """Test simple RLE encoding/decoding roundtrip."""
    # Create a simple binary mask
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 1  # 4x4 square in the middle

    # Encode
    rle = mask_to_rle(mask)

    # Decode
    decoded = rle_to_mask(rle)

    # Check they match
    np.testing.assert_array_equal(mask, decoded)


def test_mask_to_bbox():
    """Test bounding box extraction from mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:50, 30:70] = 1

    bbox = mask_to_bbox(mask)

    assert bbox == [30.0, 20.0, 70.0, 50.0]


def test_mask_to_bbox_empty():
    """Test bbox from empty mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    bbox = mask_to_bbox(mask)
    assert bbox == [0, 0, 0, 0]


def test_mask_area():
    """Test mask area calculation."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:10, 0:10] = 1  # 100 pixels

    assert mask_area(mask) == 100


def test_mask_iou():
    """Test IoU calculation."""
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[0:50, 0:50] = 1  # 2500 pixels

    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[25:75, 25:75] = 1  # 2500 pixels

    # Intersection: [25:50, 25:50] = 625 pixels
    # Union: 2500 + 2500 - 625 = 4375 pixels
    expected_iou = 625 / 4375

    iou = mask_iou(mask1, mask2)
    assert abs(iou - expected_iou) < 0.01


def test_mask_iou_identical():
    """Test IoU of identical masks."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:50, 0:50] = 1

    assert mask_iou(mask, mask) == 1.0


def test_mask_iou_disjoint():
    """Test IoU of non-overlapping masks."""
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[0:10, 0:10] = 1

    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[50:60, 50:60] = 1

    assert mask_iou(mask1, mask2) == 0.0


def test_drop_specks_keeps_the_object_and_its_large_parts():
    from core.masks import drop_specks, mask_to_bbox
    mask = np.zeros((100, 100), bool)
    mask[10:40, 10:60] = True      # the object, 1500 px
    mask[50:70, 10:30] = True      # a second large part (occluded object), 400 px
    mask[95, 95] = True            # a speck far away
    mask[90:92, 5:7] = True        # another speck
    cleaned = drop_specks(mask)
    assert cleaned[10:40, 10:60].all() and cleaned[50:70, 10:30].all()
    assert not cleaned[95, 95] and not cleaned[90:92, 5:7].any()
    assert mask_to_bbox(cleaned) == [10, 10, 60, 70]


def test_drop_specks_never_empties_a_tiny_mask():
    from core.masks import drop_specks
    mask = np.zeros((20, 20), bool)
    mask[2:4, 2:4] = True
    mask[10, 10] = True
    assert drop_specks(mask)[2:4, 2:4].all()
