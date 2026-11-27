"""Pipelines for plate localization.

Week 10 implements the two required baselines.  Both threshold the raw
grayscale image, then reduce the binary mask to its largest connected
component and report that component's bounding box as the single plate
candidate.  Pipelines A, B and C are added in Weeks 11-12.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BaselineResult:
    """Binary mask plus the selected plate candidate."""

    mask: np.ndarray
    bbox: tuple[int, int, int, int] | None
    n_components: int


def largest_component_bbox(
    mask: np.ndarray,
    min_area: int = 100,
    connectivity: int = 8,
) -> tuple[tuple[int, int, int, int] | None, int]:
    """Return the bounding box of the largest foreground component.

    Args:
        mask: binary array, any nonzero value is foreground.
        min_area: components smaller than this many pixels are ignored.
        connectivity: 4 or 8.

    Returns:
        ``((x1, y1, x2, y2), n_components)``; the box is ``None`` when no
        component reaches ``min_area``.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity)
    best_box = None
    best_area = 0
    kept = 0
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < min_area:
            continue
        kept += 1
        if area > best_area:
            best_area = area
            best_box = (x, y, x + width, y + height)
    return best_box, kept


def trivial_threshold(
    gray: np.ndarray,
    threshold: int = 127,
    min_area: int = 100,
) -> BaselineResult:
    """Trivial baseline: fixed global threshold on the raw grayscale image."""
    if gray.ndim != 2:
        raise ValueError("expected a 2-D grayscale image")
    mask = (gray > threshold).astype(np.uint8)
    bbox, n_components = largest_component_bbox(mask, min_area=min_area)
    return BaselineResult(mask=mask, bbox=bbox, n_components=n_components)


def simple_otsu_opening(
    gray: np.ndarray,
    kernel_size: int = 3,
    min_area: int = 100,
) -> BaselineResult:
    """Simple baseline: Otsu threshold followed by one square opening."""
    if gray.ndim != 2:
        raise ValueError("expected a 2-D grayscale image")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    mask = (opened > 0).astype(np.uint8)
    bbox, n_components = largest_component_bbox(mask, min_area=min_area)
    return BaselineResult(mask=mask, bbox=bbox, n_components=n_components)
