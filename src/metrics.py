"""Evaluation metrics shared by every pipeline.

The formulas follow the project specification:
``MSE = mean((A - B)^2)``, ``PSNR = 10 log10(255^2 / MSE)`` and
``IoU = |A & B| / |A | B|``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def mse(reference: np.ndarray, test: np.ndarray) -> float:
    """Mean squared error between two arrays of identical shape."""
    ref = np.asarray(reference, dtype=np.float64)
    tst = np.asarray(test, dtype=np.float64)
    if ref.shape != tst.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {tst.shape}")
    return float(np.mean((ref - tst) ** 2))


def psnr(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB; returns ``inf`` for identical inputs."""
    error = mse(reference, test)
    if error == 0.0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / error))


def iou_mask(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Intersection over union for two binary masks.

    Two empty masks are a perfect match (1.0); one empty mask scores 0.0.
    """
    pred = np.asarray(prediction).astype(bool)
    truth = np.asarray(ground_truth).astype(bool)
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {truth.shape}")
    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def iou_box(
    prediction: Sequence[float] | None,
    ground_truth: Sequence[float] | None,
) -> float:
    """Intersection over union for two ``x1, y1, x2, y2`` boxes."""
    if prediction is None or ground_truth is None:
        return 0.0
    px1, py1, px2, py2 = (float(value) for value in prediction[:4])
    gx1, gy1, gx2, gy2 = (float(value) for value in ground_truth[:4])
    inter_w = max(0.0, min(px2, gx2) - max(px1, gx1))
    inter_h = max(0.0, min(py2, gy2) - max(py1, gy1))
    intersection = inter_w * inter_h
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    truth_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pred_area + truth_area - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def best_box_iou(
    prediction: Sequence[float] | None,
    ground_truth_boxes: np.ndarray,
) -> tuple[float, int]:
    """Return the highest IoU against the reference boxes and its index."""
    best_iou = 0.0
    best_index = -1
    for index, truth in enumerate(ground_truth_boxes):
        value = iou_box(prediction, truth)
        if value > best_iou:
            best_iou = value
            best_index = index
    return best_iou, best_index


def count_error(predicted_count: int, reference_count: int) -> int:
    """Absolute error between the predicted and reference object counts."""
    return int(abs(int(predicted_count) - int(reference_count)))
