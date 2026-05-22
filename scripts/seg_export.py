# SPDX-License-Identifier: MIT
"""Sapiens2 body-part segmentation → color maps (official dome29 palette)."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Tuple

import numpy as np

_log = logging.getLogger("sap2_seg")


@lru_cache(maxsize=1)
def _palette_rgb() -> np.ndarray:
    from sapiens.dense.datasets import DOME_CLASSES_29

    max_id = max(int(k) for k in DOME_CLASSES_29.keys())
    pal = np.zeros((max(max_id + 1, 256), 3), dtype=np.uint8)
    for cid, meta in DOME_CLASSES_29.items():
        pal[int(cid)] = np.array(meta.get("color", [0, 0, 0]), dtype=np.uint8)
    return pal


def labels_to_color_rgb(label_map: np.ndarray) -> np.ndarray:
    """
    Official-style color part map (RGB float 0–1, black background).
    Matches Sapiens2 SEG visualizer palette without plate overlay.
    """
    labels = np.asarray(label_map, dtype=np.int32)
    pal = _palette_rgb()
    if labels.max() >= pal.shape[0]:
        _log.warning("Seg label id %d exceeds palette — clipping", int(labels.max()))
        labels = np.clip(labels, 0, pal.shape[0] - 1)
    color_rgb = pal[labels].astype(np.float32) / 255.0
    return np.ascontiguousarray(color_rgb)


def labels_to_human_alpha(label_map: np.ndarray) -> np.ndarray:
    """Combined human silhouette: 1 where any body part (class > 0), else 0."""
    labels = np.asarray(label_map, dtype=np.int32)
    return (labels > 0).astype(np.float32)


def labels_to_overlay_bgr(image_bgr: np.ndarray, label_map: np.ndarray, *, opacity: float = 0.5) -> np.ndarray:
    """Plate + semi-transparent seg colors (demo / QC)."""
    import cv2

    if image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr must be uint8")
    pal = _palette_rgb()
    labels = np.asarray(label_map, dtype=np.int32)
    color_bgr = pal[:, ::-1][labels]
    overlay = cv2.addWeighted(image_bgr, 1.0 - opacity, color_bgr, opacity, 0)
    return overlay


def split_person_seg_masks(
    label_map: np.ndarray,
    boxes: list,
    *,
    max_subjects: int = 4,
) -> list[np.ndarray]:
    """Per-person binary alpha from seg labels inside each bbox (for channels layout)."""
    h, w = label_map.shape
    masks: list[np.ndarray] = []
    for box in boxes[:max_subjects]:
        mask = np.zeros((h, w), dtype=np.float32)
        sy = slice(max(0, box.y1), min(h, box.y2))
        sx = slice(max(0, box.x1), min(w, box.x2))
        region = label_map[sy, sx]
        mask[sy, sx] = (region > 0).astype(np.float32)
        masks.append(mask)
    return masks


def pack_subject_alpha_channels(masks: list[np.ndarray], *, max_subjects: int = 4) -> Tuple[np.ndarray, tuple[str, ...]]:
    h, w = masks[0].shape
    packed = np.zeros((h, w, max_subjects), dtype=np.float32)
    for i, m in enumerate(masks[:max_subjects]):
        packed[:, :, i] = np.clip(m, 0.0, 1.0)
    names = ("R", "G", "B", "A")[:max_subjects]
    return packed, names
