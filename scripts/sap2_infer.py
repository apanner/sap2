# SPDX-License-Identifier: MIT
"""Sapiens2 matting + normals — high-res inference with plate-size upscale."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_log = logging.getLogger("sap2_infer")

# Training default H×W (Sapiens2)
_BASE_H = 1024
_BASE_W = 768


def vram_auto_long_edge() -> int:
    """Pick inference long edge from GPU VRAM (target ~80GB → 4096)."""
    if not torch.cuda.is_available():
        return _BASE_H
    gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if gb >= 70:
        return 4096
    if gb >= 40:
        return 3072
    if gb >= 22:
        return 2048
    if gb >= 14:
        return 1536
    return 1280


def infer_size_hw(
    plate_h: int,
    plate_w: int,
    *,
    long_edge: int = 0,
    max_h: int = 4096,
    max_w: int = 3072,
) -> tuple[int, int]:
    """Return (height, width) for pipeline, multiples of 16."""
    le = int(long_edge) if int(long_edge) > 0 else vram_auto_long_edge()
    le = min(le, max_h, max_w)
    scale = le / max(plate_h, plate_w, 1)
    h = max(16, int(round(plate_h * scale / 16) * 16))
    w = max(16, int(round(plate_w * scale / 16) * 16))
    h = min(h, max_h)
    w = min(w, max_w)
    return h, w


def patch_pipeline_size(model: Any, height: int, width: int) -> None:
    for t in getattr(model.pipeline, "transforms", []):
        if hasattr(t, "height") and hasattr(t, "width"):
            t.height = int(height)
            t.width = int(width)


def _init_task_model(
    dense_root: Path,
    config_rel: str,
    ckpt: Path,
    infer_h: int,
    infer_w: int,
    device: str,
) -> Any:
    import os

    sap2_root = dense_root.parent.parent
    if str(sap2_root) not in sys.path:
        sys.path.insert(0, str(sap2_root))
    prev = os.getcwd()
    try:
        os.chdir(dense_root)
        from sapiens.dense.models import init_model

        model = init_model(str(config_rel), str(ckpt), device=device)
        patch_pipeline_size(model, infer_h, infer_w)
        return model
    finally:
        os.chdir(prev)


def _run_matting_frame(model: Any, image_bgr: np.ndarray) -> np.ndarray:
    data = model.pipeline(dict(img=image_bgr))
    data = model.data_preprocessor(data)
    with torch.no_grad():
        out = model(data["inputs"])
    out = F.interpolate(
        out,
        size=(image_bgr.shape[0], image_bgr.shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    alpha = out.squeeze(0)[3].float().cpu().numpy().clip(0.0, 1.0)
    return alpha


def _run_normal_frame(model: Any, image_bgr: np.ndarray) -> np.ndarray:
    data = model.pipeline(dict(img=image_bgr))
    data = model.data_preprocessor(data)
    inputs = data["inputs"]
    samples = data.get("data_samples")
    with torch.no_grad():
        normal = model(inputs)
        normal = normal / torch.norm(normal, dim=1, keepdim=True).clamp(min=1e-8)
    if samples and "meta" in samples and "padding_size" in samples["meta"]:
        pad_left, pad_right, pad_top, pad_bottom = samples["meta"]["padding_size"]
        normal = normal[
            :,
            :,
            pad_top : inputs.shape[2] - pad_bottom,
            pad_left : inputs.shape[3] - pad_right,
        ]
    normal = F.interpolate(
        normal,
        size=(image_bgr.shape[0], image_bgr.shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    out = normal.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return np.ascontiguousarray(out.astype(np.float32, copy=True))


class Sap2ShotProcessor:
    """One-shot processor: JPEG plates → matte/normal EXR at full plate resolution."""

    def __init__(
        self,
        dense_root: Path,
        ckpt_root: Path,
        model_key: str = "1b",
        device: str = "cuda:0",
        inference_long_edge: int = 0,
    ):
        self.dense_root = Path(dense_root)
        self.ckpt_root = Path(ckpt_root)
        self.model_key = model_key.replace("sapiens2_", "")
        self.device = device
        self.inference_long_edge = int(inference_long_edge)
        self._matting_model = None
        self._normal_model = None
        self._infer_hw: tuple[int, int] | None = None

        from sap2_models import MODEL_CONFIGS

        self._cfg = MODEL_CONFIGS.get(self.model_key, MODEL_CONFIGS["1b"])

    def _ensure_matting(self, plate_h: int, plate_w: int) -> None:
        if self._matting_model is not None:
            return
        ih, iw = infer_size_hw(plate_h, plate_w, long_edge=self.inference_long_edge)
        self._infer_hw = (ih, iw)
        ckpt = self.ckpt_root / self._cfg["matting_ckpt"]
        _log.info("Matting infer size H×W = %d×%d (long_edge=%s)", ih, iw, self.inference_long_edge or "auto")
        self._matting_model = _init_task_model(
            self.dense_root,
            self._cfg["matting_config"],
            ckpt,
            ih,
            iw,
            self.device,
        )

    def _ensure_normal(self, plate_h: int, plate_w: int) -> None:
        if self._normal_model is not None:
            return
        ih, iw = infer_size_hw(plate_h, plate_w, long_edge=self.inference_long_edge)
        self._infer_hw = (ih, iw)
        ckpt = self.ckpt_root / self._cfg["normal_ckpt"]
        _log.info("Normal infer size H×W = %d×%d", ih, iw)
        self._normal_model = _init_task_model(
            self.dense_root,
            self._cfg["normal_config"],
            ckpt,
            ih,
            iw,
            self.device,
        )

    def process_frame_matting(self, jpeg_path: Path) -> np.ndarray:
        img = cv2.imread(str(jpeg_path))
        if img is None:
            raise FileNotFoundError(jpeg_path)
        self._ensure_matting(img.shape[0], img.shape[1])
        return _run_matting_frame(self._matting_model, img)

    def process_frame_normal(self, jpeg_path: Path) -> np.ndarray:
        img = cv2.imread(str(jpeg_path))
        if img is None:
            raise FileNotFoundError(jpeg_path)
        self._ensure_normal(img.shape[0], img.shape[1])
        return _run_normal_frame(self._normal_model, img)

    def unload(self) -> None:
        self._matting_model = None
        self._normal_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
