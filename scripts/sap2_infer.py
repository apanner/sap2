# SPDX-License-Identifier: MIT
"""Sapiens2 matting + normals — native 1024×768; optional person-crop infer."""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_log = logging.getLogger("sap2_infer")

MODEL_NATIVE_H = 1024
MODEL_NATIVE_W = 768


def patch_pipeline_size(model: Any, height: int, width: int) -> None:
    if int(height) != MODEL_NATIVE_H or int(width) != MODEL_NATIVE_W:
        _log.warning(
            "Ignoring pipeline resize %d×%d — model capacity is %d×%d only",
            height,
            width,
            MODEL_NATIVE_H,
            MODEL_NATIVE_W,
        )
        return
    for t in getattr(model.pipeline, "transforms", []):
        if hasattr(t, "height") and hasattr(t, "width"):
            t.height = int(height)
            t.width = int(width)


def _init_task_model(
    dense_root: Path,
    config_rel: str,
    ckpt: Path,
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
        model.float()
        patch_pipeline_size(model, MODEL_NATIVE_H, MODEL_NATIVE_W)
        return model
    finally:
        os.chdir(prev)


def _run_matting_frame(model: Any, image_bgr: np.ndarray, device: str) -> np.ndarray:
    data = model.pipeline(dict(img=image_bgr))
    data = model.data_preprocessor(data)
    with torch.inference_mode():
        out = model(data["inputs"])
    out = F.interpolate(
        out.float(),
        size=(image_bgr.shape[0], image_bgr.shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    alpha = out.squeeze(0)[3].float().cpu().numpy().clip(0.0, 1.0)
    return alpha


def _run_normal_frame(model: Any, image_bgr: np.ndarray, device: str) -> np.ndarray:
    data = model.pipeline(dict(img=image_bgr))
    data = model.data_preprocessor(data)
    inputs = data["inputs"]
    samples = data.get("data_samples")
    with torch.inference_mode():
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
        normal.float(),
        size=(image_bgr.shape[0], image_bgr.shape[1]),
        mode="bilinear",
        align_corners=False,
    )
    out = normal.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return np.ascontiguousarray(out.astype(np.float32, copy=True))


class Sap2ShotProcessor:
    """Full-res plates → person crop (optional) → 1024×768 infer → full-res EXR."""

    def __init__(
        self,
        dense_root: Path,
        ckpt_root: Path,
        model_key: str = "1b",
        device: str = "cuda:0",
        inference_long_edge: int = 0,
        max_megapixels: float = 0.0,
        *,
        use_person_crop: bool = True,
        person_crop_pad: float = 0.18,
        person_crop_confidence: float = 0.4,
        person_crop_smooth: float = 0.72,
        person_crop_multi: bool = True,
    ):
        self.dense_root = Path(dense_root)
        self.ckpt_root = Path(ckpt_root)
        self.model_key = model_key.replace("sapiens2_", "")
        self.device = device
        self.use_person_crop = bool(use_person_crop)
        if int(inference_long_edge) > 0 or float(max_megapixels or 0) > 0:
            _log.info("infer size overrides ignored — model native %d×%d", MODEL_NATIVE_H, MODEL_NATIVE_W)
        self._matting_model = None
        self._normal_model = None
        self._crop_tracker = None

        from sap2_models import MODEL_CONFIGS

        self._cfg = MODEL_CONFIGS.get(self.model_key, MODEL_CONFIGS["1b"])
        if self.use_person_crop:
            from person_detect import PersonCropTracker

            self._crop_tracker = PersonCropTracker(
                pad_ratio=person_crop_pad,
                min_confidence=person_crop_confidence,
                smooth_alpha=person_crop_smooth,
                merge_multi=person_crop_multi,
            )
            _log.info(
                "Person-crop infer ON (pad=%.0f%%, conf=%.2f, multi=%s)",
                person_crop_pad * 100,
                person_crop_confidence,
                person_crop_multi,
            )

    def _log_infer_mode(self, label: str, plate_h: int, plate_w: int) -> None:
        mode = "person crop → 1024×768" if self.use_person_crop else "full frame → 1024×768"
        _log.info("%s: plate %d×%d — %s → EXR full res", label, plate_h, plate_w, mode)

    def _ensure_matting(self, plate_h: int, plate_w: int) -> None:
        if self._matting_model is not None:
            return
        self._log_infer_mode("Matting", plate_h, plate_w)
        ckpt = self.ckpt_root / self._cfg["matting_ckpt"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._matting_model = _init_task_model(
            self.dense_root,
            self._cfg["matting_config"],
            ckpt,
            self.device,
        )

    def _ensure_normal(self, plate_h: int, plate_w: int) -> None:
        if self._normal_model is not None:
            return
        self._log_infer_mode("Normal", plate_h, plate_w)
        ckpt = self.ckpt_root / self._cfg["normal_ckpt"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._normal_model = _init_task_model(
            self.dense_root,
            self._cfg["normal_config"],
            ckpt,
            self.device,
        )

    def _infer_matting_crops(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        assert self._matting_model is not None
        return [_run_matting_frame(self._matting_model, c, self.device) for c in crops]

    def _infer_normal_crops(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        assert self._normal_model is not None
        return [_run_normal_frame(self._normal_model, c, self.device) for c in crops]

    def process_frame_matting(self, jpeg_path: Path) -> np.ndarray:
        img = cv2.imread(str(jpeg_path))
        if img is None:
            raise FileNotFoundError(jpeg_path)
        h, w = img.shape[:2]
        self._ensure_matting(h, w)

        if not self.use_person_crop or self._crop_tracker is None:
            return _run_matting_frame(self._matting_model, img, self.device)

        crops, boxes, _full = self._crop_tracker.crop_regions(img)
        mattes = self._infer_matting_crops(crops)
        if len(mattes) == 1:
            return self._crop_tracker.paste_matte(h, w, mattes[0], boxes[0])
        return self._crop_tracker.merge_mattes(h, w, mattes, boxes)

    def process_frame_normal(self, jpeg_path: Path) -> np.ndarray:
        img = cv2.imread(str(jpeg_path))
        if img is None:
            raise FileNotFoundError(jpeg_path)
        h, w = img.shape[:2]
        self._ensure_normal(h, w)

        if not self.use_person_crop or self._crop_tracker is None:
            return _run_normal_frame(self._normal_model, img, self.device)

        crops, boxes, _full = self._crop_tracker.crop_regions(img)
        normals = self._infer_normal_crops(crops)
        if len(normals) == 1:
            return self._crop_tracker.paste_normal(h, w, normals[0], boxes[0])
        return self._crop_tracker.merge_normals(h, w, normals, boxes)

    def unload(self) -> None:
        for attr in ("_matting_model", "_normal_model"):
            m = getattr(self, attr, None)
            if m is not None:
                del m
            setattr(self, attr, None)
        self._crop_tracker = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
