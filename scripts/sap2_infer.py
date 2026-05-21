# SPDX-License-Identifier: MIT
"""
Sapiens2 infer — official image path ([facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2)):

  Full-res BGR plate → pipeline resize 1024×768 → model → upsample to full plate → EXR

4K plates: ``image_feed_mode=auto`` uses person-crop so the model budget targets humans.
"""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path
from typing import Any, List, Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_log = logging.getLogger("sap2_infer")

# All standard dense configs: image_size = (1024, 768) H×W
MODEL_NATIVE_H = 1024
MODEL_NATIVE_W = 768

ImageFeedMode = Literal["auto", "full_res", "person_crop"]

# Above this megapixel count, auto mode prefers person crop (4K ≈ 8.3 MP)
_AUTO_CROP_MIN_MP = 1.05
_AUTO_CROP_MIN_LONG_EDGE = 1920


def resolve_use_person_crop(
    plate_h: int,
    plate_w: int,
    *,
    image_feed_mode: ImageFeedMode = "auto",
    use_person_crop_flag: bool = True,
) -> bool:
    if not use_person_crop_flag:
        return False
    if image_feed_mode == "full_res":
        return False
    if image_feed_mode == "person_crop":
        return True
    mp = plate_h * plate_w / 1_000_000.0
    long_edge = max(plate_h, plate_w)
    return mp >= _AUTO_CROP_MIN_MP or long_edge >= _AUTO_CROP_MIN_LONG_EDGE


def patch_pipeline_native(model: Any) -> None:
    """Ensure test pipeline stays at model-native 1024×768 (do not upscale)."""
    for t in getattr(model.pipeline, "transforms", []):
        if hasattr(t, "height") and hasattr(t, "width"):
            t.height = MODEL_NATIVE_H
            t.width = MODEL_NATIVE_W
        if type(t).__name__ == "MattingResize":
            t.keep_ratio = False  # official matting test: stretch to 1024×768


def _init_task_model(
    dense_root: Path,
    config_rel: str,
    ckpt: Path,
    device: str,
    *,
    is_matting: bool,
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
        model.eval()
        patch_pipeline_native(model)
        return model
    finally:
        os.chdir(prev)


def _cuda_sync_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _run_matting_on_bgr(model: Any, image_bgr: np.ndarray) -> np.ndarray:
    """Official vis_matting path: any resolution in → full resolution alpha out."""
    out_h, out_w = image_bgr.shape[:2]
    data = model.pipeline(dict(img=image_bgr))
    data = model.data_preprocessor(data)
    inputs = data["inputs"]
    with torch.no_grad():
        out = model(inputs)
    out = F.interpolate(
        out.float(),
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )
    alpha = out.squeeze(0)[3].float().cpu().numpy().clip(0.0, 1.0)
    del data, inputs, out
    return np.ascontiguousarray(alpha, dtype=np.float32)


def _run_normal_on_bgr(model: Any, image_bgr: np.ndarray) -> np.ndarray:
    out_h, out_w = image_bgr.shape[:2]
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
        normal.float(),
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )
    out = normal.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    del data, inputs, normal
    return np.ascontiguousarray(out.astype(np.float32))


class Sap2ShotProcessor:
    """
    4K (or any size) JPEG/PNG plates:
      - full_res: entire frame → Sapiens2 pipeline → full-res EXR
      - person_crop / auto: detect human → crop → pipeline → paste → full-res EXR
    """

    def __init__(
        self,
        dense_root: Path,
        ckpt_root: Path,
        model_key: str = "1b",
        device: str = "cuda:0",
        *,
        image_feed_mode: ImageFeedMode = "auto",
        use_person_crop: bool = True,
        person_crop_pad: float = 0.18,
        person_crop_confidence: float = 0.35,
        person_crop_smooth: float = 0.72,
        person_crop_multi: bool = True,
    ):
        self.dense_root = Path(dense_root)
        self.ckpt_root = Path(ckpt_root)
        self.model_key = model_key.replace("sapiens2_", "")
        self.device = device
        self.image_feed_mode: ImageFeedMode = image_feed_mode  # type: ignore[assignment]
        self.use_person_crop_flag = bool(use_person_crop)
        self._person_crop_pad = person_crop_pad
        self._person_crop_confidence = person_crop_confidence
        self._person_crop_smooth = person_crop_smooth
        self._person_crop_multi = person_crop_multi
        self._matting_model = None
        self._normal_model = None
        self._crop_tracker = None
        self._plate_logged = False

        from sap2_models import MODEL_CONFIGS

        self._cfg = MODEL_CONFIGS.get(self.model_key, MODEL_CONFIGS["1b"])

    def _crop_tracker_lazy(self) -> Any:
        if self._crop_tracker is None:
            from person_detect import PersonCropTracker

            self._crop_tracker = PersonCropTracker(
                pad_ratio=self._person_crop_pad,
                min_confidence=self._person_crop_confidence,
                smooth_alpha=self._person_crop_smooth,
                merge_multi=self._person_crop_multi,
            )
        return self._crop_tracker

    def _log_plate_once(self, h: int, w: int, use_crop: bool) -> None:
        if self._plate_logged:
            return
        self._plate_logged = True
        mp = h * w / 1e6
        mode = "person crop" if use_crop else "full frame (official)"
        _log.info(
            "4K/plate feed: %d×%d (%.2f MP) — %s → model %d×%d → EXR %d×%d",
            w,
            h,
            mp,
            mode,
            MODEL_NATIVE_W,
            MODEL_NATIVE_H,
            w,
            h,
        )

    def _read_bgr(self, jpeg_path: Path) -> np.ndarray:
        img = cv2.imread(str(jpeg_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(jpeg_path)
        return img

    def _ensure_matting(self) -> None:
        if self._matting_model is not None:
            return
        ckpt = self.ckpt_root / self._cfg["matting_ckpt"]
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing matting ckpt: {ckpt}")
        _cuda_sync_cleanup()
        self._matting_model = _init_task_model(
            self.dense_root,
            self._cfg["matting_config"],
            ckpt,
            self.device,
            is_matting=True,
        )
        _log.info("Matting model ready (native %d×%d internal)", MODEL_NATIVE_H, MODEL_NATIVE_W)

    def _ensure_normal(self) -> None:
        if self._normal_model is not None:
            return
        ckpt = self.ckpt_root / self._cfg["normal_ckpt"]
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing normal ckpt: {ckpt}")
        _cuda_sync_cleanup()
        self._normal_model = _init_task_model(
            self.dense_root,
            self._cfg["normal_config"],
            ckpt,
            self.device,
            is_matting=False,
        )
        _log.info("Normal model ready (native %d×%d + letterbox pad)", MODEL_NATIVE_H, MODEL_NATIVE_W)

    def _process_matting_bgr(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        use_crop = resolve_use_person_crop(
            h, w,
            image_feed_mode=self.image_feed_mode,
            use_person_crop_flag=self.use_person_crop_flag,
        )
        self._log_plate_once(h, w, use_crop)
        self._ensure_matting()
        assert self._matting_model is not None

        if not use_crop:
            return _run_matting_on_bgr(self._matting_model, img)

        tracker = self._crop_tracker_lazy()
        crops, boxes, _ = tracker.crop_regions(img)
        mattes = [_run_matting_on_bgr(self._matting_model, c) for c in crops]
        if len(mattes) == 1:
            return tracker.paste_matte(h, w, mattes[0], boxes[0])
        return tracker.merge_mattes(h, w, mattes, boxes)

    def _process_normal_bgr(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        use_crop = resolve_use_person_crop(
            h, w,
            image_feed_mode=self.image_feed_mode,
            use_person_crop_flag=self.use_person_crop_flag,
        )
        self._log_plate_once(h, w, use_crop)
        self._ensure_normal()
        assert self._normal_model is not None

        if not use_crop:
            return _run_normal_on_bgr(self._normal_model, img)

        tracker = self._crop_tracker_lazy()
        crops, boxes, _ = tracker.crop_regions(img)
        normals = [_run_normal_on_bgr(self._normal_model, c) for c in crops]
        if len(normals) == 1:
            return tracker.paste_normal(h, w, normals[0], boxes[0])
        return tracker.merge_normals(h, w, normals, boxes)

    def process_frame_matting(self, jpeg_path: Path) -> np.ndarray:
        img = self._read_bgr(jpeg_path)
        try:
            return self._process_matting_bgr(img)
        finally:
            _cuda_sync_cleanup()

    def process_frame_normal(self, jpeg_path: Path) -> np.ndarray:
        img = self._read_bgr(jpeg_path)
        try:
            return self._process_normal_bgr(img)
        finally:
            _cuda_sync_cleanup()

    def unload(self) -> None:
        for attr in ("_matting_model", "_normal_model"):
            m = getattr(self, attr, None)
            if m is not None:
                del m
            setattr(self, attr, None)
        self._crop_tracker = None
        _cuda_sync_cleanup()
