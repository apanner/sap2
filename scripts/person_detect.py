# SPDX-License-Identifier: MIT
"""
Person detection for SAP2 person-crop.

1. Try MobileNet-SSD DNN (optional download)
2. Fall back to OpenCV HOG people detector (always available, no download)
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

CropMode = Literal["union", "per_person", "largest"]

import cv2
import numpy as np

_log = logging.getLogger("person_detect")

PERSON_CLASS_ID_MOBILENET = 15
DNN_DIR = Path(os.environ.get("SAP2_PERSON_DET_DIR", "/content/sap2_models/person_det"))
PROTOTXT_NAME = "MobileNetSSD_deploy.prototxt"
CAFFEMODEL_NAME = "MobileNetSSD_deploy.caffemodel"

PROTOTXT_URLS = (
    "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt",
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt",
)
CAFFEMODEL_URLS = (
    "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.caffemodel",
)

DetBackend = Literal["mobilenet", "hog"]
MAX_MATTE_SUBJECTS = 4


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def clamp(self, w: int, h: int) -> "BBox":
        return BBox(
            max(0, min(self.x1, w - 1)),
            max(0, min(self.y1, h - 1)),
            max(1, min(self.x2, w)),
            max(1, min(self.y2, h)),
        )

    def expand(self, pad_ratio: float, img_w: int, img_h: int) -> "BBox":
        bw = self.x2 - self.x1
        bh = self.y2 - self.y1
        px = int(bw * pad_ratio)
        py = int(bh * pad_ratio)
        return BBox(self.x1 - px, self.y1 - py, self.x2 + px, self.y2 + py).clamp(img_w, img_h)

    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def as_slice(self) -> Tuple[slice, slice]:
        return slice(self.y1, self.y2), slice(self.x1, self.x2)


def union_bbox(boxes: List[BBox]) -> BBox:
    return BBox(
        min(b.x1 for b in boxes),
        min(b.y1 for b in boxes),
        max(b.x2 for b in boxes),
        max(b.y2 for b in boxes),
    )


def _bbox_iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = a.area() + b.area() - inter
    return inter / max(union, 1)


def sort_boxes_left_to_right(boxes: List[BBox]) -> List[BBox]:
    """Stable subject index across frames: left → right by bbox center."""
    return sorted(boxes, key=lambda b: (b.x1 + b.x2) * 0.5)


def dedupe_boxes(boxes: List[BBox], *, iou_thresh: float = 0.45) -> List[BBox]:
    """Keep highest-area box when two detections overlap heavily."""
    if len(boxes) <= 1:
        return boxes
    ranked = sorted(boxes, key=lambda b: b.area(), reverse=True)
    kept: List[BBox] = []
    for box in ranked:
        if all(_bbox_iou(box, k) < iou_thresh for k in kept):
            kept.append(box)
    return kept


def _download_file(url: str, dest: Path, *, min_bytes: int = 1) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        return True
    _log.info("Downloading %s", url)
    try:
        urllib.request.urlretrieve(url, dest)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        _log.warning("Download failed %s: %s", url, exc)
        if dest.is_file():
            dest.unlink(missing_ok=True)
        return False
    ok = dest.is_file() and dest.stat().st_size >= min_bytes
    if not ok and dest.is_file():
        dest.unlink(missing_ok=True)
    return ok


def _download_first(urls: tuple[str, ...], dest: Path, *, min_bytes: int) -> bool:
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        return True
    for url in urls:
        if _download_file(url, dest, min_bytes=min_bytes):
            _log.info("[ok] %s (%d bytes)", dest.name, dest.stat().st_size)
            return True
    return False


def ensure_person_det_weights(root: Path | None = None, *, try_download: bool = True) -> Path:
    """Optional prefetch; never raises. Missing weights → OpenCV HOG at detect time."""
    root = Path(root or DNN_DIR)
    root.mkdir(parents=True, exist_ok=True)
    if not try_download:
        return root
    got_pt = _download_first(PROTOTXT_URLS, root / PROTOTXT_NAME, min_bytes=1000)
    got_cm = _download_first(CAFFEMODEL_URLS, root / CAFFEMODEL_NAME, min_bytes=1_000_000)
    if not got_pt or not got_cm:
        _log.info(
            "MobileNet-SSD weights unavailable (prototxt=%s caffemodel=%s) — using OpenCV HOG",
            got_pt,
            got_cm,
        )
    return root


def _try_load_mobilenet(weights_dir: Path | None) -> cv2.dnn.Net | None:
    root = Path(weights_dir or DNN_DIR)
    prototxt = root / PROTOTXT_NAME
    caffemodel = root / CAFFEMODEL_NAME
    if not prototxt.is_file() or not caffemodel.is_file():
        return None
    if caffemodel.stat().st_size < 1_000_000:
        return None
    try:
        return cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
    except Exception as exc:
        _log.warning("MobileNet-SSD load failed: %s", exc)
        return None


class PersonDetector:
    """MobileNet-SSD first; OpenCV HOG if DNN missing or finds no person."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.35,
        weights_dir: Path | None = None,
        try_mobilenet_download: bool = True,
    ):
        self.min_confidence = float(min_confidence)
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._net: cv2.dnn.Net | None = None
        self._backend: DetBackend = "hog"

        force = os.environ.get("SAP2_PERSON_DET_BACKEND", "auto").strip().lower()
        if force == "hog":
            _log.info("Person detect: OpenCV HOG (forced by SAP2_PERSON_DET_BACKEND)")
            return

        if try_mobilenet_download and force != "hog":
            ensure_person_det_weights(weights_dir, try_download=True)

        self._net = _try_load_mobilenet(weights_dir)
        if self._net is not None and force != "hog":
            self._backend = "mobilenet"
            _log.info("Person detect: MobileNet-SSD DNN (OpenCV HOG fallback if no hits)")
        else:
            _log.info("Person detect: OpenCV HOG (built-in, no download)")

    def detect(self, image_bgr: np.ndarray) -> List[BBox]:
        boxes: List[BBox] = []
        if self._net is not None:
            boxes = self._detect_mobilenet(image_bgr)
        if not boxes:
            boxes = self._detect_hog(image_bgr)
        return boxes

    def _scaled_image(self, image_bgr: np.ndarray, det_long: int = 1280) -> tuple[np.ndarray, float]:
        h, w = image_bgr.shape[:2]
        if max(h, w) <= det_long:
            return image_bgr, 1.0
        scale = det_long / max(h, w)
        small = cv2.resize(
            image_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        return small, scale

    def _detect_mobilenet(self, image_bgr: np.ndarray, *, det_long: int = 1280) -> List[BBox]:
        assert self._net is not None
        h, w = image_bgr.shape[:2]
        det_img, scale = self._scaled_image(image_bgr, det_long=det_long)
        dh, dw = det_img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            det_img, scalefactor=0.007843, size=(300, 300), mean=127.5
        )
        self._net.setInput(blob)
        det = self._net.forward()
        inv = 1.0 / scale
        boxes: List[BBox] = []
        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])
            if conf < self.min_confidence:
                continue
            if int(det[0, 0, i, 1]) != PERSON_CLASS_ID_MOBILENET:
                continue
            x1 = int(det[0, 0, i, 3] * dw * inv)
            y1 = int(det[0, 0, i, 4] * dh * inv)
            x2 = int(det[0, 0, i, 5] * dw * inv)
            y2 = int(det[0, 0, i, 6] * dh * inv)
            if x2 > x1 + 8 and y2 > y1 + 8:
                boxes.append(BBox(x1, y1, x2, y2).clamp(w, h))
        return boxes

    def _detect_hog(self, image_bgr: np.ndarray, *, det_long: int = 1280) -> List[BBox]:
        h, w = image_bgr.shape[:2]
        det_img, scale = self._scaled_image(image_bgr, det_long=det_long)
        rects, _weights = self._hog.detectMultiScale(
            det_img,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.04,
            hitThreshold=0,
        )
        inv = 1.0 / scale
        boxes: List[BBox] = []
        for x, y, bw, bh in rects:
            boxes.append(
                BBox(
                    int(x * inv),
                    int(y * inv),
                    int((x + bw) * inv),
                    int((y + bh) * inv),
                ).clamp(w, h)
            )
        return boxes


class PersonCropTracker:
    """Detect → temporal smooth → crop for Sapiens2 → paste back to full plate."""

    def __init__(
        self,
        *,
        pad_ratio: float = 0.22,
        min_confidence: float = 0.28,
        smooth_alpha: float = 0.72,
        crop_mode: CropMode = "union",
        merge_multi: bool | None = None,
        weights_dir: Path | None = None,
    ):
        self.pad_ratio = float(pad_ratio)
        if merge_multi is not None and not crop_mode:
            crop_mode = "per_person" if merge_multi else "largest"
        self.crop_mode: CropMode = crop_mode  # type: ignore[assignment]
        self.smooth_alpha = float(smooth_alpha)
        self._detector = PersonDetector(
            min_confidence=min_confidence,
            weights_dir=weights_dir,
            try_mobilenet_download=True,
        )
        self._smooth: Optional[BBox] = None
        self._full_frame_fallbacks = 0

    def _pick_boxes(self, boxes: List[BBox], img_w: int, img_h: int) -> List[BBox]:
        if not boxes:
            return []
        expanded = [b.expand(self.pad_ratio, img_w, img_h) for b in boxes]
        if self.crop_mode == "largest":
            return [max(expanded, key=lambda b: b.area())]
        if self.crop_mode == "union":
            return [union_bbox(expanded).clamp(img_w, img_h)]
        return expanded

    def _smooth_one(self, box: BBox) -> BBox:
        if self._smooth is None:
            self._smooth = box
            return box
        a = self.smooth_alpha
        self._smooth = BBox(
            int(a * box.x1 + (1 - a) * self._smooth.x1),
            int(a * box.y1 + (1 - a) * self._smooth.y1),
            int(a * box.x2 + (1 - a) * self._smooth.x2),
            int(a * box.y2 + (1 - a) * self._smooth.y2),
        )
        return self._smooth

    def crop_regions(self, image_bgr: np.ndarray) -> Tuple[List[np.ndarray], List[BBox], bool]:
        h, w = image_bgr.shape[:2]
        raw = self._detector.detect(image_bgr)
        picks = self._pick_boxes(raw, w, h)
        if not picks:
            self._full_frame_fallbacks += 1
            if self._full_frame_fallbacks == 1:
                _log.warning("No person detected (MobileNet + HOG) — full-frame infer fallback")
            return [image_bgr], [BBox(0, 0, w, h)], True

        crops: List[np.ndarray] = []
        boxes: List[BBox] = []
        if self.crop_mode == "per_person" and len(picks) > 1:
            for box in picks:
                sy, sx = box.as_slice()
                crops.append(image_bgr[sy, sx].copy())
                boxes.append(box)
        else:
            box = self._smooth_one(picks[0])
            sy, sx = box.as_slice()
            crops.append(image_bgr[sy, sx].copy())
            boxes.append(box)
            if not getattr(self, "_crop_log_once", False):
                self._crop_log_once = True
                _log.info(
                    "Person crop (%s): %d det → bbox [%d,%d,%d,%d] on %dx%d",
                    self.crop_mode,
                    len(raw),
                    box.x1,
                    box.y1,
                    box.x2,
                    box.y2,
                    w,
                    h,
                )
        return crops, boxes, False

    def crop_regions_per_subject(
        self, image_bgr: np.ndarray, *, max_subjects: int = MAX_MATTE_SUBJECTS
    ) -> Tuple[List[np.ndarray], List[BBox], bool]:
        """One crop per detected person (sorted left→right), up to max_subjects."""
        h, w = image_bgr.shape[:2]
        raw = sort_boxes_left_to_right(dedupe_boxes(self._detector.detect(image_bgr)))
        if not raw:
            self._full_frame_fallbacks += 1
            if self._full_frame_fallbacks == 1:
                _log.warning("No person detected — full-frame matte fallback (1 subject)")
            return [image_bgr], [BBox(0, 0, w, h)], True

        expanded = [b.expand(self.pad_ratio, w, h) for b in raw[:max_subjects]]
        crops: List[np.ndarray] = []
        boxes: List[BBox] = []
        for box in expanded:
            sy, sx = box.as_slice()
            crops.append(image_bgr[sy, sx].copy())
            boxes.append(box)
        if not getattr(self, "_subject_crop_log_once", False):
            self._subject_crop_log_once = True
            _log.info(
                "Per-subject matte: %d people (L→R indices 0..%d) on %dx%d",
                len(boxes),
                len(boxes) - 1,
                w,
                h,
            )
        return crops, boxes, False

    @staticmethod
    def _resize_crop_rgba(crop_rgba: np.ndarray, ch: int, cw: int) -> np.ndarray:
        if crop_rgba.shape[:2] == (ch, cw):
            return crop_rgba.astype(np.float32)
        out = np.zeros((ch, cw, 4), dtype=np.float32)
        for c in range(4):
            out[:, :, c] = cv2.resize(
                crop_rgba[:, :, c], (cw, ch), interpolation=cv2.INTER_LINEAR
            )
        return out.clip(0.0, 1.0)

    @staticmethod
    def paste_matte_rgba(
        full_h: int, full_w: int, crop_rgba: np.ndarray, box: BBox
    ) -> np.ndarray:
        out = np.zeros((full_h, full_w, 4), dtype=np.float32)
        sy, sx = box.as_slice()
        ch, cw = sy.stop - sy.start, sx.stop - sx.start
        out[sy, sx] = PersonCropTracker._resize_crop_rgba(crop_rgba, ch, cw)
        return out

    @staticmethod
    def paste_matte(full_h: int, full_w: int, crop_alpha: np.ndarray, box: BBox) -> np.ndarray:
        """Legacy alpha-only paste (alpha plane from RGBA matte)."""
        if crop_alpha.ndim == 3 and crop_alpha.shape[-1] == 4:
            crop_alpha = crop_alpha[:, :, 3]
        out = np.zeros((full_h, full_w), dtype=np.float32)
        sy, sx = box.as_slice()
        ch, cw = sy.stop - sy.start, sx.stop - sx.start
        if crop_alpha.shape[:2] != (ch, cw):
            crop_alpha = cv2.resize(crop_alpha, (cw, ch), interpolation=cv2.INTER_LINEAR)
        out[sy, sx] = np.maximum(out[sy, sx], crop_alpha.astype(np.float32).clip(0.0, 1.0))
        return out

    @staticmethod
    def paste_normal(
        full_h: int, full_w: int, crop_normal: np.ndarray, box: BBox
    ) -> np.ndarray:
        out = np.zeros((full_h, full_w, 3), dtype=np.float32)
        sy, sx = box.as_slice()
        nrm = crop_normal
        ch, cw = sy.stop - sy.start, sx.stop - sx.start
        if nrm.shape[:2] != (ch, cw):
            nrm = cv2.resize(nrm, (cw, ch), interpolation=cv2.INTER_LINEAR)
        nrm = np.ascontiguousarray(nrm.astype(np.float32))
        out[sy, sx] = nrm
        return out

    def merge_mattes_rgba(
        self, full_h: int, full_w: int, mattes: List[np.ndarray], boxes: List[BBox]
    ) -> np.ndarray:
        out = np.zeros((full_h, full_w, 4), dtype=np.float32)
        for rgba, box in zip(mattes, boxes):
            sy, sx = box.as_slice()
            ch, cw = sy.stop - sy.start, sx.stop - sx.start
            layer = self._resize_crop_rgba(rgba, ch, cw)
            a_new = layer[:, :, 3]
            a_old = out[sy, sx, 3]
            take = a_new > a_old
            for c in range(3):
                out[sy, sx, c] = np.where(take, layer[:, :, c], out[sy, sx, c])
            out[sy, sx, 3] = np.maximum(a_old, a_new)
        return out

    def merge_mattes(self, full_h: int, full_w: int, mattes: List[np.ndarray], boxes: List[BBox]) -> np.ndarray:
        rgba = self.merge_mattes_rgba(full_h, full_w, mattes, boxes)
        return rgba[:, :, 3]

    def merge_normals(
        self, full_h: int, full_w: int, normals: List[np.ndarray], boxes: List[BBox]
    ) -> np.ndarray:
        if len(normals) == 1:
            return self.paste_normal(full_h, full_w, normals[0], boxes[0])
        out = np.zeros((full_h, full_w, 3), dtype=np.float32)
        weight = np.zeros((full_h, full_w), dtype=np.float32)
        for nrm, box in zip(normals, boxes):
            layer = self.paste_normal(full_h, full_w, nrm, box)
            sy, sx = box.as_slice()
            mask = layer[sy, sx].sum(axis=2) > 1e-6
            out[sy, sx][mask] = layer[sy, sx][mask]
            weight[sy, sx] = np.maximum(weight[sy, sx], mask.astype(np.float32))
        return out
