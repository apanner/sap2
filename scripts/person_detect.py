# SPDX-License-Identifier: MIT
"""OpenCV person detection + crop/paste for SAP2 native 1024×768 infer."""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

_log = logging.getLogger("person_detect")

PERSON_CLASS_ID_MOBILENET = 15
DNN_DIR = Path(os.environ.get("SAP2_PERSON_DET_DIR", "/content/sap2_models/person_det"))
PROTOTXT_NAME = "MobileNetSSD_deploy.prototxt"
CAFFEMODEL_NAME = "MobileNetSSD_deploy.caffemodel"
PROTOTXT_URL = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt"
)
CAFFEMODEL_URL = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/"
    "master/MobileNetSSD_deploy.caffemodel"
)


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


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 100_000:
        return
    _log.info("Downloading %s → %s", url, dest)
    urllib.request.urlretrieve(url, dest)


def ensure_person_det_weights(root: Path | None = None) -> Path:
    root = Path(root or DNN_DIR)
    _download_file(PROTOTXT_URL, root / PROTOTXT_NAME)
    _download_file(CAFFEMODEL_URL, root / CAFFEMODEL_NAME)
    return root


class PersonDetector:
    """MobileNet-SSD (person class) with HOG fallback."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.4,
        weights_dir: Path | None = None,
    ):
        self.min_confidence = float(min_confidence)
        self._net: cv2.dnn.Net | None = None
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._use_hog = True
        try:
            wdir = ensure_person_det_weights(weights_dir)
            prototxt = wdir / PROTOTXT_NAME
            caffemodel = wdir / CAFFEMODEL_NAME
            if prototxt.is_file() and caffemodel.is_file() and caffemodel.stat().st_size > 1_000_000:
                self._net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
                self._use_hog = False
                _log.info("Person detector: MobileNet-SSD DNN")
            else:
                _log.warning("Person DNN weights missing — using HOG fallback")
        except Exception as exc:
            _log.warning("Person DNN load failed (%s) — HOG fallback", exc)

    def detect(self, image_bgr: np.ndarray) -> List[BBox]:
        if self._net is not None:
            boxes = self._detect_dnn(image_bgr)
            if boxes:
                return boxes
        return self._detect_hog(image_bgr)

    def _detect_dnn(self, image_bgr: np.ndarray) -> List[BBox]:
        assert self._net is not None
        h, w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image_bgr, scalefactor=0.007843, size=(300, 300), mean=127.5
        )
        self._net.setInput(blob)
        det = self._net.forward()
        boxes: List[BBox] = []
        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])
            if conf < self.min_confidence:
                continue
            class_id = int(det[0, 0, i, 1])
            if class_id != PERSON_CLASS_ID_MOBILENET:
                continue
            x1 = int(det[0, 0, i, 3] * w)
            y1 = int(det[0, 0, i, 4] * h)
            x2 = int(det[0, 0, i, 5] * w)
            y2 = int(det[0, 0, i, 6] * h)
            if x2 > x1 + 8 and y2 > y1 + 8:
                boxes.append(BBox(x1, y1, x2, y2))
        return boxes

    def _detect_hog(self, image_bgr: np.ndarray) -> List[BBox]:
        h, w = image_bgr.shape[:2]
        rects, _weights = self._hog.detectMultiScale(
            image_bgr,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05,
        )
        boxes: List[BBox] = []
        for x, y, bw, bh in rects:
            boxes.append(BBox(int(x), int(y), int(x + bw), int(y + bh)).clamp(w, h))
        return boxes


class PersonCropTracker:
    """Detect → temporal smooth → crop for Sapiens2 → paste back to full plate."""

    def __init__(
        self,
        *,
        pad_ratio: float = 0.18,
        min_confidence: float = 0.4,
        smooth_alpha: float = 0.72,
        merge_multi: bool = True,
        weights_dir: Path | None = None,
    ):
        self.pad_ratio = float(pad_ratio)
        self.merge_multi = bool(merge_multi)
        self.smooth_alpha = float(smooth_alpha)
        self._detector = PersonDetector(min_confidence=min_confidence, weights_dir=weights_dir)
        self._smooth: Optional[BBox] = None
        self._full_frame_fallbacks = 0

    def _pick_boxes(self, boxes: List[BBox], img_w: int, img_h: int) -> List[BBox]:
        if not boxes:
            return []
        expanded = [b.expand(self.pad_ratio, img_w, img_h) for b in boxes]
        if self.merge_multi:
            return expanded
        return [max(expanded, key=lambda b: b.area())]

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
        """
        Returns list of crop images, boxes on full plate, used_full_frame flag.
        """
        h, w = image_bgr.shape[:2]
        raw = self._detector.detect(image_bgr)
        picks = self._pick_boxes(raw, w, h)
        if not picks:
            self._full_frame_fallbacks += 1
            if self._full_frame_fallbacks == 1:
                _log.warning("No person detected — full-frame infer fallback")
            return [image_bgr], [BBox(0, 0, w, h)], True

        crops: List[np.ndarray] = []
        boxes: List[BBox] = []
        if self.merge_multi and len(picks) > 1:
            for box in picks:
                sy, sx = box.as_slice()
                crops.append(image_bgr[sy, sx].copy())
                boxes.append(box)
        else:
            box = self._smooth_one(picks[0])
            sy, sx = box.as_slice()
            crops.append(image_bgr[sy, sx].copy())
            boxes.append(box)
        return crops, boxes, False

    @staticmethod
    def paste_matte(full_h: int, full_w: int, crop_alpha: np.ndarray, box: BBox) -> np.ndarray:
        out = np.zeros((full_h, full_w), dtype=np.float32)
        sy, sx = box.as_slice()
        ch, cw = sy.stop - sy.start, sx.stop - sx.start
        alpha = crop_alpha
        if alpha.shape[:2] != (ch, cw):
            alpha = cv2.resize(alpha, (cw, ch), interpolation=cv2.INTER_LINEAR)
        alpha = alpha.astype(np.float32).clip(0.0, 1.0)
        out[sy, sx] = np.maximum(out[sy, sx], alpha)
        return out

    @staticmethod
    def paste_normal(
        full_h: int, full_w: int, crop_normal: np.ndarray, box: BBox
    ) -> np.ndarray:
        out = np.zeros((full_h, full_w, 3), dtype=np.float32)
        sy, sx = box.as_slice()
        ch, cw = sy.stop - sy.start, sx.stop - sx.start
        nrm = crop_normal
        if nrm.shape[:2] != (ch, cw):
            nrm = cv2.resize(nrm, (cw, ch), interpolation=cv2.INTER_LINEAR)
        nrm = np.ascontiguousarray(nrm.astype(np.float32))
        out[sy, sx] = nrm
        return out

    def merge_mattes(self, full_h: int, full_w: int, mattes: List[np.ndarray], boxes: List[BBox]) -> np.ndarray:
        out = np.zeros((full_h, full_w), dtype=np.float32)
        for alpha, box in zip(mattes, boxes):
            layer = self.paste_matte(full_h, full_w, alpha, box)
            np.maximum(out, layer, out=out)
        return out

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
