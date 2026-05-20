# SPDX-License-Identifier: MIT
"""Minimal EXR read/write for SAP2 Colab (OpenImageIO)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def require_oiio() -> Any:
    try:
        import OpenImageIO as oiio
    except ImportError as exc:
        raise RuntimeError(
            "OpenImageIO required for EXR plates. pip install oiio-python"
        ) from exc
    return oiio


def pattern_to_path(plate_dir: Path, pattern: str, frame: int) -> Path:
    stem = pattern.replace("%04d", f"{frame:04d}").replace("%06d", f"{frame:06d}")
    stem = stem.replace("####", f"{frame:04d}").replace("######", f"{frame:06d}")
    if "/" in stem or "\\" in stem:
        return Path(stem)
    return plate_dir / stem


def read_plate_rgb(path: Path) -> np.ndarray:
    oiio = require_oiio()
    buf = oiio.ImageBuf(str(path))
    if buf.has_error:
        raise RuntimeError(f"OIIO read failed: {path} — {buf.geterror()}")
    spec = buf.spec()
    arr = buf.get_pixels(oiio.FLOAT)
    if arr is None:
        raise RuntimeError(f"OIIO get_pixels failed: {path}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    nch = spec.nchannels
    if nch >= 3:
        rgb = arr[..., :3]
    else:
        rgb = np.stack([arr[..., 0]] * 3, axis=-1)
    return np.ascontiguousarray(np.clip(rgb.astype(np.float32), 0.0, None))


def write_exr_float(path: Path, data: np.ndarray, *, channels: int | None = None) -> None:
    oiio = require_oiio()
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    # OIIO set_pixels requires C-contiguous H×W×C (transpose from torch is often non-contiguous)
    arr = np.ascontiguousarray(arr)
    h, w, c = arr.shape
    if channels is not None:
        c = channels
    spec = oiio.ImageSpec(w, h, c, oiio.FLOAT)
    buf = oiio.ImageBuf(spec)
    if not buf.set_pixels(oiio.ROI(0, w, 0, h, 0, 1, 0, c), arr):
        raise RuntimeError(f"OIIO write failed: {path} — {buf.geterror()}")
    if not buf.write(str(path)):
        raise RuntimeError(f"OIIO write failed: {path} — {buf.geterror()}")


def write_alpha_exr(path: Path, alpha: np.ndarray) -> None:
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    write_exr_float(path, a[..., np.newaxis], channels=1)


def write_normal_exr(path: Path, normal: np.ndarray) -> None:
    n = np.asarray(normal, dtype=np.float32)
    if n.shape[-1] != 3:
        raise ValueError(f"normal must be HxWx3, got {n.shape}")
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    write_exr_float(path, np.clip(n, -1.0, 1.0), channels=3)
