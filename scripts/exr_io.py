# SPDX-License-Identifier: MIT
"""EXR read/write for SAP2 — LAOV-style OpenImageIO (ImageInput/ImageOutput)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger("sap2_exr_io")


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


def _ensure_hwc(pixels: Any, height: int, width: int, nchannels: int) -> np.ndarray:
    arr = np.asarray(pixels, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.reshape(height, width, 1)
    elif arr.ndim == 3 and arr.shape != (height, width, nchannels):
        arr = arr.reshape(height, width, nchannels)
    return np.ascontiguousarray(arr, dtype=np.float32)


def read_exr_pixels(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read all channels as H×W×C float32 (LAOV read_plate pattern)."""
    oiio = require_oiio()
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise RuntimeError(f"OIIO open failed: {path}")
    try:
        spec = inp.spec()
        pixels = inp.read_image(format=oiio.FLOAT)
        if pixels is None:
            raise RuntimeError(f"OIIO read_image failed: {path}")
        arr = _ensure_hwc(pixels, spec.height, spec.width, spec.nchannels)
        names = tuple(spec.channelnames) if spec.channelnames else tuple(
            f"ch{i}" for i in range(arr.shape[-1])
        )
        return arr, names
    finally:
        inp.close()


def read_plate_rgb(path: Path) -> np.ndarray:
    arr, _names = read_exr_pixels(path)
    if arr.shape[-1] >= 3:
        return arr[..., :3].copy()
    return np.stack([arr[..., 0]] * 3, axis=-1)


def _to_float32_c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float32)


def write_exr_float(
    path: Path,
    data: np.ndarray,
    *,
    channels: int | None = None,
    channel_names: tuple[str, ...] | None = None,
    compression: str = "zip",
) -> None:
    """Write H×W×C float32 EXR via ImageOutput.write_image (same as LAOV oiio_io)."""
    oiio = require_oiio()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = _to_float32_c(data)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    h, w, nc = arr.shape
    c = int(channels) if channels is not None else int(nc)
    if nc < c:
        raise ValueError(f"Expected at least {c} channels, got shape {arr.shape}")
    if nc > c:
        arr = _to_float32_c(arr[:, :, :c])

    if channel_names is not None:
        if len(channel_names) != c:
            raise ValueError(f"channel_names len {len(channel_names)} != {c}")
        names = tuple(channel_names)
    elif c == 1:
        names = ("A",)
    elif c == 3:
        names = ("R", "G", "B")
    elif c == 4:
        names = ("R", "G", "B", "A")
    else:
        names = tuple(f"ch{i}" for i in range(c))

    spec = oiio.ImageSpec(w, h, c, oiio.FLOAT)
    spec.channelnames = names
    spec.attribute("compression", compression)

    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(f"No ImageOutput for {path}")
    try:
        if not out.open(str(path), spec):
            raise RuntimeError(f"OIIO open for write failed: {path}")
        if not out.write_image(arr):
            raise RuntimeError(f"OIIO write_image failed: {path}")
    finally:
        out.close()


def write_alpha_exr(path: Path, alpha: np.ndarray) -> None:
    a = np.clip(_to_float32_c(alpha), 0.0, 1.0)
    write_exr_float(path, a, channels=1, channel_names=("A",))


def write_matte_alpha_exr(path: Path, alpha: np.ndarray) -> None:
    """
    Official Sapiens2 primary output (see vis_matting --save_pred): alpha in [0, 1].
    https://github.com/facebookresearch/sapiens2 — single-channel matte for Nuke.
    """
    a = np.clip(_to_float32_c(alpha), 0.0, 1.0)
    if a.ndim == 3:
        a = a[..., 0]
    write_exr_float(path, a, channels=1, channel_names=("A",))


def write_matte_premult_exr(path: Path, matte_rgba: np.ndarray) -> None:
    """Premultiplied foreground RGB + alpha exactly as model outputs (4 ch)."""
    m = _to_float32_c(matte_rgba)
    if m.ndim != 3 or m.shape[-1] != 4:
        raise ValueError(f"matte must be HxWx4 premult RGB+A, got {m.shape}")
    m = np.clip(m, 0.0, 1.0)
    write_exr_float(path, m, channels=4, channel_names=("R", "G", "B", "A"))


def write_matte_exr(path: Path, matte_rgba: np.ndarray) -> None:
    """Default matte deliverable: alpha-only EXR (official). Premult optional via write_matte_premult_exr."""
    if matte_rgba.ndim == 3 and matte_rgba.shape[-1] == 4:
        write_matte_alpha_exr(path, matte_rgba[:, :, 3])
    else:
        write_matte_alpha_exr(path, matte_rgba)


def write_matte_subject_channels_exr(
    path: Path,
    subject_rgba: list[np.ndarray],
    *,
    max_subjects: int = 4,
) -> None:
    if not subject_rgba:
        raise ValueError("subject_rgba is empty")
    h, w = subject_rgba[0].shape[:2]
    packed = np.zeros((h, w, max_subjects), dtype=np.float32)
    for i, subj in enumerate(subject_rgba[:max_subjects]):
        if subj.shape[:2] != (h, w):
            raise ValueError(f"subject {i} shape {subj.shape} != {(h, w, 4)}")
        if subj.shape[-1] == 4:
            packed[:, :, i] = np.clip(subj[:, :, 3], 0.0, 1.0)
        else:
            packed[:, :, i] = np.clip(subj, 0.0, 1.0)
    names = ("R", "G", "B", "A")[:max_subjects]
    write_exr_float(path, packed, channels=max_subjects, channel_names=names)


def write_normal_exr(path: Path, normal: np.ndarray) -> None:
    """Unit normals in [-1, 1] as R,G,B (LAOV n.x/n.y/n.z equivalent)."""
    n = _to_float32_c(normal)
    if n.shape[-1] != 3:
        raise ValueError(f"normal must be HxWx3, got {n.shape}")
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(norm, 1e-8)
    n = np.clip(n, -1.0, 1.0)
    write_exr_float(path, n, channels=3, channel_names=("R", "G", "B"))


def verify_exr_nonzero(path: Path, *, label: str = "") -> dict[str, float]:
    """Read-back sanity check after write (logs min/max)."""
    arr, names = read_exr_pixels(path)
    stats = {
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
    }
    _log.info(
        "EXR verify %s%s: shape=%s ch=%s min=%.4f max=%.4f mean=%.4f",
        label,
        path.name,
        arr.shape,
        names,
        stats["min"],
        stats["max"],
        stats["mean"],
    )
    return stats
