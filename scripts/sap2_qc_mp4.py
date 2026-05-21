#!/usr/bin/env python3
"""QC MP4 previews for SAP2 matte + normal EXR (LAOV-style plate overlay)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from exr_io import read_exr_pixels

_log = logging.getLogger("sap2_qc_mp4")

_FRAME_RE = re.compile(r"_(\d{4,8})\.exr$", re.IGNORECASE)
_SUBJECT_DIR_RE = re.compile(r"^p(\d{2})$")

# R,G,B,A person slot colors (channels layout)
_SLOT_COLORS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
)


def _sorted_exrs(folder: Path) -> list[tuple[int, Path]]:
    if not folder.is_dir():
        return []
    by_frame: dict[int, Path] = {}
    for path in sorted(folder.glob("*.exr")):
        m = _FRAME_RE.search(path.name)
        if m:
            by_frame[int(m.group(1))] = path
    return sorted(by_frame.items(), key=lambda x: x[0])


def _channel_plane(
    pixels: np.ndarray, names: tuple[str, ...], channel: str
) -> np.ndarray | None:
    if channel not in names:
        return None
    idx = names.index(channel)
    return np.clip(pixels[..., idx].astype(np.float32), 0.0, 1.0)


def _overlay_plate(plate_rgb: np.ndarray, rgb: np.ndarray, *, alpha_scale: float = 0.55) -> np.ndarray:
    plate = np.clip(plate_rgb[..., :3], 0.0, 1.0).astype(np.float32)
    tint = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    a = np.max(tint, axis=-1, keepdims=True)
    out = plate * (1.0 - alpha_scale * a) + tint * (alpha_scale * a)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _write_mp4(frames_rgb: list[np.ndarray], out_path: Path, fps: float) -> None:
    if not frames_rgb:
        raise ValueError("no frames to encode")
    try:
        import imageio.v3 as iio
    except ImportError:
        import imageio as iio  # type: ignore[no-redef]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        out_path,
        np.stack(frames_rgb, axis=0),
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
    )


def _read_plate_bgr(
    plate_cache_dir: Path | None,
    frame_idx: int,
) -> np.ndarray | None:
    if plate_cache_dir is None:
        return None
    import cv2

    for name in (f"plate_{frame_idx:06d}.jpg", f"plate_{frame_idx:04d}.jpg"):
        path = plate_cache_dir / name
        if path.is_file():
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return None


def _matte_combined_rgb(pixels: np.ndarray, names: tuple[str, ...]) -> np.ndarray | None:
    r = _channel_plane(pixels, names, "R")
    g = _channel_plane(pixels, names, "G")
    b = _channel_plane(pixels, names, "B")
    a = _channel_plane(pixels, names, "A")
    h, w = pixels.shape[0], pixels.shape[1]
    if r is not None or g is not None or b is not None:
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        if r is not None:
            rgb[..., 0] = r
        if g is not None:
            rgb[..., 1] = g
        if b is not None:
            rgb[..., 2] = b
        return rgb
    if a is not None:
        return np.stack([a, a, a], axis=-1)
    return None


def _matte_channels_rgb(pixels: np.ndarray, names: tuple[str, ...]) -> np.ndarray | None:
    """R,G,B,A EXR planes → red/green/blue/white matte preview."""
    h, w = pixels.shape[0], pixels.shape[1]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    any_ch = False
    ch_names = ("R", "G", "B", "A")
    for i, ch in enumerate(ch_names):
        if i >= len(_SLOT_COLORS):
            break
        plane = _channel_plane(pixels, names, ch)
        if plane is None:
            continue
        any_ch = True
        color = np.array(_SLOT_COLORS[i], dtype=np.float32)
        for c in range(3):
            rgb[..., c] = np.maximum(rgb[..., c], plane * color[c])
    return rgb if any_ch else None


def _normal_display_rgb(pixels: np.ndarray, names: tuple[str, ...]) -> np.ndarray | None:
    r = _channel_plane(pixels, names, "R")
    g = _channel_plane(pixels, names, "G")
    b = _channel_plane(pixels, names, "B")
    if r is None or g is None or b is None:
        return None
    n = np.stack([r, g, b], axis=-1)
    return np.clip(n * 0.5 + 0.5, 0.0, 1.0)


def _export_stage_mp4(
    stage_dir: Path,
    qc_dir: Path,
    shot_label: str,
    suffix: str,
    vis_kind: str,
    *,
    plate_cache_dir: Path | None,
    fps: float,
) -> Path | None:
    entries = _sorted_exrs(stage_dir)
    if not entries:
        return None

    frames_rgb: list[np.ndarray] = []
    for frame_idx, exr_path in entries:
        pixels, names = read_exr_pixels(exr_path)
        if vis_kind == "matte_combined":
            rgb = _matte_combined_rgb(pixels, names)
        elif vis_kind == "matte_channels":
            rgb = _matte_channels_rgb(pixels, names)
        elif vis_kind == "normal":
            rgb = _normal_display_rgb(pixels, names)
        else:
            rgb = _matte_combined_rgb(pixels, names)
        if rgb is None:
            continue
        plate = _read_plate_bgr(plate_cache_dir, frame_idx)
        if plate is not None and plate.shape[:2] == rgb.shape[:2]:
            frames_rgb.append(_overlay_plate(plate, rgb))
        else:
            frames_rgb.append((rgb * 255.0).astype(np.uint8))

    if not frames_rgb:
        return None
    out = qc_dir / f"{shot_label}_{suffix}_qc.mp4"
    _write_mp4(frames_rgb, out, fps)
    _log.info("QC MP4 (%s): %s", suffix, out)
    return out


def _concat_horizontal(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts, axis=1)


def _export_review_mp4(
    shot_dir: Path,
    qc_dir: Path,
    shot_label: str,
    *,
    plate_cache_dir: Path | None,
    matte_layout: str,
    fps: float,
) -> Path | None:
    """Single MP4: plate | matte | normal (when folders exist)."""
    normal_dir = shot_dir / "normal"
    matte_dir = shot_dir / "matte"
    normal_entries = _sorted_exrs(normal_dir)
    if not normal_entries:
        return None

    matte_vis = "matte_channels" if matte_layout == "channels" else "matte_combined"
    frames: list[np.ndarray] = []
    for frame_idx, norm_path in normal_entries:
        norm_px, norm_names = read_exr_pixels(norm_path)
        norm_rgb = _normal_display_rgb(norm_px, norm_names)
        if norm_rgb is None:
            continue

        parts: list[np.ndarray] = []
        plate = _read_plate_bgr(plate_cache_dir, frame_idx)
        if plate is not None:
            parts.append((np.clip(plate, 0, 1) * 255.0).astype(np.uint8))

        matte_rgb = None
        if matte_layout == "separate":
            for sub in sorted(matte_dir.iterdir()) if matte_dir.is_dir() else []:
                if sub.is_dir() and _SUBJECT_DIR_RE.match(sub.name):
                    exrs = _sorted_exrs(sub)
                    exr_map = dict(exrs)
                    if frame_idx in exr_map:
                        px, names = read_exr_pixels(exr_map[frame_idx])
                        matte_rgb = _matte_combined_rgb(px, names)
                        break
        else:
            exrs = dict(_sorted_exrs(matte_dir))
            if frame_idx in exrs:
                px, names = read_exr_pixels(exrs[frame_idx])
                matte_rgb = (
                    _matte_channels_rgb(px, names)
                    if matte_layout == "channels"
                    else _matte_combined_rgb(px, names)
                )

        if matte_rgb is not None:
            if plate is not None and plate.shape[:2] == matte_rgb.shape[:2]:
                parts.append(_overlay_plate(plate, matte_rgb))
            else:
                parts.append((matte_rgb * 255.0).astype(np.uint8))
        parts.append((norm_rgb * 255.0).astype(np.uint8))
        if len(parts) >= 2:
            frames.append(_concat_horizontal(parts))

    if not frames:
        return None
    out = qc_dir / f"{shot_label}_review_qc.mp4"
    _write_mp4(frames, out, fps)
    _log.info("QC MP4 (review): %s", out)
    return out


def export_sap2_qc_mp4s(
    output_shot_dir: Path,
    *,
    plate_cache_dir: Path | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    fps: float = 24.0,
    matte_subject_layout: str = "combined",
    export_matte: bool = True,
    export_normal: bool = True,
    include_review: bool = True,
) -> dict[str, Path]:
    """
    Write QC MP4s under ``<shot>/qc/`` (LAOV-style naming).

    - ``{shot}_matte_qc.mp4`` — combined RGBA matte on plate
    - ``{shot}_matte_channels_qc.mp4`` — R/G/B/A person alphas as colors
    - ``{shot}_matte_p00_qc.mp4`` … — separate per-person folders
    - ``{shot}_normal_qc.mp4`` — normals on plate
    - ``{shot}_review_qc.mp4`` — plate | matte | normal (optional)
    """
    output_shot_dir = Path(output_shot_dir)
    qc_dir = output_shot_dir / "qc"
    shot_label = output_shot_dir.name
    layout = str(matte_subject_layout or "combined").strip().lower()
    written: dict[str, Path] = {}

    if export_matte:
        matte_dir = output_shot_dir / "matte"
        if layout == "separate" and matte_dir.is_dir():
            for sub in sorted(matte_dir.iterdir()):
                if not sub.is_dir() or not _SUBJECT_DIR_RE.match(sub.name):
                    continue
                sid = sub.name
                out = _export_stage_mp4(
                    sub,
                    qc_dir,
                    shot_label,
                    f"matte_{sid}",
                    "matte_combined",
                    plate_cache_dir=plate_cache_dir,
                    fps=fps,
                )
                if out:
                    written[f"matte_{sid}"] = out
        elif layout == "channels" and matte_dir.is_dir():
            out = _export_stage_mp4(
                matte_dir,
                qc_dir,
                shot_label,
                "matte_channels",
                "matte_channels",
                plate_cache_dir=plate_cache_dir,
                fps=fps,
            )
            if out:
                written["matte_channels"] = out
        elif matte_dir.is_dir():
            out = _export_stage_mp4(
                matte_dir,
                qc_dir,
                shot_label,
                "matte",
                "matte_combined",
                plate_cache_dir=plate_cache_dir,
                fps=fps,
            )
            if out:
                written["matte"] = out

    if export_normal:
        normal_dir = output_shot_dir / "normal"
        if normal_dir.is_dir():
            out = _export_stage_mp4(
                normal_dir,
                qc_dir,
                shot_label,
                "normal",
                "normal",
                plate_cache_dir=plate_cache_dir,
                fps=fps,
            )
            if out:
                written["normal"] = out

    if include_review and export_matte and export_normal:
        out = _export_review_mp4(
            output_shot_dir,
            qc_dir,
            shot_label,
            plate_cache_dir=plate_cache_dir,
            matte_layout=layout,
            fps=fps,
        )
        if out:
            written["review"] = out

    return written
