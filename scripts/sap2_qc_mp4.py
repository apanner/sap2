#!/usr/bin/env python3
"""QC MP4 previews for SAP2 matte + normal EXR (LAOV-style plate overlay, HD proxy)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from exr_io import read_exr_pixels

_log = logging.getLogger("sap2_qc_mp4")

_FRAME_RE = re.compile(r"_(\d{4,8})\.exr$", re.IGNORECASE)
_SUBJECT_DIR_RE = re.compile(r"^p(\d{2})$")

# Default: 1920px long edge → 1080×1920 for portrait 4K plates, 1920×1080 for landscape
DEFAULT_QC_MAX_LONG_EDGE = 1920

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


def _fit_long_edge_uint8(img: np.ndarray, max_long: int) -> np.ndarray:
    """Downscale for QC (HD): keep aspect, long edge <= max_long."""
    import cv2

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long:
        return np.ascontiguousarray(img)
    scale = max_long / float(long_edge)
    nw = max(2, int(round(w * scale)))
    nh = max(2, int(round(h * scale)))
    if nw % 2:
        nw += 1
    if nh % 2:
        nh += 1
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _resize_frame_uint8(img: np.ndarray, h: int, w: int) -> np.ndarray:
    import cv2

    if img.shape[0] == h and img.shape[1] == w:
        return np.ascontiguousarray(img)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _write_mp4(frames_rgb: list[np.ndarray], out_path: Path, fps: float) -> None:
    if not frames_rgb:
        raise ValueError("no frames to encode")
    h, w = frames_rgb[0].shape[:2]
    stack = np.stack([_resize_frame_uint8(f, h, w) for f in frames_rgb], axis=0)
    try:
        import imageio.v3 as iio

        out_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(
            out_path,
            stack,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
        )
    except Exception:
        import imageio as iio  # type: ignore[no-redef]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        iio.mimsave(
            str(out_path),
            list(stack),
            fps=fps,
            codec="libx264",
            ffmpeg_params=["-crf", "20", "-pix_fmt", "yuv420p"],
        )


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
    h, w = pixels.shape[0], pixels.shape[1]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    any_ch = False
    for i, ch in enumerate(("R", "G", "B", "A")):
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


def _frame_to_hd_uint8(
    rgb: np.ndarray,
    plate_cache_dir: Path | None,
    frame_idx: int,
    *,
    max_long: int,
    use_plate_overlay: bool,
) -> np.ndarray:
    plate = _read_plate_bgr(plate_cache_dir, frame_idx)
    if use_plate_overlay and plate is not None and plate.shape[:2] == rgb.shape[:2]:
        out = _overlay_plate(plate, rgb)
    else:
        out = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return _fit_long_edge_uint8(out, max_long)


def _export_stage_mp4(
    stage_dir: Path,
    qc_dir: Path,
    shot_label: str,
    suffix: str,
    vis_kind: str,
    *,
    plate_cache_dir: Path | None,
    fps: float,
    max_long: int,
) -> Path | None:
    entries = _sorted_exrs(stage_dir)
    if not entries:
        return None

    frames_rgb: list[np.ndarray] = []
    use_overlay = vis_kind != "normal"
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
        frames_rgb.append(
            _frame_to_hd_uint8(
                rgb,
                plate_cache_dir,
                frame_idx,
                max_long=max_long,
                use_plate_overlay=use_overlay,
            )
        )

    if not frames_rgb:
        return None
    out = qc_dir / f"{shot_label}_{suffix}_qc.mp4"
    _write_mp4(frames_rgb, out, fps)
    h, w = frames_rgb[0].shape[:2]
    _log.info("QC MP4 (%s): %s (%dx%d HD)", suffix, out, w, h)
    return out


def _export_review_mp4(
    shot_dir: Path,
    qc_dir: Path,
    shot_label: str,
    *,
    plate_cache_dir: Path | None,
    matte_layout: str,
    fps: float,
    max_long: int,
) -> Path | None:
    """HD 3-up: plate | matte | normal — fixed panel size every frame."""
    normal_dir = shot_dir / "normal"
    matte_dir = shot_dir / "matte"
    normal_entries = _sorted_exrs(normal_dir)
    if not normal_entries:
        return None

    frames: list[np.ndarray] = []
    panel_h, panel_w = 0, 0

    for frame_idx, norm_path in normal_entries:
        norm_px, norm_names = read_exr_pixels(norm_path)
        norm_rgb = _normal_display_rgb(norm_px, norm_names)
        if norm_rgb is None:
            continue

        plate = _read_plate_bgr(plate_cache_dir, frame_idx)
        matte_rgb = None
        if matte_layout == "separate":
            for sub in sorted(matte_dir.iterdir()) if matte_dir.is_dir() else []:
                if sub.is_dir() and _SUBJECT_DIR_RE.match(sub.name):
                    exr_map = dict(_sorted_exrs(sub))
                    if frame_idx in exr_map:
                        px, names = read_exr_pixels(exr_map[frame_idx])
                        matte_rgb = _matte_combined_rgb(px, names)
                        break
        else:
            exr_map = dict(_sorted_exrs(matte_dir))
            if frame_idx in exr_map:
                px, names = read_exr_pixels(exr_map[frame_idx])
                matte_rgb = (
                    _matte_channels_rgb(px, names)
                    if matte_layout == "channels"
                    else _matte_combined_rgb(px, names)
                )

        ref = plate if plate is not None else norm_rgb
        ref_u8 = _fit_long_edge_uint8(
            (np.clip(ref, 0, 1) * 255).astype(np.uint8) if ref.dtype != np.uint8 else ref,
            max_long,
        )
        ph, pw = ref_u8.shape[:2]
        if panel_h == 0:
            panel_h, panel_w = ph, pw

        black = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        if plate is not None and plate.shape[:2] == norm_rgb.shape[:2]:
            plate_p = _resize_frame_uint8(
                _fit_long_edge_uint8((np.clip(plate, 0, 1) * 255).astype(np.uint8), max_long),
                panel_h,
                panel_w,
            )
        else:
            plate_p = black.copy()

        if matte_rgb is not None and matte_rgb.shape[:2] == norm_rgb.shape[:2]:
            if plate is not None and plate.shape[:2] == matte_rgb.shape[:2]:
                matte_p = _resize_frame_uint8(
                    _overlay_plate(plate, matte_rgb), panel_h, panel_w
                )
            else:
                matte_p = _resize_frame_uint8(
                    _fit_long_edge_uint8((matte_rgb * 255).astype(np.uint8), max_long),
                    panel_h,
                    panel_w,
                )
        else:
            matte_p = black.copy()

        norm_p = _resize_frame_uint8(
            _frame_to_hd_uint8(
                norm_rgb, None, frame_idx, max_long=max_long, use_plate_overlay=False
            ),
            panel_h,
            panel_w,
        )
        frames.append(np.concatenate([plate_p, matte_p, norm_p], axis=1))

    if not frames:
        return None
    out = qc_dir / f"{shot_label}_review_qc.mp4"
    _write_mp4(frames, out, fps)
    fh, fw = frames[0].shape[:2]
    _log.info("QC MP4 (review): %s (%dx%d HD)", out, fw, fh)
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
    qc_max_long_edge: int = DEFAULT_QC_MAX_LONG_EDGE,
) -> dict[str, Path]:
    """Write HD QC MP4s under ``<shot>/qc/`` (long edge capped at ``qc_max_long_edge``)."""
    output_shot_dir = Path(output_shot_dir)
    qc_dir = output_shot_dir / "qc"
    shot_label = output_shot_dir.name
    layout = str(matte_subject_layout or "combined").strip().lower()
    max_long = max(480, int(qc_max_long_edge))
    written: dict[str, Path] = {}
    _log.info("QC encode: max long edge %d px (HD proxy)", max_long)

    stage_kw = dict(plate_cache_dir=plate_cache_dir, fps=fps, max_long=max_long)

    if export_matte:
        matte_dir = output_shot_dir / "matte"
        if layout == "separate" and matte_dir.is_dir():
            for sub in sorted(matte_dir.iterdir()):
                if not sub.is_dir() or not _SUBJECT_DIR_RE.match(sub.name):
                    continue
                try:
                    out = _export_stage_mp4(
                        sub, qc_dir, shot_label, f"matte_{sub.name}", "matte_combined", **stage_kw
                    )
                    if out:
                        written[f"matte_{sub.name}"] = out
                except Exception as exc:
                    _log.warning("QC matte %s failed: %s", sub.name, exc)
        elif layout == "channels" and matte_dir.is_dir():
            try:
                out = _export_stage_mp4(
                    matte_dir, qc_dir, shot_label, "matte_channels", "matte_channels", **stage_kw
                )
                if out:
                    written["matte_channels"] = out
            except Exception as exc:
                _log.warning("QC matte_channels failed: %s", exc)
        elif matte_dir.is_dir():
            try:
                out = _export_stage_mp4(
                    matte_dir, qc_dir, shot_label, "matte", "matte_combined", **stage_kw
                )
                if out:
                    written["matte"] = out
            except Exception as exc:
                _log.warning("QC matte failed: %s", exc)

    if export_normal:
        normal_dir = output_shot_dir / "normal"
        if normal_dir.is_dir():
            try:
                out = _export_stage_mp4(
                    normal_dir, qc_dir, shot_label, "normal", "normal", **stage_kw
                )
                if out:
                    written["normal"] = out
            except Exception as exc:
                _log.warning("QC normal failed: %s", exc)

    if include_review and export_matte and export_normal:
        try:
            out = _export_review_mp4(
                output_shot_dir,
                qc_dir,
                shot_label,
                plate_cache_dir=plate_cache_dir,
                matte_layout=layout,
                fps=fps,
                max_long=max_long,
            )
            if out:
                written["review"] = out
        except Exception as exc:
            _log.warning("QC review failed (matte/normal MP4s kept): %s", exc)

    return written
