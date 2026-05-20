# SPDX-License-Identifier: MIT
"""EXR plate → JPEG cache for Sapiens2 vis scripts (jpg/png only)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from exr_io import pattern_to_path, read_plate_rgb

_log = logging.getLogger("sap2_plate_cache")


def cache_path(cache_dir: Path, frame_idx: int) -> Path:
    return cache_dir / f"plate_{frame_idx:06d}.jpg"


def plate_cache_complete(cache_dir: Path, frame_start: int, frame_end: int) -> bool:
    n = frame_end - frame_start + 1
    if not cache_dir.is_dir():
        return False
    return len(list(cache_dir.glob("plate_*.jpg"))) >= n


def build_plate_jpeg_cache(
    plate_dir: Path,
    pattern: str,
    frame_start: int,
    frame_end: int,
    cache_dir: Path,
    *,
    jpeg_quality: int = 92,
    workers: int = 8,
    log_every: int = 25,
) -> tuple[int, int]:
    import cv2

    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = list(range(frame_start, frame_end + 1))
    h = w = 0

    def _one(frame_idx: int) -> tuple[int, int, int]:
        out = cache_path(cache_dir, frame_idx)
        if out.is_file():
            img = cv2.imread(str(out))
            if img is not None:
                return int(img.shape[0]), int(img.shape[1]), frame_idx
        path = pattern_to_path(plate_dir, pattern, frame_idx)
        rgb = read_plate_rgb(path)
        rgb_u8 = (np.clip(rgb[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(out),
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 50, 100))],
        )
        return int(rgb_u8.shape[0]), int(rgb_u8.shape[1]), frame_idx

    workers = max(1, int(workers))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, f): f for f in frames}
        for fut in as_completed(futures):
            hi, wi, _fi = fut.result()
            h, w = hi, wi
            done += 1
            if log_every > 0 and (done == 1 or done % log_every == 0 or done == len(frames)):
                _log.info("Plate JPEG cache %d/%d (%dx%d)", done, len(frames), h, w)
    return h, w
