#!/usr/bin/env python3
"""SAP2 local engine — EXR plates → JPEG cache → matting + normals → EXR (VDA-style job JSON)."""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path

_SAP2_ROOT = Path(__file__).resolve().parents[2]
_DENSE = _SAP2_ROOT / "sapiens" / "dense"
_SCRIPTS = _SAP2_ROOT / "scripts"
for p in (_SCRIPTS, str(_DENSE)):
    if p not in sys.path:
        sys.path.insert(0, str(p))

os.environ.setdefault("SAPIENS_CHECKPOINT_ROOT", str(_SAP2_ROOT / "checkpoints"))

from exr_io import write_alpha_exr, write_normal_exr  # noqa: E402
from plate_cache import build_plate_jpeg_cache, cache_path, plate_cache_complete  # noqa: E402
from sap2_infer import MODEL_NATIVE_H, MODEL_NATIVE_W, Sap2ShotProcessor  # noqa: E402

_log = logging.getLogger("sap2_engine")


def _shared(job: dict) -> dict:
    return job.get("shared") or job.get("shared_settings") or {}


def process_shot(job: dict) -> bool:
    shared = _shared(job)
    seq = job
    if "sequences" in job and job["sequences"]:
        seq = job["sequences"][0]

    shot = str(seq.get("shot_name") or job.get("shot_name") or "shot")
    plate_dir = Path(seq.get("plate_dir") or job.get("plate_dir", ""))
    pattern = str(seq.get("plate_pattern") or job.get("plate_pattern", "%06d.exr"))
    frame_start = int(seq.get("frame_start") or job.get("frame_start", 1001))
    frame_end = int(seq.get("frame_end") or job.get("frame_end", frame_start))
    output_dir = Path(
        seq.get("output_dir")
        or job.get("output_dir")
        or shared.get("output_dir")
        or (_SAP2_ROOT / "output" / shot)
    )

    ckpt_root = Path(
        shared.get("checkpoint_root")
        or os.environ.get("SAPIENS_CHECKPOINT_ROOT", _SAP2_ROOT / "checkpoints")
    )
    model_key = str(shared.get("sapiens_model", "1b"))
    long_edge = int(shared.get("inference_long_edge", 0))
    run_matting = bool(shared.get("run_matting", True))
    run_normal = bool(shared.get("run_normal", True))

    out_shot = output_dir / shot if output_dir.name != shot else output_dir
    cache_dir = out_shot / "_plate_jpeg_cache"
    matte_dir = out_shot / "matte"
    normal_dir = out_shot / "normal"
    n_frames = frame_end - frame_start + 1

    _log.info("SAP2 shot=%s frames %d-%d → %s", shot, frame_start, frame_end, out_shot)
    _log.info("Infer %d×%d native (long_edge setting %s ignored)", MODEL_NATIVE_H, MODEL_NATIVE_W, long_edge or "0")

    if bool(shared.get("use_plate_jpeg_cache", True)):
        if not plate_cache_complete(cache_dir, frame_start, frame_end):
            build_plate_jpeg_cache(
                plate_dir,
                pattern,
                frame_start,
                frame_end,
                cache_dir,
                jpeg_quality=int(shared.get("plate_jpeg_quality", 92)),
                workers=int(shared.get("plate_cache_workers", 8)),
            )

    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
    proc = Sap2ShotProcessor(
        _DENSE,
        ckpt_root,
        model_key=model_key,
        device=device,
        inference_long_edge=long_edge,
    )

    try:
        if run_matting:
            matte_dir.mkdir(parents=True, exist_ok=True)
            for fi in range(frame_start, frame_end + 1):
                exr_out = matte_dir / f"matte_{fi:06d}.exr"
                if exr_out.is_file():
                    continue
                alpha = proc.process_frame_matting(cache_path(cache_dir, fi))
                write_alpha_exr(exr_out, alpha)
                if (fi - frame_start) % 25 == 0:
                    _log.info("Matting %d/%d", fi - frame_start + 1, n_frames)

        if run_normal:
            normal_dir.mkdir(parents=True, exist_ok=True)
            for fi in range(frame_start, frame_end + 1):
                exr_out = normal_dir / f"normal_{fi:06d}.exr"
                if exr_out.is_file():
                    continue
                normal = proc.process_frame_normal(cache_path(cache_dir, fi))
                write_normal_exr(exr_out, normal)
                if (fi - frame_start) % 25 == 0:
                    _log.info("Normal %d/%d", fi - frame_start + 1, n_frames)
    finally:
        proc.unload()

    _log.info("[OK] %s matte=%s normal=%s", shot, matte_dir, normal_dir)
    return True


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if len(sys.argv) < 2:
        print("Usage: sap2_engine.py <job.json>")
        return 1
    job_path = Path(sys.argv[1])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    try:
        process_shot(job)
        return 0
    except Exception as exc:
        _log.error("SAP2 engine failed: %s", exc)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
