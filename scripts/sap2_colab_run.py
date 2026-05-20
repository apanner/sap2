#!/usr/bin/env python3
"""SAP2 Colab batch runner — Sapiens2 human matting + surface normals.

Usage on Colab::

    export SAP2_DRIVE_MOUNT=/content/drive/MyDrive
    export SAP2_RUNTIME_DATE_FOLDER=$(date +%Y%m%d)
    python scripts/sap2_colab_run.py --job-json /content/sap2_config.json

Expects batch JSON with ``model_type`` ``SAP2_STANDALONE`` (Desk or manual).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
_SAP2_ROOT = _SCRIPTS.parent
_DENSE = _SAP2_ROOT / "sapiens" / "dense"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from exr_io import write_alpha_exr, write_normal_exr  # noqa: E402
from plate_cache import build_plate_jpeg_cache, cache_path, plate_cache_complete  # noqa: E402

_log = logging.getLogger("sap2_colab_run")

SAP2_DEFAULTS: dict[str, Any] = {
    "sapiens_model": "1b",
    "run_matting": True,
    "run_normal": True,
    "export_matte_exr": True,
    "export_normal_exr": True,
    "use_plate_jpeg_cache": True,
    "plate_jpeg_quality": 92,
    "plate_cache_workers": 8,
    "qc_mp4": True,
    "batch_one_process_per_shot": True,
    "batch_gap_seconds": 5,
    "checkpoint_root": "",
    # 0 = auto from VRAM (80GB → 4096 long edge); else cap inference long edge (px)
    "inference_long_edge": 0,
}

from sap2_models import MODEL_CONFIGS  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SAP2 (Sapiens2) batch from Desk JSON.")
    p.add_argument("--job-json", required=True)
    p.add_argument("--shot-index", type=int, default=None)
    return p.parse_args()


def _drive_mount() -> Path:
    raw = os.environ.get("SAP2_DRIVE_MOUNT", "/content/drive/MyDrive")
    return Path(raw)


def _date_folder() -> str:
    return os.environ.get("SAP2_RUNTIME_DATE_FOLDER", time.strftime("%Y%m%d"))


def _checkpoint_root(shared: dict[str, Any]) -> Path:
    explicit = str(shared.get("checkpoint_root", "")).strip()
    if explicit:
        return Path(explicit)
    env = os.environ.get("SAPIENS_CHECKPOINT_ROOT", "").strip()
    if env:
        return Path(env)
    return _drive_mount() / "VDA_models" / "sapiens2_host"


def _output_root(shared: dict[str, Any]) -> Path:
    root_name = str(shared.get("sap2_output_root") or "SAP2_output")
    out_base = shared.get("output_folder_path") or shared.get("output_base")
    if out_base:
        return Path(str(out_base)) / _date_folder() / root_name
    return _drive_mount() / "VDA_output" / _date_folder() / root_name


def _merge_shared(job: dict[str, Any]) -> dict[str, Any]:
    out = dict(SAP2_DEFAULTS)
    out.update(job.get("shared") or {})
    out.update(job.get("shared_settings") or {})
    return out


def _frame_count(frame_start: int, frame_end: int) -> int:
    return max(0, frame_end - frame_start + 1)


def _exr_count(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return len(list(folder.glob("*.exr")))


def _write_image_list(cache_dir: Path, frame_start: int, frame_end: int, list_path: Path) -> None:
    lines = []
    for fi in range(frame_start, frame_end + 1):
        p = cache_path(cache_dir, fi)
        lines.append(str(p.resolve()))
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_vis(
    run_file: str,
    config_rel: str,
    ckpt: Path,
    list_file: Path,
    vis_out: Path,
    *,
    save_pred: bool = False,
    extra_args: list[str] | None = None,
) -> None:
    cmd = [
        sys.executable,
        "-u",
        run_file,
        config_rel,
        str(ckpt),
        "--input",
        str(list_file),
        "--output",
        str(vis_out),
    ]
    if save_pred:
        cmd.append("--save_pred")
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_SAP2_ROOT), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    _log.info("CMD: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(_DENSE),
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{run_file} exited {proc.returncode}")


def _export_matte_exrs(
    vis_out: Path,
    matte_dir: Path,
    frame_start: int,
    frame_end: int,
    cache_dir: Path,
) -> None:
    import numpy as np

    matte_dir.mkdir(parents=True, exist_ok=True)
    for fi in range(frame_start, frame_end + 1):
        stem = cache_path(cache_dir, fi).stem
        npy = vis_out / f"{stem}_alpha.npy"
        if not npy.is_file():
            raise FileNotFoundError(f"Missing matting npy: {npy}")
        alpha = np.load(npy)
        out_exr = matte_dir / f"matte_{fi:06d}.exr"
        if not out_exr.is_file():
            write_alpha_exr(out_exr, alpha)
    _log.info("Matte EXR export: %s (%d frames)", matte_dir, _exr_count(matte_dir))


def _export_normal_exrs(
    vis_out: Path,
    normal_dir: Path,
    frame_start: int,
    frame_end: int,
    cache_dir: Path,
) -> None:
    normal_dir.mkdir(parents=True, exist_ok=True)
    for fi in range(frame_start, frame_end + 1):
        stem = cache_path(cache_dir, fi).stem
        npy = vis_out / f"{stem}.npy"
        if not npy.is_file():
            alt = vis_out / f"{stem.replace('plate_', '')}.npy"
            npy = alt if alt.is_file() else npy
        if not npy.is_file():
            raise FileNotFoundError(f"Missing normal npy for frame {fi}: {npy}")
        import numpy as np

        normal = np.load(npy)
        frame_tag = f"{fi:06d}"
        out_exr = normal_dir / f"normal_{frame_tag}.exr"
        if not out_exr.is_file():
            write_normal_exr(out_exr, normal)
    _log.info("Normal EXR export: %s (%d frames)", normal_dir, _exr_count(normal_dir))


def _run_shot(
    seq: dict[str, Any],
    shared: dict[str, Any],
    ckpt_root: Path,
) -> None:
    shot = str(seq.get("shot_name") or seq.get("name") or "shot")
    plate_dir = Path(seq["plate_dir"])
    pattern = str(seq.get("plate_pattern") or "%06d.exr")
    frame_start = int(seq["frame_start"])
    frame_end = int(seq["frame_end"])
    n_frames = _frame_count(frame_start, frame_end)

    out_shot = _output_root(shared) / shot
    cache_dir = out_shot / "_plate_jpeg_cache"
    matte_dir = out_shot / "matte"
    normal_dir = out_shot / "normal"
    vis_matte = out_shot / "_vis_matting"
    vis_normal = out_shot / "_vis_normal"

    model_key = str(shared.get("sapiens_model", "1b")).lower().replace("sapiens2_", "")
    cfg = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS["1b"])

    _log.info("=== Shot %s frames %d-%d (%d) ===", shot, frame_start, frame_end, n_frames)

    if bool(shared.get("use_plate_jpeg_cache", True)):
        if not plate_cache_complete(cache_dir, frame_start, frame_end):
            _log.info("Building plate JPEG cache → %s", cache_dir)
            build_plate_jpeg_cache(
                plate_dir,
                pattern,
                frame_start,
                frame_end,
                cache_dir,
                jpeg_quality=int(shared.get("plate_jpeg_quality", 92)),
                workers=int(shared.get("plate_cache_workers", 8)),
            )
        else:
            _log.info("Plate JPEG cache complete — skip build")

    list_file = out_shot / "_frame_list.txt"
    _write_image_list(cache_dir, frame_start, frame_end, list_file)

    run_matting = bool(shared.get("run_matting", True))
    run_normal = bool(shared.get("run_normal", True))
    export_matte = bool(shared.get("export_matte_exr", True))
    export_normal = bool(shared.get("export_normal_exr", True))

    matte_done = _exr_count(matte_dir) >= n_frames if export_matte else True
    normal_done = _exr_count(normal_dir) >= n_frames if export_normal else True

    long_edge = int(shared.get("inference_long_edge", 0))
    device = "cuda:0"
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"

    from sap2_infer import Sap2ShotProcessor, vram_auto_long_edge

    _log.info(
        "Inference long_edge=%s (auto=%d on this GPU)",
        long_edge or "auto",
        vram_auto_long_edge(),
    )

    if run_matting and not matte_done:
        if not cfg.get("matting_config"):
            raise RuntimeError("Matting only supported for sapiens_model=1b")
        ckpt = ckpt_root / cfg["matting_ckpt"]
        if not ckpt.is_file():
            raise FileNotFoundError(f"Matting checkpoint missing: {ckpt}")
        matte_dir.mkdir(parents=True, exist_ok=True)
        proc = Sap2ShotProcessor(_DENSE, ckpt_root, model_key=model_key, device=device, inference_long_edge=long_edge)
        try:
            for fi in range(frame_start, frame_end + 1):
                exr_out = matte_dir / f"matte_{fi:06d}.exr"
                if exr_out.is_file() and export_matte:
                    continue
                alpha = proc.process_frame_matting(cache_path(cache_dir, fi))
                if export_matte:
                    write_alpha_exr(exr_out, alpha)
        finally:
            proc.unload()
    elif run_matting:
        _log.info("Skip matting — %d EXRs in %s", _exr_count(matte_dir), matte_dir)

    if run_normal and not normal_done:
        ckpt = ckpt_root / cfg["normal_ckpt"]
        if not ckpt.is_file():
            raise FileNotFoundError(f"Normal checkpoint missing: {ckpt}")
        normal_dir.mkdir(parents=True, exist_ok=True)
        proc = Sap2ShotProcessor(_DENSE, ckpt_root, model_key=model_key, device=device, inference_long_edge=long_edge)
        try:
            for fi in range(frame_start, frame_end + 1):
                exr_out = normal_dir / f"normal_{fi:06d}.exr"
                if exr_out.is_file() and export_normal:
                    continue
                normal = proc.process_frame_normal(cache_path(cache_dir, fi))
                if export_normal:
                    write_normal_exr(exr_out, normal)
        finally:
            proc.unload()
    elif run_normal:
        _log.info("Skip normal — %d EXRs in %s", _exr_count(normal_dir), normal_dir)

    if export_matte and _exr_count(matte_dir) < n_frames:
        raise RuntimeError(f"Incomplete matte EXRs: {_exr_count(matte_dir)}/{n_frames}")
    if export_normal and _exr_count(normal_dir) < n_frames:
        raise RuntimeError(f"Incomplete normal EXRs: {_exr_count(normal_dir)}/{n_frames}")

    _log.info("[OK] Shot %s → %s", shot, out_shot)


def _run_batch(job_path: Path, shot_index: int | None) -> int:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    model_type = str(job.get("model_type", "")).upper()
    if model_type and model_type != "SAP2_STANDALONE":
        _log.warning("model_type=%s (expected SAP2_STANDALONE)", model_type)

    shared = _merge_shared(job)
    sequences = list(job.get("sequences") or [])
    if not sequences:
        _log.error("No sequences in job JSON")
        return 1

    ckpt_root = _checkpoint_root(shared)
    os.environ.setdefault("SAPIENS_CHECKPOINT_ROOT", str(ckpt_root))
    _log.info("SAP2 root=%s dense=%s checkpoints=%s", _SAP2_ROOT, _DENSE, ckpt_root)

    if shot_index is not None:
        if shot_index < 0 or shot_index >= len(sequences):
            _log.error("shot-index %d out of range (0..%d)", shot_index, len(sequences) - 1)
            return 1
        sequences = [sequences[shot_index]]

    failures: list[str] = []
    for i, seq in enumerate(sequences):
        shot = str(seq.get("shot_name") or f"seq_{i}")
        try:
            _run_shot(seq, shared, ckpt_root)
        except Exception as exc:
            _log.error("[ERROR] Shot %s: %s", shot, exc)
            _log.error(traceback.format_exc())
            failures.append(shot)

    if failures:
        _log.error("Batch finished with failures: %s", ", ".join(failures))
        return 1
    _log.info("Batch finished OK (%d shots)", len(sequences))
    return 0


def _spawn_per_shot_parent(job_path: Path) -> int:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    sequences = job.get("sequences") or []
    gap = int((job.get("shared") or {}).get("batch_gap_seconds", 5))
    script = str(Path(__file__).resolve())
    failures: list[str] = []
    for idx in range(len(sequences)):
        shot = str(sequences[idx].get("shot_name") or f"seq_{idx}")
        _log.info("--- Subprocess shot %d/%d: %s ---", idx + 1, len(sequences), shot)
        cmd = [sys.executable, "-u", script, "--job-json", str(job_path), "--shot-index", str(idx)]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            failures.append(shot)
        if idx < len(sequences) - 1 and gap > 0:
            time.sleep(gap)
    return 1 if failures else 0


def main() -> int:
    configure = logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        force=True,
    )
    _ = configure
    args = _parse_args()
    job_path = Path(args.job_json).resolve()
    if not job_path.is_file():
        _log.error("Job JSON not found: %s", job_path)
        return 1

    job = json.loads(job_path.read_text(encoding="utf-8"))
    shared = _merge_shared(job)
    one_per_shot = bool(shared.get("batch_one_process_per_shot", True))
    if one_per_shot and args.shot_index is None and len(job.get("sequences") or []) > 1:
        return _spawn_per_shot_parent(job_path)
    return _run_batch(job_path, args.shot_index)


if __name__ == "__main__":
    raise SystemExit(main())
