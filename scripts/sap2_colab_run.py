#!/usr/bin/env python3
"""SAP2 Colab batch — VDA-style: read plates from Drive, process on /content, sync EXR to Drive."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
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
from plate_cache import build_plate_jpeg_cache, cache_path, plate_cache_complete, pattern_to_path  # noqa: E402

_log = logging.getLogger("sap2_colab_run")

# VDA pattern: all hot I/O on Colab local disk
SAP2_LOCAL_ROOT = Path(os.environ.get("SAP2_LOCAL_WORK", "/content/sap2_work"))

SAP2_DEFAULTS: dict[str, Any] = {
    "sapiens_model": "1b",
    "run_matting": True,
    "run_normal": True,
    "export_matte_exr": True,
    "export_normal_exr": True,
    "use_plate_jpeg_cache": True,
    "plate_jpeg_quality": 92,
    "plate_cache_workers": 12,
    "qc_mp4": False,
    "batch_one_process_per_shot": True,
    "split_pass_subprocess": True,
    "batch_gap_seconds": 5,
    "checkpoint_root": "",
    "download_models_in_colab": True,
    "inference_long_edge": 0,
    "inference_max_megapixels": 1.6,
    "use_local_workdir": True,
}

from sap2_models import MODEL_CONFIGS  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SAP2 batch from Desk JSON.")
    p.add_argument("--job-json", required=True)
    p.add_argument("--shot-index", type=int, default=None)
    p.add_argument(
        "--pass",
        dest="pass_name",
        choices=("matting", "normal", "all"),
        default="all",
    )
    return p.parse_args()


def _drive_mount() -> Path:
    return Path(os.environ.get("SAP2_DRIVE_MOUNT", "/content/drive/MyDrive"))


def _date_folder() -> str:
    return os.environ.get("SAP2_RUNTIME_DATE_FOLDER", time.strftime("%Y%m%d"))


def _resolve_on_drive(rel_or_abs: str) -> Path:
    p = str(rel_or_abs or "").strip().replace("\\", "/")
    if not p:
        return _drive_mount()
    if p.startswith("/content/drive"):
        return Path(p)
    if p.startswith("MyDrive/"):
        return _drive_mount() / p[len("MyDrive/") :]
    drive = _drive_mount()
    if str(p).startswith(str(drive).replace("\\", "/")):
        return Path(p)
    return drive / p.lstrip("/")


def _local_root() -> Path:
    return Path(os.environ.get("SAP2_LOCAL_WORK", str(SAP2_LOCAL_ROOT)))


def _drive_output_root(shared: dict[str, Any]) -> Path:
    root_name = str(shared.get("sap2_output_root") or "SAP2_output")
    out_base = str(shared.get("output_folder_path") or "VDA_output").strip().strip("/")
    return _resolve_on_drive(out_base) / _date_folder() / root_name


def _local_shot_dir(shot: str) -> Path:
    return _local_root() / _date_folder() / shot


def _drive_shot_dir(shared: dict[str, Any], shot: str) -> Path:
    return _drive_output_root(shared) / shot


def _checkpoint_root(shared: dict[str, Any]) -> Path:
    explicit = str(shared.get("checkpoint_root", "")).strip()
    if explicit and "VDA_models" not in explicit.replace("\\", "/"):
        return Path(explicit)
    env = os.environ.get("SAPIENS_CHECKPOINT_ROOT", "").strip()
    if env:
        return Path(env)
    from download_checkpoints_colab import COLAB_CHECKPOINT_ROOT

    return Path(COLAB_CHECKPOINT_ROOT)


def _ensure_models_if_needed(shared: dict[str, Any], ckpt_root: Path) -> None:
    if not bool(shared.get("download_models_in_colab", True)):
        return
    from download_checkpoints_colab import ensure_checkpoints

    _log.info("Download/check models → %s", ckpt_root)
    ensure_checkpoints(
        ckpt_root,
        sapiens_model=str(shared.get("sapiens_model", "1b")),
        run_matting=bool(shared.get("run_matting", True)),
        run_normal=bool(shared.get("run_normal", True)),
    )


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


def _sync_exr_folder(local_dir: Path, drive_dir: Path, label: str) -> int:
    """Copy local EXRs to Drive (VDA-style save at end of pass)."""
    local_dir = Path(local_dir)
    drive_dir = Path(drive_dir)
    drive_dir.mkdir(parents=True, exist_ok=True)
    if not local_dir.is_dir():
        return 0
    n = 0
    for src in sorted(local_dir.glob("*.exr")):
        dest = drive_dir / src.name
        if dest.is_file() and dest.stat().st_size == src.stat().st_size:
            continue
        shutil.copy2(src, dest)
        n += 1
    _log.info("[SAVE] %s → Drive %s (%d file(s) copied)", label, drive_dir, n)
    return n


def _preflight_plate(plate_dir: Path, pattern: str, frame_start: int) -> None:
    first = pattern_to_path(plate_dir, pattern, frame_start)
    if not first.is_file():
        raise FileNotFoundError(f"Plate not found on Drive: {first}")
    _log.info("Plate OK (Drive read): %s", first)


def _run_shot(
    seq: dict[str, Any],
    shared: dict[str, Any],
    ckpt_root: Path,
    *,
    pass_name: str = "all",
) -> None:
    shot = str(seq.get("shot_name") or seq.get("name") or "shot")
    plate_dir = _resolve_on_drive(str(seq["plate_dir"]))
    pattern = str(seq.get("plate_pattern") or "%06d.exr")
    frame_start = int(seq["frame_start"])
    frame_end = int(seq["frame_end"])
    n_frames = _frame_count(frame_start, frame_end)

    local_shot = _local_shot_dir(shot)
    drive_shot = _drive_shot_dir(shared, shot)
    cache_dir = local_shot / "_plate_jpeg_cache"
    matte_local = local_shot / "matte"
    normal_local = local_shot / "normal"
    matte_drive = drive_shot / "matte"
    normal_drive = drive_shot / "normal"

    model_key = str(shared.get("sapiens_model", "1b")).lower().replace("sapiens2_", "")
    cfg = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS["1b"])

    _log.info("=== Shot %s frames %d-%d pass=%s ===", shot, frame_start, frame_end, pass_name)
    _log.info("Local work: %s", local_shot.resolve())
    _log.info("Drive save: %s", drive_shot.resolve())
    _preflight_plate(plate_dir, pattern, frame_start)

    run_matting = bool(shared.get("run_matting", True)) and pass_name in ("matting", "all")
    run_normal = bool(shared.get("run_normal", True)) and pass_name in ("normal", "all")
    export_matte = bool(shared.get("export_matte_exr", True))
    export_normal = bool(shared.get("export_normal_exr", True))

    # Resume from Drive (persistent)
    matte_done = _exr_count(matte_drive) >= n_frames if export_matte else True
    normal_done = _exr_count(normal_drive) >= n_frames if export_normal else True

    local_shot.mkdir(parents=True, exist_ok=True)

    if bool(shared.get("use_plate_jpeg_cache", True)):
        if not plate_cache_complete(cache_dir, frame_start, frame_end):
            _log.info("EXR→JPEG on local disk (read plates once from Drive)...")
            build_plate_jpeg_cache(
                plate_dir,
                pattern,
                frame_start,
                frame_end,
                cache_dir,
                jpeg_quality=int(shared.get("plate_jpeg_quality", 92)),
                workers=int(shared.get("plate_cache_workers", 12)),
            )
        else:
            _log.info("Local JPEG cache ready — %s", cache_dir)

    long_edge = int(shared.get("inference_long_edge", 0))
    max_mp = float(shared.get("inference_max_megapixels", 1.6))
    device = "cuda:0"
    try:
        import torch

        if not torch.cuda.is_available():
            _log.warning("CUDA not available — CPU fallback (very slow)")
            device = "cpu"
        else:
            p = torch.cuda.get_device_properties(0)
            _log.info("GPU: %s %.1f GB", p.name, p.total_memory / 1e9)
    except ImportError:
        device = "cpu"

    from sap2_infer import Sap2ShotProcessor, vram_auto_long_edge

    _log.info("Infer long_edge=%s auto=%d max_mp=%.1f", long_edge or "auto", vram_auto_long_edge(), max_mp)

    if run_matting and not matte_done:
        ckpt = ckpt_root / cfg["matting_ckpt"]
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing matting ckpt: {ckpt}")
        matte_local.mkdir(parents=True, exist_ok=True)
        proc = Sap2ShotProcessor(
            _DENSE, ckpt_root, model_key=model_key, device=device,
            inference_long_edge=long_edge, max_megapixels=max_mp,
        )
        try:
            done = 0
            for fi in range(frame_start, frame_end + 1):
                exr_out = matte_local / f"matte_{fi:06d}.exr"
                if exr_out.is_file():
                    done += 1
                    continue
                alpha = proc.process_frame_matting(cache_path(cache_dir, fi))
                write_alpha_exr(exr_out, alpha)
                done += 1
                if done == 1 or done % 10 == 0 or done == n_frames:
                    _log.info("Matte local %d/%d", done, n_frames)
        finally:
            proc.unload()
        if export_matte:
            _sync_exr_folder(matte_local, matte_drive, "matte")
    elif run_matting:
        _log.info("Skip matting — %d EXRs on Drive", _exr_count(matte_drive))

    if run_normal and not normal_done:
        ckpt = ckpt_root / cfg["normal_ckpt"]
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing normal ckpt: {ckpt}")
        normal_local.mkdir(parents=True, exist_ok=True)
        proc = Sap2ShotProcessor(
            _DENSE, ckpt_root, model_key=model_key, device=device,
            inference_long_edge=long_edge, max_megapixels=max_mp,
        )
        try:
            done = 0
            for fi in range(frame_start, frame_end + 1):
                exr_out = normal_local / f"normal_{fi:06d}.exr"
                if exr_out.is_file():
                    done += 1
                    continue
                normal = proc.process_frame_normal(cache_path(cache_dir, fi))
                write_normal_exr(exr_out, normal)
                done += 1
                if done == 1 or done % 10 == 0 or done == n_frames:
                    _log.info("Normal local %d/%d", done, n_frames)
        finally:
            proc.unload()
        if export_normal:
            _sync_exr_folder(normal_local, normal_drive, "normal")
    elif run_normal:
        _log.info("Skip normal — %d EXRs on Drive", _exr_count(normal_drive))

    if export_matte and run_matting and _exr_count(matte_drive) < n_frames:
        raise RuntimeError(f"Incomplete matte on Drive: {_exr_count(matte_drive)}/{n_frames}")
    if export_normal and run_normal and _exr_count(normal_drive) < n_frames:
        raise RuntimeError(f"Incomplete normal on Drive: {_exr_count(normal_drive)}/{n_frames}")

    _log.info("[OK] Shot %s — results on Drive: %s", shot, drive_shot)


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("SAP2_DRIVE_MOUNT", "/content/drive/MyDrive")
    env.setdefault("SAP2_LOCAL_WORK", "/content/sap2_work")
    return env


def _enabled_passes(shared: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if bool(shared.get("run_matting", True)):
        out.append("matting")
    if bool(shared.get("run_normal", True)):
        out.append("normal")
    return out


def _run_batch(job_path: Path, shot_index: int | None, pass_name: str) -> int:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    shared = _merge_shared(job)
    sequences = list(job.get("sequences") or [])
    if not sequences:
        _log.error("No sequences")
        return 1

    _local_root().mkdir(parents=True, exist_ok=True)
    ckpt_root = _checkpoint_root(shared)
    if pass_name == "all":
        _ensure_models_if_needed(shared, ckpt_root)
    os.environ["SAPIENS_CHECKPOINT_ROOT"] = str(ckpt_root)

    _log.info("SAP2 local work → %s", _local_root().resolve())
    _log.info("SAP2 Drive output → %s", _drive_output_root(shared).resolve())
    _log.info("Checkpoints → %s", ckpt_root)

    if shot_index is not None:
        sequences = [sequences[shot_index]]

    failures: list[str] = []
    for i, seq in enumerate(sequences):
        shot = str(seq.get("shot_name") or f"seq_{i}")
        try:
            _run_shot(seq, shared, ckpt_root, pass_name=pass_name)
        except Exception as exc:
            _log.error("[ERROR] %s: %s", shot, exc)
            _log.error(traceback.format_exc())
            failures.append(shot)
    return 1 if failures else 0


def _spawn_subprocess(job_path: Path, shot_index: int, pass_name: str) -> int:
    cmd = [
        sys.executable, "-u", str(Path(__file__).resolve()),
        "--job-json", str(job_path), "--shot-index", str(shot_index), "--pass", pass_name,
    ]
    _log.info("Subprocess: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(_SAP2_ROOT), env=_subprocess_env(), check=False).returncode


def _run_orchestrated(job_path: Path) -> int:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    shared = _merge_shared(job)
    sequences = job.get("sequences") or []
    gap = int(shared.get("batch_gap_seconds", 5))
    failures: list[str] = []
    for idx in range(len(sequences)):
        shot = str(sequences[idx].get("shot_name") or f"seq_{idx}")
        for pn in _enabled_passes(shared):
            _log.info("--- %s / %s ---", shot, pn)
            if _spawn_subprocess(job_path, idx, pn) != 0:
                failures.append(f"{shot}:{pn}")
            if gap > 0:
                time.sleep(gap)
    return 1 if failures else 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", force=True)
    args = _parse_args()
    job_path = Path(args.job_json).resolve()
    if not job_path.is_file():
        _log.error("Missing job json: %s", job_path)
        return 1

    job = json.loads(job_path.read_text(encoding="utf-8"))
    shared = _merge_shared(job)
    n_seq = len(job.get("sequences") or [])

    if args.pass_name != "all":
        return _run_batch(job_path, args.shot_index, args.pass_name)
    if bool(shared.get("split_pass_subprocess", True)) and args.shot_index is None:
        return _run_orchestrated(job_path)
    if bool(shared.get("batch_one_process_per_shot", True)) and args.shot_index is None and n_seq > 1:
        failures = []
        for idx in range(n_seq):
            if _spawn_subprocess(job_path, idx, "all") != 0:
                failures.append(str(job["sequences"][idx].get("shot_name")))
        return 1 if failures else 0
    return _run_batch(job_path, args.shot_index, "all")


if __name__ == "__main__":
    raise SystemExit(main())
