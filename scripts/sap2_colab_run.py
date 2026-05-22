#!/usr/bin/env python3
"""SAP2 Colab batch — same I/O as VDA: /content/output local, copy to Drive when done."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
_SAP2_ROOT = _SCRIPTS.parent
_DENSE = _SAP2_ROOT / "sapiens" / "dense"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from exr_io import (  # noqa: E402
    read_exr_pixels,
    verify_exr_nonzero,
    write_matte_alpha_exr,
    write_matte_exr,
    write_matte_premult_exr,
    write_matte_seg_rgba_exr,
    write_matte_subject_channels_exr,
    write_normal_exr,
    write_exr_float,
    write_seg_color_exr,
    write_seg_id_exr,
)
from seg_export import labels_to_human_alpha, pack_subject_alpha_channels  # noqa: E402
from plate_cache import build_plate_jpeg_cache, cache_path, plate_cache_complete, pattern_to_path  # noqa: E402

_log = logging.getLogger("sap2_colab_run")

# VDA uses LOCAL_OUTPUT_PATH = '/content/output' — match exactly
LOCAL_OUTPUT_PATH = "/content/output"

SAP2_DEFAULTS: dict[str, Any] = {
    "sapiens_model": "1b",
    "run_matting": True,
    "run_normal": True,
    "export_matte_exr": True,
    "export_normal_exr": True,
    "use_plate_jpeg_cache": True,
    "plate_jpeg_quality": 92,
    "plate_cache_workers": 12,
    "qc_mp4": True,
    "qc_fps": 24.0,
    "qc_max_long_edge": 1920,
    "batch_one_process_per_shot": True,
    "split_pass_subprocess": False,
    "image_feed_mode": "auto",
    "matte_output_mode": "segmentation",
    "matte_image_feed_mode": "full_res",
    "export_matte_seg_id_exr": True,
    "export_matte_alpha_exr": True,
    "export_matte_premult_exr": False,
    "batch_gap_seconds": 5,
    "checkpoint_root": "",
    "download_models_in_colab": True,
    "inference_long_edge": 0,
    "inference_max_megapixels": 0,
    "use_person_crop": True,
    "person_crop_pad": 0.18,
    "person_crop_confidence": 0.4,
    "person_crop_smooth": 0.72,
    "person_crop_multi": True,
    "person_crop_mode": "union",
    "matte_subject_layout": "combined",
    "matte_max_subjects": 4,
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


def _collapse_drive_path(p: str) -> str:
    """Fix paths broken by strip('/') or repeated Cell2 prefixing."""
    p = str(p or "").strip().replace("\\", "/")
    if not p:
        return ""
    marker = "/MyDrive/"
    if marker in p:
        tail = p.split(marker)[-1]
        return f"/content/drive/MyDrive/{tail.lstrip('/')}"
    if p.startswith("content/drive/MyDrive/"):
        return "/" + p
    return p


def _resolve_on_drive(rel_or_abs: str) -> Path:
    p = _collapse_drive_path(rel_or_abs)
    if not p:
        return _drive_mount()
    if p.startswith("/content/drive"):
        return Path(p)
    if p.startswith("MyDrive/"):
        return _drive_mount() / p[len("MyDrive/") :]
    drive = _drive_mount()
    drive_s = str(drive).replace("\\", "/")
    if p.startswith(drive_s):
        return Path(p)
    return drive / p.lstrip("/")


def _output_folder_from_shared(shared: dict[str, Any]) -> str:
    raw = str(shared.get("output_folder_path") or "VDA_output").strip().replace("\\", "/")
    if raw.startswith("/content/drive") or raw.startswith("MyDrive/"):
        return _collapse_drive_path(raw)
    return raw.strip("/") or "VDA_output"


def _local_output_root() -> Path:
    return Path(os.environ.get("SAP2_LOCAL_OUTPUT", LOCAL_OUTPUT_PATH))


def _drive_output_root(shared: dict[str, Any]) -> Path:
    root_name = str(shared.get("sap2_output_root") or "SAP2_output")
    return _resolve_on_drive(_output_folder_from_shared(shared)) / _date_folder() / root_name


def _local_shot_dir(shot: str) -> Path:
    return _local_output_root() / shot


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


def _ensure_models_if_needed(
    shared: dict[str, Any],
    ckpt_root: Path,
    *,
    pass_name: str = "all",
) -> None:
    if not bool(shared.get("download_models_in_colab", True)):
        return
    from download_checkpoints_colab import ensure_checkpoints

    run_matting = bool(shared.get("run_matting", True)) and pass_name in ("matting", "all")
    run_normal = bool(shared.get("run_normal", True)) and pass_name in ("normal", "all")
    _log.info("Download/check models → %s (pass=%s)", ckpt_root, pass_name)
    ensure_checkpoints(
        ckpt_root,
        sapiens_model=str(shared.get("sapiens_model", "1b")),
        run_matting=run_matting,
        run_normal=run_normal,
        matte_output_mode=str(shared.get("matte_output_mode", "segmentation")),
    )


def _merge_shared(job: dict[str, Any]) -> dict[str, Any]:
    out = dict(SAP2_DEFAULTS)
    out.update(job.get("shared") or {})
    out.update(job.get("shared_settings") or {})
    return out


def _frame_count(frame_start: int, frame_end: int) -> int:
    return max(0, frame_end - frame_start + 1)


def _exr_is_empty(path: Path) -> bool:
    """True if EXR reads as all ~zero (broken ImageBuf-era writes)."""
    if not path.is_file():
        return True
    try:
        arr, _ = read_exr_pixels(path)
        return float(np.max(np.abs(arr))) < 1e-5
    except Exception:
        return True


def _matte_exr_outdated(path: Path) -> bool:
    """True if matte EXR is RGB-only (pre-alpha delivery) or old Z-only id file."""
    if not path.is_file():
        return True
    try:
        arr, names = read_exr_pixels(path)
        if path.name.startswith("matte_id_"):
            nl = {n.lower() for n in names}
            has_layers = "class_id.red" in nl or any(
                n.startswith("class_id.") for n in nl
            )
            return not has_layers
        if path.name.startswith("matte_") and "premult" not in path.name.lower():
            return "A" not in names and arr.shape[-1] < 4
        return False
    except Exception:
        return True


def _write_seg_matte_bundle(
    matte_local: Path,
    fi: int,
    labels: np.ndarray,
    color: np.ndarray,
    shared: dict[str, Any],
    *,
    exr_out: Path,
    person_masks: list[np.ndarray] | None = None,
) -> None:
    human_alpha = labels_to_human_alpha(labels)
    write_matte_seg_rgba_exr(exr_out, color, human_alpha)
    if bool(shared.get("export_matte_alpha_exr", True)):
        write_matte_alpha_exr(matte_local / f"matte_alpha_{fi:06d}.exr", human_alpha)
    if bool(shared.get("export_matte_seg_id_exr", True)):
        write_seg_id_exr(
            matte_local / f"matte_id_{fi:06d}.exr",
            labels,
            person_masks=person_masks,
            max_people=int(shared.get("matte_max_subjects", 4)),
        )
    if person_masks:
        packed, ch_names = pack_subject_alpha_channels(person_masks)
        write_exr_float(
            matte_local / f"matte_people_{fi:06d}.exr",
            packed,
            channels=packed.shape[-1],
            channel_names=ch_names,
        )


def _is_primary_matte_exr(path: Path) -> bool:
    n = path.name.lower()
    if "premult" in n or "seg_id" in n or n.startswith("matte_id"):
        return False
    return bool(re.match(r"^matte_\d+\.exr$", path.name, re.IGNORECASE))


def _resolve_matte_modes(shared: dict[str, Any]) -> tuple[str, bool, bool]:
    """Returns (layout, run_segmentation, run_alpha)."""
    mode = str(shared.get("matte_output_mode") or "segmentation").strip().lower()
    if mode not in ("segmentation", "alpha", "both"):
        mode = "segmentation"
    layout = str(shared.get("matte_subject_layout") or "combined").strip().lower()
    if layout not in ("combined", "channels", "separate"):
        layout = "combined"
    return layout, mode in ("segmentation", "both"), mode in ("alpha", "both")


def _exr_count(folder: Path, *, recursive: bool = False, matte_primary_only: bool = False) -> int:
    if not folder.is_dir():
        return 0
    if recursive:
        paths = folder.rglob("*.exr")
    else:
        paths = folder.glob("*.exr")
    if matte_primary_only:
        return sum(1 for p in paths if _is_primary_matte_exr(p))
    return sum(1 for p in paths if "premult" not in p.stem.lower())


def _matte_layout(shared: dict[str, Any]) -> str:
    layout = str(shared.get("matte_subject_layout") or "combined").strip().lower()
    return layout if layout in ("combined", "channels", "separate") else "combined"


def _matte_complete(matte_dir: Path, n_frames: int, layout: str) -> bool:
    if not matte_dir.is_dir():
        return False
    if layout == "separate":
        subdirs = sorted(
            d for d in matte_dir.iterdir() if d.is_dir() and d.name.startswith("p")
        )
        if subdirs:
            return any(
                _exr_count(d, matte_primary_only=True) >= n_frames for d in subdirs
            )
        return _exr_count(matte_dir, matte_primary_only=True) >= n_frames
    return _exr_count(matte_dir, matte_primary_only=True) >= n_frames


def _pass_done_local_or_drive(
    local_dir: Path,
    drive_dir: Path,
    n_frames: int,
    *,
    is_matte: bool = False,
    matte_layout: str = "combined",
) -> bool:
    if is_matte:
        if _matte_complete(local_dir, n_frames, matte_layout):
            return True
        return _matte_complete(drive_dir, n_frames, matte_layout)
    if _exr_count(local_dir) >= n_frames:
        return True
    return _exr_count(drive_dir) >= n_frames


def _copy_exr_folder_to_drive(local_dir: Path, drive_dir: Path, label: str) -> int:
    """Copy EXRs (top-level + p00/… subdirs) local → Drive."""
    local_dir = Path(local_dir)
    drive_dir = Path(drive_dir)
    if not local_dir.is_dir():
        _log.warning("[SAVE] No local %s dir: %s", label, local_dir)
        return 0
    drive_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(local_dir.rglob("*.exr")):
        rel = src.relative_to(local_dir)
        dest = drive_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        n += 1
    if n:
        _log.info("[OK] Saved %d %s EXR → %s", n, label, drive_dir)
    else:
        _log.warning("[SAVE] 0 %s EXR copied from %s", label, local_dir)
    return n


def _copy_qc_to_drive(local_shot: Path, drive_shot: Path) -> int:
    local_qc = Path(local_shot) / "qc"
    if not local_qc.is_dir():
        return 0
    drive_qc = Path(drive_shot) / "qc"
    drive_qc.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(local_qc.glob("*.mp4")):
        shutil.copy2(src, drive_qc / src.name)
        n += 1
    if n:
        _log.info("[OK] Saved %d QC MP4 → %s", n, drive_qc)
    return n


def _save_shot_to_drive(
    shot: str,
    shared: dict[str, Any],
    *,
    export_matte: bool,
    export_normal: bool,
) -> None:
    local_shot = _local_shot_dir(shot)
    drive_shot = _drive_shot_dir(shared, shot)
    drive_shot.mkdir(parents=True, exist_ok=True)
    if export_matte:
        _copy_exr_folder_to_drive(local_shot / "matte", drive_shot / "matte", "matte")
    if export_normal:
        _copy_exr_folder_to_drive(local_shot / "normal", drive_shot / "normal", "normal")
    _copy_qc_to_drive(local_shot, drive_shot)


def _is_last_pass(pass_name: str, shared: dict[str, Any]) -> bool:
    if pass_name == "all":
        return True
    passes = _enabled_passes(shared)
    return bool(passes) and pass_name == passes[-1]


def _preflight_plate(plate_dir: Path, pattern: str, frame_start: int) -> None:
    first = pattern_to_path(plate_dir, pattern, frame_start)
    if not first.is_file():
        raise FileNotFoundError(f"Plate not found on Drive: {first}")
    _log.info("Plate OK (read from Drive): %s", first)


def _run_shot(
    seq: dict[str, Any],
    shared: dict[str, Any],
    ckpt_root: Path,
    *,
    pass_name: str = "all",
    save_to_drive: bool = False,
) -> str:
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
    _log.info("Local output: %s", local_shot.resolve())
    _log.info("Drive deliverable (after done): %s", drive_shot.resolve())
    _preflight_plate(plate_dir, pattern, frame_start)

    run_matting = bool(shared.get("run_matting", True)) and pass_name in ("matting", "all")
    run_normal = bool(shared.get("run_normal", True)) and pass_name in ("normal", "all")
    export_matte = bool(shared.get("export_matte_exr", True))
    export_normal = bool(shared.get("export_normal_exr", True))

    matte_layout = _matte_layout(shared)
    matte_done = (
        _pass_done_local_or_drive(
            matte_local, matte_drive, n_frames, is_matte=True, matte_layout=matte_layout
        )
        if export_matte
        else True
    )
    normal_done = _pass_done_local_or_drive(normal_local, normal_drive, n_frames) if export_normal else True

    local_shot.mkdir(parents=True, exist_ok=True)
    _local_output_root().mkdir(parents=True, exist_ok=True)

    if bool(shared.get("use_plate_jpeg_cache", True)):
        if not plate_cache_complete(cache_dir, frame_start, frame_end):
            _log.info("EXR→JPEG cache on %s (read plates from Drive)...", LOCAL_OUTPUT_PATH)
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
            _log.info("JPEG cache ready: %s", cache_dir)

    feed_mode = str(shared.get("image_feed_mode") or "auto").strip().lower()
    if feed_mode not in ("auto", "full_res", "person_crop"):
        feed_mode = "auto"
    matte_feed = str(shared.get("matte_image_feed_mode") or "full_res").strip().lower()
    if matte_feed not in ("auto", "full_res", "person_crop"):
        matte_feed = "full_res"
    matte_out_mode = str(shared.get("matte_output_mode") or "segmentation").strip().lower()
    proc_kw = dict(
        image_feed_mode=feed_mode,
        matte_image_feed_mode=matte_feed,
        matte_output_mode=matte_out_mode,
        use_person_crop=bool(shared.get("use_person_crop", True)),
        person_crop_pad=float(shared.get("person_crop_pad", 0.22)),
        person_crop_confidence=float(shared.get("person_crop_confidence", 0.28)),
        person_crop_smooth=float(shared.get("person_crop_smooth", 0.72)),
        person_crop_multi=bool(shared.get("person_crop_multi", True)),
        person_crop_mode=str(shared.get("person_crop_mode") or "union"),
        matte_subject_layout=str(shared.get("matte_subject_layout") or "combined"),
        matte_max_subjects=int(shared.get("matte_max_subjects", 4)),
    )
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

    from sap2_infer import MODEL_NATIVE_H, MODEL_NATIVE_W, Sap2ShotProcessor

    _log.info(
        "Sapiens2: full-res plate → %d×%d infer → full-res EXR (feed=%s)",
        MODEL_NATIVE_H,
        MODEL_NATIVE_W,
        feed_mode,
    )

    need_proc = (run_matting and not matte_done) or (run_normal and not normal_done)
    proc: Sap2ShotProcessor | None = None
    if need_proc:
        proc = Sap2ShotProcessor(
            _DENSE, ckpt_root, model_key=model_key, device=device, **proc_kw
        )

    try:
        if run_matting and not matte_done:
            matte_layout, run_seg, run_alpha = _resolve_matte_modes(shared)
            if run_seg:
                seg_ckpt = ckpt_root / cfg.get("seg_ckpt", "")
                if not seg_ckpt.is_file():
                    raise FileNotFoundError(
                        f"Missing seg ckpt for body-part matte: {seg_ckpt}"
                    )
            if run_alpha:
                ckpt = ckpt_root / cfg["matting_ckpt"]
                if not ckpt.is_file():
                    raise FileNotFoundError(f"Missing matting ckpt: {ckpt}")
            matte_local.mkdir(parents=True, exist_ok=True)
            assert proc is not None
            _log.info(
                "Matte pass: mode=%s layout=%s (seg=%s alpha=%s)",
                matte_out_mode,
                matte_layout,
                run_seg,
                run_alpha,
            )
            done = 0
            verified_matte = False
            for fi in range(frame_start, frame_end + 1):
                exr_out = matte_local / f"matte_{fi:06d}.exr"
                if (
                    exr_out.is_file()
                    and not _exr_is_empty(exr_out)
                    and not _matte_exr_outdated(exr_out)
                ):
                    done += 1
                    continue
                if exr_out.is_file():
                    _log.warning("Re-export matte EXR (missing alpha / old format): %s", exr_out.name)

                if run_seg:
                    if matte_layout == "channels":
                        labels, color, person_masks = proc.process_frame_segmentation_subjects(
                            cache_path(cache_dir, fi)
                        )
                        _write_seg_matte_bundle(
                            matte_local,
                            fi,
                            labels,
                            color,
                            shared,
                            exr_out=exr_out,
                            person_masks=person_masks,
                        )
                    elif matte_layout == "separate":
                        labels, color, person_masks = proc.process_frame_segmentation_subjects(
                            cache_path(cache_dir, fi)
                        )
                        _write_seg_matte_bundle(
                            matte_local, fi, labels, color, shared, exr_out=exr_out
                        )
                        for si, mask in enumerate(person_masks):
                            sub_dir = matte_local / f"p{si:02d}"
                            sub_dir.mkdir(parents=True, exist_ok=True)
                            write_matte_alpha_exr(sub_dir / f"matte_{fi:06d}.exr", mask)
                    else:
                        labels, color, person_masks = proc.process_frame_segmentation_subjects(
                            cache_path(cache_dir, fi)
                        )
                        _write_seg_matte_bundle(
                            matte_local,
                            fi,
                            labels,
                            color,
                            shared,
                            exr_out=exr_out,
                            person_masks=person_masks if len(person_masks) > 1 else None,
                        )

                if run_alpha:
                    alpha_path = matte_local / f"matte_alpha_{fi:06d}.exr"
                    if matte_layout == "combined":
                        matte = proc.process_frame_matting(cache_path(cache_dir, fi))
                        if not run_seg:
                            write_matte_alpha_exr(exr_out, matte)
                        else:
                            write_matte_alpha_exr(alpha_path, matte)
                        if bool(shared.get("export_matte_premult_exr", False)):
                            write_matte_premult_exr(
                                matte_local / f"matte_premult_{fi:06d}.exr", matte
                            )
                    elif matte_layout == "channels":
                        subjects = proc.process_frame_matting_subjects(
                            cache_path(cache_dir, fi)
                        )
                        if not run_seg:
                            write_matte_subject_channels_exr(exr_out, subjects)
                        else:
                            packed = np.zeros(
                                (
                                    subjects[0].shape[0],
                                    subjects[0].shape[1],
                                    4,
                                ),
                                dtype=np.float32,
                            )
                            for i, subj in enumerate(subjects[:4]):
                                a = subj[:, :, 3] if subj.shape[-1] == 4 else subj
                                packed[:, :, i] = np.clip(a, 0.0, 1.0)
                            write_exr_float(
                                alpha_path,
                                packed,
                                channels=4,
                                channel_names=("R", "G", "B", "A"),
                            )
                    else:
                        subjects = proc.process_frame_matting_subjects(
                            cache_path(cache_dir, fi)
                        )
                        for si, subj in enumerate(subjects):
                            sub_dir = matte_local / f"p{si:02d}"
                            sub_dir.mkdir(parents=True, exist_ok=True)
                            write_matte_exr(sub_dir / f"matte_{fi:06d}.exr", subj)

                if not verified_matte:
                    verify_exr_nonzero(exr_out, label="matte")
                    verified_matte = True
                done += 1
                if done == 1 or done % 10 == 0 or done == n_frames:
                    _log.info("Matte → local %d/%d", done, n_frames)
        elif run_matting:
            _log.info("Skip matting (local=%d drive=%d)", _exr_count(matte_local), _exr_count(matte_drive))

        if run_normal and not normal_done:
            ckpt = ckpt_root / cfg["normal_ckpt"]
            if not ckpt.is_file():
                raise FileNotFoundError(f"Missing normal ckpt: {ckpt}")
            normal_local.mkdir(parents=True, exist_ok=True)
            assert proc is not None
            done = 0
            verified_normal = False
            for fi in range(frame_start, frame_end + 1):
                exr_out = normal_local / f"normal_{fi:06d}.exr"
                if exr_out.is_file() and not _exr_is_empty(exr_out):
                    done += 1
                    continue
                if exr_out.is_file():
                    _log.warning("Re-export empty normal EXR: %s", exr_out.name)
                normal = proc.process_frame_normal(cache_path(cache_dir, fi))
                write_normal_exr(exr_out, normal)
                if not verified_normal:
                    verify_exr_nonzero(exr_out, label="normal")
                    verified_normal = True
                done += 1
                if done == 1 or done % 10 == 0 or done == n_frames:
                    _log.info("Normal → local %d/%d", done, n_frames)
        elif run_normal:
            _log.info("Skip normal (local=%d drive=%d)", _exr_count(normal_local), _exr_count(normal_drive))
    finally:
        if proc is not None:
            proc.unload()

    if export_matte and run_matting and not _matte_complete(matte_local, n_frames, matte_layout):
        raise RuntimeError(
            f"Incomplete matte on local (layout={matte_layout}): "
            f"{_exr_count(matte_local, recursive=True, matte_primary_only=True)}/{n_frames}"
        )
    if export_normal and run_normal and _exr_count(normal_local) < n_frames:
        raise RuntimeError(f"Incomplete normal on local: {_exr_count(normal_local)}/{n_frames}")

    if bool(shared.get("qc_mp4", True)):
        try:
            from sap2_qc_mp4 import export_sap2_qc_mp4s

            fps = float(shared.get("qc_fps") or seq.get("fps") or shared.get("fps") or 24.0)
            matte_layout = str(shared.get("matte_subject_layout") or "combined").strip().lower()
            _log.info("%s: QC MP4 export (fps=%.3f, layout=%s)...", shot, fps, matte_layout)
            written = export_sap2_qc_mp4s(
                local_shot,
                plate_cache_dir=cache_dir if cache_dir.is_dir() else None,
                frame_start=frame_start,
                frame_end=frame_end,
                fps=fps,
                matte_subject_layout=matte_layout,
                matte_output_mode=str(shared.get("matte_output_mode", "segmentation")),
                export_matte=export_matte and run_matting,
                export_normal=export_normal and run_normal,
                qc_max_long_edge=int(shared.get("qc_max_long_edge", 1920)),
            )
            if not written:
                _log.warning("%s: QC MP4 — no previews written (missing EXR?)", shot)
        except Exception as exc:
            _log.warning("QC MP4 export failed for %s: %s", shot, exc)

    if save_to_drive:
        _log.info("[SAVE] Copying %s → Drive (%s)...", shot, drive_shot)
        try:
            _save_shot_to_drive(shot, shared, export_matte=export_matte, export_normal=export_normal)
        except Exception as exc:
            raise RuntimeError(f"Drive copy failed for {shot}: {exc}") from exc

    if save_to_drive:
        if export_matte and not _matte_complete(matte_drive, n_frames, matte_layout):
            raise RuntimeError(
                f"Incomplete matte on Drive (layout={matte_layout}): "
                f"{_exr_count(matte_drive, recursive=True, matte_primary_only=True)}/{n_frames}"
            )
        if export_normal and _exr_count(normal_drive) < n_frames:
            raise RuntimeError(f"Incomplete normal on Drive: {_exr_count(normal_drive)}/{n_frames}")
        _log.info("[OK] Shot %s on Drive: %s", shot, drive_shot)
    else:
        _log.info("[OK] Shot %s on local %s (Drive copy pending)", shot, local_shot)

    return shot


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("SAP2_DRIVE_MOUNT", "/content/drive/MyDrive")
    env.setdefault("SAP2_LOCAL_OUTPUT", LOCAL_OUTPUT_PATH)
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

    _local_output_root().mkdir(parents=True, exist_ok=True)
    ckpt_root = _checkpoint_root(shared)
    _ensure_models_if_needed(shared, ckpt_root, pass_name=pass_name)
    if bool(shared.get("use_person_crop", True)):
        _log.info(
            "Person crop: try MobileNet-SSD; if download/load fails → OpenCV HOG (built-in)"
        )
    os.environ["SAPIENS_CHECKPOINT_ROOT"] = str(ckpt_root)

    _log.info("Local output (VDA): %s", _local_output_root().resolve())
    _log.info("Drive output: %s", _drive_output_root(shared).resolve())
    _log.info("Checkpoints: %s", ckpt_root)

    if shot_index is not None:
        sequences = [sequences[shot_index]]

    # VDA: process all shots locally first, then one [SAVE] block to Drive
    batch_save_at_end = pass_name == "all" and shot_index is None and len(sequences) > 0
    save_per_shot = _is_last_pass(pass_name, shared) and not batch_save_at_end

    failures: list[str] = []
    completed: list[str] = []
    for i, seq in enumerate(sequences):
        shot = str(seq.get("shot_name") or f"seq_{i}")
        try:
            _run_shot(
                seq, shared, ckpt_root,
                pass_name=pass_name,
                save_to_drive=save_per_shot,
            )
            completed.append(shot)
        except Exception as exc:
            _log.error("[ERROR] %s: %s", shot, exc)
            _log.error(traceback.format_exc())
            failures.append(shot)

    if batch_save_at_end and completed:
        _log.info("=" * 60)
        _log.info("[SAVE] Saving results to Drive...")
        _log.info("=" * 60)
        export_matte = bool(shared.get("export_matte_exr", True))
        export_normal = bool(shared.get("export_normal_exr", True))
        for shot in completed:
            try:
                _save_shot_to_drive(
                    shot, shared,
                    export_matte=export_matte,
                    export_normal=export_normal,
                )
            except Exception as exc:
                _log.error("[ERROR] Drive save %s: %s", shot, exc)
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
    passes = _enabled_passes(shared)
    failures: list[str] = []
    for idx in range(len(sequences)):
        shot = str(sequences[idx].get("shot_name") or f"seq_{idx}")
        for pn in passes:
            _log.info("--- %s / %s ---", shot, pn)
            rc = _spawn_subprocess(job_path, idx, pn)
            if rc != 0:
                hint = ""
                if rc < 0 or rc in (134, 139):
                    hint = " (likely GPU crash/OOM — try inference_max_megapixels=1.2)"
                _log.error("Subprocess exit code %s%s", rc, hint)
                failures.append(f"{shot}:{pn}")
                break
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
