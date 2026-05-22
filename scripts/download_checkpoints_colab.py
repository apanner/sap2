#!/usr/bin/env python3
"""Download Sapiens2 checkpoints (Colab local disk by default, not Drive)."""

from __future__ import annotations

import argparse
from pathlib import Path

# Colab session storage — models re-download each new runtime unless you cache elsewhere
COLAB_CHECKPOINT_ROOT = "/content/sapiens2_checkpoints"

# repo_id, local relative path, filename on HuggingFace hub
HF_REPOS = {
    "matting_1b": (
        "facebook/sapiens2-matting-1b",
        "matting/sapiens2_1b_matting.safetensors",
        "sapiens2_1b_matting.safetensors",
    ),
    "normal_1b": (
        "facebook/sapiens2-normal-1b",
        "normal/sapiens2_1b_normal.safetensors",
        "sapiens2_1b_normal.safetensors",
    ),
    "normal_0.4b": (
        "facebook/sapiens2-normal-0.4b",
        "normal/sapiens2_0.4b_normal.safetensors",
        "sapiens2_0.4b_normal.safetensors",
    ),
    "normal_0.8b": (
        "facebook/sapiens2-normal-0.8b",
        "normal/sapiens2_0.8b_normal.safetensors",
        "sapiens2_0.8b_normal.safetensors",
    ),
    "normal_5b": (
        "facebook/sapiens2-normal-5b",
        "normal/sapiens2_5b_normal.safetensors",
        "sapiens2_5b_normal.safetensors",
    ),
    "seg_1b": (
        "facebook/sapiens2-seg-1b",
        "seg/sapiens2_1b_seg.safetensors",
        "sapiens2_1b_seg.safetensors",
    ),
    "seg_0.4b": (
        "facebook/sapiens2-seg-0.4b",
        "seg/sapiens2_0.4b_seg.safetensors",
        "sapiens2_0.4b_seg.safetensors",
    ),
    "seg_0.8b": (
        "facebook/sapiens2-seg-0.8b",
        "seg/sapiens2_0.8b_seg.safetensors",
        "sapiens2_0.8b_seg.safetensors",
    ),
    "seg_5b": (
        "facebook/sapiens2-seg-5b",
        "seg/sapiens2_5b_seg.safetensors",
        "sapiens2_5b_seg.safetensors",
    ),
}


def which_for_job(
    *,
    sapiens_model: str = "1b",
    run_matting: bool = True,
    run_normal: bool = True,
    matte_output_mode: str = "segmentation",
) -> list[str]:
    model = str(sapiens_model).lower().replace("sapiens2_", "")
    keys: list[str] = []
    mom = str(matte_output_mode or "segmentation").strip().lower()
    need_seg = run_matting and mom in ("segmentation", "both")
    need_alpha = run_matting and mom in ("alpha", "both")
    if need_seg:
        sk = f"seg_{model}"
        if sk not in HF_REPOS:
            raise ValueError(f"No seg checkpoint mapping for model {model}")
        keys.append(sk)
    if need_alpha:
        if model != "1b":
            raise ValueError("Human matting alpha requires sapiens_model=1b")
        keys.append("matting_1b")
    if run_normal:
        nk = f"normal_{model}"
        if nk not in HF_REPOS:
            raise ValueError(f"No normal checkpoint mapping for model {model}")
        keys.append(nk)
    return keys


def download_checkpoints(root: Path, which: list[str]) -> None:
    from huggingface_hub import hf_hub_download
    import shutil

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for key in which:
        repo_id, rel, hf_name = HF_REPOS[key]
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and dest.stat().st_size > 1_000_000:
            print(f"[skip] {dest}")
            continue
        print(f"[download] {repo_id}/{hf_name} → {dest}")
        cached = hf_hub_download(repo_id=repo_id, filename=hf_name)
        shutil.copy2(cached, dest)
        print(f"[ok] {dest} ({dest.stat().st_size / 1e9:.2f} GB)")


def ensure_checkpoints(
    root: str | Path | None = None,
    *,
    sapiens_model: str = "1b",
    run_matting: bool = True,
    run_normal: bool = True,
    matte_output_mode: str = "segmentation",
) -> Path:
    """Download missing weights before inference. Returns checkpoint root path."""
    ckpt_root = Path(root or COLAB_CHECKPOINT_ROOT)
    which = which_for_job(
        sapiens_model=sapiens_model,
        run_matting=run_matting,
        run_normal=run_normal,
        matte_output_mode=matte_output_mode,
    )
    download_checkpoints(ckpt_root, which)
    return ckpt_root


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default=COLAB_CHECKPOINT_ROOT,
        help="Checkpoint root (default: Colab local /content/sapiens2_checkpoints)",
    )
    p.add_argument(
        "--which",
        nargs="+",
        default=None,
        choices=list(HF_REPOS.keys()),
    )
    p.add_argument("--sapiens-model", default="1b")
    p.add_argument("--no-matting", action="store_true")
    p.add_argument("--no-normal", action="store_true")
    args = p.parse_args()
    which = args.which or which_for_job(
        sapiens_model=args.sapiens_model,
        run_matting=not args.no_matting,
        run_normal=not args.no_normal,
    )
    download_checkpoints(Path(args.root), which)


if __name__ == "__main__":
    main()
