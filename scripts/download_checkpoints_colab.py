#!/usr/bin/env python3
"""Download Sapiens2 matting + normal checkpoints to Drive (run once in Colab)."""

from __future__ import annotations

import argparse
from pathlib import Path

HF_REPOS = {
    "matting_1b": ("facebook/sapiens2-matting-1b", "matting/sapiens2_1b_matting.safetensors"),
    "normal_1b": ("facebook/sapiens2-normal-1b", "normal/sapiens2_1b_normal.safetensors"),
    "normal_0.4b": ("facebook/sapiens2-normal-0.4b", "normal/sapiens2_0.4b_normal.safetensors"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="/content/drive/MyDrive/VDA_models/sapiens2_host",
        help="Checkpoint root (SAPIENS_CHECKPOINT_ROOT)",
    )
    p.add_argument(
        "--which",
        nargs="+",
        default=["matting_1b", "normal_1b"],
        choices=list(HF_REPOS.keys()),
    )
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    for key in args.which:
        repo_id, rel = HF_REPOS[key]
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file():
            print(f"skip exists: {dest}")
            continue
        print(f"download {repo_id} → {dest}")
        cached = hf_hub_download(repo_id=repo_id, filename=dest.name)
        Path(cached).replace(dest)
        print(f"OK {dest}")


if __name__ == "__main__":
    main()
