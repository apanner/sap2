#!/usr/bin/env python3
"""SAP2 Colab dependency install (Python 3.12+, torch 2.7+, oiio)."""

from __future__ import annotations

import subprocess
import sys


PIP_PACKAGES = [
    "oiio-python>=2.5",
    "opencv-python-headless",
    "safetensors",
    "accelerate",
    "timm",
    "transformers",
    "huggingface_hub",
]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    if sys.version_info < (3, 12):
        print(
            f"WARNING: Sapiens2 requires Python >=3.12; current {sys.version}. "
            "Switch Colab runtime to Python 3.12 if install fails."
        )
    _run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    _run([sys.executable, "-m", "pip", "install", "-q"] + PIP_PACKAGES)
    from pathlib import Path

    sap2_root = Path(__file__).resolve().parent.parent
    _run([sys.executable, "-m", "pip", "install", "-q", "-e", str(sap2_root)])
    import OpenImageIO as oiio  # noqa: F401
    import torch  # noqa: F401

    print(f"OK — Python {sys.version.split()[0]} torch {torch.__version__} oiio ready", flush=True)


if __name__ == "__main__":
    main()
