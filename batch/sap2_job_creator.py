"""Create per-shot job JSON for launch_sap2_engine.bat."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict


def create_sap2_job_file(
    sequence: Dict[str, Any],
    output_base: str,
    shared: Dict[str, Any],
) -> str:
    shot = sequence.get("shot_name", "shot")
    folder = sequence.get("folder_path", "")
    base = sequence.get("base_name", "")
    sep = sequence.get("frame_separator", "_") or ""
    pad = int(sequence.get("frame_padding", 4))
    ext = str(sequence.get("format", "exr")).lower()
    fs = int(sequence.get("frame_start", 1001))
    fe = int(sequence.get("frame_end", fs))

    if sep:
        pattern = f"{base}{sep}%0{pad}d.{ext}"
    else:
        pattern = f"{base}%0{pad}d.{ext}"

    out_dir = os.path.join(output_base, shot)
    job = {
        "shot_name": shot,
        "plate_dir": folder,
        "plate_pattern": pattern,
        "frame_start": fs,
        "frame_end": fe,
        "output_dir": out_dir,
        "shared": dict(shared),
    }
    fd, path = tempfile.mkstemp(suffix="_sap2_job.json", prefix=shot + "_")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2)
    return path
