"""Run SAP2 engine per shot (VDA-style)."""
from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Callable, Dict, List, Optional

from sap2_job_creator import create_sap2_job_file

logger = logging.getLogger(__name__)


class Sap2ProcessingEngine:
    def __init__(self, sap2_root: str, batch_gap_seconds: int = 10):
        self.sap2_root = sap2_root
        self.batch_gap_seconds = batch_gap_seconds
        self.is_processing = False
        self.progress_callback: Optional[Callable[[str, str, float], None]] = None

    def set_progress_callback(self, callback: Optional[Callable[[str, str, float], None]]) -> None:
        self.progress_callback = callback

    def process_shots(
        self,
        selected_shots: Dict[str, List[Dict]],
        config: Dict,
    ) -> Dict[str, bool]:
        if self.is_processing:
            return {}
        self.is_processing = True
        results: Dict[str, bool] = {}
        batch_file = os.path.join(self.sap2_root, "launch_sap2_engine.bat")
        if not os.path.isfile(batch_file):
            logger.error("Missing %s", batch_file)
            self.is_processing = False
            return results

        shared = config.get("shared", {})
        output_base = config.get("output_base", os.path.join(self.sap2_root, "output"))
        os.makedirs(output_base, exist_ok=True)
        shots = list(selected_shots.items())
        try:
            for idx, (shot_name, sequences) in enumerate(shots):
                ok = True
                for seq in sequences:
                    job_path = create_sap2_job_file(seq, output_base, shared)
                    if self.progress_callback:
                        self.progress_callback(shot_name, "running", idx / max(len(shots), 1))
                    logger.info("SAP2 engine: %s", job_path)
                    proc = subprocess.run(
                        [batch_file, job_path],
                        cwd=self.sap2_root,
                        capture_output=False,
                    )
                    try:
                        os.remove(job_path)
                    except OSError:
                        pass
                    if proc.returncode != 0:
                        ok = False
                results[shot_name] = ok
                if idx < len(shots) - 1 and self.batch_gap_seconds > 0:
                    time.sleep(self.batch_gap_seconds)
        finally:
            self.is_processing = False
        return results
