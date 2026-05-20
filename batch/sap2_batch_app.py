"""SAP2 local batch GUI — scan EXR folders, run launch_sap2_engine.bat per shot."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

batch_dir = Path(__file__).parent
sap2_root = batch_dir.parent
sys.path.insert(0, str(batch_dir))

from sequence_scanner import scan_folder_for_shots  # noqa: E402
from sap2_processing_engine import Sap2ProcessingEngine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ProcessThread(QThread):
    finished_signal = Signal(dict)

    def __init__(self, engine, shots, config):
        super().__init__()
        self.engine = engine
        self.shots = shots
        self.config = config

    def run(self):
        self.finished_signal.emit(self.engine.process_shots(self.shots, self.config))


class Sap2BatchWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAP2 Batch — Human Matte + Normals")
        self.resize(900, 600)
        self.scan_root = ""
        self.sequences = []
        self._build_ui()
        self.engine = Sap2ProcessingEngine(str(sap2_root))

    def _build_ui(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.folder_label = QLabel("No folder selected")
        browse = QPushButton("Browse EXR folder…")
        browse.clicked.connect(self._browse)
        scan = QPushButton("Scan sequences")
        scan.clicked.connect(self._scan)
        row.addWidget(self.folder_label, 1)
        row.addWidget(browse)
        row.addWidget(scan)
        lay.addLayout(row)

        self.info = QLabel("0 sequences")
        lay.addWidget(self.info)

        proc = QPushButton("Process all scanned shots")
        proc.clicked.connect(self._process)
        lay.addWidget(proc)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log, 1)
        self.setCentralWidget(w)

    def _log(self, msg: str) -> None:
        self.log.append(msg)
        logger.info(msg)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select plate root folder")
        if d:
            self.scan_root = d
            self.folder_label.setText(d)

    def _scan(self) -> None:
        if not self.scan_root:
            QMessageBox.warning(self, "Folder", "Select a folder first.")
            return
        self.sequences = scan_folder_for_shots(self.scan_root)
        self.info.setText(f"{len(self.sequences)} sequence(s) in {self.scan_root}")
        self._log(f"Scanned {len(self.sequences)} sequence(s)")

    def _process(self) -> None:
        if not self.sequences:
            QMessageBox.warning(self, "Scan", "Scan a folder first.")
            return
        out = QFileDialog.getExistingDirectory(self, "Output base folder")
        if not out:
            return
        shots: dict = {}
        for seq in self.sequences:
            d = seq.to_dict() if hasattr(seq, "to_dict") else seq
            shot = d.get("shot_name") or Path(d["folder_path"]).name
            d["shot_name"] = shot
            shots.setdefault(shot, []).append(d)
        shared = {
            "sapiens_model": "1b",
            "run_matting": True,
            "run_normal": True,
            "use_plate_jpeg_cache": True,
            "inference_long_edge": 0,
            "checkpoint_root": str(sap2_root / "checkpoints"),
            "plate_cache_workers": 8,
        }
        config = {"output_base": out, "shared": shared}
        self._log(f"Processing {len(shots)} shot(s) → {out}")
        self.thread = ProcessThread(self.engine, shots, config)
        self.thread.finished_signal.connect(self._on_done)
        self.thread.start()

    def _on_done(self, results: dict) -> None:
        ok = sum(1 for v in results.values() if v)
        self._log(f"Done: {ok}/{len(results)} shots OK")
        QMessageBox.information(self, "SAP2", f"Finished: {ok}/{len(results)} shots succeeded.")


def main():
    app = QApplication(sys.argv)
    win = Sap2BatchWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
