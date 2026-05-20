"""
SAP2 Colab cellcode — Sapiens2 human matting + normals.
Drive: VDA_Jobs/code/{job_id}_cellcode.py
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime


def setup_sap2_cell1(drive_service):
    from google.colab import auth, drive

    print("=" * 60)
    print("SAP2 — mount Drive")
    print("=" * 60)
    auth.authenticate_user()
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive", force_remount=False)
    else:
        print("[OK] Drive already mounted")
    return drive_service


def load_sap2_config_cell2(drive_service, config_file_id):
    from googleapiclient.http import MediaIoBaseDownload

    config_path = "/content/sap2_config.json"
    request = drive_service.files().get_media(fileId=config_file_id)
    with open(config_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print("   Config download: " + str(int(status.progress() * 100)) + "%")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    date_folder = datetime.now().strftime("%Y%m%d")
    os.environ["SAP2_DRIVE_MOUNT"] = "/content/drive/MyDrive"
    os.environ["SAP2_RUNTIME_DATE_FOLDER"] = date_folder
    print("[OK] SAP2 config loaded — date folder:", date_folder)
    print("   Shots:", len(config.get("sequences") or []))
    return config_path, config


def _stream_subprocess(cmd, cwd=None, env=None):
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()


def run_sap2_cell3(config_path: str, repo_url: str = "https://github.com/apanner/sap2.git"):
    print("=" * 60)
    print("SAP2 — clone, install, run batch")
    print("=" * 60)
    clone_dir = "/content/sap2"
    if not os.path.isdir(clone_dir):
        subprocess.check_call(["git", "clone", "--depth", "1", repo_url, clone_dir])
    else:
        print("[OK] sap2 already at", clone_dir)

    subprocess.check_call([sys.executable, "scripts/colab_setup.py"], cwd=clone_dir)

    env = os.environ.copy()
    env.setdefault("SAP2_DRIVE_MOUNT", "/content/drive/MyDrive")
    env.setdefault(
        "SAPIENS_CHECKPOINT_ROOT",
        env["SAP2_DRIVE_MOUNT"] + "/VDA_models/sapiens2_host",
    )

    cmd = [
        sys.executable,
        "-u",
        "scripts/sap2_colab_run.py",
        "--job-json",
        config_path,
    ]
    print("[sap2_colab_run] starting:", " ".join(cmd), flush=True)
    rc = _stream_subprocess(cmd, cwd=clone_dir, env=env)
    if rc != 0:
        print("[STOP] SAP2 batch failed (exit %d). Scroll up for [ERROR] lines." % rc)
        raise SystemExit(rc)
    print("[OK] SAP2 batch complete")


def main_cell3_from_drive(config_file_id: str, drive_service=None):
    if drive_service is None:
        from googleapiclient.discovery import build
        from google.colab import auth

        auth.authenticate_user()
        drive_service = build("drive", "v3")
    config_path, cfg = load_sap2_config_cell2(drive_service, config_file_id)
    shared = cfg.get("shared") or cfg.get("shared_settings") or {}
    repo_url = str(shared.get("sap2_git_url", "https://github.com/apanner/sap2.git"))
    try:
        run_sap2_cell3(config_path, repo_url=repo_url)
    except Exception:
        traceback.print_exc()
        print("[STOP] Exception in SAP2 cell — see traceback above.")
        raise
