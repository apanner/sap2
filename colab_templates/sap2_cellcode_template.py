"""
SAP2 Colab cellcode — VDA I/O: LOCAL_OUTPUT_PATH=/content/output, copy to Drive when done.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime

COLAB_CHECKPOINT_ROOT = "/content/sapiens2_checkpoints"
LOCAL_OUTPUT_PATH = "/content/output"  # same as VDA


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
    shared = config.get("shared") or config.get("shared_settings") or {}
    drive = "/content/drive/MyDrive"
    out_raw = str(shared.get("output_folder_path") or "VDA_output").strip().replace("\\", "/")
    if out_raw.startswith("/content/drive/MyDrive/"):
        out_path = out_raw
    elif out_raw.startswith("/content/drive"):
        marker = "/MyDrive/"
        out_path = (
            f"{drive}/{out_raw.split(marker)[-1].lstrip('/')}"
            if marker in out_raw
            else out_raw
        )
    elif out_raw.startswith("MyDrive/"):
        out_path = f"{drive}/{out_raw[len('MyDrive/'):]}"
    else:
        out_path = f"{drive}/{out_raw.strip('/')}"
    shared["output_folder_path"] = out_path
    config["shared"] = shared
    config["shared_settings"] = shared
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    is_batch = bool(config.get("sequences"))
    date_folder = datetime.now().strftime("%Y%m%d")
    os.environ["SAP2_DRIVE_MOUNT"] = drive
    os.environ["SAP2_RUNTIME_DATE_FOLDER"] = date_folder
    os.environ["SAP2_LOCAL_OUTPUT"] = LOCAL_OUTPUT_PATH
    os.environ["SAPIENS_CHECKPOINT_ROOT"] = COLAB_CHECKPOINT_ROOT
    os.makedirs(LOCAL_OUTPUT_PATH, exist_ok=True)

    logger = logging.getLogger("sap2_colab")
    if not logger.handlers:
        h = logging.StreamHandler()
        logger.addHandler(h)
        logger.setLevel(logging.INFO)

    print("[OK] SAP2 config — batch=" + str(is_batch) + " date=" + date_folder)
    print("   Local output (process here): " + LOCAL_OUTPUT_PATH)
    print("   Drive (copy when done): " + shared["output_folder_path"] + "/" + date_folder + "/SAP2_output/<shot>/")
    print("   Models: " + COLAB_CHECKPOINT_ROOT)
    print("   Shots: " + str(len(config.get("sequences") or [])))
    return config, drive, date_folder, logger, is_batch, config_path


def _stream_subprocess(cmd, cwd=None, env=None):
    run_env = dict(env or os.environ)
    run_env["PYTHONUNBUFFERED"] = "1"
    if cmd and cmd[0] == sys.executable:
        cmd = [sys.executable, "-u", *cmd[1:]]
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=run_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()


def _run_sap2_batch(config_path: str, repo_url: str, date_folder: str) -> bool:
    clone_dir = "/content/sap2"
    if not os.path.isdir(clone_dir):
        subprocess.check_call(["git", "clone", "--depth", "1", repo_url, clone_dir])
    else:
        print("[OK] sap2 repo at", clone_dir)
        pull = subprocess.run(
            ["git", "-C", clone_dir, "pull", "--ff-only"],
            capture_output=True, text=True,
        )
        if pull.returncode == 0:
            print("[OK] git pull:", (pull.stdout or "").strip() or "up to date")

    subprocess.check_call([sys.executable, "scripts/colab_setup.py"], cwd=clone_dir)

    env = os.environ.copy()
    env["SAP2_DRIVE_MOUNT"] = "/content/drive/MyDrive"
    env["SAP2_RUNTIME_DATE_FOLDER"] = date_folder
    env["SAP2_LOCAL_OUTPUT"] = LOCAL_OUTPUT_PATH
    env["SAPIENS_CHECKPOINT_ROOT"] = COLAB_CHECKPOINT_ROOT

    cmd = [sys.executable, "scripts/sap2_colab_run.py", "--job-json", config_path]
    print("\n[sap2_colab_run] write EXR to " + LOCAL_OUTPUT_PATH + " → copy to Drive when done")
    print("       ", " ".join(cmd), flush=True)
    return _stream_subprocess(cmd, cwd=clone_dir, env=env) == 0


def process_sap2_cell3(config, drive_base_path, date_folder, logger, is_batch, config_path):
    if not is_batch:
        print("[ERROR] Batch JSON must contain sequences[]")
        return False
    shared = config.get("shared") or config.get("shared_settings") or {}
    git_url = str(
        os.environ.get("SAP2_GIT_URL")
        or shared.get("sap2_git_url")
        or "https://github.com/apanner/sap2.git"
    )
    print("\n" + "=" * 60)
    print("[SAVE] SAP2: process on " + LOCAL_OUTPUT_PATH + ", then copy to Drive")
    print("=" * 60)
    try:
        ok = _run_sap2_batch(config_path, git_url, date_folder)
    except Exception as exc:
        print("[ERROR]", exc)
        print(traceback.format_exc())
        ok = False
    return ok


def run_sap2_cell3(config_path: str, repo_url: str = "https://github.com/apanner/sap2.git"):
    date_folder = os.environ.get("SAP2_RUNTIME_DATE_FOLDER", datetime.now().strftime("%Y%m%d"))
    return _run_sap2_batch(config_path, repo_url, date_folder)
