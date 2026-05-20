"""
SAP2 Colab cellcode — Sapiens2 human matting + normals.
Uploaded to Drive: VDA_Jobs/code/{job_id}_cellcode.py
Launched from: google_desk_app/run_app_sap2.bat → Send to Colab

Models download to Colab disk (/content/sapiens2_checkpoints) during Cell 3 — not Drive.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime

# Colab session cache (re-downloaded when runtime restarts)
COLAB_CHECKPOINT_ROOT = "/content/sapiens2_checkpoints"


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
    is_batch = bool(config.get("sequences"))
    date_folder = datetime.now().strftime("%Y%m%d")
    os.environ["SAP2_DRIVE_MOUNT"] = "/content/drive/MyDrive"
    os.environ["SAP2_RUNTIME_DATE_FOLDER"] = date_folder
    os.environ["SAPIENS_CHECKPOINT_ROOT"] = COLAB_CHECKPOINT_ROOT
    logger = logging.getLogger("sap2_colab")
    if not logger.handlers:
        h = logging.StreamHandler()
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    print("[OK] SAP2 config loaded — batch=" + str(is_batch) + " date=" + date_folder)
    print("   Models will download to:", COLAB_CHECKPOINT_ROOT)
    print("   Shots:", len(config.get("sequences") or []))
    return config, "/content/drive/MyDrive", date_folder, logger, is_batch, config_path


def _stream_subprocess(cmd, cwd=None, env=None):
    run_env = dict(env or os.environ)
    run_env["PYTHONUNBUFFERED"] = "1"
    if cmd and cmd[0] == sys.executable:
        cmd = [sys.executable, "-u", *cmd[1:]]
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
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
            capture_output=True,
            text=True,
        )
        if pull.returncode == 0:
            print("[OK] git pull:", (pull.stdout or "").strip() or "up to date")
        else:
            print("[warn] git pull failed — using existing clone:", pull.stderr or pull.stdout)

    subprocess.check_call([sys.executable, "scripts/colab_setup.py"], cwd=clone_dir)

    with open(config_path, encoding="utf-8") as f:
        job = json.load(f)
    shared = job.get("shared") or job.get("shared_settings") or {}
    shared["checkpoint_root"] = COLAB_CHECKPOINT_ROOT
    shared["download_models_in_colab"] = True

    env = os.environ.copy()
    env["SAP2_DRIVE_MOUNT"] = "/content/drive/MyDrive"
    env["SAP2_RUNTIME_DATE_FOLDER"] = date_folder
    env["SAPIENS_CHECKPOINT_ROOT"] = COLAB_CHECKPOINT_ROOT

    cmd = [
        sys.executable,
        "scripts/sap2_colab_run.py",
        "--job-json",
        config_path,
    ]
    print("\n[sap2_colab_run] starting:", " ".join(cmd), flush=True)
    print("[info] Checkpoints: download to", COLAB_CHECKPOINT_ROOT, "then infer", flush=True)
    rc = _stream_subprocess(cmd, cwd=clone_dir, env=env)
    if rc != 0:
        print(
            "\n[STOP] sap2_colab_run.py exited %s. Scroll up for [download] / [ERROR]. "
            "HF login: huggingface_hub.login() if 401." % rc,
            flush=True,
        )
    return rc == 0


def process_sap2_cell3(
    config,
    drive_base_path,
    date_folder,
    logger,
    is_batch,
    config_path,
):
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
    print("SAP2 batch — install → download models → matte + normals")
    print("=" * 60)

    try:
        ok = _run_sap2_batch(config_path, git_url, date_folder)
    except Exception as exc:
        print("[ERROR] " + str(exc))
        print(traceback.format_exc())
        ok = False

    out_path = shared.get("output_folder_path", "VDA_output")
    root = shared.get("sap2_output_root", "SAP2_output")
    print(
        "\nOutputs on Drive: MyDrive/"
        + str(out_path)
        + "/"
        + date_folder
        + "/"
        + root
        + "/<shot>/matte/ normal/"
    )
    print("Models stayed on Colab disk only:", COLAB_CHECKPOINT_ROOT)
    return ok


def run_sap2_cell3(config_path: str, repo_url: str = "https://github.com/apanner/sap2.git"):
    date_folder = os.environ.get("SAP2_RUNTIME_DATE_FOLDER", datetime.now().strftime("%Y%m%d"))
    return _run_sap2_batch(config_path, repo_url, date_folder)
