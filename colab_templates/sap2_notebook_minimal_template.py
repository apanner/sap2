# SAP2 Colab — minimal 3-cell notebook (paste into Colab or upload .ipynb)
# Cell 1
from google.colab import auth
from googleapiclient.discovery import build
from google.colab import drive

auth.authenticate_user()
drive.mount("/content/drive")
drive_service = build("drive", "v3")

# Cell 2 — set CONFIG_FILE_ID from Desk upload
CONFIG_FILE_ID = "PASTE_DRIVE_FILE_ID_HERE"
from sap2_cellcode_template import load_sap2_config_cell2

config_path, config = load_sap2_config_cell2(drive_service, CONFIG_FILE_ID)

# Cell 3
import sys
sys.path.insert(0, "/content/sap2/colab_templates")
from sap2_cellcode_template import run_sap2_cell3

run_sap2_cell3(config_path)

# On failure: scroll for [sap2_colab_run] starting and [ERROR] Shot ...
