# SAP2 on Google Colab

Human **matting** + **surface normals** using [Sapiens2](https://github.com/apanner/sap2) (same 3-cell pattern as LAOV AI Matte).

## Prerequisites

1. **Colab runtime:** GPU (T4 / L4 / A100). Prefer **Python 3.12** runtime if available.
2. **Drive folders:**
   - `MyDrive/VDA_models/sapiens2_host/` — checkpoints (see below)
   - `MyDrive/VDA_Jobs/` — job JSON + cellcode from Desk (or manual)
3. **Plates:** EXR sequence per shot (`plate_dir` + `plate_pattern` in JSON)

## Models (automatic in Cell 3)

**No Drive download needed.** Cell 3 downloads weights to Colab local disk:

```
/content/sapiens2_checkpoints/
├── matting/sapiens2_1b_matting.safetensors
└── normal/sapiens2_1b_normal.safetensors
```

Re-run Cell 3 in a **new runtime** to re-download (~12 GB). Outputs still go to **Drive** under `SAP2_output/`.

## Batch JSON

See `batch/sample_sap2_job.json`. Set `model_type` to `SAP2_STANDALONE`.

## Run (3 cells)

1. Mount Drive (use `colab_templates/sap2_cellcode_template.py` → `setup_sap2_cell1`).
2. Download config by Drive file id → `load_sap2_config_cell2`.
3. `run_sap2_cell3("/content/sap2_config.json")` — clones repo, `colab_setup.py`, `sap2_colab_run.py`.

Or from shell after clone:

```bash
export SAP2_DRIVE_MOUNT=/content/drive/MyDrive
export SAP2_RUNTIME_DATE_FOLDER=$(date +%Y%m%d)
python scripts/sap2_colab_run.py --job-json /content/sap2_config.json
```

## Outputs

**Processing** writes to `/content/output/{shot}/` (same as VDA `LOCAL_OUTPUT_PATH`). Plates are read from Drive once into a local JPEG cache. Optional **person crop** (OpenCV MobileNet-SSD / HOG): detect → pad crop → Sapiens2 @ 1024×768 → paste matte/normal to full plate. **Drive copy happens only when the shot is done** (`[SAVE] Saving results to Drive...`).

Per shot deliverables under `MyDrive/VDA_output/{date}/SAP2_output/{shot_name}/`:

| Folder | Content |
|--------|---------|
| `matte/` | Alpha EXR (`matte_000001.exr`, …) |
| `normal/` | Unit normal RGB EXR |
| `_plate_jpeg_cache/` | Full-res JPEG plates for inference |
| `_vis_matting/`, `_vis_normal/` | Side-by-side QC PNGs from upstream vis |

**Resume:** Re-run Cell 3; passes skip when EXR count matches frame range.

## Local batch (Windows, VDA-style)

```bat
sap2\batch\launch_sap2_batch.bat
```

Scans a local EXR folder, builds JPEG cache, runs `launch_sap2_engine.bat` per shot. Uses VDA embedded Python if present.

**80GB GPU:** set `inference_long_edge: 0` (auto → 4096 px long edge) in JSON or Desk settings.

## Google Desk GUI

```bat
google_desk_app\run_app_sap2.bat
```

Separate settings from VDA / AI Matte. **Send to Colab** uploads `SAP2_STANDALONE` JSON + cellcode.

## Push engine to GitHub

From `sap2/` repo root (clone [apanner/sap2](https://github.com/apanner/sap2.git)):

```bat
push_to_github.bat
```

## vs LAOV AI Matte

| Use SAP2 | Use LAOV matte |
|----------|----------------|
| People / human-centric plates | Arbitrary objects, SAM3 prompts |
| Matte + normals in one model family | SAM3 → BiRefNet / ViTMatte |
| Fixed 1024×768 infer, upscaled to plate | Optional full-res refine |

Full plan: [SAP2_COLAB_PLAN.md](SAP2_COLAB_PLAN.md).
