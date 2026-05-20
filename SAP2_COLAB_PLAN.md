# SAP2 Colab — plan (LAOV-style human matte + normals)

**Repo:** [apanner/sap2](https://github.com/apanner/sap2.git) (fork of [facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2))  
**Goal:** Google Colab batch workflow like LAOV `AI_MATTE_STANDALONE` — Desk JSON → 3 cells → Drive outputs per shot.

## What SAP2 gives you (vs LAOV AI Matte)

| | LAOV AI Matte | SAP2 (Sapiens2) |
|---|---------------|-----------------|
| **Matte** | SAM3 track → BiRefNet / ViTMatte refine | Single **human matting** head (`sapiens2_1b_matting`) |
| **Normals** | — | **Surface normals** (`sapiens2_*_normal`) |
| **Input** | Any foreground (text prompts) | **Human-centric** (people in frame) |
| **Native infer res** | Full-res refine optional | **1024×768 (H×W)** internal; upsampled to plate size |
| **Python / torch** | 3.10+, torch 2.x | **≥3.12**, **torch ≥2.7** |

Use SAP2 when plates are **people / live-action humans** and you want **matte + normal** without SAM3/BiRefNet. Keep LAOV matte for multi-object / non-human plates.

## Drive layout (mirror VDA / LAOV)

```
MyDrive/
├── VDA_Jobs/
│   ├── code/{job_id}_cellcode.py      # uploaded from Desk
│   └── {job_id}_config.json           # SAP2_STANDALONE batch
├── VDA_models/
│   └── sapiens2_host/
│       ├── matting/sapiens2_1b_matting.safetensors
│       └── normal/sapiens2_1b_normal.safetensors   # or 0.4b / 0.8b / 5b
└── VDA_output/
    └── {YYYYMMDD}/
        └── SAP2_output/
            └── {shot_name}/
                ├── _plate_jpeg_cache/plate_000001.jpg
                ├── matte/           # alpha EXR (Nuke)
                ├── normal/          # RGB normal EXR
                ├── _vis_matting/    # optional QC PNGs from upstream vis
                ├── _vis_normal/
                └── qc_sap2.mp4      # optional side-by-side
```

## Batch JSON (`model_type`: `SAP2_STANDALONE`)

Desk / manual JSON (same shape as LAOV batch):

```json
{
  "model_type": "SAP2_STANDALONE",
  "batch_id": "sap2_batch_001",
  "shared": {
    "sapiens_model": "1b",
    "run_matting": true,
    "run_normal": true,
    "export_matte_exr": true,
    "export_normal_exr": true,
    "use_plate_jpeg_cache": true,
    "plate_jpeg_quality": 92,
    "plate_cache_workers": 8,
    "qc_mp4": true,
    "batch_one_process_per_shot": true,
    "batch_gap_seconds": 5,
    "checkpoint_root": "/content/drive/MyDrive/VDA_models/sapiens2_host"
  },
  "sequences": [
    {
      "shot_name": "TB_073_050",
      "plate_dir": "/content/drive/MyDrive/plates/TB_073_050",
      "plate_pattern": "TB_073_050.%04d.exr",
      "frame_start": 1001,
      "frame_end": 1100
    }
  ]
}
```

## Colab cells (3-cell pattern)

| Cell | Action |
|------|--------|
| **1** | Mount Drive, auth |
| **2** | Download config from Drive file id; set `SAP2_DRIVE_MOUNT`, date folder |
| **3** | Clone `apanner/sap2`, `pip install -e .`, `colab_setup.py`, run `sap2_colab_run.py` |

Templates: `colab_templates/sap2_cellcode_template.py`, `sap2_notebook_minimal_template.py`.

## Runner phases (`scripts/sap2_colab_run.py`)

1. **Plate cache** — parallel EXR→JPEG (OpenImageIO); skip if cache complete.
2. **Matting** — `vis_matting.py` → `_alpha.npy` per frame → **EXR** in `matte/`.
3. **Normals** — `vis_normal.py` → `.npy` → **EXR** in `normal/` (unit vectors, -1 outside optional later).
4. **Resume** — skip pass if output EXR count ≥ frame count.
5. **Batch** — optional one subprocess per shot (VRAM reset), continue on failure.
6. **QC** — optional MP4 (plate | alpha | normal false-color).

## Models (one-time on Drive)

From [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md):

| Task | HuggingFace | Local path under `sapiens2_host/` |
|------|-------------|-----------------------------------|
| Matting | [facebook/sapiens2-matting-1b](https://huggingface.co/facebook/sapiens2-matting-1b) | `matting/sapiens2_1b_matting.safetensors` |
| Normal 1B | [facebook/sapiens2-normal-1b](https://huggingface.co/facebook/sapiens2-normal-1b) | `normal/sapiens2_1b_normal.safetensors` |

`scripts/download_checkpoints_colab.py` can pull HF weights into Drive (run once in Cell 2 or a setup notebook).

## Desk integration (phase 2)

- New lane `SAP2_STANDALONE` in `google_desk_app` (mirror `ai_matte_config_generator.py`).
- Upload cellcode + JSON to `VDA_Jobs/`.
- Nuke read nodes: `matte/*.exr`, `normal/*.exr` under `SAP2_output/{date}/{shot}/`.

## Risks / constraints

1. **Colab Python** — must be **3.12+**; use Runtime → change runtime or `!python --version` before install.
2. **torch 2.7** — first run may reinstall torch (long Cell 3).
3. **4K plates** — inference still 1024×768; output upsampled to full plate (same as upstream demos).
4. **Normals without seg mask** — full-frame normal; optional `seg_dir` later for body-only export.
5. **License** — Sapiens2 proprietary license; checkpoints from Meta HF.

## Implementation status

| Item | Status |
|------|--------|
| Plan (this doc) | ✅ |
| `sap2_colab_run.py` + high-res `sap2_infer.py` | ✅ |
| `deploy/external_engine/sap2_engine.py` | ✅ local VDA-style engine |
| `launch_sap2_engine.bat` | ✅ |
| `batch/launch_sap2_batch.bat` + GUI | ✅ |
| `run_app_sap2.bat` + Desk settings | ✅ |
| `sap2_config_generator.py` | ✅ |
| Cellcode + notebook template | ✅ |
| `COLAB.md` quick start | ✅ |
| Git push `push_to_github.bat` | ✅ |

## Citation

```bibtex
@article{khirodkarsapiens2,
  title={Sapiens2},
  author={Khirodkar, Rawal and Wen, He and Martinez, Julieta and Dong, Yuan and Su, Zhaoen and Saito, Shunsuke},
  journal={arXiv preprint arXiv:2604.21681},
  year={2026}
}
```
