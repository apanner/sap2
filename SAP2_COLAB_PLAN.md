# SAP2 Colab — plan (LAOV-style human matte + normals)

**Repo:** [apanner/sap2](https://github.com/apanner/sap2.git) (fork of [facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2))  
**Goal:** Google Colab batch workflow like LAOV `AI_MATTE_STANDALONE` — Desk JSON → 3 cells → Drive outputs per shot.

## What SAP2 gives you (vs LAOV AI Matte)

| | LAOV AI Matte | SAP2 (Sapiens2) |
|---|---------------|-----------------|
| **Matte** | SAM3 track → BiRefNet / ViTMatte refine | Single **human matting** head (`sapiens2_1b_matting`) |
| **Normals** | — | **Surface normals** (`sapiens2_*_normal`) |
| **Input** | Any foreground (text prompts) | **Human-centric** (people in frame) |
| **Native infer res** | Full-res refine optional | **1024×768 (H×W)** internal; **EXR at full plate size** |
| **Python / torch** | 3.10+, torch 2.x | **≥3.12**, **torch ≥2.7** |

## Image data (official Sapiens2 + SAP2 4K)

Per [facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2):

- Models are trained at **1024×768 (H×W)** (not plate resolution).
- Inference accepts **any-size** JPG/PNG/EXR→JPEG: `cv2.imread` → `model.pipeline` resizes → model → **bilinear upsample to original H×W**.
- **Matting:** stretch to 1024×768 (`keep_ratio=False`).
- **Normals:** letterbox + pad to 1024×768, then unpad.

### SAP2 feed modes for 4K plates (e.g. 2160×3840)

| `image_feed_mode` | Behavior |
|-------------------|----------|
| **`auto`** (default) | **≥1080p / >1 MP** → OpenCV person detect → crop → infer → paste to **full 4K EXR** |
| **`full_res`** | Whole 4K frame → official pipeline (stretch to 1024×768) → **full 4K EXR** |
| **`person_crop`** | Always detect + crop (best matte detail on wide shots) |

You always get **full-resolution EXR** on disk; only the **internal** infer uses 1024×768.

## I/O layout (VDA-style — same as depth lane)

| Stage | Path |
|-------|------|
| **Plates (read)** | Drive: `plate_dir` in job JSON |
| **Hot work** | Colab local: `/content/output/{shot}/` (JPEG cache, matte/normal EXR) — **same path as VDA** |
| **Models** | Colab local: `/content/sapiens2_checkpoints` (Cell 3 download) |
| **Deliverables (save)** | Drive: `MyDrive/VDA_output/{YYYYMMDD}/SAP2_output/{shot}/matte|normal/` |

Resume checks **Drive** EXR counts; infer missing frames locally; `shutil.copy2` to Drive **once when shot is done**.

```
/content/output/{shot}/               # ephemeral — same as VDA LOCAL_OUTPUT_PATH
├── _plate_jpeg_cache/
├── matte/
└── normal/

MyDrive/VDA_output/{date}/SAP2_output/{shot}/   # persistent
├── matte/
│   ├── matte_000001.exr          # layout=combined or channels
│   ├── p00/matte_000001.exr      # layout=separate (per person)
│   └── p01/…
└── normal/
├── qc/                              # when qc_mp4=true (LAOV-style)
│   ├── {shot}_matte_qc.mp4
│   ├── {shot}_normal_qc.mp4
│   └── {shot}_review_qc.mp4         # plate | matte | normal
```

### QC MP4 (`qc_mp4`, default **true**)

Same idea as LAOV: after EXR infer, build H.264 previews under `qc/` with **plate overlay** from the JPEG cache.

| File | Content |
|------|---------|
| `{shot}_matte_qc.mp4` | Combined matte (premult RGB or alpha) on plate |
| `{shot}_matte_channels_qc.mp4` | R/G/B/A person masks as red/green/blue/white |
| `{shot}_matte_p00_qc.mp4` | Per-person folder (`layout=separate`) |
| `{shot}_normal_qc.mp4` | Surface normals (0.5·n+0.5) on plate |
| `{shot}_review_qc.mp4` | 3-up: plate \| matte \| normal |

Set `"qc_mp4": false` in JSON to skip. `"qc_fps": 24` optional. **`qc_max_long_edge": 1920`** (default) downscales 4K plates to HD for fast MP4s (portrait → ~1080×1920).

### EXR I/O (LAOV-aligned)

Writes use **`ImageOutput.write_image`** + zip compression (same as LAOV `oiio_io.write_exr`), not `ImageBuf.set_pixels` (that path produced **black/empty EXRs**).

- **Matte:** straight RGB + **A** (readable in Nuke; premult from model is unpremulted on write)
- **Normal:** unit **R,G,B** in [-1, 1] (like LAOV `n.x` / `n.y` / `n.z`)

After `git pull`, re-run Cell 3 — empty old EXRs are **auto re-exported** (detected via read-back).

### Multi-person matte (`matte_subject_layout`)

| Layout | Output | Nuke |
|--------|--------|------|
| **`combined`** (default) | One RGBA EXR — premult RGB + A, all people (union crop) | Single Read; one comp matte |
| **`channels`** | One EXR — **R,G,B,A = alpha** for person 0,1,2,3 (left→right) | One Read; ShuffleCopy / Dot Product per channel |
| **`separate`** | `matte/p00/`, `matte/p01/`, … full-plate RGBA per person | One Read per person; most flexible |

Subject index is **left → right** by bbox center (stable across frames). Max **4** people (`matte_max_subjects`).

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
    "image_feed_mode": "auto",
    "use_person_crop": true,
    "person_crop_pad": 0.22,
    "person_crop_mode": "union",
    "matte_subject_layout": "combined",
    "qc_mp4": true,
    "qc_fps": 24,
    "output_folder_path": "/content/drive/MyDrive/VDA_output"
  },
  "sequences": [
    {
      "shot_name": "073_020",
      "plate_dir": "/content/drive/MyDrive/plates/TB_073_050",
      "plate_pattern": "TB_073_050_plate_v001_%06d.exr",
      "frame_start": 1001,
      "frame_end": 1050
    }
  ]
}
```

## Colab cells

| Cell | Action |
|------|--------|
| **1** | Mount Drive, auth |
| **2** | Download config from Drive file id; set `SAP2_LOCAL_OUTPUT=/content/output` |
| **3** | `git pull` sap2 → `colab_setup.py` → `sap2_colab_run.py` (one process per shot: matting + normal) |

## Models (Colab local, not Drive)

Cell 3 downloads to `/content/sapiens2_checkpoints`:

- `matting/sapiens2_1b_matting.safetensors`
- `normal/sapiens2_1b_normal.safetensors`

Person detect: try MobileNet-SSD (`/content/sap2_models/person_det/`); **if that fails → OpenCV HOG** (no download). Force HOG only: `SAP2_PERSON_DET_BACKEND=hog`.
