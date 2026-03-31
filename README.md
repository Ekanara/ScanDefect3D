# ScanDefect3D

**3D construction-defect detection using point clouds and deep learning.**

ScanDefect3D supports the full pipeline — from raw point cloud preprocessing through semantic segmentation, defect detection, and multi-task model training — built on top of Open3D, PointNet/PointNet++, and a custom OpenTrench3D multi-task architecture.

---

## Features

- Load, visualize, downsample, and denoise point clouds (Open3D)
- Train single-task segmentation models: PointNet and PointNet++ (SSG)
- Train multi-task OpenTrench3D models with two heads:
  - **Semantic head** — classifies points into classes 0–4
  - **Defect head** — binary defect classification (normal / defect)
- Run inference from `.npz` point cloud files or directly from images (URL or local)
- Save checkpoints and visualize predictions as colored `.ply` files

---

## Repository Structure

```
ScanDefect3D/
├── configs/
├── scripts/
│   ├── check_gpu.py
│   ├── prepare_synthetic_dataset.py
│   ├── prepare_opentrench_defect_dataset.py
│   ├── train.py
│   ├── train_multitask.py
│   ├── infer.py
│   ├── infer_image_defect.py
│   └── visualize_prediction.py
├── src/scandefect3d/
│   ├── data/
│   ├── models/
│   └── utils/
├── pyproject.toml
└── requirements.txt
```

---

## Setup

### 1. Create a virtual environment (Windows / PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install PyTorch (GPU) and project dependencies

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

Verify your CUDA/GPU setup:

```powershell
python scripts/check_gpu.py
```

---

## Data Preparation

### Option A — OpenTrench3D (real data, recommended)

**Step 1: Download the raw dataset**

Source: [OpenTrench3D on Kaggle](https://www.kaggle.com/datasets/hestogpony/opentrench3d/data)

Download manually via the Kaggle UI, or use the CLI:

```powershell
pip install kaggle
kaggle datasets download -d hestogpony/opentrench3d -p data/OpenTrench3D --unzip
```

Expected folder layout after extraction:

```
data/OpenTrench3D/OpenTrench3D/
  <scene_folder_1>/*.ply
  <scene_folder_2>/*.ply
  ...
```

**Step 2: Convert raw `.ply` files to training `.npz` format**

```powershell
python scripts/prepare_opentrench_defect_dataset.py ^
  --input-root data/OpenTrench3D/OpenTrench3D ^
  --output-root data/opentrench3d_defect_multitask_v2_strong ^
  --variants-per-scene 2 ^
  --max-points-per-scene 50000
```

This generates `train/`, `val/`, and `test/` splits. Each `.npz` sample contains `points`, `semantic_labels`, and `defect_labels`.

### Option B — Synthetic dataset (quick start)

```powershell
python scripts/prepare_synthetic_dataset.py \
  --output-root data/synthetic \
  --train-count 300 \
  --val-count 60 \
  --test-count 40
```

---

## Training

### Single-task: PointNet

```powershell
python scripts/train.py \
  --dataset-type npz \
  --data-root data/synthetic \
  --train-split train \
  --val-split val \
  --model pointnet \
  --num-classes 6 \
  --input-dim 3 \
  --epochs 30 \
  --batch-size 16 \
  --num-points 4096 \
  --exp-name pointnet_synthetic
```

### Single-task: PointNet++

```powershell
python scripts/train.py \
  --dataset-type npz \
  --data-root data/synthetic \
  --train-split train \
  --val-split val \
  --model pointnet2 \
  --num-classes 6 \
  --input-dim 3 \
  --epochs 30 \
  --batch-size 8 \
  --num-points 4096 \
  --exp-name pointnet2_synthetic
```

### Multi-task: PointNet++ + Transformer (recommended)

```powershell
python scripts/train_multitask.py \
  --model pointnet2_transformer \
  --data-root data/opentrench3d_defect_multitask_v2_strong_samplesplit \
  --defect-task-level scene \
  --defect-scene-threshold 0.001 \
  --defect-scene-pooling topk \
  --defect-scene-topk-ratio 0.03 \
  --defect-pos-weight 12 \
  --defect-loss-weight 0.5 \
  --semantic-loss-weight 2.0 \
  --train-defect-sampling-ratio 0.55 \
  --hard-stop-epoch 5 \
  --exp-name pointnet2_transformer_v5_semstrong
```

Checkpoints are saved to `checkpoints/<exp-name>/`.

---

## Inference

### From a point cloud file

```powershell
python scripts/infer.py \
  --checkpoint checkpoints/pointnet_synthetic/best.pt \
  --input data/synthetic/test/test_0000.npz \
  --output-dir outputs \
  --num-points 4096 \
  --num-votes 8 \
  --defect-class-ids 4,5 \
  --save-colored-ply
```

### From an image (URL or local file)

This pipeline converts an image to a pseudo point cloud using luminance-based pseudo depth, then runs the multi-task model to produce a scene-level defect prediction (`defect` / `normal`) and a point-level defect map.

**From URL:**
```powershell
python scripts/infer_image_defect.py \
  --checkpoint checkpoints/pointnet2_transformer_v5_semstrong/best.pt \
  --image-url URL \ # Insert Image URL here
  --output-dir outputs/image_infer \
  --num-votes 4 \
  --max-points 60000 \
  --num-points 4096 \
  --save-colored-ply
```

**From a local file:**
```powershell
python scripts/infer_image_defect.py \
  --checkpoint checkpoints/pointnet2_transformer_v5_semstrong/best.pt \
  --image-path path/to/your/local_image.jpg \
  --output-dir outputs/image_infer_local \
  --num-votes 4 \
  --max-points 60000 \
  --num-points 4096 \
  --save-colored-ply
```

---

## Visualization

```powershell
python scripts/visualize_prediction.py \
  --pred-npz outputs/test_0000_pred.npz \
  --defect-class-ids 4,5
```

---

## Notes

- **Class 4 (Misc)** is ignored by default in the semantic loss for OpenTrench3D multi-task training.
- **`--hard-stop-epoch 5`** is set in the sample commands above for fast iteration; increase or remove this flag for full training runs.
- In scene-level defect mode, lowering `--defect-scene-threshold` increases recall but may also increase false positives.