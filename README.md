# ScanDefect3D

ScanDefect3D is a 3D construction-defect detection project built on point clouds and deep learning.

The project supports:
- Point cloud preprocessing (Open3D)
- Semantic segmentation (PointNet / PointNet++)
- Defect detection
- OpenTrench3D multi-task training (semantic head + defect head)

## Features

- Load, visualize, downsample, and denoise point clouds
- Train single-task segmentation models:
  - `PointNet`
  - `PointNet++ (SSG)`
- Train multi-task OpenTrench3D models:
  - Head 1: semantic classes `0..4`
  - Head 2: defect classification `0/1`
- Save checkpoints and run inference/visualization scripts

## Repository Structure

```text
ScanDefect3D/
  configs/
  scripts/
    check_gpu.py
    prepare_synthetic_dataset.py
    prepare_opentrench_defect_dataset.py
    train.py
    train_multitask.py
    infer.py
    visualize_prediction.py
  src/scandefect3d/
    data/
    models/
    utils/
  pyproject.toml
  requirements.txt
```

## 1) Create Virtual Environment

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2) Install PyTorch (GPU) and Dependencies

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

Verify CUDA/GPU:

```powershell
python scripts/check_gpu.py
```

## 3) Prepare Data

### Synthetic quick dataset

```powershell
python scripts/prepare_synthetic_dataset.py --output-root data/synthetic --train-count 300 --val-count 60 --test-count 40
```

### OpenTrench3D multi-task defect dataset

```powershell
python scripts/prepare_opentrench_defect_dataset.py ^
  --input-root data/OpenTrench3D/OpenTrench3D ^
  --output-root data/opentrench3d_defect_multitask_v2_strong ^
  --variants-per-scene 2 ^
  --max-points-per-scene 50000
```

## 4) Train Models

### Single-task: PointNet

```powershell
python scripts/train.py --dataset-type npz --data-root data/synthetic --train-split train --val-split val --model pointnet --num-classes 6 --input-dim 3 --epochs 30 --batch-size 16 --num-points 4096 --exp-name pointnet_synthetic
```

### Single-task: PointNet++

```powershell
python scripts/train.py --dataset-type npz --data-root data/synthetic --train-split train --val-split val --model pointnet2 --num-classes 6 --input-dim 3 --epochs 30 --batch-size 8 --num-points 4096 --exp-name pointnet2_synthetic
```

### Multi-task (recommended): PointNet++ + Transformer context

```powershell
python scripts/train_multitask.py --model pointnet2_transformer --data-root data/opentrench3d_defect_multitask_v2_strong_samplesplit --defect-task-level scene --defect-scene-threshold 0.001 --defect-scene-pooling topk --defect-scene-topk-ratio 0.03 --defect-pos-weight 12 --defect-loss-weight 0.5 --semantic-loss-weight 2.0 --train-defect-sampling-ratio 0.55 --hard-stop-epoch 5 --exp-name pointnet2_transformer_v5_semstrong
```

Checkpoints are saved in:

```text
checkpoints/<exp-name>/
```

## 5) Inference

```powershell
python scripts/infer.py --checkpoint checkpoints/pointnet_synthetic/best.pt --input data/synthetic/test/test_0000.npz --output-dir outputs --num-points 4096 --num-votes 8 --defect-class-ids 4,5 --save-colored-ply
```

## 6) Visualization

```powershell
python scripts/visualize_prediction.py --pred-npz outputs/test_0000_pred.npz --defect-class-ids 4,5
```

## Notes

- `class 4 (Misc)` is ignored by default in semantic loss for OpenTrench3D multi-task training.
- For fast iteration, `--hard-stop-epoch 5` is enabled in sample commands.
- In scene-level defect mode, very low thresholds can increase recall but also increase false positives.
