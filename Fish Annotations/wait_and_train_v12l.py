#!/usr/bin/env python3
"""
Waits for BOTH yolo11m and yolo12m to finish, then trains YOLO12l (large).
Run in background: nohup python3 wait_and_train_v12l.py > train_log_v12l.txt 2>&1 &
"""

import time
from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path("/mnt/raid7tb/dileep/Suhas/Fish Annotations")
DATASET_YAML = ROOT_DIR / "fish_yolo_dataset" / "data.yaml"
PROJECT_DIR = str(ROOT_DIR / "yolo_training_runs")
SEED = 42

# Wait for BOTH current runs to finish
YOLO11_BEST = ROOT_DIR / "yolo_training_runs" / "fish_yolo11m_full" / "weights" / "best.pt"
YOLO12M_BEST = ROOT_DIR / "yolo_training_runs" / "fish_yolo12m_full" / "weights" / "best.pt"

print("=" * 60)
print("⏳ Waiting for YOLO11m + YOLO12m to both finish...")
print("=" * 60)

while not (YOLO11_BEST.exists() and YOLO12M_BEST.exists()):
    missing = []
    if not YOLO11_BEST.exists(): missing.append("YOLO11m")
    if not YOLO12M_BEST.exists(): missing.append("YOLO12m")
    print(f"   Still waiting for: {', '.join(missing)}", flush=True)
    time.sleep(60)

print("\n✅ Both YOLO11m and YOLO12m done!")
print("🚀 Starting YOLO12l (Large) training in 15 seconds...")
time.sleep(15)

print("=" * 60)
print("🚀 Training YOLO12l — Large Model for Max Accuracy")
print("=" * 60)

model = YOLO("yolo12l.pt")

results = model.train(
    data=str(DATASET_YAML),
    epochs=150,
    imgsz=640,
    batch=12,           # Slightly smaller batch for large model
    patience=30,
    device="0",
    project=PROJECT_DIR,
    name="fish_yolo12l_full",
    exist_ok=True,
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    # Optimisation
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=5,
    cos_lr=True,
    workers=8,
    seed=SEED,
    verbose=True,
    plots=True,
)

print("\n" + "=" * 60)
print("✅ YOLO12l Training Complete!")
print(f"   Best model: {results.save_dir}/weights/best.pt")
print("=" * 60)
