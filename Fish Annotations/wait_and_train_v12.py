#!/usr/bin/env python3
"""
Waits for YOLO11m to finish, then trains YOLO12m (latest model with pretrained weights)
on the same fish dataset.
Run in background: nohup python3 wait_and_train_v12.py > train_log_v12.txt 2>&1 &

Note: YOLOv13 architecture exists but has NO pretrained weights yet.
      YOLO12m is the actual latest model with downloadable pretrained weights.
"""

import time
from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path("/mnt/raid7tb/dileep/Suhas/Fish Annotations")
DATASET_YAML = ROOT_DIR / "fish_yolo_dataset" / "data.yaml"
PROJECT_DIR = str(ROOT_DIR / "yolo_training_runs")
YOLO11_BEST = ROOT_DIR / "yolo_training_runs" / "fish_yolo11m_full" / "weights" / "best.pt"
SEED = 42

print("=" * 60)
print("⏳ Waiting for YOLO11m training to complete...")
print(f"   Watching: {YOLO11_BEST}")
print("=" * 60)

while not YOLO11_BEST.exists():
    time.sleep(30)
    print(".", end="", flush=True)

print(f"\n✅ YOLO11m done! best.pt found.")
print("🚀 Starting YOLO12m training in 15 seconds...")
time.sleep(15)

# Train YOLO12m — the latest model with actual pretrained weights
print("=" * 60)
print("🚀 Training YOLO12m — Latest YOLO with Pretrained Weights")
print("=" * 60)

model = YOLO("yolo12m.pt")

results = model.train(
    data=str(DATASET_YAML),
    epochs=150,
    imgsz=640,
    batch=16,
    patience=30,
    device="0",
    project=PROJECT_DIR,
    name="fish_yolo12m_full",
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
print("✅ YOLO12m Training Complete!")
print(f"   Best model: {results.save_dir}/weights/best.pt")
print("=" * 60)
