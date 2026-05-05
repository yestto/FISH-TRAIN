import os
from pathlib import Path
from ultralytics import YOLO

ROOT_DIR = Path("/mnt/raid7tb/dileep/Suhas/Fish Annotations")
DATASET_YAML = ROOT_DIR / "fish_yolo_dataset" / "data.yaml"
PROJECT_DIR = str(ROOT_DIR / "yolo_training_runs")
SEED = 42

def train_yolov13():
    """Train the absolute newest YOLO architecture."""
    print("=" * 60)
    print("🚀 Starting YOLOv13m Training")
    print(f"   Using dataset: {DATASET_YAML}")
    print("=" * 60)
    
    model = YOLO("yolov13m.pt")
    
    results = model.train(
        data=str(DATASET_YAML),
        epochs=150,
        imgsz=640,
        batch=16,
        patience=30,
        device="0",
        project=PROJECT_DIR,
        name="fish_yolov13m_full",
        exist_ok=True,
        # Augmentation for robustness
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
    return results

if __name__ == "__main__":
    if not DATASET_YAML.exists():
        print("❌ Dataset not found! Run process_and_train.py first!")
    else:
        train_yolov13()
