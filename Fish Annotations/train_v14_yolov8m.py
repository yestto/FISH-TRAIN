"""
train_v14_yolov8m.py
====================
YOLOv8m detection at imgsz=1280 with enhanced augmentation.
Expected improvement: mAP50-95 ~95-96% (from 92.1% at imgsz=640).
Run FIRST (detection, no seg labels needed).
"""

from ultralytics import YOLO
import torch

DATA_YAML   = "/mnt/raid7tb/dileep/Suhas/Fish Annotations/fish_yolo_dataset/data.yaml"
SAVE_DIR    = "/mnt/raid7tb/dileep/Suhas/Fish Annotations/yolo_training_runs"
PROJECT     = SAVE_DIR
RUN_NAME    = "fish_yolov8m_1280"

def main():
    print("=== YOLOv8m Training at imgsz=1280 ===")
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    model = YOLO("yolov8m.pt")

    results = model.train(
        data        = DATA_YAML,
        epochs      = 200,
        imgsz       = 1280,
        batch       = 8,           # reduce from default if OOM → try 6 or 4
        device      = 0,
        workers     = 8,
        project     = PROJECT,
        name        = RUN_NAME,
        exist_ok    = True,
        # ── Core augmentation ──
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        degrees     = 5.0,         # slight rotation (fish tilt)
        translate   = 0.1,
        scale       = 0.5,
        fliplr      = 0.5,
        flipud      = 0.0,         # fish don't swim upside down
        mosaic      = 1.0,
        mixup       = 0.1,
        copy_paste  = 0.1,
        # ── LR schedule ──
        lr0         = 0.01,
        lrf         = 0.001,
        warmup_epochs = 5,
        cos_lr      = True,
        # ── Regularization ──
        weight_decay = 0.0005,
        # ── Logging ──
        plots       = True,
        verbose     = True,
        patience    = 50,          # early stopping
    )

    # ── Final validation ──
    print("\n=== Final Validation ===")
    metrics = model.val()
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"\nWeights saved at: {PROJECT}/{RUN_NAME}/weights/best.pt")

if __name__ == "__main__":
    main()
