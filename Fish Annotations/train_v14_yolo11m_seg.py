"""
train_v14_yolo11m_seg.py
=========================
YOLO11m SEGMENTATION at imgsz=1280.
Requires segmentation labels from: convert_bbox_to_seg.py
Expected improvement: mAP50-95 ~96-97% (vs 92.1% current detection baseline).
Run AFTER convert_bbox_to_seg.py.
"""

from ultralytics import YOLO
import torch

SEG_DATA_YAML = "/mnt/raid7tb/dileep/Suhas/Fish Annotations/fish_yolo_dataset_seg/data.yaml"
SAVE_DIR      = "/mnt/raid7tb/dileep/Suhas/Fish Annotations/yolo_training_runs"
RUN_NAME      = "fish_yolo11m_seg_1280"

def main():
    print("=== YOLO11m-seg Training at imgsz=1280 ===")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    model = YOLO("yolo11m-seg.pt")

    results = model.train(
        data        = SEG_DATA_YAML,
        epochs      = 200,
        imgsz       = 1280,
        batch       = 6,           # seg uses more VRAM than det
        device      = 0,
        workers     = 8,
        project     = SAVE_DIR,
        name        = RUN_NAME,
        exist_ok    = True,
        # ── Augmentation ──
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        degrees     = 5.0,
        translate   = 0.1,
        scale       = 0.5,
        fliplr      = 0.5,
        flipud      = 0.0,
        mosaic      = 1.0,
        mixup       = 0.1,
        copy_paste  = 0.1,
        # ── LR ──
        lr0         = 0.01,
        lrf         = 0.001,
        warmup_epochs = 5,
        cos_lr      = True,
        weight_decay = 0.0005,
        # ── Logging ──
        plots       = True,
        verbose     = True,
        patience    = 50,
    )

    print("\n=== Final Validation ===")
    metrics = model.val()
    print(f"  mAP50 (box):     {metrics.box.map50:.4f}")
    print(f"  mAP50-95 (box):  {metrics.box.map:.4f}")
    print(f"  mAP50 (mask):    {metrics.seg.map50:.4f}")
    print(f"  mAP50-95 (mask): {metrics.seg.map:.4f}")
    print(f"\nWeights: {SAVE_DIR}/{RUN_NAME}/weights/best.pt")

if __name__ == "__main__":
    main()
