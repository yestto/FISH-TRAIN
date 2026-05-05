"""
compare_models.py
=================
Validates all trained YOLO models and prints a comparison table.
Run after any/all models finish training.
"""

from ultralytics import YOLO
import os

DATA_DET = "/mnt/raid7tb/dileep/Suhas/Fish Annotations/fish_yolo_dataset/data.yaml"
DATA_SEG  = "/mnt/raid7tb/dileep/Suhas/Fish Annotations/fish_yolo_dataset_seg/data.yaml"
RUNS_DIR  = "/mnt/raid7tb/dileep/Suhas/Fish Annotations/yolo_training_runs"

MODELS = [
    # (display_name, weights_path, data_yaml, type)
    ("YOLO11m det  640",  f"{RUNS_DIR}/fish_yolo11m_full/weights/best.pt",      DATA_DET, "det"),
    ("YOLO12m det  640",  f"{RUNS_DIR}/fish_yolo12m_full/weights/best.pt",      DATA_DET, "det"),
    ("YOLOv8m det  1280", f"{RUNS_DIR}/fish_yolov8m_1280/weights/best.pt",      DATA_DET, "det"),
    ("YOLO11m seg  1280", f"{RUNS_DIR}/fish_yolo11m_seg_1280/weights/best.pt",  DATA_SEG, "seg"),
    ("YOLO12m seg  1280", f"{RUNS_DIR}/fish_yolo12m_seg_1280/weights/best.pt",  DATA_SEG, "seg"),
]

def main():
    print("\n" + "=" * 70)
    print(f"{'Model':<22} | {'mAP50':>7} | {'mAP50-95':>9} | {'Mask mAP50':>10} | {'Mask mAP50-95':>13}")
    print("=" * 70)

    for name, weights, data, mtype in MODELS:
        if not os.path.exists(weights):
            print(f"  {name:<20} | {'(not trained yet)':>45}")
            continue
        try:
            model   = YOLO(weights)
            metrics = model.val(data=data, imgsz=1280 if "1280" in name else 640,
                                verbose=False, plots=False)

            map50    = metrics.box.map50
            map5095  = metrics.box.map
            seg50    = metrics.seg.map50    if mtype == "seg" else "—"
            seg5095  = metrics.seg.map      if mtype == "seg" else "—"

            seg50_s   = f"{seg50:.4f}"   if isinstance(seg50, float)   else seg50
            seg5095_s = f"{seg5095:.4f}" if isinstance(seg5095, float) else seg5095

            print(f"  {name:<20} | {map50:>7.4f} | {map5095:>9.4f} | {seg50_s:>10} | {seg5095_s:>13}")
        except Exception as e:
            print(f"  {name:<20} | ERROR: {e}")

    print("=" * 70)
    print("\nDone.")

if __name__ == "__main__":
    main()
