"""
wait_and_train_seg_pipeline.py
==============================
Waits for YOLOv8m training to finish, then runs YOLO11m-seg, then YOLO12m-seg.
Run in a separate terminal so all 3 models train sequentially overnight.
"""

import subprocess
import time
import os
import sys

SUHAS_DIR = "/mnt/raid7tb/dileep/Suhas/Fish Annotations"
SCRIPTS = [
    ("YOLOv8m det  1280",  "train_v14_yolov8m.py",    "train_log_yolov8m_1280.txt"),
    ("YOLO11m-seg  1280",  "train_v14_yolo11m_seg.py", "train_log_yolo11m_seg_1280.txt"),
    ("YOLO12m-seg  1280",  "train_v14_yolo12m_seg.py", "train_log_yolo12m_seg_1280.txt"),
]

def run_training(name, script, logfile):
    log_path = os.path.join(SUHAS_DIR, logfile)
    print(f"\n{'='*60}")
    print(f"  Starting: {name}")
    print(f"  Log: {log_path}")
    print(f"{'='*60}\n", flush=True)

    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            ["conda", "run", "-n", "su_t", "python", script],
            cwd=SUHAS_DIR,
            stdout=log,
            stderr=subprocess.STDOUT
        )
        # Poll and print last line every 2 minutes
        while proc.poll() is None:
            time.sleep(120)
            try:
                with open(log_path, "r") as f:
                    lines = [l for l in f.readlines() if l.strip()]
                    if lines:
                        print(f"  [{name}] {lines[-1].strip()}", flush=True)
            except Exception:
                pass
        rc = proc.returncode
        status = "✅ DONE" if rc == 0 else f"❌ FAILED (exit {rc})"
        print(f"\n  {name}: {status}\n", flush=True)
        return rc == 0

if __name__ == "__main__":
    print("=== Sequential YOLO Training Pipeline ===")
    print("Will train: YOLOv8m → YOLO11m-seg → YOLO12m-seg")
    print("Logs saved per model in the Suhas/Fish Annotations folder.")

    for name, script, logfile in SCRIPTS:
        success = run_training(name, script, logfile)
        if not success:
            print(f"Pipeline stopped: {name} failed. Check log.")
            sys.exit(1)

    print("\n=== All 3 models trained! ===")
    print("Run compare_models.py to see the mAP50-95 comparison table.")
