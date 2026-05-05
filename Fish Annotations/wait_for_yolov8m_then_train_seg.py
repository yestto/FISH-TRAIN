"""
wait_for_yolov8m_then_train_seg.py
===================================
Waits for the already-running YOLOv8m training to finish,
then sequentially trains YOLO11m-seg and YOLO12m-seg.
"""

import subprocess
import time
import os
import sys

SUHAS_DIR = "/mnt/raid7tb/dileep/Suhas/Fish Annotations"
YOLOV8M_WEIGHTS = os.path.join(SUHAS_DIR, "yolo_training_runs/fish_yolov8m_1280/weights/best.pt")

SEG_SCRIPTS = [
    ("YOLO11m-seg 1280", "train_v14_yolo11m_seg.py", "train_log_yolo11m_seg_1280.txt"),
    ("YOLO12m-seg 1280", "train_v14_yolo12m_seg.py", "train_log_yolo12m_seg_1280.txt"),
]

def wait_for_yolov8m():
    print("Waiting for YOLOv8m to finish (checking every 5 min)...", flush=True)
    while not os.path.exists(YOLOV8M_WEIGHTS):
        time.sleep(300)  # check every 5 minutes
        print(f"  Still training... [{time.strftime('%H:%M')}]", flush=True)
    print(f"\n✅ YOLOv8m finished! Weights found at:\n   {YOLOV8M_WEIGHTS}\n", flush=True)

def run_training(name, script, logfile):
    log_path = os.path.join(SUHAS_DIR, logfile)
    print(f"\n{'='*60}", flush=True)
    print(f"  Starting: {name}", flush=True)
    print(f"  Log: {log_path}", flush=True)
    print(f"{'='*60}\n", flush=True)

    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            ["conda", "run", "-n", "su_t", "python", script],
            cwd=SUHAS_DIR,
            stdout=log,
            stderr=subprocess.STDOUT
        )
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
        status = "DONE" if rc == 0 else f"FAILED (exit {rc})"
        print(f"\n  {name}: {status}\n", flush=True)
        return rc == 0

if __name__ == "__main__":
    print("=== Seg Training Queue (waits for YOLOv8m first) ===")
    wait_for_yolov8m()

    for name, script, logfile in SEG_SCRIPTS:
        ok = run_training(name, script, logfile)
        if not ok:
            print(f"Stopped: {name} failed. Check log.")
            sys.exit(1)

    print("\n=== All segmentation models done! ===")
    print("Run: python compare_models.py")
