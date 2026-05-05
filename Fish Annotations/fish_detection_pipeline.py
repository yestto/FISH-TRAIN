"""
Fish Detection Pipeline - Complete End-to-End Script
=====================================================
Extracts frames from videos, converts MOT 1.1 annotations to YOLO format,
splits into train/val/test sets, and trains YOLOv8 model.

Usage:
    python fish_detection_pipeline.py --prepare    # Prepare dataset only
    python fish_detection_pipeline.py --train      # Train YOLOv8 model
    python fish_detection_pipeline.py --all        # Full pipeline
"""

import os
import sys
import cv2
import csv
import json
import shutil
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
ANNOTATIONS_DIR = Path(r"c:\Users\shain\Downloads\fish_annotated_MOT-1.1-downloads")
VIDEOS_DIR = Path(r"c:\Users\shain\Downloads\FISH DATASET\FISH DATASET")
OUTPUT_DIR = ANNOTATIONS_DIR / "fish_yolo_dataset"

# Dataset splits
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Frame sampling: extract every Nth frame to avoid near-duplicate training data
FRAME_SAMPLE_RATE = 5  # Every 5th frame

# Single class detection
CLASS_NAME = "fish"
CLASS_ID = 0

# ============================================================================
# VIDEO-TO-ANNOTATION MAPPING
# ============================================================================

# Map annotation folder names to video paths (relative to VIDEOS_DIR)
ANNOTATION_TO_VIDEO_MAP = {
    # Fish 01
    "fish01-front":   ("fish01/front view", "front.mp4.mp4"),
    "fish1_top":      ("fish01/top view", None),  # Will auto-detect mp4
    
    # Fish 02
    "fish2-front":    ("fish2/front view", None),
    "fish2_top":      ("fish2/top view", None),
    
    # Fish 03
    "fish3_front":    ("fish3/front view", None),
    "fish3_top":      ("fish3/top view", None),
    
    # Fish 04
    "fish4_front":    ("fish4/front view", None),
    "fish4_top":      ("fish4/top view", None),
    
    # Fish 05
    "fish5_front":    ("fish5/front view", None),
    "fish5_top_moto": ("fish5/top view", None),
    
    # Fish 06
    "fish_front 06":  ("fish6/front view", None),
    "fish6-top":      ("fish6/top view", None),
    
    # Fish 07
    "fish-7-front":   ("fish7/front view", None),
    "fish-7-top":     ("fish7/top view", None),
    
    # Fish 08
    "fish8_front":    ("fish8/front view", None),
    "fish8_top":      ("fish8/top view", None),
    
    # Fish 09
    "fish-9-front":   ("fish9/front view", None),
    "fish-9-top":     ("fish9/top view", None),
    
    # Fish 11
    "fish 11 front":  ("fish11/front view", None),
    "fish 11 top":    ("fish11/top view", None),
    
    # Fish 12
    "fish 12-front":  ("fish12/front view", None),
    "fish 12 top":    ("fish12/top view", None),
    
    # Fish 13
    "fish 13 front":  ("fish13/front view", None),
    "fish top 13":    ("fish13/top view", None),
    
    # Fish 14
    "fish14_front":   ("fish14/front view", None),
    "fish14_top":     ("fish14/top view", None),
    
    # Fish 15
    "fish15_front":   ("fish15/front view", None),
    "fish15_top":     ("fish15/top view", None),
}


def find_video_file(video_dir: Path, filename: str = None) -> Path:
    """Find the video file in a directory. Auto-detect if filename is None."""
    if filename:
        video_path = video_dir / filename
        if video_path.exists():
            return video_path
    # Auto-detect mp4
    mp4_files = list(video_dir.glob("*.mp4"))
    if mp4_files:
        return mp4_files[0]
    raise FileNotFoundError(f"No video file found in {video_dir}")


def parse_mot_annotations(gt_file: Path) -> dict:
    """
    Parse MOT 1.1 ground truth file.
    Format: frame_id, track_id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    
    Returns: dict[frame_id] -> list of (x, y, w, h) bounding boxes
    """
    annotations = defaultdict(list)
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            # track_id = int(parts[1])
            bb_left = float(parts[2])
            bb_top = float(parts[3])
            bb_width = float(parts[4])
            bb_height = float(parts[5])
            annotations[frame_id].append((bb_left, bb_top, bb_width, bb_height))
    return annotations


def bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert MOT bbox (x_left, y_top, width, height) in pixels 
    to YOLO format (class_id, x_center, y_center, width, height) normalized [0,1].
    """
    x_left, y_top, w, h = bbox
    
    x_center = (x_left + w / 2.0) / img_width
    y_center = (y_top + h / 2.0) / img_height
    norm_w = w / img_width
    norm_h = h / img_height
    
    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))
    
    return CLASS_ID, x_center, y_center, norm_w, norm_h


def extract_and_convert(annotation_name: str, video_subdir: str, video_filename: str,
                        output_images_dir: Path, output_labels_dir: Path,
                        sample_rate: int = FRAME_SAMPLE_RATE) -> dict:
    """
    Extract frames from video and create corresponding YOLO label files.
    Returns stats dict.
    """
    # Find video file
    video_dir = VIDEOS_DIR / video_subdir
    if not video_dir.exists():
        print(f"  ⚠ Video directory not found: {video_dir}")
        return {"status": "skipped", "reason": "video_dir_not_found"}
    
    try:
        video_path = find_video_file(video_dir, video_filename)
    except FileNotFoundError as e:
        print(f"  ⚠ {e}")
        return {"status": "skipped", "reason": "video_not_found"}
    
    # Parse annotations
    gt_file = ANNOTATIONS_DIR / annotation_name / "gt" / "gt.txt"
    if not gt_file.exists():
        print(f"  ⚠ Annotation file not found: {gt_file}")
        return {"status": "skipped", "reason": "annotation_not_found"}
    
    annotations = parse_mot_annotations(gt_file)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ⚠ Cannot open video: {video_path}")
        return {"status": "skipped", "reason": "video_open_failed"}
    
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    extracted = 0
    skipped = 0
    frame_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        
        # Sample every Nth frame
        if frame_id % sample_rate != 0:
            skipped += 1
            continue
        
        # Check if this frame has annotations
        if frame_id not in annotations:
            skipped += 1
            continue
        
        # Create unique filename
        safe_name = annotation_name.replace(" ", "_").replace("-", "_")
        img_filename = f"{safe_name}_frame_{frame_id:06d}.jpg"
        label_filename = f"{safe_name}_frame_{frame_id:06d}.txt"
        
        # Save image
        img_path = output_images_dir / img_filename
        cv2.imwrite(str(img_path), frame)
        
        # Save YOLO label
        label_path = output_labels_dir / label_filename
        with open(label_path, 'w') as f:
            for bbox in annotations[frame_id]:
                yolo_bbox = bbox_to_yolo(bbox, img_width, img_height)
                f.write(f"{yolo_bbox[0]} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f} {yolo_bbox[4]:.6f}\n")
        
        extracted += 1
    
    cap.release()
    
    return {
        "status": "ok",
        "video": str(video_path.name),
        "resolution": f"{img_width}x{img_height}",
        "total_frames": total_frames,
        "annotation_frames": len(annotations),
        "extracted_frames": extracted,
        "skipped_frames": skipped,
    }


def prepare_dataset():
    """
    Full data preparation pipeline:
    1. Extract frames from all videos
    2. Convert MOT annotations to YOLO format
    3. Split into train/val/test
    """
    print("=" * 70)
    print("🐟 FISH DETECTION DATASET PREPARATION")
    print("=" * 70)
    
    # Create temporary staging area
    staging_images = OUTPUT_DIR / "_staging" / "images"
    staging_labels = OUTPUT_DIR / "_staging" / "labels"
    staging_images.mkdir(parents=True, exist_ok=True)
    staging_labels.mkdir(parents=True, exist_ok=True)
    
    # Process each annotation folder
    all_stats = {}
    total_extracted = 0
    
    for ann_name, (video_subdir, video_file) in ANNOTATION_TO_VIDEO_MAP.items():
        print(f"\n📂 Processing: {ann_name}")
        print(f"   Video dir: {video_subdir}")
        
        stats = extract_and_convert(
            ann_name, video_subdir, video_file,
            staging_images, staging_labels,
            sample_rate=FRAME_SAMPLE_RATE
        )
        
        all_stats[ann_name] = stats
        if stats["status"] == "ok":
            total_extracted += stats["extracted_frames"]
            print(f"   ✅ Extracted {stats['extracted_frames']} frames "
                  f"({stats['resolution']}, {stats['total_frames']} total)")
        else:
            print(f"   ❌ Skipped: {stats.get('reason', 'unknown')}")
    
    print(f"\n{'=' * 70}")
    print(f"📊 Total extracted frames: {total_extracted}")
    print(f"{'=' * 70}")
    
    if total_extracted == 0:
        print("❌ No frames extracted! Check video paths.")
        return
    
    # Get all image files
    all_images = sorted(list(staging_images.glob("*.jpg")))
    print(f"\n🔀 Splitting {len(all_images)} images into train/val/test...")
    
    # Shuffle with seed for reproducibility
    random.seed(42)
    random.shuffle(all_images)
    
    # Split
    n_total = len(all_images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train + n_val]
    test_images = all_images[n_train + n_val:]
    
    print(f"   Train: {len(train_images)}")
    print(f"   Val:   {len(val_images)}")
    print(f"   Test:  {len(test_images)}")
    
    # Create final directory structure
    for split_name, split_images in [("train", train_images), ("val", val_images), ("test", test_images)]:
        split_img_dir = OUTPUT_DIR / split_name / "images"
        split_lbl_dir = OUTPUT_DIR / split_name / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in split_images:
            # Copy image
            shutil.copy2(str(img_path), str(split_img_dir / img_path.name))
            
            # Copy label
            label_name = img_path.stem + ".txt"
            label_src = staging_labels / label_name
            if label_src.exists():
                shutil.copy2(str(label_src), str(split_lbl_dir / label_name))
    
    # Create data.yaml
    data_yaml = OUTPUT_DIR / "data.yaml"
    yaml_content = f"""# Fish Detection Dataset - YOLOv8
# Auto-generated by fish_detection_pipeline.py

path: {OUTPUT_DIR.as_posix()}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: fish

# Dataset info
nc: 1
"""
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Dataset created at: {OUTPUT_DIR}")
    print(f"   data.yaml: {data_yaml}")
    
    # Save stats report
    stats_file = OUTPUT_DIR / "dataset_stats.json"
    report = {
        "total_extracted": total_extracted,
        "train_count": len(train_images),
        "val_count": len(val_images),
        "test_count": len(test_images),
        "sample_rate": FRAME_SAMPLE_RATE,
        "per_video_stats": all_stats,
    }
    with open(stats_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Cleanup staging
    print("\n🧹 Cleaning up staging files...")
    shutil.rmtree(str(OUTPUT_DIR / "_staging"))
    
    print("✅ Dataset preparation complete!\n")
    return report


def train_yolov8(model_size='s', epochs=100, imgsz=640, batch=16):
    """Train YOLOv8 model on the prepared fish dataset."""
    from ultralytics import YOLO
    
    data_yaml = OUTPUT_DIR / "data.yaml"
    if not data_yaml.exists():
        print("❌ Dataset not prepared! Run with --prepare first.")
        return None
    
    print("=" * 70)
    print(f"🏋️ TRAINING YOLOv8{model_size} Fish Detector")
    print("=" * 70)
    
    # Load pretrained model
    model_name = f"yolov8{model_size}.pt"
    print(f"\n📦 Loading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    # Train
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch}")
    print(f"   Data: {data_yaml}")
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=f"fish_detector_yolov8{model_size}",
        project=str(OUTPUT_DIR / "runs"),
        patience=20,          # Early stopping
        save=True,            # Save checkpoints
        save_period=10,       # Save every 10 epochs
        plots=True,           # Generate training plots
        verbose=True,
        workers=0,            # Fix Windows paging file / DLL reload errors
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.5,          # Fish can be upside down
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )
    
    print("\n✅ Training complete!")
    print(f"   Best model saved at: {results.save_dir}/weights/best.pt")
    
    return results


def evaluate_model(model_path=None, model_size='s'):
    """Evaluate the trained model on the test set."""
    from ultralytics import YOLO
    
    if model_path is None:
        # Find latest trained model
        runs_dir = OUTPUT_DIR / "runs"
        model_dirs = sorted(runs_dir.glob(f"fish_detector_yolov8{model_size}*"))
        if not model_dirs:
            print("❌ No trained model found! Run with --train first.")
            return
        model_path = model_dirs[-1] / "weights" / "best.pt"
    
    print("=" * 70)
    print(f"📊 EVALUATING MODEL: {model_path}")
    print("=" * 70)
    
    model = YOLO(str(model_path))
    
    # Validate on test set
    data_yaml = OUTPUT_DIR / "data.yaml"
    results = model.val(
        data=str(data_yaml),
        split="test",
        verbose=True,
        plots=True,
    )
    
    print("\n📈 Test Results:")
    print(f"   mAP@0.5:      {results.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"   Precision:     {results.box.mp:.4f}")
    print(f"   Recall:        {results.box.mr:.4f}")
    
    # Export model to ONNX
    print("\n📦 Exporting model to ONNX...")
    model.export(format="onnx")
    print("✅ Model exported!")
    
    return results


def visualize_predictions(model_path=None, model_size='s', num_samples=10):
    """Visualize predictions on random test images."""
    from ultralytics import YOLO
    
    if model_path is None:
        runs_dir = OUTPUT_DIR / "runs"
        model_dirs = sorted(runs_dir.glob(f"fish_detector_yolov8{model_size}*"))
        if not model_dirs:
            print("❌ No trained model found!")
            return
        model_path = model_dirs[-1] / "weights" / "best.pt"
    
    model = YOLO(str(model_path))
    
    # Get test images
    test_images = list((OUTPUT_DIR / "test" / "images").glob("*.jpg"))
    if not test_images:
        print("❌ No test images found!")
        return
    
    # Sample random images
    samples = random.sample(test_images, min(num_samples, len(test_images)))
    
    # Create predictions directory
    pred_dir = OUTPUT_DIR / "predictions"
    pred_dir.mkdir(exist_ok=True)
    
    print(f"\n🔍 Running inference on {len(samples)} test images...")
    for img_path in samples:
        results = model(str(img_path), verbose=False)
        annotated = results[0].plot()
        output_path = pred_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(output_path), annotated)
    
    print(f"✅ Predictions saved to: {pred_dir}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fish Detection Pipeline")
    parser.add_argument("--prepare", action="store_true", help="Prepare dataset from videos + MOT annotations")
    parser.add_argument("--train", action="store_true", help="Train YOLOv8 model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model")
    parser.add_argument("--visualize", action="store_true", help="Visualize predictions")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--model-size", type=str, default="s", choices=["n", "s", "m", "l", "x"],
                        help="YOLOv8 model size (default: s)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--sample-rate", type=int, default=5, help="Frame sample rate (default: 5)")
    
    args = parser.parse_args()
    
    # Update global sample rate if specified
    if args.sample_rate != FRAME_SAMPLE_RATE:
        FRAME_SAMPLE_RATE = args.sample_rate
    
    if args.all:
        args.prepare = True
        args.train = True
        args.evaluate = True
        args.visualize = True
    
    if not any([args.prepare, args.train, args.evaluate, args.visualize]):
        parser.print_help()
        sys.exit(1)
    
    if args.prepare:
        prepare_dataset()
    
    if args.train:
        train_yolov8(
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
        )
    
    if args.evaluate:
        evaluate_model(model_size=args.model_size)
    
    if args.visualize:
        visualize_predictions(model_size=args.model_size)
