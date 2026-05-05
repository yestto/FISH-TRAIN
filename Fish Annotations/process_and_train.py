import os
import shutil
import random
import yaml
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

# Configuration
ROOT_DIR = Path("/mnt/raid7tb/dileep/Suhas/Fish Annotations")
YOLO_DATASET_DIR = ROOT_DIR / "fish_yolo_dataset"
PROJECT_DIR = str(ROOT_DIR / "yolo_training_runs")
TRAIN_RATIO = 0.85
SEED = 42

random.seed(SEED)

def setup_directories():
    """Create fresh YOLO dataset structure."""
    if YOLO_DATASET_DIR.exists():
        print(f"🧹 Cleaning existing dataset directory: {YOLO_DATASET_DIR}")
        shutil.rmtree(YOLO_DATASET_DIR)
        
    for split in ['train', 'val']:
        (YOLO_DATASET_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

def read_mot_gt(gt_path: Path):
    """Read MOT format ground truth and return frame dict."""
    boxes_by_frame = {}
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                frame_id = int(parts[0])
                # MOT format: frame_id, track_id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
                left, top, width, height = map(float, parts[2:6])
                if frame_id not in boxes_by_frame:
                    boxes_by_frame[frame_id] = []
                boxes_by_frame[frame_id].append((left, top, width, height))
    return boxes_by_frame

def convert_to_yolo_format(box, img_width, img_height):
    """Convert top-left MOT box to normalized YOLO center box."""
    left, top, width, height = box
    
    # Calculate center
    x_center = left + (width / 2)
    y_center = top + (height / 2)
    
    # Normalize by image dimensions
    x_norm = max(0.0, min(1.0, x_center / img_width))
    y_norm = max(0.0, min(1.0, y_center / img_height))
    w_norm = max(0.0, min(1.0, width / img_width))
    h_norm = max(0.0, min(1.0, height / img_height))
    
    return x_norm, y_norm, w_norm, h_norm

def process_dataset():
    """Process all valid fish directories and convert to YOLO."""
    print("=" * 60)
    print("🐟 Starting MOT-to-YOLO Dataset Conversion")
    print(f"   Root: {ROOT_DIR}")
    print("=" * 60)
    
    setup_directories()
    all_frames = []
    
    for fish_dir in sorted(ROOT_DIR.iterdir()):
        if not fish_dir.is_dir() or 'yolo' in fish_dir.name.lower() or 'training' in fish_dir.name.lower():
            continue
            
        gt_file = fish_dir / 'gt' / 'gt.txt'
        if not gt_file.exists():
            print(f"⚠️ Skipping {fish_dir.name} - No gt.txt found")
            continue
            
        print(f"Processing {fish_dir.name}...")
        boxes_by_frame = read_mot_gt(gt_file)
        
        # Find all range directories containing frames
        range_dirs = [d for d in fish_dir.iterdir() if d.is_dir() and 'range' in d.name]
        
        for range_dir in range_dirs:
            for img_path in range_dir.glob('*.jpg'):
                # Extract frame ID from filename (e.g., frame_123.jpg -> 123)
                try:
                    frame_id = int(img_path.stem.split('_')[1])
                except (IndexError, ValueError):
                    continue
                
                if frame_id in boxes_by_frame:
                    all_frames.append({
                        'img_path': img_path,
                        'boxes': boxes_by_frame[frame_id],
                        'prefix': f"{fish_dir.name.replace(' ', '_').replace('-', '_')}"
                    })

    # Shuffle and split dataset
    random.shuffle(all_frames)
    split_idx = int(len(all_frames) * TRAIN_RATIO)
    splits = {
        'train': all_frames[:split_idx],
        'val': all_frames[split_idx:]
    }
    
    print(f"\nTotal annotated frames found: {len(all_frames)}")
    print(f"Train split: {len(splits['train'])}")
    print(f"Val split: {len(splits['val'])}")
    
    stats = {'train': 0, 'val': 0, 'corrupt': 0}
    
    # Process images and labels
    for split, frames in splits.items():
        print(f"\nWriting {split} set...")
        for item in frames:
            img_path = item['img_path']
            prefix = item['prefix']
            
            # Check for corrupted images
            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except (UnidentifiedImageError, OSError):
                stats['corrupt'] += 1
                continue
                
            # Create new filenames
            new_img_name = f"{prefix}_{img_path.name}"
            new_lbl_name = f"{prefix}_{img_path.stem}.txt"
            
            dest_img = YOLO_DATASET_DIR / split / 'images' / new_img_name
            dest_lbl = YOLO_DATASET_DIR / split / 'labels' / new_lbl_name
            
            # Copy image
            shutil.copy2(img_path, dest_img)
            
            # Write YOLO labels
            with open(dest_lbl, 'w') as f:
                for box in item['boxes']:
                    x, y, w, h = convert_to_yolo_format(box, img_width, img_height)
                    f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    
            stats[split] += 1
            
    print("\n✅ Conversion Complete!")
    print(f"   Successfully written Train: {stats['train']}")
    print(f"   Successfully written Val: {stats['val']}")
    if stats['corrupt'] > 0:
        print(f"   ⚠️ Skipped Corrupt/Empty: {stats['corrupt']}")

    # Create dataset.yaml
    yaml_content = {
        'path': str(YOLO_DATASET_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'fish'}
    }
    
    yaml_path = YOLO_DATASET_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    return yaml_path

def train_yolo11(yaml_path):
    """Train YOLO11m on the generated dataset."""
    print("=" * 60)
    print("🚀 Starting YOLO11m Training on NEW Dataset")
    print("=" * 60)
    
    model = YOLO("yolo11m.pt")
    
    results = model.train(
        data=str(yaml_path),
        epochs=150,
        imgsz=640,
        batch=16,
        patience=30,
        device="0",
        project=PROJECT_DIR,
        name="fish_yolo11m_full",
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
    yaml_path = process_dataset()
    train_yolo11(yaml_path)
