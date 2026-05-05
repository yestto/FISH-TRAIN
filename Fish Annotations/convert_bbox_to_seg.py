"""
convert_bbox_to_seg.py
======================
Converts YOLO bounding-box labels (5 values: class cx cy w h)
to YOLO segmentation labels (polygon: class x1 y1 x2 y2 ... xn yn)
by approximating each box as an 8-point ellipse.

Run ONCE before training any *-seg model.
"""

import os
import shutil
import numpy as np
from pathlib import Path

SRC_ROOT = "/mnt/raid7tb/dileep/Suhas/Fish Annotations/fish_yolo_dataset"
DST_ROOT = "/mnt/raid7tb/dileep/Suhas/Fish Annotations/fish_yolo_dataset_seg"

N_POINTS = 16  # number of polygon points per fish (more = tighter ellipse)


def bbox_to_ellipse_polygon(cx, cy, w, h, n=N_POINTS):
    """
    Convert normalized bbox (cx, cy, w, h) to a normalized n-point
    ellipse polygon. Clips all values to [0, 1].
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Semi-axes
    a = w / 2.0  # horizontal
    b = h / 2.0  # vertical
    xs = np.clip(cx + a * np.cos(angles), 0.0, 1.0)
    ys = np.clip(cy + b * np.sin(angles), 0.0, 1.0)
    # Interleave x, y
    points = np.column_stack([xs, ys]).flatten()
    return points


def convert_label_file(src_path, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    lines_out = []
    with open(src_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls = parts[0]
                cx, cy, w, h = map(float, parts[1:])
                poly = bbox_to_ellipse_polygon(cx, cy, w, h)
                poly_str = " ".join(f"{v:.6f}" for v in poly)
                lines_out.append(f"{cls} {poly_str}")
            elif len(parts) > 5:
                # Already polygon format — copy as-is
                lines_out.append(line.strip())
    with open(dst_path, "w") as f:
        f.write("\n".join(lines_out) + "\n")


def convert_split(split):
    src_img_dir = Path(SRC_ROOT) / split / "images"
    src_lbl_dir = Path(SRC_ROOT) / split / "labels"
    dst_img_dir = Path(DST_ROOT) / split / "images"
    dst_lbl_dir = Path(DST_ROOT) / split / "labels"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Symlink images (saves disk space)
    for img in src_img_dir.iterdir():
        dst = dst_img_dir / img.name
        if not dst.exists():
            dst.symlink_to(img.resolve())

    # Convert labels
    label_files = list(src_lbl_dir.glob("*.txt"))
    print(f"  {split}: converting {len(label_files)} label files...")
    for lbl in label_files:
        dst_lbl = dst_lbl_dir / lbl.name
        convert_label_file(lbl, dst_lbl)
    print(f"  {split}: done.")


def write_data_yaml():
    yaml_content = f"""names:
  0: fish
path: {DST_ROOT}
train: train/images
val: val/images
"""
    with open(os.path.join(DST_ROOT, "data.yaml"), "w") as f:
        f.write(yaml_content)
    print(f"  Written: {DST_ROOT}/data.yaml")


if __name__ == "__main__":
    print("=== BBox → Segmentation Polygon Conversion ===")
    for split in ["train", "val"]:
        convert_split(split)
    write_data_yaml()
    print("\nConversion complete!")
    print(f"Segmentation dataset at: {DST_ROOT}")
    print("Now train with: yolo11m-seg.pt or yolo12m-seg.pt using this dataset.")
