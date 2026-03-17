import argparse
import random
import time
from pathlib import Path


def load_classes(classes_path: Path):
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found at {classes_path}")
    with open(classes_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def make_label_file(label_path: Path, class_id: int, boxes=None, img_w=640, img_h=480):
    """
    Writes a YOLO label file.
    If boxes is None, creates a full-frame box.
    boxes should be a list of [x1, y1, x2, y2, cls] in pixel coordinates.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        if boxes:
            for b in boxes:
                # b = [x1, y1, x2, y2, cls]
                bw = max(1, b[2] - b[0])
                bh = max(1, b[3] - b[1])
                xc = b[0] + bw / 2
                yc = b[1] + bh / 2
                # Normalize
                f.write(f"{b[4]} {xc/img_w:.6f} {yc/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}\n")
        else:
            # Fallback to full-frame box
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


def load_annotations(dataset_path: Path):
    """Try to find and load _annotations.txt or similar."""
    ann_file = dataset_path / "images" / "train" / "_annotations.txt"
    if not ann_file.exists():
        ann_file = dataset_path / "_annotations.txt"
    
    if not ann_file.exists():
        return {}

    print(f"📖 Found annotation file: {ann_file.name}")
    data = {}
    with open(ann_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2: continue
            img_name = parts[0]
            # Expecting x1,y1,x2,y2,class
            box_parts = [p.split(',') for p in parts[1:]]
            boxes = []
            for bp in box_parts:
                if len(bp) == 5:
                    boxes.append([float(x) for x in bp])
            data[img_name] = boxes
    return data


def main():
    p = argparse.ArgumentParser(description="Prepare YOLO dataset with real boxes")
    p.add_argument("--dataset", default="data/dataset(tubes)", help="Root dataset folder")
    p.add_argument("--train-pct", type=float, default=0.9, help="Fraction of images to use for training")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--img-w", type=int, default=512, help="Original image width (for normalization)")
    p.add_argument("--img-h", type=int, default=512, help="Original image height (for normalization)")
    p.add_argument("--force", action="store_true", help="Regenerate labels")
    args = p.parse_args()

    base = Path(args.dataset).resolve()
    images_dir = base / "images"
    labels_dir = base / "labels"
    classes_file = base / "classes.txt"
    data_yaml = base / "data.yaml"

    classes = load_classes(classes_file)
    class_map = build_class_map(classes)
    
    # Load real boxes if they exist
    all_annotations = load_annotations(base)
    if all_annotations:
        print(f"🎯 Loaded boxes for {len(all_annotations)} images.")

    imgs = collect_image_paths(images_dir)
    if not imgs:
        raise SystemExit(f"❌ No images found under {images_dir}")

    labels_dir.mkdir(parents=True, exist_ok=True)

    prepared = []
    for img in imgs:
        rel = img.relative_to(images_dir)
        
        # Priority 1: Use boxes from annotations file
        boxes = all_annotations.get(img.name)
        
        class_id = None
        if not boxes:
            # Priority 2: Try folder name
            folder = rel.parts[0] if len(rel.parents) > 1 else None
            if folder:
                normalized = normalize_name(folder)
                for name, idx in class_map.items():
                    if normalize_name(name) == normalized:
                        class_id = idx
                        break
        
        # Skip if no class info at all
        if boxes is None and class_id is None:
            continue

        label_path = labels_dir / (img.stem + ".txt")
        if args.force or not label_path.exists():
            # We assume image size is constant for normalization if using raw pixel coords
            # Usually Roboflow exports are 640x640 or 512x512
            make_label_file(label_path, class_id, boxes=boxes, img_w=args.img_w, img_h=args.img_h)

        prepared.append((img, label_path))

    # Split into train/val
    random.seed(args.seed)
    random.shuffle(prepared)
    split_idx = int(len(prepared) * args.train_pct)
    train = prepared[:split_idx]
    val = prepared[split_idx:]

    def copy_split(items, split_name):
        dst_img = images_dir / split_name
        dst_lbl = labels_dir / split_name
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path in items:
            from shutil import copy2
            copy2(img_path, dst_img / img_path.name)
            copy2(lbl_path, dst_lbl / lbl_path.name)

    print(f"Splitting {len(prepared)}: {len(train)} train / {len(val)} val")
    copy_split(train, "train")
    copy_split(val, "val")

    # Update data.yaml
    yaml_text = f"path: {base.as_posix()}\ntrain: images/train\nval: images/val\n\nnames:\n"
    for i, name in enumerate(classes):
        yaml_text += f"  {i}: '{name}'\n"
    with open(data_yaml, "w") as f:
        f.write(yaml_text)

    print("✅ Dataset prepared with actual boxes where available.")


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
