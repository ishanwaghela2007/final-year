import argparse
from pathlib import Path

from prepare_dataset import load_classes, make_label_file


def label_unlabeled_train_images(
    dataset_root: Path,
    default_class: int,
    overwrite: bool = False,
) -> None:
    """
    Create or update YOLO label files for images in images/train.

    - If overwrite=False: only creates labels for images that do not
      already have a corresponding labels/train .txt file.
    - If overwrite=True: rewrites labels for all matching images.

    Bounding box covers the full image (single-tube assumption).
    """
    base = dataset_root.resolve()
    images_dir = base / "images" / "train"
    labels_dir = base / "labels" / "train"
    classes_file = base / "classes.txt"

    if not images_dir.exists():
        raise SystemExit(f"images/train not found at {images_dir}")

    labels_dir.mkdir(parents=True, exist_ok=True)

    classes = load_classes(classes_file)
    if not classes:
        raise SystemExit(f"No classes found in {classes_file}")

    if default_class < 0 or default_class >= len(classes):
        raise SystemExit(
            f"Invalid default class index {default_class}. "
            f"Valid range: 0..{len(classes) - 1}"
        )

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    created = 0
    overwritten = 0
    skipped_existing = 0

    for img in images_dir.iterdir():
        if not img.is_file() or img.suffix.lower() not in exts:
            continue

        label_path = labels_dir / f"{img.stem}.txt"
        if label_path.exists() and not overwrite:
            skipped_existing += 1
            continue

        if label_path.exists() and overwrite:
            overwritten += 1
        else:
            created += 1

        make_label_file(label_path, default_class)
        print(
            f"Label {label_path.name} -> class {default_class} "
            f"('{classes[default_class]}')"
        )

    print(
        f"Labeling complete. Created {created} new labels, "
        f"overwrote {overwritten} labels, "
        f"skipped {skipped_existing} images that already had labels."
    )


def resolve_class_index(classes, class_id: int | None, class_name: str | None) -> int:
    if class_id is not None:
        if 0 <= class_id < len(classes):
            return class_id
        raise SystemExit(
            f"class-id {class_id} out of range. "
            f"Valid range: 0..{len(classes) - 1}"
        )

    if class_name is not None:
        lookup = {c.strip().lower(): i for i, c in enumerate(classes)}
        key = class_name.strip().lower()
        if key not in lookup:
            raise SystemExit(
                f"class-name '{class_name}' not found in classes.txt. "
                f"Available: {classes}"
            )
        return lookup[key]

    # Fallback: ask user once which index to use
    print("Available classes:")
    for i, c in enumerate(classes):
        print(f"  {i}: {c}")
    selected = input("Enter class index to use for these images: ").strip()
    try:
        idx = int(selected)
    except ValueError:
        raise SystemExit(f"Invalid index '{selected}'. Expected an integer.")
    if not (0 <= idx < len(classes)):
        raise SystemExit(
            f"Index {idx} out of range. Valid range: 0..{len(classes) - 1}"
        )
    return idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create or update YOLO labels for images/train."
    )
    parser.add_argument(
        "--dataset",
        default="data/dataset(tubes)",
        help="Root dataset folder containing images/ and labels/ subfolders",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Class index to use for these images",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default=None,
        help="Class name (from classes.txt) to use for these images",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite labels even if they already exist",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset)

    # Load classes once to resolve the desired index
    classes = load_classes(dataset_root / "classes.txt")
    class_idx = resolve_class_index(classes, args.class_id, args.class_name)

    label_unlabeled_train_images(
        dataset_root=dataset_root,
        default_class=class_idx,
        overwrite=args.overwrite,
    )

