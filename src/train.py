import argparse
from ultralytics import YOLO
import os
from pathlib import Path


def train_model(args):
    print("🚀 Loading YOLOv11 Nano model...")
    model = YOLO("yolo11n.pt") 

    # Robust path handling
    base_dir = Path(__file__).resolve().parent.parent # This points to 'ml/main'
    data_dir = base_dir / 'data' / 'dataset(tubes)'

    # Read classes from classes.txt
    classes_file = data_dir / 'classes.txt'
    if not classes_file.exists():
        print(f"Error: classes.txt not found at {classes_file}")
        return

    with open(classes_file, 'r') as f:
        company_classes = {i: line.strip() for i, line in enumerate(f.readlines()) if line.strip()}

    print("Detected classes:", company_classes)

    # Create YAML config
    names_yaml = "\n".join([f"  {k}: '{v}'" for k, v in company_classes.items()])
    
    # Use train/val split folders if they exist (created by prepare_dataset.py)
    train_folder = "images/train" if (data_dir / "images" / "train").exists() else "images"
    val_folder = "images/val" if (data_dir / "images" / "val").exists() else "images"

    yaml_content = f'''
path: {data_dir.as_posix()} 
train: {train_folder}
val: {val_folder}

names:
{names_yaml}
'''
    
    yaml_path = data_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"🏋️ Starting training: epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, patience={args.patience}")
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        patience=args.patience,
        
        # ── Heavy Augmentation (useful for small datasets) ──
        augment=True,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=15.0,
        translate=0.2,
        scale=0.6,
        shear=5.0,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.1,
        erasing=0.2,
        
        # ── Regularization ──
        dropout=0.1,
        weight_decay=0.001,
        
        # ── Learning rate ──
        lr0=0.01,
        lrf=0.001,
        warmup_epochs=10,
        
        # ── Output ──
        project=str(base_dir / "runs" / "detect"),
        name=args.name
    )

    print("✅ Training complete!")
    print(f"📁 Weights saved to: {base_dir / 'runs' / 'detect' / args.name / 'weights'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train YOLOv11 model on your dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    parser.add_argument("--name", default="brand_experiment3", help="Run name (output folder)")

    args = parser.parse_args()
    train_model(args)
