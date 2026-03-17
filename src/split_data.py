import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, train_pct=0.9):
    # Resolve paths relative to this script location for safety
    # Assumes structure: ml/src/split_data.py -> data is at ml/data
    base_dir = Path(__file__).resolve().parent.parent
    source_path = base_dir / 'data'
    images_dir = source_path / 'images'
    labels_dir = source_path / 'labels'
    
    # 1. Clear old splits
    if images_dir.exists(): shutil.rmtree(images_dir)
    if labels_dir.exists(): shutil.rmtree(labels_dir)

    # 2. Create folders
    for split in ['train', 'val']:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    # 3. Get images
    all_files = os.listdir(source_path)
    images = [f for f in all_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if not images:
        print("❌ No images found in ml/data/! Please dump your images there first.")
        return

    random.shuffle(images)
    
    # 4. Split
    split_idx = int(len(images) * train_pct)
    train_files = images[:split_idx]
    val_files = images[split_idx:]

    # 5. Move Function
    def move_files(file_list, split_name):
        for img_name in file_list:
            src_img = source_path / img_name
            src_txt = source_path / (os.path.splitext(img_name)[0] + '.txt')
            
            dest_img = images_dir / split_name / img_name
            dest_txt = labels_dir / split_name / src_txt.name
            
            shutil.copy(src_img, dest_img)
            if src_txt.exists():
                shutil.copy(src_txt, dest_txt)

    print(f"🔄 Processing {len(images)} images...")
    move_files(train_files, 'train')
    move_files(val_files, 'val')
    print(f"✅ Done! Data split inside 'ml/data/'.")

if __name__ == "__main__":
    split_dataset('../data')
