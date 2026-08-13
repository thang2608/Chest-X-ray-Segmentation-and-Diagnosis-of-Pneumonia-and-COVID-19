from pathlib import Path
from PIL import Image
import numpy as np
import random
from shutil import copy2    
RANDOM_SEED = 42

CLASSES = ["COVID", "Lung_Opacity", "Normal"]

PROCESS_DIR = Path("data/processed")

SPLIT_DIR = Path("data/split")

SPLITS = {
    "train" : 0.7,
    "val"   : 0.15,
    "test"  : 0.15,
}

random.seed(RANDOM_SEED)

# Tạo folders
for split_names in SPLITS:
    output_image_dir = SPLIT_DIR / split_names / "images"
    output_mask_dir = SPLIT_DIR / split_names / "masks"

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

def copy_pairs(image_paths, mask_dir, split_names):

    for image_path in image_paths:

        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            print(f"WARNING: Missing mask: {image_path.name}")
            continue

        copy2(image_path, SPLIT_DIR / split_names / "images")
        copy2(mask_path, SPLIT_DIR / split_names / "masks")

for class_name in CLASSES:

    image_dir = PROCESS_DIR / class_name / "images"
    mask_dir = PROCESS_DIR / class_name / "masks" 

    image_paths = list(image_dir.glob("*.png"))

    random.shuffle(image_paths)

    # Splitting data using index slicing
    n = len(image_paths)

    train_end = int(n * SPLITS["train"])
    val_end = int(train_end + n * SPLITS["val"])

    train_path = image_paths[:train_end]
    val_path = image_paths[train_end:val_end]
    test_path = image_paths[val_end:]

    copy_pairs(train_path, mask_dir, "train")
    copy_pairs(val_path, mask_dir, "val")
    copy_pairs(test_path, mask_dir, "test")

# Kiểm tra size của từng tập

# for split in ["train", "val", "test"]:
#     images_dir = SPLIT_DIR / split / "images"
#     masks_dir = SPLIT_DIR / split / "masks"

#     n_images = len(list(images_dir.iterdir()))
#     n_masks = len(list(masks_dir.iterdir()))

#     print(f"{split}:")
#     print(f"  images: {n_images}")
#     print(f"  masks : {n_masks}")