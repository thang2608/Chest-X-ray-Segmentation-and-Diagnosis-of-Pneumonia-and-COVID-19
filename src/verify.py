from pathlib import Path
from PIL import Image
import numpy as np


DATA_DIR = Path("data/processed")

CLASSES = [
    "COVID",
    "Lung_Opacity",
    "Normal"
]

for class_name in CLASSES:

    image_dir = DATA_DIR / class_name / "images"
    mask_dir = DATA_DIR / class_name / "masks"

    image_path = next(image_dir.glob("*.png"))
    mask_path = mask_dir / image_path.name

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    print(f"\n===== {class_name} =====")
    print("Image:", image.size)
    print("Mask :", mask.size)

    mask_array = np.array(mask)

    print(
        "Mask unique values:",
        np.unique(mask_array)
    )