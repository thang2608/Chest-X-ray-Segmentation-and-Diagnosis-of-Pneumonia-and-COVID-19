from pathlib import Path
from PIL import Image
import numpy as np
import random


DATASET_DIR = Path("COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset")

OUTPUT_DIR = Path("data/processed")

CLASSES = ["COVID", "Lung_Opacity", "Normal"]

IMAGE_SIZE = (224, 224)

MAX_IMAGES_PER_CLASS = 3000

RANDOM_SEED = 42

LABELS = {
    "Normal": 1,
    "Lung_Opacity": 2,
    "COVID": 3,
}


# PREPROCESSING

random.seed(RANDOM_SEED)

for class_name in CLASSES:

    image_dir = DATASET_DIR / class_name / "images"
    mask_dir = DATASET_DIR / class_name / "masks"

    output_image_dir = OUTPUT_DIR / class_name / "images"
    output_mask_dir = OUTPUT_DIR / class_name / "masks"

    # Create output folders
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(image_dir.glob("*.png"))

    # Randomly select maximum 3000 images
    if len(image_paths) > MAX_IMAGES_PER_CLASS:
        image_paths = random.sample(
            image_paths,
            MAX_IMAGES_PER_CLASS
        )

    print(
        f"{class_name}: processing {len(image_paths)} images..."
    )

    # Process images
    for image_path in image_paths:

        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            print(
                f"WARNING: Missing mask: {image_path.name}"
            )
            continue

        # Load image and mask
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Resize image
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR
        )

        # Resize mask
        mask = mask.resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST
        )

        # Convert mask to binary
        mask_array = np.array(mask)

        mask_binary = mask_array > 127

        mask_array = np.zeros_like(mask_array, dtype=np.uint8)
        mask_array[mask_binary] = LABELS[class_name]

        mask = Image.fromarray(mask_array)

        # Save processed image
        image.save(
            output_image_dir / image_path.name
        )

        # Save processed mask
        mask.save(
            output_mask_dir / image_path.name
        )

    print(
        f"{class_name}: preprocessing completed."
    )


print("\nAll preprocessing completed!")



