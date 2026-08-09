from pathlib import Path
from PIL import Image
import numpy as np

DATASET_DIR = Path(
    "COVID-19_Radiography_Dataset"
)

OUTPUT_DIR = Path(
    "data/processed"
)

CLASSES = ["COVID", "Lung_Opacity", "Normal"]

IMAGE_SIZE = (224, 224)


# =========================
# PREPROCESSING
# =========================

for class_name in CLASSES:

    image_dir = DATASET_DIR / class_name / "images"
    mask_dir = DATASET_DIR / class_name / "masks"

    output_image_dir = (
        OUTPUT_DIR / class_name / "images"
    )

    output_mask_dir = (
        OUTPUT_DIR / class_name / "masks"
    )

    # Create output folders
    output_image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_mask_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Process every image
    for image_path in image_dir.glob("*.png"):

        # =========================
        # IMAGE
        # =========================

        image = Image.open(image_path).convert("L")

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR
        )

        image.save(
            output_image_dir / image_path.name
        )

        # =========================
        # MASK
        # =========================

        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            print(
                f"WARNING: Missing mask: "
                f"{image_path.name}"
            )
            continue

        mask = Image.open(mask_path).convert("L")

        # IMPORTANT:
        # Use NEAREST for segmentation masks
        mask = mask.resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST
        )

        # Convert mask to binary
        mask_array = np.array(mask)

        mask_array = (
            mask_array > 127
        ).astype(np.uint8) * 255

        mask = Image.fromarray(mask_array)

        mask.save(
            output_mask_dir / mask_path.name
        )

    print(
        f"{class_name}: preprocessing completed."
    )


print("\nAll preprocessing completed!")