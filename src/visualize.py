from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random


DATA_DIR = Path("data/processed")

CLASSES = [
    "COVID",
    "Lung_Opacity",
    "Normal"
]


fig, axes = plt.subplots(
    3, 3,
    figsize=(12, 12)
)


for row, class_name in enumerate(CLASSES):

    image_dir = DATA_DIR / class_name / "images"
    mask_dir = DATA_DIR / class_name / "masks"

    image_paths = list(image_dir.glob("*.png"))

    # Random sample
    image_path = random.choice(image_paths)
    mask_path = mask_dir / image_path.name

    image = Image.open(image_path)
    mask = Image.open(mask_path)


    # X-ray
    axes[row, 0].imshow(
        image,
        cmap="gray"
    )

    axes[row, 0].set_title(
        f"{class_name} - X-ray"
    )

    axes[row, 0].axis("off")


    # Mask
    axes[row, 1].imshow(
        mask,
        cmap="gray"
    )

    axes[row, 1].set_title(
        f"{class_name} - Mask"
    )

    axes[row, 1].axis("off")


    # Overlay
    axes[row, 2].imshow(
        image,
        cmap="gray"
    )

    axes[row, 2].imshow(
        mask,
        alpha=0.4
    )

    axes[row, 2].set_title(
        f"{class_name} - Overlay"
    )

    axes[row, 2].axis("off")


plt.tight_layout()
plt.show()