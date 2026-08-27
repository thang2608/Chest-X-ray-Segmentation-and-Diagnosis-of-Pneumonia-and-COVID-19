from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

# ---- Hằng số dùng chung  ----
CLASS_TO_IDX = {"Normal": 0, "Lung_Opacity": 1, "COVID": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_TO_IDX)

IMAGE_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]   # ImageNet stats
STD  = [0.229, 0.224, 0.225]

# ---- Transforms ----
def get_train_transforms(image_size=IMAGE_SIZE):
    return A.Compose([
        A.Resize(*image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=IMAGE_SIZE):
    return A.Compose([
        A.Resize(*image_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_train_transforms_seg(image_size=IMAGE_SIZE):
    return A.Compose([
        A.Resize(*image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, border_mode=0, p=0.5),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})

def get_val_transforms_seg(image_size=IMAGE_SIZE):
    return A.Compose([
        A.Resize(*image_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})

# ---- Parse label từ prefix filename ----
def _parse_label(filename: str) -> int:
    """
    File name convention: 'COVID-123.png', 'Normal-42.png', 'Lung_Opacity-7.png'
    """
    for cls_name, idx in CLASS_TO_IDX.items():
        if filename.startswith(cls_name):
            return idx
    raise ValueError(f"Cannot parse label from filename: {filename}")

# ---- Classification Dataset ----
class ChestXrayClassificationDataset(Dataset):
    def __init__(self, split_dir: str, transform: Optional[Callable] = None):
        self.image_dir = Path(split_dir) / "images"
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.transform = transform
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No PNG found in {self.image_dir}")
        print(f"📊 Dataset loaded: {len(self.image_paths)} images from {split_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        image = _load_image(path)   # H, W, 3
        label = _parse_label(path.name)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

# ---- Segmentation Dataset ----
class ChestXraySegmentationDataset(Dataset):
    def __init__(self, split_dir: str, transform: Optional[Callable] = None):
        self.image_dir = Path(split_dir) / "images"
        self.mask_dir  = Path(split_dir) / "masks"
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.transform = transform
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No PNG found in {self.image_dir}")
        print(f"📊 Segmentation dataset loaded: {len(self.image_paths)} images from {split_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        msk_path = self.mask_dir / img_path.name
        image = _load_image(img_path)
        mask  = np.array(Image.open(msk_path).convert("L"))    # H, W
        # Binarize mask về {0, 1} bất kể pixel value gốc (1/2/3)
        mask = (mask > 0).astype(np.float32)
        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask)
        return image, mask.unsqueeze(0)   # mask shape (1, H, W)
