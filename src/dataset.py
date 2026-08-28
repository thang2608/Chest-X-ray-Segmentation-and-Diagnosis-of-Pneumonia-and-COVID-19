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

# ---- Crop ảnh theo bounding box của mask phổi ----
def crop_to_lung_bbox(image: np.ndarray, mask: np.ndarray, padding: float = 0.1) -> np.ndarray:
    """Cắt `image` theo bounding box của vùng phổi trong `mask`, có đệm biên.

    Dùng để loại bỏ vật lý vùng NGOÀI phổi (nơi watermark/artifact gây shortcut
    learning thường nằm — xem docs/LY_THUYET.md Phần VIII, docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md
    Phần 4.1) khỏi input của classifier, thay vì chỉ hy vọng model tự học bỏ qua nó.

    Args:
        image: (H, W, 3) hoặc (H, W) — ảnh cần cắt, CÙNG kích thước với mask.
        mask: (H, W) — mask nhị phân (giá trị >0 = phổi), dùng để tính bounding box.
        padding: tỉ lệ đệm thêm quanh bounding box (theo % kích thước box mỗi chiều)
            — tránh cắt sát rìa phổi làm mất chi tiết biên hữu ích cho chẩn đoán.

    Returns:
        Ảnh đã cắt, kích thước nhỏ hơn hoặc bằng ảnh gốc. Trả nguyên `image` không đổi
        nếu `mask` rỗng (không tìm thấy phổi — an toàn, tránh crash khi mask lỗi thay
        vì trả về ảnh rỗng vô nghĩa).
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        return image

    H, W = mask.shape[:2]
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    pad_y = int((y1 - y0) * padding)
    pad_x = int((x1 - x0) * padding)

    y0 = max(0, y0 - pad_y)
    y1 = min(H, y1 + pad_y + 1)
    x0 = max(0, x0 - pad_x)
    x1 = min(W, x1 + pad_x + 1)

    return image[y0:y1, x0:x1]


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
    def __init__(
        self,
        split_dir: str,
        transform: Optional[Callable] = None,
        crop_to_lung: bool = False,
        crop_padding: float = 0.1,
    ):
        """
        crop_to_lung: nếu True, cắt ảnh theo bounding box mask phổi (ground-truth,
            đọc từ split_dir/masks/) TRƯỚC khi áp transform — xem crop_to_lung_bbox().
            Dùng để train phiên bản classifier "đã tối ưu" (xem
            docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md, notebooks/train_classifier_cropped.ipynb).
            Mặc định False — KHÔNG đổi hành vi các chỗ đã dùng class này trước đó
            (src/shortcut_iou.py, notebooks/evaluate_local.ipynb, train_classifier.ipynb).
        crop_padding: truyền thẳng cho crop_to_lung_bbox(), chỉ có tác dụng khi
            crop_to_lung=True.
        """
        self.image_dir = Path(split_dir) / "images"
        self.mask_dir = Path(split_dir) / "masks"
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.transform = transform
        self.crop_to_lung = crop_to_lung
        self.crop_padding = crop_padding
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No PNG found in {self.image_dir}")
        print(f"📊 Dataset loaded: {len(self.image_paths)} images from {split_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        image = _load_image(path)   # H, W, 3
        label = _parse_label(path.name)
        if self.crop_to_lung:
            mask = np.array(Image.open(self.mask_dir / path.name).convert("L"))
            image = crop_to_lung_bbox(image, (mask > 0).astype(np.uint8), padding=self.crop_padding)
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
