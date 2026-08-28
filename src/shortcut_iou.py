import csv
import os
import random
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    ChestXrayClassificationDataset,
    get_val_transforms,
)
from src.gradcam import generate_gradcam
from src.model import load_classifier
from src.unet import build_unet


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def binarize(x: np.ndarray, thresh: float) -> np.ndarray:
    return (x > thresh).astype(np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(a, b).sum()
    return float(inter / union)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice = 2|A∩B| / (|A|+|B|). Trả 1.0 nếu cả A và B đều rỗng (trùng khớp hoàn hảo
    theo quy ước), tránh chia 0/0."""
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    inter = np.logical_and(a, b).sum()
    return float(2 * inter / denom)


def containment(a: np.ndarray, b: np.ndarray) -> float:
    """Tỉ lệ vùng A nằm trong B: |A∩B| / |A|.

    Dùng để phân biệt 2 nguyên nhân IoU thấp: (a) heatmap NHỎ nhưng nằm TRỌN trong
    phổi — containment cao, vô hại (model tập trung đúng 1 vùng tổn thương cụ thể);
    (b) heatmap nằm phần lớn NGOÀI phổi — containment thấp, dấu hiệu shortcut thật
    (xem docs/LY_THUYET.md Phần VIII.5). Trả 0.0 nếu A rỗng (heatmap không có pixel
    nào vượt ngưỡng — hiếm nhưng có thể xảy ra).
    """
    a = a.astype(bool)
    b = b.astype(bool)
    if a.sum() == 0:
        return 0.0
    inter = np.logical_and(a, b).sum()
    return float(inter / a.sum())


def load_gt_mask(image_path: Path, mask_dir: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_dir / image_path.name).convert("L"))
    return (mask > 0).astype(np.uint8)


@torch.no_grad()
def predict_lung_mask(unet: torch.nn.Module, img_tensor: torch.Tensor) -> np.ndarray:
    device = next(unet.parameters()).device
    x = img_tensor.unsqueeze(0).to(device)
    logits = unet(x)
    mask = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
    return mask


def run_shortcut_analysis(
    classifier_path: str,
    unet_path: str,
    test_split_dir: str,
    mask_source: Literal["gt", "unet"] = "gt",
    gradcam_thresh: float = 0.5,
    device: Optional[str] = None,
) -> tuple[dict[str, list[float]], list[dict]]:
    """Trả về (ious_per_class, records); đồng thời lưu toàn bộ records ra CSV trong
    figures/ (mỗi lần gọi, không cần chạy lại Grad-CAM cho các phân tích sau này).

    ious_per_class: {tên_lớp: [iou_ảnh_1, iou_ảnh_2, ...]} — dùng để in bảng thống kê.
    records: [{"path": Path, "class": str, "iou": float, "containment": float}, ...]
        — 1 dòng/ảnh. "containment" = |heatmap∩phổi|/|heatmap| — tỉ lệ vùng heatmap
        thực sự nằm trong phổi, giúp phân biệt "heatmap nhỏ nhưng đúng trong phổi"
        (containment cao, IoU vẫn có thể thấp — vô hại) với "heatmap ở ngoài phổi"
        (containment thấp — dấu hiệu shortcut thật, xem docs/LY_THUYET.md Phần VIII.5).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    clf = load_classifier(classifier_path, num_classes=len(CLASS_TO_IDX), device=device, verbose=False)

    unet = build_unet(pretrained=False)
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet.to(device).eval()

    ds = ChestXrayClassificationDataset(test_split_dir, get_val_transforms())
    mask_dir = Path(test_split_dir) / "masks"

    ious_per_class: dict[str, list[float]] = {c: [] for c in CLASS_TO_IDX}
    records: list[dict] = []

    for i in tqdm(range(len(ds)), desc=f"shortcut analysis (mask={mask_source}, t={gradcam_thresh})"):
        img, label = ds[i]
        path = ds.image_paths[i]

        # target_class=label (NHÃN THẬT, không phải model dự đoán): câu hỏi đang
        # trả lời là "khi model chẩn đoán ĐÚNG, nó nhìn vào đâu" — không phải phân
        # tích lỗi (khác với api/inference.py, nơi dùng pred_idx để giải thích cho
        # người dùng, xem docs/QUY_TRINH_CODE.md Phần 8.4).
        heatmap = generate_gradcam(clf, img, target_class=label)
        cam_bin = binarize(heatmap, gradcam_thresh)

        if mask_source == "gt":
            lung_mask = load_gt_mask(path, mask_dir)
        else:
            lung_mask = predict_lung_mask(unet, img)

        score = iou(cam_bin, lung_mask)
        cont = containment(cam_bin, lung_mask)
        cls_name = IDX_TO_CLASS[label]
        ious_per_class[cls_name].append(score)
        records.append({"path": path, "class": cls_name, "iou": score, "containment": cont})

    print(f"\n===== Mask source: {mask_source} | thresh={gradcam_thresh} =====")
    for cls in ious_per_class:
        arr = np.array(ious_per_class[cls])
        conts = np.array([r["containment"] for r in records if r["class"] == cls])
        print(
            f"{cls:15s} n={len(arr):4d} mean IoU={arr.mean():.3f} "
            f"median IoU={np.median(arr):.3f} std IoU={arr.std():.3f}  |  "
            f"mean containment={conts.mean():.3f}"
        )

    figures_dir = Path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = figures_dir / f"shortcut_records_{mask_source}_t{gradcam_thresh}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class", "iou", "containment"])
        writer.writeheader()
        for r in records:
            writer.writerow(
                {"path": str(r["path"]), "class": r["class"], "iou": r["iou"], "containment": r["containment"]}
            )
    print(f"Đã lưu {len(records)} dòng chi tiết vào {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 4))
    for cls, scores in ious_per_class.items():
        ax.hist(scores, bins=20, alpha=0.5, label=cls)
    ax.set_xlabel("IoU(Grad-CAM, lung mask)")
    ax.set_ylabel("Count")
    ax.set_title(f"Shortcut analysis — mask={mask_source}, thresh={gradcam_thresh}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / f"shortcut_iou_{mask_source}_t{gradcam_thresh}.png", dpi=120)
    plt.close(fig)

    return ious_per_class, records


if __name__ == "__main__":
    # Chạy 2 lần cho báo cáo: so sánh nguồn mask ground-truth vs. dự đoán từ U-Net
    # (docs/TUTORIAL.md Phần 11.2 — chứng minh U-Net đủ tốt để thay ground-truth
    # khi deploy, nơi không có mask thật cho ảnh mới).
    run_shortcut_analysis(
        classifier_path="weights/best_classifier.pth",
        unet_path="weights/best_unet.pth",
        test_split_dir="data/split/test",
        mask_source="gt",
        gradcam_thresh=0.5,
    )
    run_shortcut_analysis(
        classifier_path="weights/best_classifier.pth",
        unet_path="weights/best_unet.pth",
        test_split_dir="data/split/test",
        mask_source="unet",
        gradcam_thresh=0.5,
    )
