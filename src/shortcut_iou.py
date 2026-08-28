import csv
import os
import random
from pathlib import Path
from typing import Literal, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    ChestXrayClassificationDataset,
    crop_to_lung_bbox,
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
    crop_to_lung: bool = False,
    crop_padding: float = 0.1,
) -> tuple[dict[str, list[float]], list[dict]]:
    """Trả về (ious_per_class, records); đồng thời lưu toàn bộ records ra CSV trong
    figures/ (mỗi lần gọi, không cần chạy lại Grad-CAM cho các phân tích sau này).

    ious_per_class: {tên_lớp: [iou_ảnh_1, iou_ảnh_2, ...]} — dùng để in bảng thống kê.
    records: [{"path": Path, "class": str, "iou": float, "containment": float}, ...]
        — 1 dòng/ảnh. "containment" = |heatmap∩phổi|/|heatmap| — tỉ lệ vùng heatmap
        thực sự nằm trong phổi, giúp phân biệt "heatmap nhỏ nhưng đúng trong phổi"
        (containment cao, IoU vẫn có thể thấp — vô hại) với "heatmap ở ngoài phổi"
        (containment thấp — dấu hiệu shortcut thật, xem docs/LY_THUYET.md Phần VIII.5).

    crop_to_lung: đánh giá phiên bản "đã tối ưu" (classifier train trên ảnh crop theo
        mask — xem notebooks/train_classifier_cropped.ipynb). Khi True, PHẢI truyền
        `classifier_path` trỏ tới checkpoint train trên ảnh crop (vd
        weights/best_classifier_cropped.pth) — hàm không tự kiểm tra được điều này,
        dùng nhầm checkpoint sẽ cho kết quả vô nghĩa (train/serve skew).

        ⚠️ LƯU Ý KHI SO SÁNH SỐ LIỆU TRƯỚC/SAU: khi crop_to_lung=True, cả heatmap
        VÀ mask so sánh đều được cắt theo CÙNG bounding box trước khi tính IoU/
        containment — nghĩa là mask "sau crop" chiếm tỉ lệ diện tích khung hình LỚN
        HƠN nhiều so với mask "trước crop" (đã loại phần lớn nền), nên containment
        trung bình tự nhiên cao hơn dù model có thực sự "học tốt hơn" hay không —
        đây là hiệu ứng hình học, không phải hiệu ứng model. Chỉ số ít bị ảnh hưởng
        bởi hiệu ứng này nhất, nên ưu tiên khi so sánh trước/sau, là **% ảnh IoU=0
        tuyệt đối** (đo việc heatmap có giao nhau với phổi hay không, không phụ
        thuộc tỉ lệ diện tích tương đối). Xem docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    clf = load_classifier(classifier_path, num_classes=len(CLASS_TO_IDX), device=device, verbose=False)

    unet = build_unet(pretrained=False)
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet.to(device).eval()

    # transform=None: cần ảnh numpy THÔ để tự crop tay trước khi resize/normalize —
    # áp get_val_transforms() thủ công bên dưới, sau bước crop (nếu có).
    ds = ChestXrayClassificationDataset(test_split_dir, transform=None)
    transform = get_val_transforms()
    mask_dir = Path(test_split_dir) / "masks"

    ious_per_class: dict[str, list[float]] = {c: [] for c in CLASS_TO_IDX}
    records: list[dict] = []

    for i in tqdm(range(len(ds)), desc=f"shortcut analysis (mask={mask_source}, crop={crop_to_lung}, t={gradcam_thresh})"):
        image_np, label = ds[i]
        path = ds.image_paths[i]

        if mask_source == "gt":
            lung_mask = load_gt_mask(path, mask_dir)
        else:
            # Cần mask trên ảnh CHƯA crop để làm căn cứ crop — luôn tính trên ảnh gốc.
            full_tensor = transform(image=image_np)["image"]
            lung_mask = predict_lung_mask(unet, full_tensor)

        if crop_to_lung:
            image_for_model = crop_to_lung_bbox(image_np, lung_mask, padding=crop_padding)
            # Cắt CHÍNH mask đó theo cùng box — để so khớp đúng hệ toạ độ với heatmap
            # (xem cảnh báo về hiệu ứng hình học trong docstring ở trên).
            compare_mask = crop_to_lung_bbox(lung_mask, lung_mask, padding=crop_padding)
        else:
            image_for_model = image_np
            compare_mask = lung_mask

        img = transform(image=image_for_model)["image"]

        # target_class=label (NHÃN THẬT, không phải model dự đoán): câu hỏi đang
        # trả lời là "khi model chẩn đoán ĐÚNG, nó nhìn vào đâu" — không phải phân
        # tích lỗi (khác với api/inference.py, nơi dùng pred_idx để giải thích cho
        # người dùng, xem docs/QUY_TRINH_CODE.md Phần 8.4).
        heatmap = generate_gradcam(clf, img, target_class=label)
        cam_bin = binarize(heatmap, gradcam_thresh)

        # heatmap luôn có kích thước = kích thước input model (224×224 sau transform),
        # bất kể compare_mask (đã crop, chưa resize) đang kích thước gì — resize lại
        # để so khớp đúng pixel-to-pixel trước khi tính IoU/containment.
        if compare_mask.shape != cam_bin.shape:
            compare_mask = cv2.resize(
                compare_mask, (cam_bin.shape[1], cam_bin.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        score = iou(cam_bin, compare_mask)
        cont = containment(cam_bin, compare_mask)
        cls_name = IDX_TO_CLASS[label]
        ious_per_class[cls_name].append(score)
        records.append({"path": path, "class": cls_name, "iou": score, "containment": cont})

    tag = f"{mask_source}{'_cropped' if crop_to_lung else ''}"
    print(f"\n===== Mask source: {mask_source} | crop_to_lung={crop_to_lung} | thresh={gradcam_thresh} =====")
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

    csv_path = figures_dir / f"shortcut_records_{tag}_t{gradcam_thresh}.csv"
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
    ax.set_title(f"Shortcut analysis — mask={mask_source}, crop={crop_to_lung}, thresh={gradcam_thresh}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / f"shortcut_iou_{tag}_t{gradcam_thresh}.png", dpi=120)
    plt.close(fig)

    return ious_per_class, records


if __name__ == "__main__":
    # Chạy 2 lần cho báo cáo (bản BASELINE, chưa tối ưu): so sánh nguồn mask
    # ground-truth vs. dự đoán từ U-Net (docs/TUTORIAL.md Phần 11.2 — chứng minh
    # U-Net đủ tốt để thay ground-truth khi deploy, nơi không có mask thật).
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

    # Bản ĐÃ TỐI ƯU (sau khi train notebooks/train_classifier_cropped.ipynb và có
    # weights/best_classifier_cropped.pth) — bỏ comment 2 lệnh dưới để so sánh:
    #
    # run_shortcut_analysis(
    #     classifier_path="weights/best_classifier_cropped.pth",
    #     unet_path="weights/best_unet.pth",
    #     test_split_dir="data/split/test",
    #     mask_source="gt",
    #     gradcam_thresh=0.5,
    #     crop_to_lung=True,
    # )
    # run_shortcut_analysis(
    #     classifier_path="weights/best_classifier_cropped.pth",
    #     unet_path="weights/best_unet.pth",
    #     test_split_dir="data/split/test",
    #     mask_source="unet",
    #     gradcam_thresh=0.5,
    #     crop_to_lung=True,
    # )
