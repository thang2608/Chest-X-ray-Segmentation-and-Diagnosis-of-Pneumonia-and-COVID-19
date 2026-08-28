"""Load model 1 lần lúc startup, cung cấp predict_image() dùng chung cho api/main.py.

Nếu weights/best_classifier.pth hoặc weights/best_unet.pth chưa tồn tại (chưa train),
tự động fallback sang backbone pretrained ImageNet (đầu ra ngẫu nhiên với classifier,
mask vô nghĩa với U-Net), để backend vẫn khởi động và phục vụ được — dự đoán khi đó
không có ý nghĩa nhưng đúng shape/HTTP contract, đủ để test luồng API/UI trước khi có
model thật. Xem docs/QUY_TRINH_CODE.md Phần 8.3-8.4.

Ngoài classifier, còn load U-Net để tính "độ tin cậy giải thích": so khớp Grad-CAM
heatmap với mask phổi U-Net dự đoán bằng IoU/containment (cùng công thức dùng trong
src/shortcut_iou.py để kiểm định shortcut learning toàn tập test) — trả về NGAY trong
mỗi response, biến việc kiểm định thành một chỉ số minh bạch cho người dùng cuối, không
chỉ nằm trong báo cáo. Xem docs/LY_THUYET.md Phần VIII.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from src.dataset import IDX_TO_CLASS, IMAGE_SIZE, NUM_CLASSES, get_val_transforms
from src.gradcam import generate_gradcam, overlay_heatmap
from src.model import build_classifier, load_classifier
from src.shortcut_iou import binarize, containment, dice, iou, predict_lung_mask
from src.unet import build_unet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = Path("weights/best_classifier.pth")
UNET_WEIGHTS_PATH = Path("weights/best_unet.pth")
DATA_PROCESSED_DIR = Path("data/processed")
DATASET_CLASSES = ("COVID", "Lung_Opacity", "Normal")
GRADCAM_THRESH = 0.5
LOW_TRUST_CONTAINMENT = 0.3  # dưới ngưỡng này: cảnh báo model có thể đang nhìn ngoài phổi

_classifier = None
_unet = None
_transform = get_val_transforms()
_model_is_trained: bool = False
_unet_is_trained: bool = False

BASE_DISCLAIMER = (
    "Kết quả chỉ mang tính tham khảo, KHÔNG thay thế chẩn đoán y khoa chính thức. "
    "Vui lòng tham vấn bác sĩ chuyên khoa."
)
UNTRAINED_WARNING = (
    f"⚠️ CẢNH BÁO: không tìm thấy {WEIGHTS_PATH} — đang dùng backbone EfficientNet-B3 "
    "pretrained ImageNet với classification head KHỞI TẠO NGẪU NHIÊN (chưa huấn luyện). "
    "Dự đoán này KHÔNG có ý nghĩa chẩn đoán, chỉ phục vụ kiểm tra luồng kỹ thuật."
)


def load_models() -> None:
    """Gọi 1 lần lúc server khởi động (xem api/main.py::lifespan)."""
    global _classifier, _model_is_trained, _unet, _unet_is_trained

    if WEIGHTS_PATH.exists():
        try:
            _classifier = load_classifier(
                str(WEIGHTS_PATH), num_classes=NUM_CLASSES, device=DEVICE, verbose=True
            )
            _model_is_trained = True
            print(f"[inference] OK: loaded trained classifier from {WEIGHTS_PATH} on device={DEVICE}")
        except Exception as exc:
            # File tồn tại nhưng không load được (hỏng / sai kiến trúc / mid-write) —
            # KHÔNG để lỗi này làm sập server, rơi xuống nhánh fallback bên dưới.
            print(
                f"[inference] WARNING: found {WEIGHTS_PATH} but failed to load it "
                f"({type(exc).__name__}: {exc}) — falling back to UNTRAINED classifier."
            )
    else:
        print(
            f"[inference] WARNING: {WEIGHTS_PATH} does not exist — falling back to "
            "UNTRAINED classifier (ImageNet backbone + random head). Predictions will "
            "be garbage; use only to verify API/UI wiring."
        )

    if _classifier is None:
        _classifier = build_classifier(num_classes=NUM_CLASSES, pretrained=True, verbose=True)
        _classifier.to(DEVICE).eval()
        _model_is_trained = False

    if UNET_WEIGHTS_PATH.exists():
        try:
            _unet = build_unet(pretrained=False, verbose=False)
            _unet.load_state_dict(torch.load(UNET_WEIGHTS_PATH, map_location=DEVICE))
            _unet.to(DEVICE).eval()
            _unet_is_trained = True
            print(f"[inference] OK: loaded trained U-Net from {UNET_WEIGHTS_PATH} on device={DEVICE}")
        except Exception as exc:
            print(
                f"[inference] WARNING: found {UNET_WEIGHTS_PATH} but failed to load it "
                f"({type(exc).__name__}: {exc}) — falling back to UNTRAINED U-Net."
            )
    else:
        print(
            f"[inference] WARNING: {UNET_WEIGHTS_PATH} does not exist — falling back to "
            "UNTRAINED U-Net. lung_overlap_* trong response sẽ không đáng tin cậy."
        )

    if _unet is None:
        _unet = build_unet(pretrained=True, verbose=False)
        _unet.to(DEVICE).eval()
        _unet_is_trained = False


def _find_ground_truth_mask(filename: Optional[str]) -> Optional[np.ndarray]:
    """Tìm mask thật trong data/processed/<class>/masks/<filename> — chỉ có kết quả
    khi ảnh upload TRÙNG TÊN với 1 ảnh trong dataset gốc (đúng tình huống demo: upload
    lại ảnh mẫu từ data/split/test/images/ để so sánh U-Net với ground truth). Với ảnh
    thật sự mới (không có trong dataset), luôn trả None — đây là hành vi bình thường,
    không phải lỗi.
    """
    if not filename:
        return None
    for cls in DATASET_CLASSES:
        mask_path = DATA_PROCESSED_DIR / cls / "masks" / filename
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))
            return (mask > 0).astype(np.uint8)
    return None


def _encode_png_base64(image_rgb: np.ndarray) -> str:
    pil_img = Image.fromarray(image_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict_image(pil_image: Image.Image, filename: Optional[str] = None) -> dict:
    """Hàm dùng chung — api/main.py gọi hàm này cho mỗi request POST /predict.

    filename: tên file gốc lúc upload (từ UploadFile.filename) — dùng để tìm mask
        ground-truth nếu ảnh trùng tên với 1 ảnh trong dataset gốc (xem
        _find_ground_truth_mask). Không bắt buộc — None vẫn chạy bình thường, chỉ là
        sẽ không có số liệu so sánh U-Net vs ground truth.
    """
    if _classifier is None:
        raise RuntimeError("Model chưa được load — gọi load_models() lúc startup")

    image_np = np.array(pil_image.convert("RGB"))
    image_resized = cv2.resize(image_np, IMAGE_SIZE)

    img_tensor = _transform(image=image_np)["image"]

    with torch.no_grad():
        logits = _classifier(img_tensor.unsqueeze(0).to(DEVICE))
        probs = logits.softmax(dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_class = IDX_TO_CLASS[pred_idx]

    # NGOÀI khối no_grad ở trên — Grad-CAM cần backward pass, không được bọc no_grad().
    heatmap = generate_gradcam(_classifier, img_tensor, target_class=pred_idx)
    overlay = overlay_heatmap(image_resized, heatmap)

    # Chỉ số tin cậy giải thích — so khớp heatmap với mask phổi U-Net dự đoán
    # (cùng công thức src/shortcut_iou.py dùng để kiểm định shortcut learning).
    cam_bin = binarize(heatmap, GRADCAM_THRESH)
    lung_mask = predict_lung_mask(_unet, img_tensor)
    overlap_iou = iou(cam_bin, lung_mask)
    overlap_containment = containment(cam_bin, lung_mask)
    unet_mask_base64 = _encode_png_base64((lung_mask * 255).astype(np.uint8))

    # So sánh U-Net vs ground truth — chỉ có nếu ảnh upload trùng tên ảnh trong dataset.
    gt_mask = _find_ground_truth_mask(filename)
    gt_mask_found = gt_mask is not None
    unet_vs_gt_dice: Optional[float] = None
    unet_vs_gt_iou: Optional[float] = None
    if gt_mask_found:
        unet_vs_gt_dice = dice(lung_mask, gt_mask)
        unet_vs_gt_iou = iou(lung_mask, gt_mask)

    disclaimer = BASE_DISCLAIMER
    if not _model_is_trained:
        disclaimer = f"{disclaimer} {UNTRAINED_WARNING}"
    if not _unet_is_trained:
        disclaimer = (
            f"{disclaimer} ⚠️ U-Net chưa được huấn luyện — chỉ số lung_overlap_* "
            "bên dưới KHÔNG đáng tin cậy."
        )
    elif overlap_containment < LOW_TRUST_CONTAINMENT:
        pct_outside = (1 - overlap_containment) * 100
        disclaimer = (
            f"{disclaimer} ⚠️ LƯU Ý: model đang tập trung khoảng {pct_outside:.0f}% vào "
            "vùng NGOÀI phổi cho ảnh này (theo kiểm định Grad-CAM so với mask U-Net) — "
            "kết quả có thể không đáng tin cậy, xem docs/LY_THUYET.md Phần VIII."
        )

    return {
        "predicted_class": pred_class,
        "confidence": float(probs[pred_idx]),
        "probabilities": {IDX_TO_CLASS[i]: float(p) for i, p in enumerate(probs)},
        "heatmap_overlay_base64": _encode_png_base64(overlay),
        "disclaimer": disclaimer,
        "lung_overlap_iou": overlap_iou,
        "lung_overlap_containment": overlap_containment,
        "unet_mask_base64": unet_mask_base64,
        "gt_mask_found": gt_mask_found,
        "unet_vs_gt_dice": unet_vs_gt_dice,
        "unet_vs_gt_iou": unet_vs_gt_iou,
    }
