"""Load model 1 lần lúc startup, cung cấp predict_image() dùng chung cho api/main.py.

Nếu weights/best_classifier.pth chưa tồn tại (chưa train), tự động fallback sang
backbone EfficientNet-B3 pretrained ImageNet + head khởi tạo ngẫu nhiên, để backend
vẫn khởi động và phục vụ được — dự đoán khi đó vô nghĩa nhưng đúng shape/HTTP contract,
đủ để test luồng API/UI trước khi có model thật. Xem docs/QUY_TRINH_CODE.md Phần 8.3-8.4.
"""

import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.dataset import IDX_TO_CLASS, IMAGE_SIZE, NUM_CLASSES, get_val_transforms
from src.gradcam import generate_gradcam, overlay_heatmap
from src.model import build_classifier, load_classifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = Path("weights/best_classifier.pth")

_classifier = None
_transform = get_val_transforms()
_model_is_trained: bool = False

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
    global _classifier, _model_is_trained

    if WEIGHTS_PATH.exists():
        try:
            _classifier = load_classifier(
                str(WEIGHTS_PATH), num_classes=NUM_CLASSES, device=DEVICE, verbose=True
            )
            _model_is_trained = True
            print(f"[inference] OK: loaded trained weights from {WEIGHTS_PATH} on device={DEVICE}")
            return
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

    _classifier = build_classifier(num_classes=NUM_CLASSES, pretrained=True, verbose=True)
    _classifier.to(DEVICE).eval()
    _model_is_trained = False


def _encode_png_base64(image_rgb: np.ndarray) -> str:
    pil_img = Image.fromarray(image_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict_image(pil_image: Image.Image) -> dict:
    """Hàm dùng chung — api/main.py gọi hàm này cho mỗi request POST /predict."""
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

    disclaimer = BASE_DISCLAIMER
    if not _model_is_trained:
        disclaimer = f"{disclaimer} {UNTRAINED_WARNING}"

    return {
        "predicted_class": pred_class,
        "confidence": float(probs[pred_idx]),
        "probabilities": {IDX_TO_CLASS[i]: float(p) for i, p in enumerate(probs)},
        "heatmap_overlay_base64": _encode_png_base64(overlay),
        "disclaimer": disclaimer,
    }
