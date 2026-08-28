"""Model AI thật cho backend/frontend — thay logic MOCK (giả theo tên file) trước đây.

Nạp classifier (EfficientNet-B3) + U-Net đã train từ weights/ ở GỐC REPO (dùng chung với
phần còn lại của dự án, xem app/core/config.py::CLASSIFIER_WEIGHTS/UNET_WEIGHTS — không
copy trùng file .pth vào backend/weights/), chạy pipeline classification + Grad-CAM +
lung-segmentation thật. Xem docs/THAY_DOI_TICH_HOP_BACKEND.md cho đầy đủ quyết định thiết
kế (ánh xạ field với schemas/prediction.py, vì sao chỉ 3 lớp thay vì 4, vì sao
dice/precision/recall là số tổng hợp trên tập test thay vì tính riêng từng ảnh).
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image

# app.core.config đã tự thêm REPO_ROOT vào sys.path khi import — import nó TRƯỚC để
# "import src.xxx" bên dưới hoạt động, bất kể uvicorn chạy với cwd nào.
from app.core import config as _config  # noqa: F401  (side-effect: sys.path đã có REPO_ROOT)

from src.dataset import IDX_TO_CLASS, IMAGE_SIZE, NUM_CLASSES, get_val_transforms
from src.gradcam import generate_gradcam, overlay_heatmap
from src.model import build_classifier, load_classifier
from src.shortcut_iou import binarize, containment, iou, predict_lung_mask
from src.unet import build_unet

# Tên hiển thị cho người dùng. Model chỉ có 3 lớp THẬT (Viral Pneumonia KHÔNG được train —
# bị loại khỏi pipeline từ src/preprocess.py đầu dự án, xem CLAUDE.md) — KHÔNG dùng chữ
# "Pneumonia" để tránh người dùng hiểu nhầm đây là chẩn đoán viêm phổi do virus.
DISPLAY_NAME = {"Normal": "Normal", "Lung_Opacity": "Lung Opacity", "COVID": "COVID-19"}

GRADCAM_THRESH = 0.5
LOW_TRUST_CONTAINMENT = 0.3  # dưới ngưỡng này: cảnh báo model có thể đang nhìn ngoài phổi

# Số liệu TỔNG HỢP trên val set (1350 ảnh, KHÔNG PHẢI tính riêng cho từng ảnh upload —
# ảnh mới không có ground-truth để tính per-image) — lấy từ lần train + đánh giá gần nhất,
# xem docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md. Đây là val set, CHƯA phải test set chính thức
# (xem Phần 4.4 của báo cáo đó) — cập nhật lại 3 số này bằng notebooks/evaluate_local.ipynb
# mỗi khi train lại model.
AGGREGATE_METRICS = {
    "dice_score": 0.9862,   # U-Net, Val Dice
    "precision": 0.9075,    # Classifier, Macro Precision (val set)
    "recall": 0.9067,       # Classifier, Macro Recall (val set)
}


class MedicalSegmentationModel:
    """Bọc quanh classifier + U-Net thật, giữ NGUYÊN interface `predict_and_save()` mà
    backend/app/routers/predict.py đang gọi — không đổi chữ ký hàm, không đổi
    schemas/prediction.py."""

    def __init__(self, classifier_path: str, unet_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier_is_trained = False
        self.unet_is_trained = False

        # --- Classifier: cùng cơ chế fallback graceful đã dùng trong api/inference.py ---
        classifier_p = Path(classifier_path)
        if classifier_p.exists():
            try:
                self.classifier = load_classifier(
                    str(classifier_p), num_classes=NUM_CLASSES, device=self.device, verbose=True
                )
                self.classifier_is_trained = True
            except Exception as exc:
                print(
                    f"[ai_engine] WARNING: khong load duoc classifier tu {classifier_p} "
                    f"({type(exc).__name__}: {exc}) - fallback backbone ImageNet pretrained."
                )
        else:
            print(f"[ai_engine] WARNING: khong tim thay {classifier_p} - fallback backbone ImageNet pretrained.")

        if not self.classifier_is_trained:
            self.classifier = build_classifier(num_classes=NUM_CLASSES, pretrained=True, verbose=True)
            self.classifier.to(self.device).eval()

        # --- U-Net: cùng cơ chế fallback ---
        unet_p = Path(unet_path)
        if unet_p.exists():
            try:
                self.unet = build_unet(pretrained=False, verbose=False)
                self.unet.load_state_dict(torch.load(str(unet_p), map_location=self.device))
                self.unet.to(self.device).eval()
                self.unet_is_trained = True
            except Exception as exc:
                print(
                    f"[ai_engine] WARNING: khong load duoc U-Net tu {unet_p} "
                    f"({type(exc).__name__}: {exc}) - fallback encoder pretrained."
                )
        else:
            print(f"[ai_engine] WARNING: khong tim thay {unet_p} - fallback encoder pretrained.")

        if not self.unet_is_trained:
            self.unet = build_unet(pretrained=True, verbose=False)
            self.unet.to(self.device).eval()

        self.transform = get_val_transforms()
        print(
            f"[ai_engine] READY | classifier_trained={self.classifier_is_trained} "
            f"unet_trained={self.unet_is_trained} device={self.device}"
        )

    def predict_and_save(
        self, image_path: str, mask_output_path: str, heatmap_output_path: str
    ) -> Dict[str, Any]:
        """Đọc ảnh từ đĩa, chạy classifier + Grad-CAM + U-Net, lưu 2 ảnh kết quả ra đĩa,
        trả về dict khớp field mà backend/app/routers/predict.py cần (disease, confidence,
        metrics) để build PredictionResponse."""
        t0 = time.time()

        pil_image = Image.open(image_path).convert("RGB")
        image_np = np.array(pil_image)                       # (H0,W0,3) RGB uint8
        image_resized = cv2.resize(image_np, IMAGE_SIZE)      # (224,224,3) RGB — nền cho overlay/mask

        img_tensor = self.transform(image=image_np)["image"]

        with torch.no_grad():
            logits = self.classifier(img_tensor.unsqueeze(0).to(self.device))
            probs = logits.softmax(dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_class = IDX_TO_CLASS[pred_idx]                   # "Normal" | "Lung_Opacity" | "COVID"
        disease = DISPLAY_NAME[pred_class]

        # NGOÀI torch.no_grad() ở trên — Grad-CAM cần backward pass thật.
        heatmap = generate_gradcam(self.classifier, img_tensor, target_class=pred_idx)
        heatmap_overlay = overlay_heatmap(image_resized, heatmap)  # (224,224,3) RGB

        lung_mask = predict_lung_mask(self.unet, img_tensor)   # (224,224) {0,1}
        # Mask hiển thị: tô xanh lá bán trong suốt lên vùng phổi trên nền ảnh gốc —
        # dễ nhìn hơn mask nhị phân trần (trắng/đen).
        mask_display = image_resized.copy()
        lung_bool = lung_mask > 0
        mask_display[lung_bool] = (
            0.5 * mask_display[lung_bool].astype(np.float32) + 0.5 * np.array([0, 255, 0], dtype=np.float32)
        ).astype(np.uint8)

        # Chỉ số tin cậy giải thích — so khớp heatmap với mask phổi, cùng công thức
        # src/shortcut_iou.py dùng để kiểm định shortcut learning toàn tập test.
        cam_bin = binarize(heatmap, GRADCAM_THRESH)
        iou_score = iou(cam_bin, lung_mask)
        cont = containment(cam_bin, lung_mask)

        lung_area = int(lung_mask.sum())
        affected_lung_area = (
            float(np.logical_and(cam_bin.astype(bool), lung_bool).sum() / lung_area * 100)
            if lung_area > 0
            else 0.0
        )

        # PIL/src.gradcam trả RGB — cv2.imwrite cần BGR, đổi lại trước khi ghi đĩa.
        cv2.imwrite(heatmap_output_path, cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_output_path, cv2.cvtColor(mask_display, cv2.COLOR_RGB2BGR))

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        metrics = dict(AGGREGATE_METRICS)
        metrics["iou_score"] = round(float(iou_score), 3)
        metrics["affected_lung_area"] = round(affected_lung_area, 1)
        metrics["inference_time_ms"] = elapsed_ms

        warning = None
        if not self.classifier_is_trained:
            warning = (
                "CANH BAO: classifier CHUA duoc huan luyen (dang dung backbone ImageNet "
                "pretrained + head khoi tao ngau nhien). Ket qua KHONG co y nghia chan doan."
            )
        elif not self.unet_is_trained:
            warning = "CANH BAO: U-Net chua duoc huan luyen - mask phoi va iou_score khong dang tin cay."
        elif cont < LOW_TRUST_CONTAINMENT:
            warning = (
                f"LUU Y: model dang tap trung khoang {100 * (1 - cont):.0f}% vao vung NGOAI "
                "phoi cho anh nay - ket qua co the khong dang tin cay (xem docs/LY_THUYET.md Phan VIII)."
            )

        return {
            "disease": disease,
            "confidence": round(float(probs[pred_idx]) * 100, 1),  # schema quy ước 0-100%
            "metrics": metrics,
            "warning": warning,  # đọc ở routers/predict.py, ghép vào disclaimer nếu có
        }
