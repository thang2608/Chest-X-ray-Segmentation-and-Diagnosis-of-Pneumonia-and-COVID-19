from typing import Optional

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.model import get_gradcam_target_layer


def generate_gradcam(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    target_class: Optional[int] = None,
) -> np.ndarray:
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    elif img_tensor.dim() != 4:
        raise ValueError(f"img_tensor phải có 3 hoặc 4 chiều, nhận shape {tuple(img_tensor.shape)}")

    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    # eval() bắt buộc (BatchNorm/Dropout ổn định) NHƯNG không bọc torch.no_grad()
    # ở ngoài — Grad-CAM cần build computation graph để backward logit[target_class]
    # theo activation map. Bọc no_grad() ở đây là bug kinh điển nhất, cho heatmap toàn 0.
    model.eval()

    if target_class is None:
        with torch.no_grad():
            logits = model(img_tensor)
            target_class = int(logits.argmax(dim=1).item())

    target_layer = get_gradcam_target_layer(model)
    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        # grayscale_cam: (batch=1, H, W), float32, [0,1] — thư viện tự làm GAP(gradient),
        # tổ hợp có trọng số, ReLU, resize, chuẩn hoá (docs/LY_THUYET.md Phần VII.2).
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

    return grayscale_cam[0].astype(np.float32)


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    if heatmap.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"heatmap {heatmap.shape} và image_rgb {image_rgb.shape[:2]} phải cùng (H, W)"
        )

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = alpha * heatmap_color.astype(np.float32) + (1 - alpha) * image_rgb.astype(np.float32)
    return overlay.astype(np.uint8)
