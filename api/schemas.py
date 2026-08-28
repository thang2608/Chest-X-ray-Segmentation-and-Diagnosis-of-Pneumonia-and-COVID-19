from typing import Dict, Optional

from pydantic import BaseModel


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    heatmap_overlay_base64: str
    disclaimer: str
    lung_overlap_iou: float           # IoU(Grad-CAM heatmap, mask phổi từ U-Net)
    lung_overlap_containment: float   # tỉ lệ vùng heatmap nằm TRONG phổi — xem src/shortcut_iou.py
    unet_mask_base64: str             # mask phổi U-Net dự đoán, PNG grayscale encode base64
    gt_mask_found: bool                # True nếu ảnh upload trùng tên 1 ảnh trong dataset gốc
    unet_vs_gt_dice: Optional[float] = None  # chỉ có giá trị khi gt_mask_found=True
    unet_vs_gt_iou: Optional[float] = None
