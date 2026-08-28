from typing import Dict

from pydantic import BaseModel


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    heatmap_overlay_base64: str
    disclaimer: str
    lung_overlap_iou: float           # IoU(Grad-CAM heatmap, mask phổi từ U-Net)
    lung_overlap_containment: float   # tỉ lệ vùng heatmap nằm TRONG phổi — xem src/shortcut_iou.py
