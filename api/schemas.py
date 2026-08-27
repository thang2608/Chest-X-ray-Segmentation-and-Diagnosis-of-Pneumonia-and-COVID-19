from typing import Dict

from pydantic import BaseModel


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    heatmap_overlay_base64: str
    disclaimer: str
