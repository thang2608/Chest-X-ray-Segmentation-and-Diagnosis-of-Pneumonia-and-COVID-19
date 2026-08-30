from typing import Optional
from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    dice_score: float = Field(
        ...,
        description=(
            "Dice Coefficient của U-Net khi phân đoạn phổi, đo TRÊN TOÀN TẬP VAL lúc "
            "train (không phải tính riêng cho ảnh này — ảnh mới không có ground-truth "
            "để tính per-image). Xem docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md."
        ),
        example=0.986,
    )
    iou_score: float = Field(
        ...,
        description=(
            "IoU giữa vùng Grad-CAM 'nóng' và mask phổi (U-Net) CHO ẢNH NÀY — chỉ số tin "
            "cậy giải thích, thấp nghĩa là model có thể đang nhìn ra ngoài phổi để chẩn "
            "đoán (shortcut learning), xem docs/LY_THUYET.md Phần VIII."
        ),
        example=0.42,
    )
    precision: float = Field(
        ...,
        description="Macro Precision của classifier, đo TRÊN TOÀN TẬP VAL lúc train (không phải riêng ảnh này, %)",
        example=90.8,
    )
    recall: float = Field(
        ...,
        description="Macro Recall của classifier, đo TRÊN TOÀN TẬP VAL lúc train (không phải riêng ảnh này, %)",
        example=90.7,
    )
    affected_lung_area: float = Field(
        ...,
        description="Tỷ lệ vùng phổi (theo mask U-Net) trùng với vùng Grad-CAM 'nóng' — ước lượng diện tích phổi model đang chú ý khi chẩn đoán (%)",
        example=18.4,
    )
    pointing_game: Optional[str] = Field(
        "Hit",
        description="Kết quả Pointing Game: 'Hit' nếu điểm kích hoạt Grad-CAM cực đại nằm trong phổi, 'Miss' nếu nằm ngoài",
        example="Hit",
    )
    soft_containment: Optional[float] = Field(
        95.0,
        description="Soft Containment: Tỷ lệ năng lượng kích hoạt Grad-CAM nằm trọn trong vùng phổi (%)",
        example=94.5,
    )
    inference_time_ms: float = Field(
        ...,
        description="Thời gian mô hình AI xử lý và suy luận (mili-giây)",
        example=38.5,
    )


class PredictionResponse(BaseModel):
    record_code: str = Field(..., description="Mã phiên phân tích hình ảnh", example="RAD-20260828-A1B2")
    message: str = Field(
        ...,
        description="Thông báo trạng thái kết quả xử lý",
        example="Phân tích thành công",
    )
    disease: str = Field(
        ...,
        description=(
            "Tên bệnh được mô hình dự đoán — model hiện chỉ train 3 lớp: "
            "COVID-19, Lung Opacity, Normal (Viral Pneumonia CHƯA được hỗ trợ, "
            "xem docs/THAY_DOI_TICH_HOP_BACKEND.md)."
        ),
        example="COVID-19",
    )
    confidence: float = Field(
        ...,
        description="Tỷ lệ phần trăm độ tin cậy của kết quả dự đoán (từ 0 đến 100%)",
        example=92.5,
    )
    raw_image_url: str = Field(
        ...,
        description="Đường dẫn URL ảnh X-quang gốc đã tải lên",
        example="http://localhost:8000/static/raw/sample.png",
    )
    result_image_url: str = Field(
        ...,
        description=(
            "Đường dẫn URL ảnh mask VÙNG PHỔI (từ U-Net, tô xanh lá) chồng lên ảnh gốc — "
            "model hiện KHÔNG phân đoạn vùng tổn thương cụ thể, chỉ định vị vùng phổi."
        ),
        example="http://localhost:8000/static/results/sample.png",
    )
    heatmap_url: str = Field(
        ...,
        description="Đường dẫn URL ảnh bản đồ nhiệt giải thích vùng tổn thương (Grad-CAM Heatmap)",
        example="http://localhost:8000/static/heatmaps/sample.png",
    )
    metrics: Optional[EvaluationMetrics] = Field(
        default=None,
        description="Các chỉ số định lượng đánh giá chất lượng phân tích của mô hình",
    )
    disclaimer: str = Field(
        default="Kết quả phân tích từ mô hình AI chỉ mang tính chất tham khảo, hỗ trợ sàng lọc và không thay thế cho chẩn đoán y khoa chính thức từ bác sĩ chuyên khoa.",
        description="Khuyến cáo y tế và tuyên bố từ chối trách nhiệm",
        example="Kết quả phân tích từ mô hình AI chỉ mang tính chất tham khảo, hỗ trợ sàng lọc và không thay thế cho chẩn đoán y khoa chính thức từ bác sĩ chuyên khoa.",
    )
    created_at: Optional[str] = Field(None, description="Thời gian phân tích", example="2026-08-28 00:45:00")


class HealthResponse(BaseModel):
    status: str = Field(
        ...,
        description="Trạng thái hoạt động của server",
        example="Hệ thống đang hoạt động tốt",
    )
    version: str = Field(
        ...,
        description="Phiên bản hiện tại của API",
        example="1.0.0",
    )
