from typing import Optional
from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    dice_score: float = Field(
        ...,
        description="Chỉ số Dice Coefficient (F1-score) đánh giá độ trùng khớp vùng phân đoạn tổn thương (0 đến 1.0)",
        example=0.892,
    )
    iou_score: float = Field(
        ...,
        description="Chỉ số IoU (Intersection over Union / Jaccard Index) đo độ bao phủ vùng tổn thương (0 đến 1.0)",
        example=0.815,
    )
    precision: float = Field(
        ...,
        description="Độ chính xác trong việc khoanh vùng tổn thương (%)",
        example=93.8,
    )
    recall: float = Field(
        ...,
        description="Độ nhạy / Tỷ lệ phát hiện đúng vùng tổn thương (%)",
        example=91.5,
    )
    affected_lung_area: float = Field(
        ...,
        description="Tỷ lệ diện tích phổi bị tổn thương so với tổng diện tích phổi (%)",
        example=18.4,
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
        description="Tên bệnh được mô hình dự đoán (COVID-19, Viral Pneumonia, Normal, Lung Opacity)",
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
        description="Đường dẫn URL ảnh kết quả phân đoạn tổn thương (Segmentation Mask)",
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
