import os
from pathlib import Path

# Thư mục gốc của Backend (backend/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Thư mục Frontend
FRONTEND_DIR = BASE_DIR / "frontend"
SAMPLES_DIR = FRONTEND_DIR / "samples"

# Các thư mục lưu trữ tĩnh
UPLOADS_DIR = BASE_DIR / "uploads"
RAW_IMAGES_DIR = UPLOADS_DIR / "raw"
RESULT_IMAGES_DIR = UPLOADS_DIR / "results"
HEATMAPS_DIR = UPLOADS_DIR / "heatmaps"

# Thư mục chứa model weights
WEIGHTS_DIR = BASE_DIR / "weights"
DEFAULT_MODEL_WEIGHTS = WEIGHTS_DIR / "yolov10-medical-seg.pt"

# Tự động khởi tạo các thư mục lưu trữ nếu chưa có
os.makedirs(RAW_IMAGES_DIR, exist_ok=True)
os.makedirs(RESULT_IMAGES_DIR, exist_ok=True)
os.makedirs(HEATMAPS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Khuyến cáo y tế mặc định
DEFAULT_MEDICAL_DISCLAIMER = (
    "Kết quả phân tích từ mô hình AI chỉ mang tính chất tham khảo, "
    "hỗ trợ sàng lọc và không thay thế cho chẩn đoán y khoa chính thức từ bác sĩ chuyên khoa."
)

# Thông tin ứng dụng
APP_TITLE = "Chest X-Ray Segmentation & Diagnosis API"
APP_DESCRIPTION = "API phát hiện và phân đoạn tổn thương phổi do COVID-19 và Viêm phổi từ ảnh X-quang"
APP_VERSION = "1.0.0"
