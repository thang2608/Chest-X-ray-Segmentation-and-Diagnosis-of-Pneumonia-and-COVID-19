import os
import sys
from pathlib import Path

# Windows console mac dinh dung codepage cp1252, khong encode duoc tieng Viet co dau
# (vd "đã sẵn sàng") -> crash toan bo app khi print() gap ky tu nhu vay (UnicodeEncodeError).
# Ep stdout/stderr sang UTF-8 ngay tu dau, truoc khi bat ky module nao khac print gi.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Thư mục gốc của Backend (backend/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Gốc toàn bộ repo (chứa src/, weights/, data/ dùng CHUNG với phần còn lại của dự án —
# xem docs/THAY_DOI_TICH_HOP_BACKEND.md). Thêm vào sys.path ở đây (import sớm nhất,
# mọi module khác trong backend/app/ đều import config trước tiên) để "import src.xxx"
# hoạt động dù uvicorn được chạy với cwd = backend/ (nơi package "app" sống).
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Thư mục Frontend
FRONTEND_DIR = BASE_DIR / "frontend"
SAMPLES_DIR = FRONTEND_DIR / "samples"

# Các thư mục lưu trữ tĩnh
UPLOADS_DIR = BASE_DIR / "uploads"
RAW_IMAGES_DIR = UPLOADS_DIR / "raw"
RESULT_IMAGES_DIR = UPLOADS_DIR / "results"
HEATMAPS_DIR = UPLOADS_DIR / "heatmaps"

# Thư mục chứa model weights — tìm ở REPO_ROOT/weights/ hoặc backend/weights/
WEIGHTS_DIR = REPO_ROOT / "weights" if (REPO_ROOT / "weights").exists() else BASE_DIR / "weights"

def _find_weight_file(filename: str) -> Path:
    p_repo = REPO_ROOT / "weights" / filename
    p_backend = BASE_DIR / "weights" / filename
    if p_repo.exists():
        return p_repo
    if p_backend.exists():
        return p_backend
    return p_repo

# --- Các model phân loại & segmentation của nhánh demo ---
CLASSIFIER_WEIGHTS = _find_weight_file("best_classifier.pth")
CROPPED_CLASSIFIER_WEIGHTS = _find_weight_file("best_classifier_cropped.pth")
BLACKOUT_CLASSIFIER_WEIGHTS = _find_weight_file("best_classifier_blackout.pth")
UNET_WEIGHTS = _find_weight_file("best_unet.pth")

# --- Model nhận diện YOLO ---
YOLO_MEDICAL_SEG_WEIGHTS = _find_weight_file("yolov10-medical-seg.pt")

# --- Khởi tạo mặc định ---
DEFAULT_MODEL_WEIGHTS = CROPPED_CLASSIFIER_WEIGHTS if CROPPED_CLASSIFIER_WEIGHTS.exists() else CLASSIFIER_WEIGHTS

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
