import os
import sys
from pathlib import Path

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

# Thư mục chứa model weights — DÙNG CHUNG weights/ ở gốc repo (đã có best_classifier.pth
# và best_unet.pth từ notebooks/train_classifier.ipynb, train_unet.ipynb) — KHÔNG copy
# trùng file .pth nặng (~140MB) vào backend/weights/ riêng.
WEIGHTS_DIR = REPO_ROOT / "weights"
CLASSIFIER_WEIGHTS = WEIGHTS_DIR / "best_classifier.pth"
CROPPED_CLASSIFIER_WEIGHTS = WEIGHTS_DIR / "best_classifier_cropped.pth"  # bản "đã tối ưu"
# — xem notebooks/train_classifier_cropped.ipynb, docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md.
BLACKOUT_CLASSIFIER_WEIGHTS = WEIGHTS_DIR / "best_classifier_blackout.pth"  # xoá pixel
# ngoài hình dạng phổi thay vì chỉ crop bounding box — giảm shortcut RẤT mạnh nhưng
# Macro F1 val giảm xuống DƯỚI CẢ baseline (0.8659 < 0.9057, xem
# docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md Phần 5.5) — KHÔNG phải bản "tốt hơn" đơn thuần.
# ai_engine.py ưu tiên CROPPED_CLASSIFIER_WEIGHTS trước (cân bằng tốt nhất); file này
# chỉ dùng khi CROPPED thiếu/lỗi, không tự động "leo thang" lên vì có sẵn.
UNET_WEIGHTS = WEIGHTS_DIR / "best_unet.pth"
DEFAULT_MODEL_WEIGHTS = CLASSIFIER_WEIGHTS  # giữ tên cũ để tương thích ngược nếu có chỗ khác import

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
