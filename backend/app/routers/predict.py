import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.core.config import (
    DEFAULT_MEDICAL_DISCLAIMER,
    HEATMAPS_DIR,
    RAW_IMAGES_DIR,
    RESULT_IMAGES_DIR,
    SAMPLES_DIR,
)
from app.schemas.prediction import PredictionResponse

router = APIRouter()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class SampleImageInfo(BaseModel):
    id: str
    name: str
    category: str
    image_url: str
    description: str


@router.get("/samples", response_model=List[SampleImageInfo], tags=["Samples"])
def get_sample_images(request: Request):
    """
    Trả về danh sách các ảnh X-quang mẫu để người dùng thử nghiệm nhanh
    """
    base_url = str(request.base_url).rstrip("/")
    samples = [
        {
            "id": "sample_covid",
            "name": "COVID-19",
            "category": "COVID-19",
            "image_url": f"{base_url}/static/samples/sample_covid.png",
            "description": "Ảnh X-quang bệnh nhân có tổn thương đông đặc do COVID-19",
        },
        # "sample_pneumonia" đã bỏ khỏi danh sách: model chỉ train 3 lớp
        # (Normal/Lung_Opacity/COVID) — KHÔNG có lớp Viral Pneumonia, xem
        # docs/THAY_DOI_TICH_HOP_BACKEND.md và CLAUDE.md (preprocess.py loại lớp này
        # từ đầu dự án). File ảnh mẫu vẫn còn ở frontend/samples/ nếu sau này train
        # thêm lớp thứ 4.
        {
            "id": "sample_opacity",
            "name": "Lung Opacity",
            "category": "Lung Opacity",
            "image_url": f"{base_url}/static/samples/sample_opacity.png",
            "description": "Ảnh X-quang vùng mờ phổi (Lung Opacity)",
        },
        {
            "id": "sample_normal",
            "name": "Normal",
            "category": "Normal",
            "image_url": f"{base_url}/static/samples/sample_normal.png",
            "description": "Ảnh X-quang phổi bình thường, không ghi nhận tổn thương",
        },
    ]
    return samples


@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def upload_and_predict(request: Request, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Vui lòng tải lên file ảnh định dạng hợp lệ (.jpg, .png).",
        )

    file_ext = Path(file.filename).suffix.lower() if file.filename else ".png"
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng ảnh {file_ext} không được hỗ trợ. Chỉ chấp nhận các định dạng {ALLOWED_EXTENSIONS}.",
        )

    # 1. Tạo mã ca phân tích ngẫu nhiên theo phiên
    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    random_hex = uuid.uuid4().hex[:6].upper()
    record_code = f"RAD-{timestamp_str}-{random_hex}"

    original_name = Path(file.filename).stem if file.filename else "scan"
    unique_filename = f"{original_name}_{uuid.uuid4().hex[:8]}_{int(time.time())}{file_ext}"

    raw_path = RAW_IMAGES_DIR / unique_filename
    result_path = RESULT_IMAGES_DIR / unique_filename
    heatmap_path = HEATMAPS_DIR / unique_filename

    # 2. Lưu file tạm thời để mô hình xử lý
    try:
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể lưu file ảnh: {str(e)}")

    # 3. Lấy AI model từ app state
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Mô hình AI chưa sẵn sàng.")

    # 4. Gọi model xử lý segmentation mask & heatmap
    try:
        prediction_data = model.predict_and_save(
            str(raw_path), str(result_path), str(heatmap_path)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi chạy mô hình dự đoán: {str(e)}")

    # 4b. Cổng gác OOD (src/ood_detector.py, xem ai_engine.py::OOD_GATE_ENABLED) — ảnh
    # không giống X-quang phổi (màu sắc/hình dạng mask bất thường) bị từ chối TẠI ĐÂY,
    # KHÔNG trả một chẩn đoán bịa. File raw vẫn đã lưu ở bước 2 để có record, nhưng
    # không có result/heatmap image nào được tạo ra cho ca này.
    if prediction_data.get("invalid_input"):
        raise HTTPException(
            status_code=422,
            detail="Ảnh tải lên không phải ảnh x quang phổi",
        )

    # 5. Đường dẫn URL trả về cho trình duyệt
    base_url = str(request.base_url).rstrip("/")
    raw_image_url = f"{base_url}/static/raw/{unique_filename}"
    result_image_url = f"{base_url}/static/results/{unique_filename}"
    heatmap_url = f"{base_url}/static/heatmaps/{unique_filename}"

    # Ghép cảnh báo động (model chưa train / shortcut learning nghi vấn — xem
    # ai_engine.py::predict_and_save) vào SAU câu khuyến cáo y tế cố định, thay vì
    # trả disclaimer tĩnh như trước — không đổi schema (vẫn 1 field "disclaimer": str).
    disclaimer = DEFAULT_MEDICAL_DISCLAIMER
    warning = prediction_data.get("warning")
    if warning:
        disclaimer = f"{disclaimer} {warning}"

    return {
        "record_code": record_code,
        "message": "Phân tích thành công",
        "disease": prediction_data["disease"],
        "confidence": prediction_data["confidence"],
        "probabilities": prediction_data.get("probabilities", {}),
        "raw_image_url": raw_image_url,
        "result_image_url": result_image_url,
        "heatmap_url": heatmap_url,
        "metrics": prediction_data.get("metrics"),
        "disclaimer": disclaimer,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }