from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    DEFAULT_MODEL_WEIGHTS,
    FRONTEND_DIR,
    HEATMAPS_DIR,
    RAW_IMAGES_DIR,
    RESULT_IMAGES_DIR,
    SAMPLES_DIR,
)
from app.routers import predict
from app.schemas.prediction import HealthResponse
from app.services.ai_engine import MedicalSegmentationModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo mô hình AI và nạp vào app state
    app.state.model = MedicalSegmentationModel(str(DEFAULT_MODEL_WEIGHTS))
    print("[INFO] Server đã sẵn sàng nhận ảnh phân tích (Chế độ Privacy-First).")

    yield  # Server chạy

    print("[INFO] Đang tắt server và giải phóng tài nguyên...")
    del app.state.model


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount các thư mục tĩnh
app.mount("/static/raw", StaticFiles(directory=str(RAW_IMAGES_DIR)), name="static_raw")
app.mount("/static/results", StaticFiles(directory=str(RESULT_IMAGES_DIR)), name="static_results")
app.mount("/static/heatmaps", StaticFiles(directory=str(HEATMAPS_DIR)), name="static_heatmaps")
app.mount("/static/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="static_samples")

# Gắn Router
app.include_router(predict.router, prefix="/api/v1")


@app.get("/api/health", response_model=HealthResponse, tags=["Health Check"])
def health_check():
    return {
        "status": "Hệ thống đang hoạt động tốt",
        "version": APP_VERSION,
    }


# Route phục vụ giao diện Web Frontend tại trang chủ
@app.get("/", include_in_schema=False)
def serve_frontend():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "status": "Hệ thống đang hoạt động tốt",
        "version": APP_VERSION,
        "docs_url": "/docs",
    }