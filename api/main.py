from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from api.db import init_db, log_prediction
from api.inference import load_models, predict_image
from api.schemas import PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    init_db()
    yield


app = FastAPI(title="Chest X-ray Diagnosis API", lifespan=lifespan)


@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File phải là ảnh (Content-Type phải bắt đầu bằng 'image/')",
        )

    image_bytes = file.file.read()
    try:
        pil_image = Image.open(BytesIO(image_bytes))
        pil_image.load()  # ép giải mã ngay — Image.open() chỉ đọc header, ảnh hỏng phần
        # thân sẽ không bị phát hiện nếu không gọi .load() bên trong try/except này
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Không đọc được ảnh — file có thể bị hỏng hoặc sai định dạng",
        )

    result = predict_image(pil_image, filename=file.filename)
    log_prediction(result["predicted_class"], result["confidence"])
    return result
