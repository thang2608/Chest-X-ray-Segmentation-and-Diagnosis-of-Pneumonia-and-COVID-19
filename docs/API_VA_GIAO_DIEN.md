# Backend API & Giao diện Gradio — Kiến trúc, Chức năng, Cách chạy

*Giải thích 2 server tạo nên demo: FastAPI backend (`api/`) và Gradio UI (`app.py`) — vai trò từng file, luồng xử lý 1 request, và tính năng "chỉ số tin cậy" mới thêm sau khi phát hiện shortcut learning.*

---

## 1. Vì sao 2 server tách biệt, không gộp làm 1

```
┌─────────────────────┐        HTTP POST        ┌──────────────────────┐
│   Gradio UI (7860)   │ ───── /predict ────────► │  FastAPI backend      │
│   app.py             │ ◄──── JSON response ──── │  (8000), api/         │
└─────────────────────┘                          └──────────────────────┘
   Người dùng thao tác ở đây                        Model AI sống ở đây
```

Đây là kiến trúc **client-server** kinh điển, tách theo đúng thiết kế gốc của dự án (`pipeline.md`):

- **`api/` (FastAPI, cổng 8000)** — "bộ não": load model 1 lần, giữ trong bộ nhớ, xử lý mọi request suy luận (inference). Không biết gì về giao diện.
- **`app.py` (Gradio, cổng 7860)** — "khuôn mặt": chỉ lo hiển thị, không tự chứa logic AI nào — mọi lần bấm "Chẩn đoán" đều gọi HTTP sang backend.

**Lợi ích của việc tách rời:** backend có thể test độc lập bằng `curl`/Swagger UI (`/docs`) mà không cần mở trình duyệt; UI có thể đổi (Gradio → React → mobile app) mà không đụng vào logic model; hai server chạy 2 tiến trình riêng — nếu UI treo, backend vẫn phục vụ request khác bình thường.

---

## 2. Backend — `api/`

### 2.1. `api/schemas.py` — hợp đồng dữ liệu

```python
class PredictResponse(BaseModel):
    predicted_class: str              # "Normal" | "Lung_Opacity" | "COVID"
    confidence: float                 # xác suất lớp dự đoán, 0.0-1.0
    probabilities: Dict[str, float]   # xác suất cả 3 lớp
    heatmap_overlay_base64: str       # ảnh Grad-CAM overlay, PNG encode base64
    disclaimer: str                   # khuyến cáo y tế + các cảnh báo tự động (xem mục 5)
    lung_overlap_iou: float           # MỚI — IoU(Grad-CAM, mask phổi U-Net)
    lung_overlap_containment: float   # MỚI — % vùng heatmap nằm trong phổi
```

Không có logic — chỉ định nghĩa **hình dạng** dữ liệu. FastAPI dùng class này để tự kiểm tra response đúng shape trước khi trả về, và tự sinh tài liệu Swagger tại `/docs`.

### 2.2. `api/db.py` — ghi log

`init_db()` tạo bảng SQLite `predictions` (nếu chưa có) tại `data/predictions.db`; `log_prediction(predicted_class, confidence)` ghi 1 dòng mỗi lần có kết quả — phục vụ tra cứu lịch sử sau này, không ảnh hưởng tới response trả về người dùng.

### 2.3. `api/inference.py` — trái tim của backend

**Trạng thái toàn cục** (load 1 lần lúc khởi động, dùng lại cho mọi request — tránh load model lặp lại mỗi lần, rất tốn thời gian):

```
_classifier   ← EfficientNet-B3, từ weights/best_classifier.pth
_unet         ← U-Net (ResNet-34 encoder), từ weights/best_unet.pth
_model_is_trained, _unet_is_trained   ← cờ đánh dấu đang dùng weights thật hay fallback
```

**`load_models()` — cơ chế fallback "không bao giờ crash":**

```
weights/best_classifier.pth tồn tại?
   ├── Có, load được       → dùng model thật, _model_is_trained = True
   ├── Có, nhưng load lỗi  → in cảnh báo, rơi xuống nhánh dưới (không crash server)
   └── Không tồn tại        → build_classifier(pretrained=True) — backbone ImageNet
                               + head NGẪU NHIÊN, _model_is_trained = False

(logic tương tự áp dụng độc lập cho U-Net)
```

Thiết kế này cho phép backend **luôn khởi động thành công** dù chưa có weights — hữu ích lúc test luồng API/UI trước khi model train xong (đã dùng thật trong dự án này).

**`predict_image(pil_image)` — pipeline xử lý 1 ảnh, gọi bởi mỗi request:**

```
PIL.Image (bất kỳ size/mode)
   │
   ▼
convert RGB → resize 224×224 (cv2, để overlay sau) + transform chuẩn hoá (cho model)
   │
   ▼
classifier(img) → softmax → predicted_class + confidence + probabilities (3 lớp)
   │
   ▼
generate_gradcam(classifier, img, target_class=predicted) → heatmap (H,W)
   │                       ⚠️ CẦN gradient — chạy NGOÀI khối torch.no_grad()
   ▼
overlay_heatmap(ảnh gốc, heatmap) → ảnh overlay, encode base64
   │
   ▼
predict_lung_mask(unet, img) → mask phổi nhị phân
   │
   ▼
so khớp heatmap vs. mask phổi → lung_overlap_iou, lung_overlap_containment (mục 5)
   │
   ▼
đóng gói JSON theo PredictResponse, trả về api/main.py
```

### 2.4. `api/main.py` — route HTTP

```python
@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...)):
    # 1. Validate content-type phải là ảnh → HTTP 400 nếu không
    # 2. Đọc bytes, thử Image.open() + .load() → HTTP 400 nếu ảnh hỏng
    # 3. Gọi predict_image() (mục 2.3)
    # 4. log_prediction() ghi SQLite
    # 5. Trả JSON
```

`lifespan` context manager gọi `load_models()` + `init_db()` **1 lần duy nhất** lúc `uvicorn` khởi động — không load lại mỗi request.

---

## 3. Frontend — `app.py`

```python
def diagnose(image):
    # 1. Convert RGB, encode PNG, gửi HTTP POST sang API_URL (mặc định localhost:8000/predict)
    # 2. Bọc trong try/except — nếu backend chưa chạy, hiện lỗi rõ ràng trong UI
    #    thay vì Gradio ném lỗi mù mờ
    # 3. Giải mã ảnh overlay từ base64, trả về cho các ô hiển thị
```

`gr.Blocks` định nghĩa layout: ô upload ảnh + nút "Chẩn đoán" bên trái; ô overlay Grad-CAM + bảng xác suất + ô kết luận + **ô "Độ tin cậy giải thích" (mới)** bên phải. `API_URL` đọc từ biến môi trường, mặc định `http://localhost:8000/predict` — đổi được khi deploy (backend và UI chạy ở địa chỉ khác nhau).

---

## 4. Sơ đồ luồng 1 request đầy đủ (từ lúc bấm nút tới lúc thấy kết quả)

```
Người dùng bấm "Chẩn đoán"
   │
   ▼
app.py::diagnose()  ──HTTP POST (ảnh PNG)──►  api/main.py::predict()
                                                    │
                                          validate + đọc ảnh
                                                    │
                                                    ▼
                                    api/inference.py::predict_image()
                                                    │
                                    classifier → Grad-CAM → U-Net → so khớp
                                                    │
                                                    ▼
                                          api/db.py::log_prediction()
                                                    │
                                                    ▼
                                    JSON (PredictResponse) ──HTTP response──►
   │
   ▼
app.py::diagnose() giải mã, trả về Gradio
   │
   ▼
Người dùng thấy: nhãn + %, overlay Grad-CAM, thanh xác suất, kết luận + disclaimer,
                 độ tin cậy giải thích
```

---

## 5. Tính năng mới: "Độ tin cậy giải thích" (`lung_overlap_iou` / `lung_overlap_containment`)

### Vì sao thêm

Kiểm định `shortcut_iou.py` trên toàn bộ test set phát hiện classifier **thường xuyên nhìn ra ngoài phổi** để chẩn đoán (đặc biệt lớp COVID: 76% ảnh có heatmap chủ yếu ngoài phổi, 27% hoàn toàn trật) — nghi vấn shortcut learning (model học watermark/logo thay vì bệnh lý thật, xem `docs/LY_THUYET.md` Phần VIII). Thay vì chỉ giấu phát hiện này trong báo cáo, tính năng này đưa nó **trực tiếp vào từng lần dự đoán** — minh bạch thật, đúng tinh thần "hệ thống giải thích được" của đề tài.

### Cách tính (tái dùng nguyên hàm từ `src/shortcut_iou.py`, không viết lại logic)

```
heatmap (Grad-CAM, giá trị liên tục [0,1])
   │  nhị phân hoá ngưỡng 0.5
   ▼
cam_bin (0/1)              lung_mask (0/1, từ U-Net)
   │                              │
   └──────────────┬───────────────┘
                   ▼
   IoU = |cam_bin ∩ lung_mask| / |cam_bin ∪ lung_mask|
   containment = |cam_bin ∩ lung_mask| / |cam_bin|   ← % vùng heatmap nằm TRONG phổi
```

### Cách đọc

| containment | Ý nghĩa |
|---|---|
| ≥ 0.3 | Không cảnh báo — heatmap chủ yếu nằm trong phổi |
| < 0.3 | Tự động thêm cảnh báo vào `disclaimer`: *"model đang tập trung khoảng X% vào vùng NGOÀI phổi cho ảnh này — kết quả có thể không đáng tin cậy"* |

Nếu U-Net đang ở chế độ fallback (chưa train), disclaimer ghi rõ chỉ số này **không đáng tin cậy** thay vì im lặng hiển thị số sai lệch.

### Đã xác nhận hoạt động đúng (test thật, không chỉ lý thuyết)

Gọi API với đúng ảnh `COVID-1094.png` (đã biết trước từ phân tích CSV offline: IoU=0.000, containment=0.000) — API tính **live** ra đúng y hệt 0.000/0.000 và tự động kèm cảnh báo "100% ngoài phổi" trong `disclaimer`. Xác nhận công thức nhất quán giữa lúc phân tích hàng loạt (`shortcut_iou.py`) và lúc phục vụ thời gian thực (`api/inference.py`).

---

## 6. Cách chạy / dừng

```powershell
# Terminal 1 — backend
python -m uvicorn api.main:app --port 8000

# Terminal 2 — UI (giữ terminal 1 chạy)
python app.py
```

Mở `http://127.0.0.1:7860` (UI) hoặc `http://localhost:8000/docs` (Swagger, test API trực tiếp). Dừng bằng `Ctrl+C` ở cả 2 terminal.

**Yêu cầu trước khi chạy:** `pip install -r requirements.txt` (đã có đủ `fastapi`, `uvicorn`, `gradio`, `requests`, `torch`...); `weights/best_classifier.pth` và `weights/best_unet.pth` nên có sẵn (không bắt buộc — thiếu vẫn chạy được ở chế độ "chưa train", disclaimer tự cảnh báo).

---

## 7. Trạng thái đã kiểm chứng trong phiên làm việc này

- Backend load đúng weights thật (`[inference] OK: loaded trained classifier...`, `...loaded trained U-Net...`).
- 3 request `POST /predict` qua `curl` — cả 3 trả `200 OK`, JSON đúng schema.
- SQLite ghi log đúng (`data/predictions.db`, xác nhận bằng `SELECT * FROM predictions`).
- Gradio UI phản hồi `HTTP 200` tại `127.0.0.1:7860`.
- Chỉ số tin cậy khớp chính xác với số liệu đã tính offline trên toàn bộ test set.
- Cả 2 server đã được dừng sau khi test xong (không còn chạy nền).
