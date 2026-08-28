# Thay đổi: Tích hợp AI thật vào backend `backend/app/` (thay Gradio)

*Changelog kỹ thuật — dùng cho báo cáo. Ghi lại quyết định + bằng chứng kiểm chứng, không chỉ mô tả code.*

---

## 1. Vì sao đổi hướng backend

Dự án có 2 nhánh phát triển song song không tương thích:

- **`demo`** — `api/` (FastAPI) + `app.py` (Gradio), có đầy đủ AI thật (classifier F1 0.906, U-Net Dice 0.986, Grad-CAM, phân tích shortcut learning).
- **`main`** (qua PR #6, `feat/backend-web`, do chủ repo tạo) — `backend/app/{routers,services,schemas,core}/` (FastAPI) + `backend/frontend/index.html` (giao diện HTML/JS tĩnh, không dùng Gradio). Kiến trúc chuyên nghiệp hơn cho việc deploy thật, nhưng phần AI (`ai_engine.py`) lúc merge **hoàn toàn là mock**: `self.model = "MOCK_MODEL_LOADED"`, kết quả bịa theo tên file ảnh (`if "normal" in path: confidence=96.8`...), còn nhắc tới kiến trúc YOLO (khác hẳn EfficientNet-B3+U-Net đã train).

**Quyết định:** giữ kiến trúc `backend/app/` + HTML tĩnh của chủ repo (đã merge `main`, dự định là hướng chính thức), nhưng **thay toàn bộ phần mock bằng pipeline AI thật** đã xây dựng trên `demo`. Thực hiện trên branch mới `feat/backend-integration` (tạo từ `demo`, lấy thêm thư mục `backend/` từ `origin/feat/backend-web` bằng `git checkout origin/feat/backend-web -- backend/` — không dùng `git merge` vì `main` đã xoá sạch `src/`/`docs/`/`notebooks/` lúc branch ra, merge trực tiếp sẽ tạo hàng chục conflict "xoá vs sửa").

---

## 2. Danh sách file đã sửa

| File | Thay đổi |
|---|---|
| `backend/app/core/config.py` | Thêm `REPO_ROOT` + tự thêm vào `sys.path` (để `import src.xxx` hoạt động dù `uvicorn` chạy với cwd=`backend/`, khác cwd chứa `src/`). `WEIGHTS_DIR` đổi từ `backend/weights/` sang trỏ **thẳng về `weights/` ở gốc repo** — dùng chung checkpoint đã train (`best_classifier.pth`, `best_unet.pth`, ~140MB), không copy trùng file nặng. Thêm hằng số `CLASSIFIER_WEIGHTS`, `UNET_WEIGHTS`. |
| `backend/app/core/__init__.py` | Export thêm `REPO_ROOT`, `CLASSIFIER_WEIGHTS`, `UNET_WEIGHTS`. |
| `backend/app/main.py` | `lifespan()` gọi `MedicalSegmentationModel(str(CLASSIFIER_WEIGHTS), str(UNET_WEIGHTS))` — 2 tham số thay vì 1 (trước đây chỉ truyền 1 đường dẫn `.pt` không tồn tại). |
| `backend/app/services/ai_engine.py` | **Viết lại hoàn toàn**, xem mục 3. |
| `backend/app/routers/predict.py` | Bỏ mục "sample_pneumonia" khỏi `/samples` (model không có lớp này). Nối cảnh báo động (`prediction_data["warning"]`) vào sau `DEFAULT_MEDICAL_DISCLAIMER` thay vì trả disclaimer tĩnh — schema `disclaimer: str` không đổi. |
| `backend/app/schemas/prediction.py` | Sửa docstring các field `disease`, `result_image_url`, `dice_score`, `precision`, `recall`, `iou_score`, `affected_lung_area` cho khớp ý nghĩa thật (xem mục 4) — không đổi tên field/kiểu dữ liệu nào, `index.html` không cần sửa gì để đọc response. |
| `backend/frontend/index.html` | Bỏ 1 sample-card "Viral Pneumonia" trong lưới ảnh mẫu. |
| `backend/requirements.txt` | Thêm `torch`, `torchvision`, `albumentations`, `segmentation-models-pytorch`, `grad-cam` (trước đó chỉ có package nhẹ đủ cho mock). |

---

## 3. `ai_engine.py` — thay mock bằng pipeline thật

Giữ nguyên chữ ký `MedicalSegmentationModel.predict_and_save(image_path, mask_output_path, heatmap_output_path) -> dict` mà `routers/predict.py` gọi — không đổi interface, chỉ đổi bên trong.

**Luồng xử lý mới** (tái dùng nguyên vẹn `src/` đã có, không viết lại logic AI):

```
image_path (đĩa)
   │  PIL → numpy RGB → resize 224×224
   ▼
classifier (EfficientNet-B3, src.model.load_classifier) → softmax → disease + confidence
   │
   ▼ (NGOÀI torch.no_grad — cần backward)
generate_gradcam (src.gradcam) → heatmap → overlay_heatmap → LƯU heatmap_output_path
   │
   ▼
predict_lung_mask (U-Net, src.unet + src.shortcut_iou) → mask phổi → tô xanh lá → LƯU mask_output_path
   │
   ▼
iou()/containment() (src.shortcut_iou) → iou_score, affected_lung_area, cảnh báo nếu containment thấp
   │
   ▼
return {disease, confidence, metrics, warning}
```

**Cơ chế fallback graceful** — giống hệt nguyên tắc đã dùng ở `api/inference.py` trên `demo`: nếu `best_classifier.pth`/`best_unet.pth` không tồn tại hoặc load lỗi, tự động dùng backbone pretrained ImageNet (không crash server), và gắn cờ `*_is_trained=False` để cảnh báo qua `warning`.

---

## 4. 3 quyết định thiết kế (ánh xạ field cũ ↔ ý nghĩa mới)

Schema gốc (`EvaluationMetrics`, `PredictionResponse`) được thiết kế cho một mô hình khác (phân đoạn trực tiếp vùng tổn thương, 4 lớp bệnh) — không khớp 100% với hệ thống đã xây (phân loại + U-Net định vị PHỔI + Grad-CAM giải thích, 3 lớp). Quyết định ánh xạ, không đổi schema:

1. **`disease` — chỉ 3 lớp thật, không phải 4.** Model chỉ train `Normal`/`Lung_Opacity`/`COVID` (`Viral Pneumonia` bị loại khỏi `src/preprocess.py` từ đầu dự án, xem `CLAUDE.md`). Tên hiển thị: `{"Normal": "Normal", "Lung_Opacity": "Lung Opacity", "COVID": "COVID-19"}` — **không dùng chữ "Pneumonia"** để tránh người dùng hiểu nhầm là chẩn đoán viêm phổi do virus (lớp mình không có).
2. **`result_image_url` — đổi ý nghĩa từ "mask vùng tổn thương" sang "mask vùng phổi".** Hệ thống không có model phân đoạn tổn thương cụ thể (chỉ có U-Net định vị phổi) — field này giờ trả về ảnh phổi tô xanh lá chồng lên ảnh gốc, docstring đã sửa lại cho đúng, không giả vờ có tính năng chưa có.
3. **`dice_score`/`precision`/`recall` — số tổng hợp trên val set, không phải per-image.** Ảnh người dùng upload mới không có ground-truth để tính per-image (cùng vấn đề đã gặp và xử lý y hệt khi thiết kế `unet_vs_gt_dice` trên bản Gradio trước đây). Lấy từ `AGGREGATE_METRICS` hằng số trong `ai_engine.py` — nguồn số liệu: `docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md` (U-Net Val Dice 0.9862; Classifier Macro Precision 0.9075, Macro Recall 0.9067 — **lưu ý đây là val set, chưa phải test set chính thức**, xem Phần 4.4 báo cáo đó). `iou_score` và `affected_lung_area` **là số tính per-image thật** (Grad-CAM vs mask phổi cho đúng ảnh vừa upload) — khác 3 số trên.

**Việc còn để ngỏ, chưa làm (do thời gian):** `dice_score`/`precision`/`recall` hiện là hằng số tĩnh, không tự cập nhật khi train lại model. Khi có `notebooks/evaluate_local.ipynb` chạy lại (đặc biệt sau khi có bản crop-mask), cần sửa tay 3 số trong `AGGREGATE_METRICS`, hoặc nâng cấp thành tính động có cache (như thiết kế `/metrics` từng bàn cho bản Gradio) — chưa cấp thiết vì đây là số hiển thị tham khảo, không ảnh hưởng logic chẩn đoán.

---

## 5. Kiểm chứng — chạy server thật, không chỉ đọc code

Chạy `cd backend && uvicorn app.main:app --port 8001`, test bằng `curl`:

| Test | Kết quả |
|---|---|
| `GET /api/health` | 200, đúng format |
| `GET /api/v1/samples` | 200, còn đúng 3 mẫu (Viral Pneumonia đã bỏ) |
| `POST /api/v1/predict` — `COVID-1094.png` (đã biết trước IoU=0.000 từ CSV phân tích offline) | `iou_score: 0.0` — **khớp tuyệt đối** với số liệu offline. `disclaimer` tự động kèm "LUU Y: model dang tap trung khoang 100% vao vung NGOAI phoi..." |
| `POST /api/v1/predict` — `Normal-1000.png` | `disease: "Normal"`, `confidence: 99.8` — hợp lý |
| 3 URL ảnh tĩnh (`/static/raw`, `/static/results`, `/static/heatmaps`) | Cả 3 trả 200, file thật được ghi đúng đĩa |
| `POST /api/v1/predict` với file không phải ảnh (`.md`) | 400 rõ ràng (`"Vui lòng tải lên file ảnh định dạng hợp lệ"`), server không sập |

Kết quả `iou_score` khớp tuyệt đối giữa lúc phân tích hàng loạt (`src/shortcut_iou.py`, chạy offline trên toàn test set) và lúc phục vụ thời gian thực (`ai_engine.py`, chạy live cho 1 ảnh) — xác nhận không có sai lệch công thức giữa 2 đường code dùng chung `src/`.
