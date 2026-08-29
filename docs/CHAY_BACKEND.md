# Chạy Backend (FastAPI + frontend HTML) — tham khảo nhanh

## Lệnh chạy

```powershell
cd backend
uvicorn app.main:app --port 8000
```

- **Bắt buộc chạy với `cwd = backend/`** (không phải gốc repo) — package `app` (`app.main:app`) chỉ import được từ trong `backend/`. `backend/app/core/config.py` tự thêm gốc repo vào `sys.path` lúc import để `import src.xxx` vẫn hoạt động (đọc model/dataset dùng chung), nhưng bản thân lệnh `uvicorn` vẫn phải chạy từ `backend/`.
- Đổi `--port 8000` nếu cổng đó đang bận (ví dụ dùng `8001`/`8002` để test song song không đụng server đang chạy).
- Thêm `--reload` khi đang sửa code backend để tự khởi động lại lúc lưu file (không dùng khi benchmark/đo thời gian suy luận).
- Mở `http://127.0.0.1:8000` → giao diện web (`backend/frontend/index.html`). Xem `http://127.0.0.1:8000/docs` cho Swagger UI của API.

## Dừng server

Tìm PID đang giữ cổng rồi kill (Windows):
```powershell
netstat -ano | findstr :8000
taskkill /F /PID <pid>
```

## Ưu tiên chọn phiên bản model — chạy tự động, không cần chỉnh gì

Server tự phát hiện file trọng số nào có trong `weights/` (ở **gốc repo**, dùng chung với `src/`, không phải `backend/weights/`) và tự chọn bản "tốt nhất đang có" theo thứ tự **cropped > blackout > baseline**. Toàn bộ logic nằm ở đây:

- **`backend/app/core/config.py`** (dòng ~29-37): khai báo 3 đường dẫn — `CLASSIFIER_WEIGHTS` (baseline), `CROPPED_CLASSIFIER_WEIGHTS`, `BLACKOUT_CLASSIFIER_WEIGHTS` — đều trỏ vào `weights/` gốc repo.
- **`backend/app/main.py`** (`lifespan()`, dòng ~26-38): lúc server khởi động, truyền cả 3 đường dẫn vào `MedicalSegmentationModel(...)`.
- **`backend/app/services/ai_engine.py`** (`MedicalSegmentationModel.__init__`, dòng ~98-135): đây là nơi **quyết định thật** — thử load `cropped_classifier_path` trước; nếu file không tồn tại hoặc load lỗi mới thử `blackout_classifier_path`; nếu cả 2 đều không có mới rơi về `classifier_path` (baseline); nếu baseline cũng thiếu thì fallback ImageNet-pretrained (model chưa train, chỉ để test luồng kỹ thuật). Kết quả set 2 cờ `self.crop_mode`/`self.blackout_mode`, in ra log dòng `[ai_engine] READY | ... crop_mode=... blackout_mode=...` lúc khởi động — xem log này để biết chắc server đang chạy bản nào.

**Vì sao ưu tiên cropped chứ không phải bản mới nhất (blackout)?** Xem `docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md` Phần 5.5-5.6 — blackout giảm shortcut learning rất mạnh nhưng đổi lấy Macro F1 thấp hơn cả baseline, nên **không tự "leo thang"** lên dùng nó chỉ vì file mới hơn. Muốn ép server dùng bản cụ thể để test/demo: xoá tạm (hoặc đổi tên) file `.pth` cao ưu tiên hơn khỏi `weights/` trước khi khởi động — không có biến môi trường/flag riêng cho việc này hiện tại.

Cùng cơ chế 3 tầng này cũng áp dụng cho `api/inference.py` (backend Gradio cũ, `load_models()`) — cùng thứ tự ưu tiên, độc lập codebase.
