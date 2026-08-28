# Nhật ký thay đổi: Tối ưu Shortcut Learning bằng Crop theo Mask U-Net

*Ghi lại toàn bộ thay đổi code cho hướng xử lý đã chọn ở `docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md` Phần 5 — dùng trực tiếp cho phần "Phương pháp tối ưu" của báo cáo sau này.*

**Trạng thái tại thời điểm viết:** toàn bộ hạ tầng code đã hoàn thiện và test kỹ thuật (không crash, dữ liệu ra đúng shape/khoảng giá trị). **CHƯA train phiên bản "đã tối ưu"** — chưa có số liệu Accuracy/F1/IoU "sau" thật để đối chiếu. Bước tiếp theo bắt buộc là chạy `notebooks/train_classifier_cropped.ipynb` trên Colab.

---

## 1. Ý tưởng & vì sao chọn hướng này

Phát hiện ở `docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md` Phần 3-4: Grad-CAM heatmap của classifier thường nằm **ngoài** vùng phổi (đặc biệt COVID: 76% ảnh containment < 0.3, 27.3% IoU=0 tuyệt đối) — nghi vấn shortcut learning (model học watermark/artifact thay vì bệnh lý thật).

**Giải pháp:** cắt ảnh theo bounding box của mask phổi (từ U-Net hoặc ground-truth) **trước khi** đưa vào classifier — loại bỏ vật lý vùng có thể chứa watermark khỏi input, đúng thiết kế gốc trong `description.md`. Train lại classifier trên ảnh đã crop, **giữ nguyên mọi hyperparameter khác** (đúng tinh thần ablation study — chỉ đổi 1 biến duy nhất, để kết luận "crop có tác dụng" có giá trị khoa học).

---

## 2. Danh sách file đã sửa

### 2.1. `src/dataset.py` — hàm crop dùng chung

**Thêm mới:**
```python
def crop_to_lung_bbox(image: np.ndarray, mask: np.ndarray, padding: float = 0.1) -> np.ndarray
```
Cắt ảnh theo bounding box của vùng >0 trong mask, có đệm biên `padding` (10% mỗi chiều theo mặc định — tránh cắt sát rìa phổi mất chi tiết biên). Trả nguyên ảnh gốc nếu mask rỗng (an toàn, không crash).

**Sửa `ChestXrayClassificationDataset`:** thêm 2 tham số mới `crop_to_lung: bool = False`, `crop_padding: float = 0.1` — **mặc định `False`, không đổi hành vi ở mọi nơi đã dùng class này trước đó** (`src/shortcut_iou.py`, `notebooks/evaluate_local.ipynb`, `notebooks/train_classifier.ipynb` gốc). Khi `crop_to_lung=True`, dataset tự đọc thêm mask ground-truth (từ `split_dir/masks/`) và crop ảnh trước khi áp transform.

**Đã test:** ảnh gốc 224×224 → crop còn ~193-224×200-224 tuỳ ca cụ thể (kích thước phổi khác nhau mỗi ảnh) — đúng như kỳ vọng.

### 2.2. `notebooks/train_classifier_cropped.ipynb` — notebook train mới

Bản sao **y hệt** `notebooks/train_classifier.ipynb` gốc (giữ file gốc nguyên vẹn để so sánh), chỉ khác đúng 2 chỗ:
- `CKPT_PATH = "weights/best_classifier_cropped.pth"` (thay vì `best_classifier.pth`) — **không ghi đè checkpoint baseline**.
- `ChestXrayClassificationDataset(..., crop_to_lung=True)` cho cả `train_ds` và `val_ds`.
- Tên file hình lưu ra đổi thành `loss_f1_curves_cropped.png`, `confusion_matrix_val_cropped.png` (không ghi đè hình của bản gốc).

Mọi hyperparameter khác (3 pha, LR, epoch, batch size, early stopping...) **giữ nguyên 100%**.

### 2.3. `src/shortcut_iou.py` — đánh giá shortcut learning cho bản "sau"

**Thêm hàm:** `dice(a, b)` — Dice coefficient giữa 2 mask nhị phân (dùng cho so sánh U-Net vs ground-truth ở `api/inference.py`, mục 2.4).

**Sửa `run_shortcut_analysis()`:** thêm 2 tham số `crop_to_lung: bool = False`, `crop_padding: float = 0.1`. Khi `True`:
1. Lấy mask phổi (gt hoặc U-Net dự đoán, như trước).
2. Crop cả ảnh **và** mask theo cùng 1 bounding box (dùng `crop_to_lung_bbox` 2 lần).
3. Chạy Grad-CAM trên ảnh **đã crop**.
4. Resize mask đã crop khớp kích thước heatmap (luôn 224×224 sau transform) trước khi tính IoU/containment.

**⚠️ Lưu ý quan trọng đã ghi trong docstring (cần trích dẫn khi viết báo cáo):** containment "sau crop" tự nhiên cao hơn "trước crop" một phần vì **hiệu ứng hình học** (mask sau crop chiếm tỉ lệ diện tích khung hình lớn hơn nhiều, không phải vì model học tốt hơn). Chỉ số đáng tin cậy nhất để so sánh trước/sau là **% ảnh có IoU = 0 tuyệt đối** (không phụ thuộc tỉ lệ diện tích tương đối).

**Đã test kỹ thuật** (dùng tạm checkpoint baseline, chỉ để kiểm tra code chạy đúng — không phải số liệu thật): pipeline crop → Grad-CAM → resize mask → so khớp chạy không lỗi, IoU/containment ra giá trị hợp lệ trong [0,1].

### 2.4. `api/inference.py` — tích hợp crop vào luồng serve

**Thêm hằng số:** `CROPPED_WEIGHTS_PATH = Path("weights/best_classifier_cropped.pth")`, `CROP_PADDING = 0.1` (phải khớp giá trị dùng lúc train).

**Sửa `load_models()`:** ưu tiên tự động dùng `CROPPED_WEIGHTS_PATH` nếu file tồn tại (đặt `_crop_mode=True`); nếu chưa có (đúng trạng thái hiện tại), fallback về `WEIGHTS_PATH` như cũ (`_crop_mode=False`) — **hành vi hiện tại không đổi cho tới khi checkpoint mới thực sự xuất hiện**.

**Sửa `predict_image()`:** tách 2 nhánh rõ ràng theo `_crop_mode`:
- **`_crop_mode=False` (baseline):** logic giữ **y hệt** bản gốc, không đổi 1 dòng hành vi.
- **`_crop_mode=True` (tối ưu):** U-Net chạy TRƯỚC classifier để lấy mask → crop ảnh → classifier + Grad-CAM chạy trên ảnh đã crop → overlay hiển thị trên khung ảnh đã crop.

**Đã kiểm chứng bằng regression test thật** (chạy server, gọi `/predict` với đúng ảnh `COVID-1094.png` đã biết kết quả từ trước): output **khớp tuyệt đối byte-for-byte** với trước khi sửa (`confidence=0.9998502731323242`, `lung_overlap_iou=0.0`, `unet_vs_gt_dice=0.992090395480226`) — xác nhận không có regression nào cho demo hiện tại.

**`PredictResponse` (api/schemas.py): KHÔNG đổi field nào** — mọi thay đổi nằm trong logic tính toán nội bộ, giữ nguyên hợp đồng dữ liệu cho phía UI.

---

## 3. Các bước còn lại để có số liệu "sau tối ưu" thật

1. Push nhánh chứa các thay đổi này lên GitHub, chạy `notebooks/train_classifier_cropped.ipynb` trên Colab (theo `docs/HUONG_DAN_TRAIN_COLAB.md`, chỉ khác tên notebook).
2. Tải `weights/best_classifier_cropped.pth` về đúng thư mục `weights/` ở máy local — `api/inference.py` sẽ **tự động** chuyển sang dùng bản này (không cần sửa code thêm).
3. Chạy lại `src/shortcut_iou.py` với `crop_to_lung=True` (đã có sẵn code mẫu comment trong khối `if __name__ == "__main__":`) — lấy số liệu IoU/containment "sau" để so sánh với bảng ở `docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md` Phần 3.
4. Chạy `notebooks/evaluate_local.ipynb` (cần cập nhật để dùng `crop_to_lung=True` khi tạo dataset test) để lấy Accuracy/F1 "sau" — so sánh trực tiếp với bảng ở Phần 1.
5. Cập nhật `docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md` thêm Phần C (kết quả sau tối ưu) theo đúng cấu trúc "trước/sau" đã thống nhất.

---

## 4. Trạng thái Git — CHƯA commit/push

Các thay đổi ở mục 2 hiện chỉ nằm trên máy local, **chưa commit**. Lý do: `api/inference.py` (mục 2.4) đang được một thành viên khác trong nhóm xem/chỉnh sửa song song — cần xác nhận thời điểm phù hợp để tránh xung đột trước khi đẩy lên nhánh chung `demo`. Thiết kế "tự động tương thích ngược" (mục 2.4) giúp việc này an toàn hơn: merge sớm hay muộn đều không ảnh hưởng tới demo đang chạy, vì nhánh crop chỉ kích hoạt khi có file checkpoint mới.
