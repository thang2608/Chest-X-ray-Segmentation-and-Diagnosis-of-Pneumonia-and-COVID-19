# Hướng dẫn Train trên Google Colab (branch `demo`)

*Dùng khi máy local không có GPU CUDA — chạy `train_classifier.ipynb` và `train_unet.ipynb` trên GPU miễn phí của Google Colab, thông qua một branch riêng (`demo`) tách khỏi `main`/`modelLead`.*

---

## Tổng quan luồng làm việc

```
Máy local (CPU)                    GitHub (branch demo)              Google Colab (GPU T4)
────────────────                   ─────────────────────             ─────────────────────
git checkout -b demo
git commit + git push  ──────────► branch demo có code              !git clone -b demo ...
                                                                       │
                                                                       ▼
                                                          Kaggle API tải dataset thô
                                                                       │
                                                                       ▼
                                                       python src/preprocess.py + split_data.py
                                                                       │
                                                                       ▼
                                                        %run notebooks/train_classifier.ipynb
                                                        %run notebooks/train_unet.ipynb
                                                                       │
                                                                       ▼
weights/best_classifier.pth  ◄──────────────────────  files.download(...)
weights/best_unet.pth        ◄──────────────────────  files.download(...)
```

---

## Bước A — Tạo branch `demo` và đẩy code lên GitHub (chạy ở máy local)

```powershell
# 1. Tạo branch demo từ trạng thái hiện tại (giữ nguyên mọi file đang có, không đụng main)
git checkout -b demo

# 2. Commit riêng phần sửa lỗi nhỏ (path preprocess.py, .gitignore)
git add .gitignore src/preprocess.py
git commit -m "fix: correct dataset path, ignore weights/figures artifacts"

# 3. Commit phần notebook U-Net vừa hoàn thiện
git add notebooks/train_unet.ipynb
git commit -m "feat(unet): complete train_unet.ipynb training loop"

# 4. Commit phần backend + docs còn lại
git add api app.py docs Dockerfile requirements.txt figures weights
git commit -m "feat: add backend API scaffold, Gradio UI, docs"

# 5. Đẩy branch demo lên GitHub — branch MỚI, KHÔNG ảnh hưởng main/modelLead
git push -u origin demo
```

Kiểm tra `git status` sau bước 4 phải sạch (`nothing to commit, working tree clean`) trước khi push.

> **Lưu ý:** `demo` là branch chung trên GitHub (cùng repo với team) — push lên là cả team nhìn thấy được. Nếu muốn, báo trước với team để tránh nhầm với `modelLead`.

---

## Bước B — Mở Colab, bật GPU, clone branch `demo`

1. Vào [colab.research.google.com](https://colab.research.google.com) → **New notebook**.
2. **Runtime → Change runtime type → GPU (T4)** → Save.
3. Cell đầu tiên:

```python
!git clone -b demo https://github.com/thang2608/Chest-X-ray-Segmentation-and-Diagnosis-of-Pneumonia-and-COVID-19.git
%cd Chest-X-ray-Segmentation-and-Diagnosis-of-Pneumonia-and-COVID-19
!pip install -q -r requirements-model.txt
```

**Không cần cài `torch`/`torchvision`** — Colab đã có sẵn bản CUDA đúng; đây chính là lý do `requirements-model.txt` cố tình không liệt kê 2 package đó (xem cảnh báo ở đầu file).

---

## Bước C — Tải dataset từ Kaggle (nhanh hơn nhiều so với upload tay ~9000 ảnh)

### C.1. Lấy API token (`kaggle.json`)

1. Đăng nhập [kaggle.com](https://kaggle.com) (tạo tài khoản miễn phí nếu chưa có).
2. Bấm ảnh đại diện (góc trên phải) → **Settings** → cuộn xuống mục **API** → **"Create New Token"**.
3. Trình duyệt tự tải về file `kaggle.json` (chứa `{"username": "...", "key": "..."}`) — đây là bí mật, không chia sẻ/commit lên Git.

### C.2. Upload token lên Colab

```python
from google.colab import files
files.upload()  # bấm "Choose Files", chọn đúng kaggle.json vừa tải
```

### C.3. Đặt đúng vị trí + quyền truy cập

```python
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

Thư viện `kaggle` quy ước cứng tìm token tại `~/.kaggle/kaggle.json` — sai vị trí sẽ báo lỗi "Could not find kaggle.json" ở bước sau. `chmod 600` giới hạn quyền đọc/ghi chỉ cho chủ file (thư viện `kaggle` từ chối chạy nếu file token có quyền quá mở, vì nó chứa API key).

### C.4. Tải dataset

```python
!kaggle datasets download -d tawsifurrahman/covid19-radiography-database
```

`tawsifurrahman/covid19-radiography-database` là "slug" lấy trực tiếp từ URL trang Kaggle của dataset. Lệnh tải về file `covid19-radiography-database.zip` (~1-2GB) vào thư mục hiện tại.

### C.5. Giải nén — điểm dễ sai nhất

File zip của Kaggle **thường đã có sẵn 1 lớp thư mục `COVID-19_Radiography_Dataset/` bên trong nó**. Nếu giải nén thêm vào một thư mục cũng đặt tên như vậy (`unzip ... -d COVID-19_Radiography_Dataset`), kết quả sẽ bị **lồng 2 lớp** — đúng lỗi khiến `src/preprocess.py` từng phải sửa `DATASET_DIR` ở máy local. Cách an toàn: giải nén **không chỉ định `-d`**, để zip tự tạo đúng thư mục gốc của nó:

```python
!unzip -q covid19-radiography-database.zip
!rm covid19-radiography-database.zip   # xoá zip cho đỡ tốn dung lượng, không bắt buộc
```

**Luôn kiểm tra lại cấu trúc** trước khi chạy `preprocess.py`:

```python
!ls COVID-19_Radiography_Dataset/
```

- **Đúng** (1 lớp): thấy trực tiếp `COVID/ Lung_Opacity/ Normal/ 'Viral Pneumonia/'`.
- **Sai** (vẫn thấy `COVID-19_Radiography_Dataset/` lồng bên trong) → gộp lại:
  ```python
  !mv COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/* COVID-19_Radiography_Dataset/
  !rmdir COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset
  ```

---

## Bước D — Chạy data pipeline ngay trên Colab

```python
!python src/preprocess.py
!python src/split_data.py
```

Kỳ vọng: `data/split/{train,val,test}/images/` có tổng cộng ~9000 ảnh (train ~6300, val ~1350, test ~1350).

---

## Bước E — Chạy notebook train

### E.1. `%run` KHÔNG hoạt động với file `.ipynb` — dùng `nbformat` + `exec()` thay thế

⚠️ **Đã gặp lỗi thật, đừng dùng cách này:** `%run notebooks/train_classifier.ipynb` báo lỗi
`OSError: File 'notebooks/train_classifier.ipynb.py' not found` — lệnh magic `%run` của
IPython **chỉ hỗ trợ file `.py`**, tự động thêm đuôi `.py` vào bất kỳ tên file nào không có
đuôi thực thi được, kể cả khi bạn gõ rõ `.ipynb`.

Cũng **không** mở file qua Files pane (📁 bên trái) — bấm vào `.ipynb` chỉ ra **panel xem
trước** tĩnh (cell ghi `<undefined>`), không kết nối runtime, không chạy được gì.

**Cách đúng, đã kiểm chứng chạy được:** ở lại notebook thiết lập (nơi đã `git clone`/tải
dataset — đừng đóng nó), thêm cell mới, chạy:

```python
!pip install -q nbformat   # vô hại nếu Colab đã có sẵn

import nbformat

def run_notebook(path: str):
    """Đọc file .ipynb bằng nbformat, thực thi TỪNG cell code lần lượt bằng exec()
    ngay trong runtime hiện tại (biến, import, dataset đã tải đều dùng chung) —
    đúng hành vi 'chạy cả notebook' mà %run KHÔNG làm được với .ipynb."""
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            exec(cell.source, globals())

run_notebook("notebooks/train_classifier.ipynb")
```

Sau khi chạy xong, lặp lại với notebook khác (đổi đường dẫn), ví dụ:
```python
run_notebook("notebooks/train_unet.ipynb")
```

### E.2. (Tuỳ chọn) Thêm 1 cell "an toàn" trước khi gọi `run_notebook`, phòng lỡ ở nhầm thư mục

```python
%cd /content/Chest-X-ray-Segmentation-and-Diagnosis-of-Pneumonia-and-COVID-19
```
Vô hại nếu đã đúng thư mục; báo lỗi ngay (`No such file or directory`) nếu vì lý do gì đó chưa đúng — giúp phát hiện sớm thay vì gặp `ModuleNotFoundError: No module named 'src'` khó hiểu khi `run_notebook()` chạy tới cell import bên trong.

### E.3. Nội dung khi chạy — `train_classifier.ipynb` có 10 cell, theo `docs/TUTORIAL.md` Phần 8:

| Cell | Làm gì | Dấu hiệu chạy đúng |
|---|---|---|
| 1 | Kiểm tra GPU | `CUDA available: True`, tên GPU (Tesla T4) |
| 2 | `set_seed(42)` | Không output |
| 3 | Import | Không `ModuleNotFoundError` |
| 4 | Config (batch size, LR, epoch từng pha) | Không output |
| 5 | Tạo `DataLoader` | Không `RuntimeError: No PNG found` (nếu có → Bước D chưa chạy trong đúng runtime này) |
| 6 | Build model + hàm `run_epoch` | In số tham số EfficientNet-B3 |
| 7 | Khởi tạo `history` | Không output |
| 8 | **Vòng lặp train 3 pha** — lâu nhất | Progress bar mỗi epoch, dòng `Ep 01 train_loss=... val_f1=...`, `Saved best_f1=...` khi cải thiện |
| 9 | Vẽ biểu đồ + confusion matrix | 2 hình hiện ngay trong notebook, lưu vào `figures/` |

Cell 8 (3 pha, tối đa 3+15+5=23 epoch, có early stopping) là lâu nhất — trên T4, mỗi epoch thường **30 giây–1.5 phút** → cả notebook khoảng **15-40 phút**.

Sau khi thấy `BEST VAL MACRO F1: 0.xxxx`, chạy tiếp `run_notebook("notebooks/train_unet.ipynb")` (1 pha, tối đa 25 epoch, theo dõi Dice/IoU — nhẹ hơn, khoảng **10-25 phút**).

### E.4. Phát hiện lỗi sớm

`run_notebook()` chạy tuần tự từng cell — nếu 1 cell lỗi, traceback hiện ra chỉ đúng dòng gây lỗi trong file `.ipynb` tương ứng (đọc kỹ thông báo lỗi, đối chiếu với bảng cell ở trên để biết đang hỏng ở bước nào). Nếu cần debug sâu hơn theo kiểu chạy từng cell một: copy nội dung các cell nghi vấn từ file `.ipynb` (mở bằng Read/editor bất kỳ, hoặc xem trong Files pane) dán thành các cell riêng trong chính notebook thiết lập, chạy từng cell bằng `Shift+Enter` — đảm bảo chắc chắn dùng đúng runtime đang có sẵn dataset/package, không phụ thuộc cách Colab xử lý file preview.

### E.5. Train thêm biến thể classifier (cropped / blackout) — KHÔNG cần lặp lại Bước C/D

`train_classifier_cropped.ipynb` và `train_classifier_blackout.ipynb` dùng CHUNG `data/split/`
đã tạo ở Bước D (đọc mask ground-truth trực tiếp từ `data/split/*/masks/`, **không cần**
`weights/best_unet.pth`) — nếu vẫn đang ở cùng runtime Colab vừa chạy xong Bước D (hoặc vừa
train xong 1 biến thể khác), chỉ cần chạy thẳng:

```python
run_notebook("notebooks/train_classifier_blackout.ipynb")
```

Nếu là phiên Colab MỚI (session trước đã bị ngắt) thì vẫn phải làm lại đủ Bước B→D trước
(clone lại repo — nhớ `git pull` hoặc clone lại để có notebook mới nhất trên branch `demo`
— rồi tải/giải nén/preprocess/split dataset) vì mọi thứ trong `/content` mất sạch khi
runtime bị thu hồi.

---

## Bước F — Tải checkpoint về máy sau khi train xong

```python
from google.colab import files
files.download("weights/best_classifier.pth")        # bản baseline (nếu vừa train)
files.download("weights/best_classifier_cropped.pth")   # bản crop (nếu vừa train)
files.download("weights/best_classifier_blackout.pth")  # bản blackout (nếu vừa train)
files.download("weights/best_unet.pth")               # chỉ cần train 1 lần, dùng chung
```

Copy các file tải về vào thư mục `weights/` ở máy local — đúng vị trí `api/inference.py`
và `backend/app/services/ai_engine.py` đã viết sẵn để tự động load, ưu tiên theo thứ tự
**blackout > cropped > baseline** (file "tối ưu nhất" đang có). Không cần sửa code:
`load_models()`/`MedicalSegmentationModel.__init__()` tự phát hiện file tồn tại.

---

## Lưu ý quan trọng

- Phiên Colab miễn phí bị ngắt sau **~90 phút không tương tác** hoặc **~12 tiếng liên tục** — nếu train lâu, thỉnh thoảng quay lại tab để tránh bị ngắt giữa chừng.
- Notebook hiện **chưa hỗ trợ resume-from-checkpoint** — chỉ lưu `best_*.pth` khi có cải thiện; nếu phiên bị ngắt giữa chừng, phải chạy lại từ đầu (mất tiến trình các epoch chưa lưu checkpoint mới nhất).
- Nếu train nhiều phiên và không muốn tải lại dataset mỗi lần: mount Google Drive (`from google.colab import drive; drive.mount('/content/drive')`), copy dataset đã giải nén vào Drive, các lần sau chỉ cần copy từ Drive thay vì tải lại từ Kaggle.
- Tài liệu liên quan: `docs/TUTORIAL.md` (Phần 8-9, giải thích từng cell), `docs/LY_THUYET.md` (Phần I-VI, nền tảng toán học của quá trình train), `docs/QUY_TRINH_CODE.md` (Phần 5, bản đồ hàm/biến trong notebook).
