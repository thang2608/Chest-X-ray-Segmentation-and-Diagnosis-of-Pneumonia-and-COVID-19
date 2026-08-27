# TUTORIAL — Hướng dẫn xây dựng toàn bộ hệ thống Chest X-ray Segmentation & Diagnosis

*Từ lý thuyết ML/DL đã học đến một hệ thống chạy được thật: tiền xử lý dữ liệu → huấn luyện U-Net + EfficientNet-B3 → giải thích bằng Grad-CAM → backend FastAPI → giao diện Gradio → đóng gói Docker.*

Tài liệu này viết cho người **vừa học xong lý thuyết ML/DL, chưa từng lắp một pipeline thật từ đầu đến cuối**. Mỗi phần đều có: khái niệm cần hiểu trước, code khung (skeleton) để tham khảo, giải thích từng dòng quan trọng, các lỗi thường gặp (gotchas), và cách tự kiểm tra (sanity check) trước khi đi tiếp.

> **Lưu ý quan trọng:** Các file `.py` và `.ipynb` trong repo hiện đang là **file rỗng/placeholder** (xem cấu trúc thư mục ở Phần 1.4). Nhiệm vụ của bạn là đọc từng phần dưới đây, **tự gõ lại code** vào đúng file tương ứng — không copy-paste mù. Code trong tài liệu là bản tối thiểu chạy được, không phải bản production; bạn được khuyến khích chỉnh sửa khi đã hiểu.

---

## Mục lục

0. Cách dùng tài liệu này
1. Tổng quan dự án & kiến trúc toàn hệ thống
2. Ôn tập nhanh nền tảng lý thuyết cần dùng
3. Chuẩn bị môi trường & công cụ
4. Giai đoạn 1 — Data pipeline (`preprocess.py`, `split_data.py`, `verify.py`, `visualize.py`)
5. Giai đoạn 2 — `src/dataset.py`: nền móng dữ liệu cho model
6. Giai đoạn 3 — `src/model.py`: EfficientNet-B3 classifier
7. Giai đoạn 4 — `src/unet.py`: U-Net segmentation
8. Giai đoạn 5 — `notebooks/train_classifier.ipynb`: huấn luyện classifier
9. Giai đoạn 6 — `notebooks/train_unet.ipynb`: huấn luyện U-Net
10. Giai đoạn 7 — `src/gradcam.py`: giải thích bằng Grad-CAM
11. Giai đoạn 8 — `src/shortcut_iou.py`: kiểm định shortcut learning
12. Giai đoạn 9 — `api/`: backend FastAPI (`POST /predict`)
13. Giai đoạn 10 — `app.py`: giao diện Gradio
14. Giai đoạn 11 — `Dockerfile` & triển khai lên Hugging Face Spaces
15. Debug playbook tổng hợp
16. Lộ trình đề xuất & checklist kiểm thử cuối
17. Đánh giá mô hình (Evaluation) toàn diện
18. Viết report
19. Tài liệu tham khảo

---

## 0. Cách dùng tài liệu này

Tài liệu chia thành **11 giai đoạn (Phần 4 → 14)**, đúng theo thứ tự bạn nên code — thứ tự này **không tùy ý**, nó phản ánh dependency thật giữa các file:

```
(1) preprocess.py, split_data.py   ─── tạo data/split/ trên đĩa
        │
        ▼
(2) src/dataset.py                 ─── hằng số + Dataset dùng chung
        │
        ├──► (3) src/model.py  ────┐
        │                          │
        ├──► (4) src/unet.py       │
        │         │                ▼
        │         │      (5) train_classifier.ipynb ──► weights/best_classifier.pth
        │         │      (6) train_unet.ipynb        ──► weights/best_unet.pth
        │         │                │
        ├─────────┴──► (7) src/gradcam.py  ◄─── cần classifier đã train
        │                          │
        └──────────────► (8) src/shortcut_iou.py ◄─── cần cả classifier + U-Net + gradcam
                                   │
                                   ▼
                    (9) api/  ──► (10) app.py ──► (11) Docker
```

Lý do thứ tự này bắt buộc:

- `dataset.py` phải làm đầu tiên vì `CLASS_TO_IDX`, `IMAGE_SIZE`, `MEAN`, `STD` là **hằng số toàn dự án** — `model.py`, `gradcam.py`, `api/inference.py` đều import lại từ đây. Đổi sau sẽ phải sửa hàng loạt chỗ.
- `model.py` + `unet.py` phải xong trước khi mở notebook train, vì notebook chỉ **orchestrate** (gọi hàm, lặp epoch) — logic kiến trúc model nằm trong file `.py` để sau này `api/inference.py` import lại được y hệt lúc train.
- `gradcam.py` và `shortcut_iou.py` để cuối cùng phần model vì cần trọng số (`weights/*.pth`) đã train xong mới test được.
- `api/` cần cả 3 thứ trên (model đã train, gradcam) mới có gì để phục vụ; `app.py` cần `api/` chạy được mới có endpoint để gọi; Docker đóng gói sau cùng khi mọi thứ đã chạy được ở local.

Ước lượng thời gian nếu làm tập trung, không tính thời gian train chờ GPU chạy xong:

| Giai đoạn | Thời gian |
|---|---|
| Data pipeline (đã có sẵn — chỉ cần chạy & hiểu) | 0.5 ngày |
| `dataset.py` | 0.5 ngày |
| `model.py` + `unet.py` | 0.5 ngày |
| `train_classifier.ipynb` (viết + train + tune) | 1.5–2 ngày |
| `train_unet.ipynb` | 1–1.5 ngày |
| `gradcam.py` | 0.5 ngày |
| `shortcut_iou.py` + phân tích | 0.5–1 ngày |
| Backend FastAPI (`api/`) | 1–1.5 ngày |
| Frontend Gradio (`app.py`) | 0.5–1 ngày |
| Docker + deploy Hugging Face Spaces | 0.5–1 ngày |
| **TỔNG** | **~8–11 ngày làm việc thuần** |

Mỗi giai đoạn (Phần 4–14) đều có cấu trúc 6 mục cố định, học theo:

1. **Mục đích & API contract** — file này để làm gì, ai import gì từ nó.
2. **Kiến thức nền tảng cần nắm** — lý thuyết/tool phải hiểu *trước khi* code, không phải tra cứu giữa chừng.
3. **Skeleton code** — khung tối thiểu chạy được.
4. **Chi tiết implementation** — giải thích các quyết định thiết kế không hiển nhiên.
5. **Gotchas** — lỗi/bẫy đã biết trước, để không mất hàng giờ debug.
6. **Cách tự test** — sanity check trước khi coi là "xong", trước khi qua giai đoạn tiếp theo.

---

## 1. Tổng quan dự án & kiến trúc toàn hệ thống

### 1.1. Bài toán

Xây dựng một hệ thống web có thể giải thích được (*explainable*), nhận một ảnh X-quang ngực và trả về:

1. Chẩn đoán: **Normal / Lung Opacity / COVID-19** (bài toán phân loại 3 lớp).
2. Vùng phổi được khoanh vùng (bài toán phân đoạn/segmentation nhị phân).
3. Heatmap giải thích model đang "nhìn" vào đâu để ra quyết định (Grad-CAM).
4. Log lại lịch sử dự đoán để tra cứu sau (SQLite).

### 1.2. Vì sao cần *hai* model chứ không phải một?

Đây là điểm dễ nhầm nhất với người mới. Bạn **không train một model làm mọi thứ**, mà có hai model độc lập, mỗi model một việc:

- **U-Net (bộ định vị phổi):** nhận ảnh X-quang thô, xuất ra mask nhị phân "đâu là phổi, đâu không phải". Nó không quan tâm bệnh gì — chỉ quan tâm hình học giải phẫu.
- **EfficientNet-B3 (bộ chẩn đoán):** nhận ảnh (đã hoặc chưa qua U-Net, tùy thiết kế) và phân loại thành 3 lớp bệnh lý.

Grad-CAM sau đó lấy gradient từ EfficientNet-B3 để vẽ heatmap; U-Net dùng để **kiểm chứng** heatmap đó có thực sự nằm trong vùng phổi hay không (Phần 11 — shortcut learning). Hai model này được train **hoàn toàn tách biệt**, bằng hai notebook khác nhau, trên hai dataset con khác nhau (ảnh gốc + mask cho U-Net; ảnh gốc + nhãn lớp cho classifier).

### 1.3. Kiến trúc tổng thể end-to-end

```
                         ┌─────────────────────────┐
                         │   Gradio UI (app.py)     │  ← bác sĩ upload ảnh X-quang
                         └────────────┬─────────────┘
                                      │ HTTP POST (ảnh)
                                      ▼
                         ┌─────────────────────────┐
                         │  FastAPI backend (api/)  │
                         │   POST /predict          │
                         └────────────┬─────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
     ┌─────────────────┐   ┌───────────────────┐   ┌────────────────────┐
     │ EfficientNet-B3  │   │ U-Net (lung mask)  │   │ Grad-CAM            │
     │ (classifier)     │   │                    │   │ (dùng classifier)   │
     └────────┬─────────┘   └─────────┬──────────┘   └──────────┬─────────┘
              │                       │                          │
              └──────────────┬────────┴──────────────────────────┘
                              ▼
                  Gộp kết quả: nhãn, % tin cậy, heatmap overlay
                              │
                     ┌────────┴────────┐
                     ▼                 ▼
              Trả JSON về UI     Ghi log vào SQLite (api/db.py)
```

Đối chiếu với pipeline mô tả trong `pipeline.md` của dự án:

1. **Input:** nhận ảnh X-quang ngực từ Gradio UI.
2. **Tiền xử lý:** resize 224×224, chuẩn hóa theo ImageNet stats (CLAHE là tùy chọn nâng cao, xem Phần 4).
3. **Inference & XAI:** EfficientNet-B3 phân loại + Grad-CAM trích heatmap (kết hợp U-Net lọc vùng phổi để kiểm tra heatmap có "trung thực" không).
4. **Output & Logging:** hiển thị chẩn đoán, % tin cậy, ảnh overlay heatmap, khuyến cáo y tế, ghi log SQLite.

### 1.4. Cấu trúc thư mục đầy đủ của dự án

```
Chest-X-ray-Segmentation-and-Diagnosis-of-Pneumonia-and-COVID-19/
├── README.md                     # hướng dẫn tải dataset + chạy pipeline (đã có)
├── requirements.txt              # dependency cho api/ + app.py (backend, UI)
├── requirements-model.txt        # dependency riêng cho phần train model (nặng, có CUDA)
├── Dockerfile                    # đóng gói backend + UI để deploy
├── .gitignore
├── data/                         # KHÔNG commit — sinh ra khi chạy script (gitignored)
│   ├── processed/                #   ảnh + mask đã resize, gắn nhãn theo lớp
│   └── split/                    #   train/val/test sau khi chia
├── COVID-19_Radiography_Dataset/ # dataset thô tải từ Kaggle (gitignored)
├── src/
│   ├── __init__.py
│   ├── preprocess.py             # ĐÃ CÓ — resize + gắn nhãn mask
│   ├── split_data.py             # ĐÃ CÓ — chia train/val/test
│   ├── verify.py                 # ĐÃ CÓ — kiểm tra nhanh 1 ảnh/lớp
│   ├── visualize.py              # ĐÃ CÓ — vẽ ảnh/mask/overlay
│   ├── dataset.py                # Giai đoạn 2 — PyTorch Dataset + transforms
│   ├── model.py                  # Giai đoạn 3 — EfficientNet-B3 classifier
│   ├── unet.py                   # Giai đoạn 4 — U-Net segmentation
│   ├── gradcam.py                # Giai đoạn 7 — Grad-CAM heatmap
│   └── shortcut_iou.py           # Giai đoạn 8 — kiểm định shortcut learning
├── notebooks/
│   ├── train_classifier.ipynb    # Giai đoạn 5
│   └── train_unet.ipynb          # Giai đoạn 6
├── weights/                      # trọng số .pth sau khi train (gitignored, giữ .gitkeep)
│   ├── best_classifier.pth
│   └── best_unet.pth
├── figures/                      # biểu đồ/hình xuất ra để đưa vào báo cáo (gitignored)
├── api/
│   ├── __init__.py
│   ├── main.py                   # Giai đoạn 9 — FastAPI app, endpoint /predict
│   ├── schemas.py                # Pydantic request/response models
│   ├── inference.py              # load model 1 lần, hàm predict dùng chung
│   └── db.py                     # ghi/đọc log SQLite
├── app.py                        # Giai đoạn 10 — Gradio UI, gọi sang api/
└── docs/
    └── TUTORIAL.md                # chính là file bạn đang đọc
```

### 1.5. Bảng công cụ & lý do chọn

| Công cụ | Vai trò | Vì sao chọn |
|---|---|---|
| **PyTorch** | Framework deep learning | Linh hoạt, dễ debug (eager execution), hệ sinh thái model pretrained lớn nhất cho vision. |
| **Torchvision** | Cung cấp EfficientNet-B3 pretrained ImageNet | Không phải tự cài kiến trúc + trọng số tay. |
| **Albumentations** | Augmentation ảnh | Nhanh hơn `torchvision.transforms` 2–3 lần; hỗ trợ augment *đồng bộ* ảnh + mask — bắt buộc cho segmentation. |
| **segmentation-models-pytorch (SMP)** | Cung cấp kiến trúc U-Net + encoder pretrained | Tiết kiệm thời gian code tay U-Net; encoder pretrained hội tụ nhanh hơn nhiều với dataset chỉ ~9k ảnh. |
| **pytorch-grad-cam** | Sinh heatmap Grad-CAM | Thư viện chuẩn, được cite rộng rãi, tránh tự code lại toán gradient dễ sai. |
| **scikit-learn** | Tính Macro F1, Precision, Recall, Confusion Matrix | Chuẩn học thuật, dễ so sánh với báo cáo khác. |
| **FastAPI** | Backend API | Type-safe (Pydantic), tự sinh docs OpenAPI (`/docs`), async tốt cho I/O ảnh. |
| **Uvicorn** | ASGI server chạy FastAPI | Server chuẩn đi kèm FastAPI. |
| **Gradio** | Giao diện demo | Dựng UI upload-ảnh-xem-kết-quả trong vài chục dòng code, không cần biết frontend. |
| **SQLite** | Lưu log dự đoán | Không cần server DB riêng, 1 file `.db`, đủ cho demo/đồ án. |
| **Docker** | Đóng gói | Đảm bảo môi trường chạy giống hệt nhau ở mọi máy, cần thiết để deploy lên Hugging Face Spaces. |
| **Hugging Face Spaces** | Hosting demo miễn phí | Free tier CPU đủ cho inference (không cần train ở đây), tích hợp sẵn Docker/Gradio. |

---

## 2. Ôn tập nhanh nền tảng lý thuyết cần dùng

Phần này **không dạy lại ML/DL từ đầu** (bạn đã học xong lý thuyết) mà chỉ nối lại các khái niệm bạn đã biết với *đúng chỗ* chúng xuất hiện trong dự án này, để khi đọc các phần sau bạn không bị hụt.

### 2.1. CNN & vì sao ảnh y tế cần transfer learning

CNN học feature qua các lớp convolution xếp chồng: lớp nông học cạnh/góc, lớp sâu học pattern phức tạp (texture, hình dạng cơ quan). Dataset của bạn có ~9.000 ảnh sau khi giới hạn `MAX_IMAGES_PER_CLASS = 3000` — quá nhỏ để train một CNN sâu **từ đầu** (random init) mà không overfit nặng. Giải pháp chuẩn: **transfer learning** — dùng model đã train trên ImageNet (1.28 triệu ảnh, 1000 lớp vật thể đời thường), giữ lại phần "biết nhìn hình dạng, texture, cạnh" (feature extractor), chỉ thay/ huấn luyện lại phần phân loại cuối cho bài toán của bạn.

### 2.2. Compound scaling — vì sao chọn EfficientNet-B3

Paper gốc: Tan & Le, *"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"* (ICML 2019). Ý tưởng: thay vì tăng riêng lẻ độ sâu (số layer), độ rộng (số channel) hoặc độ phân giải ảnh input, EfficientNet tăng **đồng thời cả ba** theo một tỷ lệ tối ưu tìm bằng grid search — cho hiệu quả tham số tốt hơn hẳn scale ngẫu nhiên.

Series B0→B7, mỗi bước tăng ~1.4 lần FLOPs. B3 là điểm cân bằng cho ảnh y tế: B0 (nhỏ nhất) dễ underfit trên task fine-grained (phân biệt COVID vs Lung Opacity nhìn khá giống nhau); B5 trở lên nặng, dễ overfit khi dataset chỉ ~9k ảnh và train chậm. B3 (~12M tham số) là lựa chọn thực dụng.

### 2.3. Fine-tuning theo pha (progressive unfreezing)

Nếu unfreeze toàn bộ backbone và train với learning rate cao ngay từ đầu, gradient lớn từ classification head (đang random, loss cao) sẽ lan ngược vào backbone và phá hỏng feature ImageNet đã học — gọi là **catastrophic forgetting**. Giải pháp: train theo 3 pha, mở khóa dần:

1. **Pha 1 — Warm-up head:** đóng băng backbone, chỉ train lớp Linear cuối, LR cao (1e-3). Mục đích: đưa head từ random về trạng thái "biết" trước khi cho gradient chạm vào feature.
2. **Pha 2 — Fine-tune vài block cuối:** mở khóa 2–3 block cuối, LR trung bình (1e-4). Các block gần cuối học feature high-level đặc thù cho bài toán.
3. **Pha 3 — Full fine-tune (tùy chọn):** mở khóa toàn bộ, LR rất thấp (1e-5), chỉ làm nếu val F1 ở pha 2 vẫn còn cải thiện.

### 2.4. Segmentation: U-Net, skip connection, loss

U-Net (Ronneberger et al., MICCAI 2015) có cấu trúc encoder–decoder đối xứng hình chữ U, với **skip connection** nối trực tiếp feature map encoder sang decoder cùng cấp. Lý do cần skip connection: quá trình downsample (pooling) ở encoder làm mất thông tin không gian chi tiết (vị trí pixel chính xác); decoder một mình không phục hồi lại được — skip connection "trả lại" thông tin pixel-level đó để mask output sắc nét ở biên.

Với bài toán nhị phân (phổi/không phổi), hai loss phổ biến:

- **BCE (Binary Cross-Entropy):** đo lỗi từng pixel độc lập. Vấn đề: bị chi phối bởi lớp đa số (nền chiếm nhiều pixel hơn phổi).
- **Dice Loss** (`1 − Dice coefficient`): đo trực tiếp độ chồng lấp (overlap) giữa mask dự đoán và ground truth, không bị lệch bởi mất cân bằng lớp.
- Kết hợp `0.5·BCE + 0.5·Dice` là lựa chọn phổ biến và ổn định nhất — dùng mặc định.

Hai metric đánh giá overlap: **Dice** = `2|A∩B| / (|A|+|B|)` và **IoU (Jaccard)** = `|A∩B| / |A∪B|`. Quan hệ: `Dice = 2·IoU / (1+IoU)` — Dice luôn ≥ IoU, khoan dung hơn ở giá trị thấp. Quy ước: báo cáo cả hai.

### 2.5. Grad-CAM — giải thích quyết định của classifier

Grad-CAM (Selvaraju et al., ICCV 2017) trả lời câu hỏi: "model nhìn vào pixel nào của ảnh để ra quyết định lớp c?". Các bước:

1. Forward pass, lấy activation map `A^k` của một conv layer mục tiêu (thường là layer conv cuối cùng trước global average pooling).
2. Lấy điểm số (logit) `y^c` của lớp `c` cần giải thích.
3. Tính gradient `∂y^c/∂A^k`, global-average-pool theo không gian để ra trọng số `α_k^c` cho từng kênh `k`.
4. Heatmap = `ReLU(Σ_k α_k^c · A^k)` — tổng có trọng số các activation map, chỉ giữ phần đóng góp dương.
5. Resize heatmap lên kích thước ảnh gốc, chuẩn hóa về `[0, 1]`.

Chọn layer mục tiêu quan trọng: quá nông (layer đầu) → heatmap chỉ là bộ dò cạnh, không mang tính "semantic" (không phản ánh model đang nghĩ gì về bệnh lý); sau global average pooling → mất hoàn toàn thông tin không gian, không vẽ heatmap được. Layer conv cuối cùng (`features[-1]` trong EfficientNet-B3 của torchvision) là điểm cân bằng chuẩn.

### 2.6. Shortcut learning — vì sao cần kiểm chứng thêm

Một model có accuracy cao trên test set **không đồng nghĩa** nó học đúng đặc điểm bệnh lý. Nó có thể học một "đường tắt" (shortcut) — ví dụ watermark, chữ ký máy chụp, viền ảnh khác nhau giữa các nguồn dữ liệu — miễn là đặc điểm đó tương quan với nhãn trong tập train.

Ví dụ kinh điển được trích trong tài liệu gốc của dự án: Zech et al. (*PLoS Medicine*, 2018) cho thấy một CNN train trên X-quang ở bệnh viện A không transfer sang bệnh viện B, vì model đã học "dấu hiệu bệnh viện" (một loại watermark/token) thay vì bệnh lý thật. Ý tưởng kiểm chứng trong dự án này: nếu Grad-CAM heatmap của model **tập trung trong vùng phổi** (đo bằng IoU với lung mask), model nhiều khả năng học đúng; nếu heatmap thường xuyên nằm ngoài phổi, đó là dấu hiệu cảnh báo shortcut. Đây là lý do U-Net và Grad-CAM phải làm trước, `shortcut_iou.py` mới có input để chạy.

---

## 3. Chuẩn bị môi trường & công cụ

### 3.1. Cài đặt package

Tách riêng hai file requirement (đã tạo sẵn trong repo) để backend nhẹ khi deploy — không cần Albumentations/scikit-learn/torchinfo (chỉ dùng lúc train) trên server production:

- **`requirements-model.txt`** — dùng khi train (máy có GPU hoặc Colab/Kaggle): `torch`, `torchvision`, `albumentations`, `segmentation-models-pytorch`, `grad-cam`, `scikit-learn`, `matplotlib`, `seaborn`, `torchinfo`, `tqdm`, `Pillow`, `opencv-python-headless`.
- **`requirements.txt`** — dùng khi chạy backend/UI: `fastapi`, `uvicorn`, `gradio`, `torch`, `torchvision`, `segmentation-models-pytorch`, `grad-cam`, `opencv-python-headless`, `Pillow`.

Cài PyTorch **qua kênh chính thức** tại [pytorch.org](https://pytorch.org) (chọn OS + CUDA version rồi copy lệnh pip) thay vì cài thẳng từ `requirements.txt`, vì bản CUDA phải khớp driver máy bạn — cài sai kênh là nguồn lỗi phổ biến nhất khi mới bắt đầu.

### 3.2. Kiểm tra GPU trước khi train

Chạy đoạn sau trong Python REPL **trước khi** mở notebook train:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
if torch.cuda.is_available():
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
```

**Vì sao bước này không phải thừa thãi:** EfficientNet-B3 chạy trên CPU chậm hơn GPU khoảng 50–100 lần — một epoch mất 5 phút trên GPU có thể thành 5–8 tiếng trên CPU. Trên Colab/Kaggle, runtime đôi khi tự fallback về CPU (hết quota GPU, quên đổi runtime type) mà bạn không để ý, chỉ phát hiện khi thấy epoch chạy siêu chậm — lúc đó đã mất thời gian. In VRAM ra giúp chọn đúng `BATCH_SIZE` **trước khi** train, tránh vừa train vừa bị CUDA out of memory giữa chừng.

Gợi ý batch size theo VRAM (EfficientNet-B3, ảnh 224×224):

| VRAM | Batch size |
|---|---|
| 4GB (Colab free T4 / GTX 1650) | 16, bật mixed precision |
| 8GB (RTX 3070) | 32 |
| 16GB+ (A100 / P100 Kaggle) | 64–128 |
| Chỉ có CPU | Không train được EfficientNet-B3 — hạ xuống ResNet-18 hoặc dùng Colab |

### 3.3. Đặt seed để reproducible

Cell đầu tiên của **mọi** notebook train, và đầu file `shortcut_iou.py`, phải có:

```python
import random, os
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Đánh đổi tốc độ để lấy tính lặp lại — bật khi cần so sánh nghiêm ngặt
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

**Vì sao cần:** PyTorch, NumPy, `random` mỗi cái có bộ sinh số ngẫu nhiên riêng, dùng cho khởi tạo trọng số, shuffle DataLoader, augmentation (Albumentations dùng cả `random` lẫn `numpy.random`). Không set seed → mỗi lần chạy lại notebook, cùng một code, ra **kết quả khác nhau**. Điều này gây khó khi cần trả lời "chạy lại có ra đúng F1 như báo cáo không?", và khi debug (so sánh LR 1e-3 vs 1e-4) sẽ không biết chênh lệch là do đổi LR hay do nhiễu ngẫu nhiên giữa hai lần chạy.

**Vì sao đặt lại ở cả hai notebook:** mỗi notebook là một tiến trình Python độc lập (kernel riêng) — seed không tự "kế thừa" giữa hai notebook.

**Trade-off của `cudnn.deterministic=True`:** cuDNN có nhiều thuật toán convolution khác nhau cho cùng một phép tính; mặc định (`benchmark=True`) nó tự chọn thuật toán *nhanh nhất* dựa trên benchmark lúc chạy — nhưng thuật toán nhanh nhất có thể cho kết quả số học khác nhỏ (do thứ tự cộng floating-point khác nhau). Đặt `deterministic=True` ép cuDNN dùng thuật toán cố định → chậm hơn ~10–20% nhưng đảm bảo kết quả lặp lại bit-for-bit. Với báo cáo học thuật nên bật; nếu chỉ đang thử nghiệm nhanh có thể tạm tắt.

---

## 4. Giai đoạn 1 — Data pipeline

Bốn file trong `src/` — `preprocess.py`, `split_data.py`, `verify.py`, `visualize.py` — **đã được viết sẵn** trong repo. Phần này giải thích chúng để bạn hiểu rõ dữ liệu đi qua những bước gì trước khi tới `dataset.py`, không phải để bạn viết lại.

### 4.1. Chuẩn bị dataset thô

Tải **COVID-19 Radiography Database** từ Kaggle, giải nén vào thư mục gốc repo sao cho có cấu trúc:

```
COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/
├── COVID/{images,masks}/
├── Lung_Opacity/{images,masks}/
├── Normal/{images,masks}/
└── Viral Pneumonia/{images,masks}/
```

Dataset này đặc biệt vì mỗi ảnh X-quang đã có sẵn **mask phổi** do bên tạo dataset cung cấp (không phải bạn tự vẽ) — đây chính là ground-truth dùng để train U-Net ở Giai đoạn 4 và để so sánh IoU ở Giai đoạn 8.

### 4.2. `preprocess.py` — chuẩn hóa & gắn nhãn mask

Đọc lại [src/preprocess.py](../src/preprocess.py). Với mỗi lớp trong `CLASSES = ["COVID", "Lung_Opacity", "Normal"]` (lưu ý: **`Viral Pneumonia` không được đưa vào** — pipeline hiện tại chỉ phân biệt 3 lớp, không phải 4):

1. Lấy tối đa `MAX_IMAGES_PER_CLASS = 3000` ảnh/lớp (random sample với `RANDOM_SEED = 42`) — để cân bằng số lượng giữa các lớp, tránh mất cân bằng nghiêm trọng.
2. Resize ảnh về `224×224` bằng nội suy **bilinear** (mượt, phù hợp ảnh xám liên tục); resize mask bằng **nearest neighbor** (giữ nguyên biên nhị phân sắc nét, bilinear sẽ làm mờ biên mask và tạo giá trị pixel không hợp lệ).
3. Nhị phân hóa mask (`> 127` → phổi) rồi **gắn lại giá trị pixel theo lớp** thay vì 0/1 thuần: `LABELS = {"Normal": 1, "Lung_Opacity": 2, "COVID": 3}`. Nghĩa là mask output không chỉ là "phổi/nền" mà còn mã hóa luôn lớp bệnh vào giá trị pixel — hữu ích nếu sau này bạn muốn phân tích riêng theo lớp, nhưng với U-Net (chỉ cần nhị phân) bạn sẽ phải nhị phân hóa lại ở `dataset.py` (xem Phần 5.4).
4. Bỏ qua ảnh không có mask tương ứng, in cảnh báo — dấu hiệu dataset có thể thiếu một vài file, không phải lỗi code.

Chạy: `python src\preprocess.py` (từ thư mục gốc repo). Kết quả ghi vào `data/processed/<class>/{images,masks}/`.

### 4.3. `split_data.py` — chia train/val/test

Đọc lại [src/split_data.py](../src/split_data.py). Với mỗi lớp, shuffle danh sách ảnh (cùng `RANDOM_SEED = 42` để tái lập được), rồi cắt theo tỉ lệ `SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}` bằng index slicing, copy cặp ảnh+mask sang `data/split/<split>/{images,masks}/`.

> **Lưu ý kỹ thuật cần biết trước khi tự viết `dataset.py`:** công thức `val_end = train_end + n * SPLITS["val"]` cộng một chỉ số đã được tính theo `train_end` (số nguyên) với một tỉ lệ mới của `n` — cách tính này vẫn cho ra 3 tập không chồng lấp và tổng đúng bằng `n`, nhưng tỉ lệ **val/test thực tế có thể lệch nhẹ so với 15/15 danh nghĩa** tùy vào `n` của từng lớp (do làm tròn số nguyên ở `train_end`). Sau khi chạy, nên tự đếm lại số ảnh mỗi tập (`verify.py` không làm việc này, bạn có thể thêm đoạn đếm nhanh bằng `len(list(dir.iterdir()))` nếu cần số chính xác cho báo cáo).

Chạy: `python src\split_data.py` (sau khi đã chạy `preprocess.py`).

### 4.4. `verify.py` và `visualize.py` — sanity check bằng mắt

[src/verify.py](../src/verify.py) in kích thước ảnh/mask và các giá trị pixel duy nhất (`np.unique`) của mask cho 1 ảnh/lớp — chạy ngay sau `preprocess.py` để xác nhận: kích thước đúng `224×224`, và mask chỉ chứa `{0, 1, 2, 3}` (không có giá trị lạ do lỗi resize/threshold).

[src/visualize.py](../src/visualize.py) vẽ lưới 3×3 (ảnh gốc / mask / overlay) cho một ảnh ngẫu nhiên mỗi lớp bằng matplotlib — cách nhanh nhất để phát hiện lỗi bằng mắt (mask lệch vị trí so với ảnh, mask trống, ảnh bị hỏng) mà số liệu không lộ ra.

**Quy tắc nên tuân theo:** sau *mỗi* lần chạy `preprocess.py` hoặc thay đổi logic tiền xử lý, chạy lại cả hai script này trước khi đi tiếp — lỗi dữ liệu phát hiện muộn (ví dụ ở bước train) tốn thời gian debug hơn rất nhiều so với phát hiện ngay ở đây.

---

## 5. Giai đoạn 2 — `src/dataset.py`: nền móng dữ liệu cho model

### 5.1. Mục đích & API contract

File này định nghĩa **dữ liệu thô đi vào model**: (a) khai báo hằng số dùng chung cho cả train và serve (`api/`), (b) đọc cấu trúc `data/split/<split>/{images,masks}/`, (c) trả về tensor đã chuẩn hóa + augment, sẵn sàng feed vào model.

Phải export ra ngoài (không đổi tên sau khi các file khác đã import):

- Hằng số: `CLASS_TO_IDX`, `IDX_TO_CLASS`, `IMAGE_SIZE`, `MEAN`, `STD`.
- Class: `ChestXrayClassificationDataset` (trả `(image, label)`) và `ChestXraySegmentationDataset` (trả `(image, mask)`).
- Hàm: `get_train_transforms()`, `get_val_transforms()` — trả về `Albumentations.Compose`.

### 5.2. Kiến thức nền tảng cần nắm trước khi code

**PyTorch `Dataset` & `DataLoader`.** `torch.utils.data.Dataset` chỉ cần override 2 method: `__len__(self)` và `__getitem__(self, idx)`. `DataLoader` bọc quanh nó để lấy batch, với các tham số quan trọng: `batch_size`, `shuffle`, `num_workers` (0–8, thường 4 — số tiến trình con đọc dữ liệu song song), `pin_memory=True` khi có GPU (giúp copy tensor lên VRAM nhanh hơn), `drop_last=True` cho tập train (tránh batch cuối bị lẻ, gây bất ổn cho BatchNorm vì thống kê batch nhỏ không đại diện). Tham khảo: [PyTorch Data Loading Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).

**Chuẩn hóa ảnh (Normalization).** Đưa giá trị pixel về khoảng chuẩn (mean≈0, std≈1) giúp optimizer hội tụ nhanh hơn, tránh gradient bùng nổ. Với transfer learning từ ImageNet, **bắt buộc** dùng đúng thống kê ImageNet: `MEAN = [0.485, 0.456, 0.406]`, `STD = [0.229, 0.224, 0.225]` — đây là điều kiện ràng buộc của trọng số pretrained; dùng thống kê khác, feature extractor sẽ nhận input lệch phân phối so với lúc nó được train, hoạt động sai. Ảnh X-quang là ảnh xám (1 kênh) nhưng EfficientNet-B3 pretrained kỳ vọng input 3 kênh (RGB) — chuyển bằng cách **lặp kênh xám thành R=G=B** (`.convert("L").convert("RGB")`), không phải áp colormap giả màu. Câu hỏi kinh điển: "có nên tự tính mean/std trên chính tập train thay vì dùng số ImageNet không?" — **Không**, khi vẫn dùng trọng số pretrained. **Có**, nếu bạn quyết định train from scratch (không khuyến khích với dataset nhỏ như ở đây).

**Augmentation cho ảnh y tế — nguyên tắc: augmentation phải giữ nguyên ý nghĩa y khoa của ảnh.**

| Augmentation | Có nên dùng? | Lý do |
|---|---|---|
| HorizontalFlip | Được | Giải phẫu người đối xứng trái-phải (dù tim lệch trái — vẫn là lựa chọn phổ biến trong benchmark X-quang) |
| VerticalFlip | **Không bao giờ** | X-quang có chiều lên-xuống cố định; lật dọc tạo ảnh không tồn tại trong thực tế |
| Rotation nhỏ (±10°) | Được | Mô phỏng bệnh nhân đứng hơi nghiêng khi chụp |
| ShiftScaleRotate nhẹ | Được | Tương tự trên |
| RandomBrightnessContrast nhẹ | Được | Mô phỏng sai khác giữa các máy chụp |
| GaussianNoise, ElasticTransform | Cân nhắc | Dễ tạo artifact giả giống tổn thương |
| CutOut / RandomErasing | **Tránh** | Có thể vô tình xóa đúng vùng tổn thương, khiến model học sai |
| MixUp / CutMix | Không dùng ở giai đoạn đầu | Chỉ thử nếu muốn mở rộng nghiên cứu |

**Albumentations vs `torchvision.transforms`.** Albumentations nhanh hơn 2–3 lần, và quan trọng hơn: hỗ trợ áp **cùng một phép augmentation cho cả ảnh và mask cùng lúc** (`additional_targets={"mask": "mask"}`) — bắt buộc phải có cho `ChestXraySegmentationDataset`, nếu không ảnh bị xoay/lật mà mask không xoay/lật theo thì cặp dữ liệu sẽ sai lệch hoàn toàn. Một điểm dễ gây bug: `ToTensorV2()` của Albumentations chỉ chuyển numpy array sang tensor, **không** tự chia cho 255 để scale về `[0,1]` — phải đặt `A.Normalize(...)` (tự chia 255 rồi chuẩn hóa) **trước** `ToTensorV2()` trong `Compose`, không được đảo thứ tự.

### 5.3. Skeleton code

```python
# src/dataset.py
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---- Hằng số dùng chung (KHÔNG đổi khi các file khác đã import) ----
CLASS_TO_IDX = {"Normal": 0, "Lung_Opacity": 1, "COVID": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_TO_IDX)

IMAGE_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]  # ImageNet stats
STD = [0.229, 0.224, 0.225]

# ---- Transforms ----
def get_train_transforms(image_size=IMAGE_SIZE):
    return A.Compose([
        A.Resize(*image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                            rotate_limit=10, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1,
                                    contrast_limit=0.1, p=0.5),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=IMAGE_SIZE):
    return A.Compose([
        A.Resize(*image_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_train_transforms_seg(image_size=IMAGE_SIZE):
    return A.Compose([
        A.Resize(*image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                            rotate_limit=10, border_mode=0, p=0.5),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})

# ---- Parse label từ tiền tố tên file ----
def _parse_label(filename: str) -> int:
    """File convention: 'COVID-123.png', 'Normal-42.png', 'Lung_Opacity-7.png'."""
    for cls_name, idx in CLASS_TO_IDX.items():
        if filename.startswith(cls_name):
            return idx
    raise ValueError(f"Cannot parse label from filename: {filename}")

# ---- Classification Dataset ----
class ChestXrayClassificationDataset(Dataset):
    def __init__(self, split_dir: str, transform: Optional[Callable] = None):
        self.image_dir = Path(split_dir) / "images"
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.transform = transform
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No PNG found in {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[idx]
        image = np.array(Image.open(path).convert("RGB"))  # H, W, 3
        label = _parse_label(path.name)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

# ---- Segmentation Dataset ----
class ChestXraySegmentationDataset(Dataset):
    def __init__(self, split_dir: str, transform: Optional[Callable] = None):
        self.image_dir = Path(split_dir) / "images"
        self.mask_dir = Path(split_dir) / "masks"
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        msk_path = self.mask_dir / img_path.name
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(msk_path).convert("L"))  # H, W
        # Nhị phân hóa mask về {0, 1} bất kể pixel value gốc (1/2/3)
        mask = (mask > 0).astype(np.float32)
        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return image, mask.unsqueeze(0)  # mask shape (1, H, W)
```

### 5.4. Chi tiết implementation

**Vì sao nhị phân hóa mask về `{0, 1}` cho U-Net?** Mask từ `preprocess.py` có giá trị pixel theo lớp (Normal=1, Lung_Opacity=2, COVID=3) — mã hóa đó dùng để phân biệt lớp *khi cần*, nhưng U-Net chỉ cần phân biệt "phổi/không phổi" nhị phân. Vì vậy gộp mọi giá trị `> 0` thành `1`.

**Vì sao `.unsqueeze(0)` trên mask?** U-Net output có shape `(N, 1, H, W)`. Loss BCE cần target cùng shape. Sau `ToTensorV2()`, mask từ Albumentations có shape `(H, W)` — thêm chiều kênh để thành `(1, H, W)`; khi `DataLoader` gộp batch sẽ ra `(N, 1, H, W)`, đúng shape để so với output model.

**Class weights cho dataset mất cân bằng.** Sau `preprocess.py`, mỗi lớp có tối đa 3000 ảnh — khá cân bằng, có thể bỏ qua. Nếu muốn chặt chẽ hơn (ví dụ một lớp thực tế có ít ảnh hơn giới hạn):

```python
from collections import Counter

def compute_class_weights(dataset):
    labels = [dataset[i][1] for i in range(len(dataset))]  # chậm nếu dataset lớn — nên cache lại
    cnt = Counter(labels)
    total = sum(cnt.values())
    weights = [total / (NUM_CLASSES * cnt[i]) for i in range(NUM_CLASSES)]
    return torch.tensor(weights, dtype=torch.float32)

# Dùng: criterion = nn.CrossEntropyLoss(weight=weights.to(device))
```

### 5.5. Gotchas

- Thứ tự `Normalize` → `ToTensorV2` **không được đảo** — `Normalize` cần numpy array đầu vào, `ToTensorV2` mới chuyển sang tensor.
- Không cache toàn bộ ảnh vào RAM trong `__init__` (dataset ~9k ảnh × 224² × 3 kênh ≈ 1.3GB, vẫn chấp nhận được ở quy mô này nhưng sẽ vỡ RAM nếu scale lên).
- `num_workers` trên Windows: nếu `Dataset` chứa lambda hoặc closure không pickle được sẽ crash khi `num_workers > 0`. Đặt `num_workers=0` lúc debug, tăng lên sau khi mọi thứ đã chạy ổn.
- `convert("RGB")` trên ảnh xám sẽ **lặp kênh** L → R,G,B (không phải áp colormap giả) — đúng hành vi mong muốn.
- Cache danh sách đường dẫn file trong `__init__`, **không** glob lại mỗi lần gọi `__getitem__` (rất chậm).

### 5.6. Cách tự test

```python
# Test nhanh trong REPL / notebook
from src.dataset import (
    ChestXrayClassificationDataset, ChestXraySegmentationDataset,
    get_train_transforms, get_train_transforms_seg,
    CLASS_TO_IDX, IDX_TO_CLASS,
)

# 1. Classification
ds = ChestXrayClassificationDataset("data/split/train", get_train_transforms())
print("Size:", len(ds))
img, label = ds[0]
print("Image shape:", img.shape, "dtype:", img.dtype)  # torch.Size([3, 224, 224]) float32
print("Label:", label, "=", IDX_TO_CLASS[label])
print("Pixel range:", img.min().item(), img.max().item())  # ~[-2.1, 2.6] sau Normalize

# 2. Kiểm tra phân phối nhãn
from collections import Counter
labels = [ds[i][1] for i in range(200)]  # sample 200
print(Counter(labels))  # phải có đủ cả 3 lớp

# 3. Segmentation
seg_ds = ChestXraySegmentationDataset("data/split/train", get_train_transforms_seg())
img, mask = seg_ds[0]
print("Image:", img.shape, "Mask:", mask.shape, "unique:", mask.unique().tolist())
# Mask shape phải là (1, 224, 224), unique phải là [0.0, 1.0]
```

Nếu cả 3 khối trên chạy không lỗi và giá trị in ra đúng như comment, coi như `dataset.py` đã sẵn sàng cho Giai đoạn 3.

---

## 6. Giai đoạn 3 — `src/model.py`: EfficientNet-B3 classifier

### 6.1. Mục đích & API contract

File duy nhất chịu trách nhiệm build classifier. Chốt signature: **`build_classifier(num_classes: int = 3, pretrained: bool = True) -> nn.Module`**. Signature này sẽ được `api/inference.py` gọi lại để load trọng số — không đổi tên/tham số sau khi đã dùng ở nơi khác.

### 6.2. Kiến trúc chi tiết cần biết để debug

`torchvision.models.efficientnet_b3` có 3 thuộc tính chính: `features` (backbone convolution), `avgpool` (global average pooling), `classifier` (một `Sequential` gồm `Dropout` + `Linear`). Số feature đầu vào của lớp Linear cuối: `model.classifier[1].in_features = 1536` với B3. Layer mục tiêu cho Grad-CAM (cần biết trước khi làm Giai đoạn 7): `model.features[-1]` — block convolution cuối cùng, ngay trước global average pooling.

### 6.3. Skeleton code

```python
# src/model.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

def build_classifier(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """
    EfficientNet-B3 với head thay bằng Linear(num_classes).
    KHÔNG đổi signature — api/inference.py import hàm này để load weights.
    """
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b3(weights=weights)

    in_features = model.classifier[1].in_features  # 1536
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng toàn bộ features, chỉ train classifier."""
    for p in model.features.parameters():
        p.requires_grad = False


def unfreeze_last_blocks(model: nn.Module, num_blocks: int = 2) -> None:
    """Mở khóa N block cuối của features cho pha 2."""
    total_blocks = len(model.features)
    for i, block in enumerate(model.features):
        for p in block.parameters():
            p.requires_grad = (i >= total_blocks - num_blocks)


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

### 6.4. Chi tiết implementation

**Vì sao lưu `state_dict` thay vì cả object model?**

```python
# ĐÚNG
torch.save(model.state_dict(), "weights/best_classifier.pth")

# Load ở backend:
model = build_classifier(num_classes=3, pretrained=False)
model.load_state_dict(torch.load("weights/best_classifier.pth", map_location="cpu"))
model.eval()

# SAI — save cả object, dễ vỡ khi PyTorch version khác hoặc thiếu đúng file định nghĩa class
torch.save(model, "weights/best_classifier.pth")
```

`state_dict` là một `dict` thuần `{"layer.weight": tensor, ...}` — portable, không phụ thuộc file `.py` định nghĩa kiến trúc lúc load. Save cả object dùng pickle nội bộ → khi load bắt buộc phải có đúng class/file y hệt lúc save, dễ vỡ khi đổi máy hoặc đổi version PyTorch.

**`map_location` khi load ở backend.** Nếu backend chạy trên môi trường chỉ có CPU (ví dụ Hugging Face Spaces free tier), mà trọng số được train và save trên GPU rồi load không chỉ định `map_location="cpu"`, sẽ báo lỗi. Quy ước: **luôn** `map_location="cpu"` khi load ở phía serve, sau đó tự `.to(device)` nếu cần.

### 6.5. Gotchas

- Dùng `weights=EfficientNet_B3_Weights.IMAGENET1K_V1` — không dùng chuỗi `"IMAGENET1K_V1"` (deprecated warning trong torchvision mới).
- `EfficientNet_B3_Weights.DEFAULT` có thể trỏ sang phiên bản khác trong tương lai — chỉ định `V1` tường minh để đảm bảo tái lập.
- Sau khi freeze/unfreeze, **phải tạo lại optimizer** — optimizer chỉ cập nhật tham số nó "biết" tại thời điểm khởi tạo. Cách chuẩn: `optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=...)`.
- Nếu val loss ra NaN ngay epoch đầu → gần như chắc chắn LR quá cao khi vừa unfreeze; hoặc quên gọi `.eval()` khiến BatchNorm học sai thống kê từ batch nhỏ.

### 6.6. Cách tự test

```python
from src.model import build_classifier, count_trainable_params, freeze_backbone
from torchinfo import summary
import torch

model = build_classifier(num_classes=3)
print(count_trainable_params(model))  # ~10.7M tham số
summary(model, input_size=(1, 3, 224, 224))

# Forward pass giả
x = torch.randn(2, 3, 224, 224)
out = model(x)
print(out.shape)          # torch.Size([2, 3])
print(out.softmax(1))     # phải sum theo dim 1 = 1

# Freeze
freeze_backbone(model)
print("After freeze:", count_trainable_params(model))  # ~3.8K (chỉ head)
```

---

## 7. Giai đoạn 4 — `src/unet.py`: U-Net segmentation

### 7.1. Mục đích & API contract

Signature: **`build_unet(in_channels: int = 3, out_channels: int = 1, pretrained: bool = True) -> nn.Module`**. Output: logits shape `(N, 1, H, W)` — **chưa** qua sigmoid (để dùng `BCEWithLogitsLoss`, ổn định số học hơn áp sigmoid rồi mới tính BCE thủ công).

### 7.2. Kiến thức nền tảng cần nắm

**ConvTranspose vs Upsample + Conv trong decoder.** `ConvTranspose` (deconvolution) có tham số học được nhưng dễ gây "checkerboard artifact" (vệt caro) nếu stride không chia hết kernel size. `Upsample` (nearest/bilinear) + `Conv` không có artifact này, ít tham số hơn, chất lượng tương đương trong hầu hết trường hợp — `segmentation-models-pytorch` mặc định dùng cách này.

**Vì sao dùng thư viện `segmentation-models-pytorch` (SMP) thay vì code tay U-Net?** Tiết kiệm đáng kể thời gian code + debug so với tự viết encoder/decoder/skip connection. Quan trọng hơn: SMP cho phép dùng **encoder pretrained trên ImageNet** (ResNet, EfficientNet, MobileNet...) — với dataset chỉ ~9k ảnh, encoder pretrained giúp hội tụ nhanh hơn và tổng quát tốt hơn nhiều so với train from scratch. Đánh đổi: trong báo cáo phải ghi rõ "sử dụng `segmentation_models_pytorch` với encoder ResNet-34", không thể nhận là tự cài đặt U-Net từ đầu — chấp nhận được vì đề tài không yêu cầu re-implement kiến trúc.

**Loss cho segmentation nhị phân** — đã giải thích khái niệm ở Phần 2.4; ở đây chốt công thức triển khai: `0.5·BCE + 0.5·Dice` làm mặc định. **Focal Loss** chỉ cần cân nhắc khi lớp mục tiêu cực hiếm (<5% pixel) — lung mask thường chiếm 30–40% diện tích ảnh nên không cần.

### 7.3. Skeleton code

```python
# src/unet.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def build_unet(
    in_channels: int = 3,
    out_channels: int = 1,
    pretrained: bool = True,
    encoder_name: str = "resnet34",
) -> nn.Module:
    """
    U-Net với encoder ResNet-34 pretrained ImageNet.
    Output: logits shape (N, out_channels, H, W) — CHƯA sigmoid.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=out_channels,
    )
    return model


# ---- Loss kết hợp BCE + Dice ----
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        p = probs.contiguous().view(probs.size(0), -1)
        t = target.contiguous().view(target.size(0), -1)
        inter = (p * t).sum(1)
        dice = (2 * inter + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)
        dice_loss = 1 - dice.mean()
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


# ---- Metric ----
@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) > thresh).float()
    inter = (pred * target).sum()
    return (2 * inter + 1) / (pred.sum() + target.sum() + 1)

@torch.no_grad()
def iou_score(logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) > thresh).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + 1) / (union + 1)
```

### 7.4. Chi tiết implementation

**Vì sao `in_channels=3` chứ không phải `1`?** Encoder ResNet-34 pretrained ImageNet kỳ vọng input 3 kênh (RGB). Nếu đặt `in_channels=1`, SMP sẽ tự khởi tạo lại lớp conv đầu tiên một cách ngẫu nhiên (không dùng được pretrained cho layer đó) → mất lợi thế pretrained ngay từ lớp đầu. Cách xử lý sạch nhất: giữ 3 kênh, ảnh xám đã được convert sang RGB (lặp kênh) ngay tại `Dataset` (Phần 5).

**Số hạng làm mượt (`+1`) trong Dice/IoU.** Nếu một ảnh trong batch không có phổi (mask toàn 0) và model cũng dự đoán toàn 0, phép chia sẽ ra `0/0 = NaN`. Cộng `smooth=1` vào cả tử và mẫu tránh chia cho 0, đồng thời làm gradient mượt hơn khi giá trị overlap gần 0.

### 7.5. Gotchas

- `BCEWithLogitsLoss` kỳ vọng **logits** (chưa sigmoid), không phải xác suất — nếu lỡ áp sigmoid trước rồi mới truyền vào, loss sẽ sai (tương đương double sigmoid).
- Mask input phải là kiểu `float`, không phải `long` — để `long` sẽ báo lỗi kiểu "result type Float can't be cast to Long".
- Output của SMP có shape `(N, classes, H, W)` — với segmentation nhị phân (`classes=1`), chú ý chiều này khi tính metric để không nhầm với `N`.
- `encoder_name` khác nhau tạo ra kiến trúc khác nhau — nếu đã train xong với `resnet34` mà load trọng số vào `build_unet(encoder_name="efficientnet-b0")` sẽ báo lỗi shape mismatch. Ghi rõ encoder đã dùng vào tên file hoặc metadata khi lưu.

### 7.6. Cách tự test

```python
from src.unet import build_unet, BCEDiceLoss, dice_score, iou_score
import torch

model = build_unet()
x = torch.randn(2, 3, 224, 224)
y = torch.randint(0, 2, (2, 1, 224, 224)).float()

logits = model(x)
print(logits.shape)  # torch.Size([2, 1, 224, 224])

loss_fn = BCEDiceLoss()
loss = loss_fn(logits, y)
print("Loss:", loss.item())  # nên > 0, < 2
print("Dice:", dice_score(logits, y).item())
print("IoU:", iou_score(logits, y).item())
```

---

## 8. Giai đoạn 5 — `notebooks/train_classifier.ipynb`

### 8.1. Kiến thức nền tảng cần nắm

**Training loop chuẩn của PyTorch** — cấu trúc mà mọi vòng lặp train/eval trong dự án đều tuân theo:

```python
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()   # xóa gradient cũ
        logits = model(x)       # forward
        loss = criterion(logits, y)
        loss.backward()         # backward
        optimizer.step()        # cập nhật tham số

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            ...  # tính val loss + metric

    scheduler.step()  # cập nhật LR sau mỗi epoch
```

**Mixed precision (khuyến nghị bật).** Dùng `torch.cuda.amp.autocast` + `GradScaler` — giảm VRAM sử dụng 30–50%, tăng tốc 1.5–2 lần trên GPU có Tensor Core (T4, V100, A100, dòng RTX 20/30/40). **Không** mang lại lợi ích trên GPU đời cũ không có Tensor Core (K80, dòng GTX 10xx).

**Early stopping.** Theo dõi **val Macro F1**, không phải val loss — loss thấp không đảm bảo F1 cao khi dữ liệu hơi mất cân bằng (loss có thể thấp vì model chỉ dự đoán tốt lớp đa số). Patience 5–7 epoch: nếu qua từng đó epoch liên tiếp F1 không cải thiện, dừng train và giữ lại checkpoint tốt nhất đã lưu.

**Lựa chọn LR scheduler.** `CosineAnnealingLR` — LR giảm mượt theo đường cosine, không cần tune nhiều tham số, phù hợp cho fine-tuning; đây là lựa chọn mặc định khuyến nghị (`T_max = num_epochs`). `ReduceLROnPlateau` giảm LR khi metric ngừng cải thiện, cần tune `factor` và `patience`. `OneCycleLR` hội tụ nhanh nhưng phức tạp hơn để tune — chỉ nên thử sau khi đã có baseline ổn với Cosine.

### 8.2. Cấu trúc notebook đề xuất

**Cell 1 — Imports + seed**

```python
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score
from tqdm import tqdm
from pathlib import Path

from src.dataset import (
    ChestXrayClassificationDataset,
    get_train_transforms, get_val_transforms,
    NUM_CLASSES, IDX_TO_CLASS,
)
from src.model import (
    build_classifier, freeze_backbone,
    unfreeze_last_blocks, unfreeze_all,
    count_trainable_params,
)

# set_seed(42)  # đã định nghĩa ở Phần 3.3, dán vào cell trước cell này
```

**Cell 2 — Config**

```python
SPLIT_DIR = "data/split"
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "weights/best_classifier.pth"
Path("weights").mkdir(exist_ok=True)

# LR theo 3 pha
LR_HEAD_ONLY = 1e-3
LR_LAST_BLOCKS = 1e-4
LR_ALL = 1e-5

EPOCHS_P1 = 3   # warm-up head
EPOCHS_P2 = 15  # fine-tune block cuối
EPOCHS_P3 = 5   # full fine-tune (tùy chọn)
PATIENCE = 5
```

**Cell 3 — DataLoader**

```python
train_ds = ChestXrayClassificationDataset(f"{SPLIT_DIR}/train", get_train_transforms())
val_ds = ChestXrayClassificationDataset(f"{SPLIT_DIR}/val", get_val_transforms())

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
```

**Cell 4 — Model + loss + hàm chạy 1 epoch**

```python
model = build_classifier(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
criterion = nn.CrossEntropyLoss()  # thêm weight=... nếu dữ liệu mất cân bằng
scaler = GradScaler()

def run_epoch(loader, train: bool, optimizer=None):
    model.train() if train else model.eval()
    losses, ys, ps = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train:
                optimizer.zero_grad()
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            losses.append(loss.item())
            ys.extend(y.cpu().tolist())
            ps.extend(logits.argmax(1).cpu().tolist())
    macro_f1 = f1_score(ys, ps, average="macro")
    return np.mean(losses), macro_f1
```

**Cell 5 — Vòng lặp train 3 pha với early stopping**

```python
def train_phase(phase_name, epochs, lr, best_f1):
    print(f"\n=== Phase: {phase_name} | LR={lr} | trainable={count_trainable_params(model)} ===")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    patience_ctr = 0
    for ep in range(epochs):
        tr_loss, tr_f1 = run_epoch(train_loader, train=True, optimizer=optimizer)
        va_loss, va_f1 = run_epoch(val_loader, train=False)
        scheduler.step()
        print(f"Ep {ep+1:02d} train_loss={tr_loss:.4f} tr_f1={tr_f1:.4f} "
              f"val_loss={va_loss:.4f} val_f1={va_f1:.4f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  Saved best_f1={best_f1:.4f}")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stop at epoch {ep+1}")
                break
    return best_f1

best_f1 = 0.0
# Pha 1
freeze_backbone(model)
best_f1 = train_phase("head-only", EPOCHS_P1, LR_HEAD_ONLY, best_f1)

# Pha 2
unfreeze_last_blocks(model, num_blocks=2)
best_f1 = train_phase("last-2-blocks", EPOCHS_P2, LR_LAST_BLOCKS, best_f1)

# Pha 3 (tùy chọn)
unfreeze_all(model)
best_f1 = train_phase("all", EPOCHS_P3, LR_ALL, best_f1)

print(f"\nBEST VAL MACRO F1: {best_f1:.4f}")
```

**Cell 6 — Vẽ biểu đồ loss/F1 + confusion matrix.** Hai hình cần lưu vào `figures/` để đưa vào báo cáo: (1) loss/F1 theo epoch, train và val chồng lên nhau trên cùng một trục; (2) confusion matrix trên tập val với model tốt nhất, vẽ bằng `seaborn.heatmap`.

### 8.3. Gotchas

- Phải **tạo lại optimizer** sau mỗi lần freeze/unfreeze (đã nhắc ở Phần 6.5).
- `GradScaler` đôi khi báo "Inf detected in gradient" nếu LR quá nhỏ khiến việc scale loss không ổn định — có thể tự động skip step; nếu skip liên tục nhiều epoch, nên tắt AMP để kiểm tra.
- Checkpoint hiện tại chỉ lưu `state_dict` của model — nếu muốn resume train giữa chừng, phải lưu thêm `optimizer.state_dict()`, `scheduler.state_dict()`, `epoch`, `best_f1` vào một `dict` rồi save.
- `train_loader` dùng `shuffle=True` nhưng `val_loader` dùng `shuffle=False` — val phải deterministic để so sánh công bằng giữa các epoch.

---

## 9. Giai đoạn 6 — `notebooks/train_unet.ipynb`

Cấu trúc gần như giống hệt `train_classifier.ipynb` (Phần 8), với các khác biệt sau:

- Dataset: `ChestXraySegmentationDataset` thay vì `ChestXrayClassificationDataset`, dùng `get_train_transforms_seg()` / `get_val_transforms()` tương ứng.
- Model: `build_unet()` thay `build_classifier()`.
- Loss: `BCEDiceLoss` thay `CrossEntropyLoss`.
- Metric theo dõi: Dice + IoU, không phải F1.
- **Không cần chia 3 pha** — encoder của U-Net đã pretrained, có thể unfreeze toàn bộ ngay từ đầu với LR vừa phải (1e-4), vì phần decoder luôn được train from scratch nên không có nguy cơ "catastrophic forgetting" như classifier.

### 9.1. Config đề xuất

```python
BATCH_SIZE = 16   # U-Net tốn VRAM hơn classifier vì có thêm decoder
EPOCHS = 25
LR = 1e-4
CKPT_PATH = "weights/best_unet.pth"
```

### 9.2. Vòng lặp — tính Dice + IoU thay vì F1

```python
dice_list, iou_list = [], []
for x, y in loader:
    x, y = x.to(DEVICE), y.to(DEVICE)
    with autocast():
        logits = model(x)
        loss = loss_fn(logits, y)
    if train:
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    dice_list.append(dice_score(logits, y).item())
    iou_list.append(iou_score(logits, y).item())
mean_dice = np.mean(dice_list); mean_iou = np.mean(iou_list)
```

### 9.3. Chỉ tiêu chấp nhận được

Segmentation phổi là bài toán tương đối "dễ" so với phân loại bệnh lý (phổi có độ tương phản mạnh với xương sườn và mô mềm xung quanh trên ảnh X-quang). Mục tiêu: **Dice > 0.90** trên tập val. Nếu Dice < 0.85 sau 15 epoch, kiểm tra theo thứ tự: augmentation có đang quá mạnh không, LR có đặt sai không, và **mask có bị nhị phân hóa sai không** (kiểm tra lại `np.unique` trên mask, phải chỉ còn `{0.0, 1.0}`).

### 9.4. Sanity check cuối cùng

Sau khi train xong, vẽ 5 ảnh mẫu dạng `[ảnh gốc | mask ground-truth | mask dự đoán]` cạnh nhau. Nếu mask dự đoán bao đúng vùng phổi (không tràn ra ngoài lồng ngực, không bị thủng lỗ giữa phổi), coi như **pass** và có thể chuyển sang Giai đoạn 7.

---

## 10. Giai đoạn 7 — `src/gradcam.py`: giải thích bằng Grad-CAM

### 10.1. Mục đích & API contract

Signature: **`generate_gradcam(model, img_tensor, target_class=None) -> np.ndarray`**. Quy ước kiểu trả về:

- `numpy.ndarray` shape `(H, W) = (224, 224)`.
- dtype: `float32`.
- Giá trị trong khoảng `[0.0, 1.0]` (đã chuẩn hóa).
- **Không phải** ảnh RGB — chỉ là heatmap 1 kênh; backend/frontend tự áp colormap (JET hoặc VIRIDIS) để overlay lên ảnh gốc (xem Phần 12).

### 10.2. Grad-CAM vs các biến thể

- **Grad-CAM** (2017): baseline, được cite rộng rãi nhất — dùng trong dự án này.
- **Grad-CAM++** (2018): xử lý tốt hơn trường hợp nhiều vùng cùng đóng góp cho một lớp (multi-instance).
- **Score-CAM** (2019): không cần gradient, ổn định hơn nhưng chậm hơn nhiều.
- Với quy mô đồ án, Grad-CAM chuẩn là đủ. Có thể thử Grad-CAM++ như một "extended experiment" trong báo cáo nếu còn thời gian.

Thư viện dùng: [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) (`pip install grad-cam`). Class chính: `GradCAM`, `GradCAMPlusPlus`, `ScoreCAM`... API cơ bản: `cam = GradCAM(model=model, target_layers=[target_layer]); grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(class_idx)])`.

### 10.3. Skeleton code

```python
# src/gradcam.py
from typing import Optional
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def _get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    EfficientNet-B3: block conv cuối cùng trước global average pooling.
    Nếu đổi kiến trúc classifier, phải cập nhật lại hàm này.
    """
    return model.features[-1]


def generate_gradcam(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """
    Args:
        model: classifier đã load weights, ở eval mode.
        img_tensor: (3, H, W) hoặc (1, 3, H, W), đã normalize.
        target_class: index lớp muốn giải thích. None = lớp model dự đoán (argmax).

    Returns:
        heatmap: np.ndarray shape (H, W), float32, range [0, 1].
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    model.eval()  # KHÔNG dùng torch.no_grad() — Grad-CAM cần gradient

    if target_class is None:
        with torch.no_grad():
            logits = model(img_tensor)
            target_class = int(logits.argmax(dim=1).item())

    target_layer = _get_target_layer(model)
    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        # shape: (batch=1, H, W), float32, [0, 1]

    return grayscale_cam[0]  # (H, W)
```

### 10.4. Chi tiết implementation

**Vì sao có `model.eval()` nhưng KHÔNG có `torch.no_grad()`?** Đây là bẫy kinh điển nhất của phần này. `model.eval()` bật chế độ eval → BatchNorm dùng thống kê đã lưu (không cập nhật theo batch hiện tại), Dropout không zero-out ngẫu nhiên nữa — cần thiết để dự đoán ổn định, có thể tái lập. `torch.no_grad()` thì **tắt hẳn autograd**, không xây dựng computation graph — nhưng Grad-CAM cần gradient của điểm số theo activation map, bắt buộc phải có graph, nên **không được** dùng `no_grad()` ở đây. Người mới hay copy nguyên khối "eval + no_grad" từ code inference thông thường sang Grad-CAM, kết quả là heatmap ra toàn số 0.

**Overlay heatmap lên ảnh gốc** (việc này làm ở `api/inference.py`, không phải trong `gradcam.py` — file này chỉ trả heatmap thô):

```python
import cv2
import numpy as np

def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
    """image_rgb: (H, W, 3) uint8. heatmap: (H, W) float32 [0,1]."""
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return (alpha * heatmap_color + (1 - alpha) * image_rgb).astype(np.uint8)
```

### 10.5. Gotchas

- **Bug #1** (đã cảnh báo ở trên): quên bỏ `no_grad()` → heatmap toàn 0.
- **Bug #2:** báo lỗi "output has no grad_fn" — thường do model đang bị bọc trong `DataParallel`, hoặc `target_layer` không nằm trong computation graph thực sự được dùng ở forward pass. Luôn test trên model gốc (chưa wrap) trước.
- **Bug #3:** truyền `target_class` là chuỗi (`"COVID"`) thay vì số nguyên (`2`) — `ClassifierOutputTarget` yêu cầu `int`.
- **Bug #4:** heatmap thô có kích thước nhỏ hơn ảnh gốc (ví dụ 7×7 với input 224 qua EfficientNet-B3) và không tự resize — `pytorch-grad-cam` đã upsample sẵn cho bạn trong ví dụ trên, nhưng nếu tự viết logic khác phải nhớ `cv2.resize` về đúng kích thước ảnh gốc.
- Version của `pytorch-grad-cam` đôi khi đổi signature giữa các bản (ví dụ 1.4 và 1.5 khác nhau) — pin `>=1.5.0` trong `requirements-model.txt` (đã có sẵn trong repo).

### 10.6. Cách tự test

```python
from src.model import build_classifier
from src.dataset import ChestXrayClassificationDataset, get_val_transforms, IDX_TO_CLASS
from src.gradcam import generate_gradcam
import matplotlib.pyplot as plt
import torch

model = build_classifier(num_classes=3, pretrained=False)
model.load_state_dict(torch.load("weights/best_classifier.pth", map_location="cpu"))
model.eval()

ds = ChestXrayClassificationDataset("data/split/test", get_val_transforms())
img, label = ds[0]

heatmap = generate_gradcam(model, img)
print(heatmap.shape, heatmap.dtype, heatmap.min(), heatmap.max())
# (224, 224) float32, min gần 0, max gần 1 — nếu min=max=0 nghĩa là quên bỏ no_grad

plt.subplot(121); plt.imshow(img.permute(1,2,0).numpy()); plt.title(f"GT: {IDX_TO_CLASS[label]}")
plt.subplot(122); plt.imshow(heatmap, cmap='jet'); plt.title("Grad-CAM")
plt.show()
```

---

## 11. Giai đoạn 8 — `src/shortcut_iou.py`: kiểm định shortcut learning

### 11.1. Mục đích

Đây là điểm cộng nghiên cứu (research contribution) của báo cáo, đã giải thích khái niệm ở Phần 2.6. Câu hỏi cần trả lời: model có thực sự nhìn vào vùng phổi (và lý tưởng là vùng tổn thương) để chẩn đoán, hay đang bám vào watermark/chữ/khung ảnh — một "đường tắt" tình cờ tương quan với nhãn trong tập train?

Nếu Grad-CAM tập trung **đúng** vào vùng phổi → IoU với lung mask **cao** → model học đúng. Ngược lại, IoU thấp là cảnh báo shortcut.

Tài liệu bắt buộc đọc trước khi viết phần phân tích trong báo cáo: Geirhos et al., *"Shortcut Learning in Deep Neural Networks"* (Nature Machine Intelligence, 2020) — khung lý thuyết tổng quát; và Zech et al., *"Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs"* (PLoS Medicine, 2018) — case study cụ thể trên chính domain X-quang ngực, rất phù hợp trích dẫn ở phần Motivation.

### 11.2. So sánh hai nguồn mask & công thức IoU

- **Ground-truth lung mask:** có sẵn trong dataset gốc (do bên tạo dataset gán nhãn). Chính xác nhất, nhưng **không tồn tại** với ảnh mới khi deploy thật.
- **U-Net predicted mask:** tự dự đoán, dùng được cho ảnh bất kỳ. Kém chính xác hơn ground-truth khoảng 5–10% Dice (Phần 9.3).
- So sánh IoU với cả hai nguồn cho phép: (1) kiểm chứng U-Net đủ tốt để thay thế ground-truth, (2) mô phỏng đúng điều kiện thực tế khi deploy (không có ground-truth mask).

`IoU(A, B) = |A ∩ B| / |A ∪ B|`. Với heatmap Grad-CAM, cần **nhị phân hóa trước** bằng một ngưỡng (thường 0.3–0.5). Lung mask đã sẵn nhị phân. Nên chạy sweep nhiều ngưỡng `[0.3, 0.4, 0.5, 0.6, 0.7]` thay vì chỉ báo cáo một con số duy nhất, để chứng minh kết luận không nhạy cảm với lựa chọn ngưỡng.

### 11.3. Skeleton code

```python
# src/shortcut_iou.py
from pathlib import Path
from typing import Literal
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from src.model import build_classifier
from src.unet import build_unet
from src.dataset import (
    ChestXrayClassificationDataset, get_val_transforms,
    IDX_TO_CLASS, CLASS_TO_IDX,
)
from src.gradcam import generate_gradcam


def binarize(x: np.ndarray, thresh: float) -> np.ndarray:
    return (x > thresh).astype(np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union > 0 else 0.0


def load_gt_mask(image_path: Path, mask_dir: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_dir / image_path.name).convert("L"))
    return (mask > 0).astype(np.uint8)


@torch.no_grad()
def predict_lung_mask(unet: torch.nn.Module, img_tensor: torch.Tensor) -> np.ndarray:
    """Trả về mask nhị phân (H, W) từ U-Net."""
    device = next(unet.parameters()).device
    x = img_tensor.unsqueeze(0).to(device)
    logits = unet(x)
    mask = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
    return mask


def run_shortcut_analysis(
    classifier_path: str,
    unet_path: str,
    test_split_dir: str,
    mask_source: Literal["gt", "unet"] = "gt",
    gradcam_thresh: float = 0.5,
    device: str = "cuda",
):
    # 1. Load model
    clf = build_classifier(num_classes=3, pretrained=False)
    clf.load_state_dict(torch.load(classifier_path, map_location=device))
    clf.to(device).eval()

    unet = build_unet(pretrained=False)
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet.to(device).eval()

    # 2. Test loader
    ds = ChestXrayClassificationDataset(test_split_dir, get_val_transforms())
    mask_dir = Path(test_split_dir) / "masks"

    ious_per_class = {c: [] for c in CLASS_TO_IDX}

    for i in tqdm(range(len(ds))):
        img, label = ds[i]
        path = ds.image_paths[i]

        # Grad-CAM — dùng target_class=label (ground-truth), không phải dự đoán,
        # để phân tích "khi model đúng thì nó nhìn đâu"
        heatmap = generate_gradcam(clf, img, target_class=label)
        cam_bin = binarize(heatmap, gradcam_thresh)

        if mask_source == "gt":
            lung_mask = load_gt_mask(path, mask_dir)
        else:
            lung_mask = predict_lung_mask(unet, img)

        score = iou(cam_bin, lung_mask)
        ious_per_class[IDX_TO_CLASS[label]].append(score)

    # 3. Báo cáo
    print(f"\n===== Mask source: {mask_source} | thresh={gradcam_thresh} =====")
    for cls, scores in ious_per_class.items():
        arr = np.array(scores)
        print(f"{cls:15s} n={len(arr):4d} mean IoU={arr.mean():.3f} "
              f"median={np.median(arr):.3f} std={arr.std():.3f}")

    # 4. Lưu histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    for cls, scores in ious_per_class.items():
        ax.hist(scores, bins=20, alpha=0.5, label=cls)
    ax.set_xlabel("IoU(Grad-CAM, lung mask)")
    ax.set_ylabel("Count")
    ax.set_title(f"Shortcut analysis — mask={mask_source}, thresh={gradcam_thresh}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"figures/shortcut_iou_{mask_source}_t{gradcam_thresh}.png", dpi=120)
    return ious_per_class


if __name__ == "__main__":
    # Chạy 2 lần cho báo cáo
    run_shortcut_analysis("weights/best_classifier.pth", "weights/best_unet.pth",
                           "data/split/test", mask_source="gt", gradcam_thresh=0.5)
    run_shortcut_analysis("weights/best_classifier.pth", "weights/best_unet.pth",
                           "data/split/test", mask_source="unet", gradcam_thresh=0.5)
```

### 11.4. Cách diễn giải kết quả

| Mean IoU | Diễn giải |
|---|---|
| > 0.5 | Model rất tập trung vào phổi, ít dấu hiệu shortcut — kết quả lý tưởng cho báo cáo |
| 0.3 – 0.5 | Chấp nhận được — Grad-CAM có thể chỉ tập trung vào một vùng **nhỏ** trong phổi (đúng vùng tổn thương thật) chứ không cần phủ toàn bộ diện tích phổi |
| < 0.2 | Dấu hiệu shortcut — model đang nhìn ra ngoài phổi. Nên visualize 10 ảnh có IoU thấp nhất để kiểm tra bằng mắt xem heatmap rơi vào đâu |

Nên so sánh IoU giữa 3 lớp: nếu lớp COVID có IoU thấp hẳn so với hai lớp còn lại, đó có thể là dấu hiệu dữ liệu COVID được gộp từ nhiều nguồn khác nhau (nhiều watermark/chữ hơn) — một giả thuyết đáng đưa vào phần thảo luận của báo cáo.

### 11.5. Gotchas

- Phải chỉ định `target_class=label` (nhãn thật), **không phải** lớp model dự đoán — mục đích là phân tích "khi model đưa ra chẩn đoán đúng, nó nhìn vào đâu", không phải phân tích lỗi.
- Chạy trên toàn bộ test set (~1350 ảnh) có thể mất 5–10 phút vì Grad-CAM cần một lượt backward pass cho mỗi ảnh — dùng `tqdm` để theo dõi tiến độ, và cân nhắc lưu kết quả trung gian nếu chạy nhiều cấu hình threshold.
- Ngưỡng 0.5 chỉ là mặc định — nên chạy sweep `[0.3, 0.4, 0.5, 0.6, 0.7]` và vẽ biểu đồ IoU theo threshold để chứng minh kết luận ổn định, không phụ thuộc vào một lựa chọn ngưỡng tình cờ.

---

## 12. Giai đoạn 9 — `api/`: backend FastAPI

### 12.1. Mục đích & API contract

Expose một endpoint duy nhất theo `pipeline.md`: **`POST /predict`**, nhận một ảnh X-quang, trả về JSON gồm chẩn đoán, % tin cậy, ảnh overlay heatmap, và ghi log vào SQLite. Backend **không train gì cả** — chỉ load trọng số đã có sẵn trong `weights/` và chạy inference.

Chia trách nhiệm trong `api/`:

- **`schemas.py`** — định nghĩa cấu trúc response bằng Pydantic (để FastAPI tự validate + tự sinh docs).
- **`inference.py`** — load model **một lần duy nhất** lúc khởi động server (không load lại mỗi request — rất tốn thời gian), cung cấp một hàm `predict_image()` dùng chung.
- **`db.py`** — ghi/đọc log dự đoán vào file SQLite.
- **`main.py`** — khởi tạo `FastAPI()` app, định nghĩa route `/predict`, gọi sang `inference.py` và `db.py`.

### 12.2. Kiến thức nền tảng cần nắm

**Vì sao cần load model một lần khi khởi động, không phải mỗi request?** Load `state_dict` từ đĩa và khởi tạo kiến trúc EfficientNet-B3 mất khoảng vài trăm ms đến vài giây — nếu làm việc này ở đầu mỗi request, độ trễ (latency) sẽ cộng dồn không cần thiết và có thể gây timeout khi nhiều người dùng cùng lúc. Giải pháp chuẩn: load model ở **module level** hoặc trong sự kiện `startup` của FastAPI, giữ trong biến toàn cục (hoặc trong `app.state`), tái sử dụng cho mọi request.

**Đường bất đồng bộ (async) trong FastAPI.** FastAPI hỗ trợ `async def` cho route handler — hữu ích khi có I/O chờ đợi (đọc file, gọi DB). Tuy nhiên, forward pass của PyTorch là **đồng bộ, chiếm CPU** (blocking) — nếu gọi trực tiếp trong một `async def` mà không cẩn thận, nó sẽ chặn toàn bộ event loop, khiến các request khác phải đợi. Với một demo/đồ án quy mô nhỏ (không có nhiều request đồng thời), dùng route đồng bộ (`def` thường, không `async`) là đơn giản và an toàn nhất — Starlette (nền của FastAPI) tự chạy các route đồng bộ trong một threadpool riêng.

**Pydantic models cho request/response.** Định nghĩa rõ schema của response (tên trường, kiểu dữ liệu) giúp FastAPI tự validate output và tự sinh tài liệu OpenAPI tại `/docs` — bạn có thể mở trình duyệt, vào `http://localhost:8000/docs`, thử gọi API trực tiếp mà không cần viết `curl` hay Postman.

**Trả ảnh overlay heatmap về client như thế nào?** Có hai lựa chọn phổ biến: (1) encode ảnh sang **base64** và nhúng thẳng vào JSON response (đơn giản, phù hợp khi client là web/Gradio hiển thị trực tiếp); (2) trả về một URL riêng để client tải ảnh (phức tạp hơn, cần lưu file tạm, phù hợp khi ảnh lớn hoặc cần cache). Với quy mô đồ án, **base64 trong JSON** là lựa chọn đơn giản và đủ dùng.

### 12.3. Skeleton code

**`api/schemas.py`**

```python
# api/schemas.py
from pydantic import BaseModel

class PredictResponse(BaseModel):
    predicted_class: str          # "Normal" | "Lung_Opacity" | "COVID"
    confidence: float             # xác suất của lớp dự đoán, 0.0–1.0
    probabilities: dict[str, float]  # xác suất từng lớp, dùng để vẽ bar chart ở UI
    heatmap_overlay_base64: str   # ảnh overlay Grad-CAM, PNG encode base64
    disclaimer: str               # khuyến cáo y tế, luôn đính kèm
```

**`api/inference.py`**

```python
# api/inference.py
import base64
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image

from src.dataset import get_val_transforms, IDX_TO_CLASS
from src.model import build_classifier
from src.gradcam import generate_gradcam

DEVICE = "cpu"  # Hugging Face Spaces free tier chỉ có CPU — xem Phần 14

_classifier = None
_transform = get_val_transforms()

def load_models():
    """Gọi một lần lúc server khởi động — xem api/main.py."""
    global _classifier
    _classifier = build_classifier(num_classes=3, pretrained=False)
    _classifier.load_state_dict(
        torch.load("weights/best_classifier.pth", map_location=DEVICE)
    )
    _classifier.to(DEVICE).eval()


def _overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return (alpha * heatmap_color + (1 - alpha) * image_rgb).astype(np.uint8)


def _encode_png_base64(image_rgb: np.ndarray) -> str:
    pil_img = Image.fromarray(image_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict_image(pil_image: Image.Image) -> dict:
    """Hàm dùng chung — api/main.py và có thể cả app.py (nếu gọi trực tiếp, không qua HTTP) đều gọi hàm này."""
    if _classifier is None:
        raise RuntimeError("Model chưa được load — gọi load_models() lúc startup")

    image_np = np.array(pil_image.convert("RGB"))
    image_resized = cv2.resize(image_np, (224, 224))

    img_tensor = _transform(image=image_np)["image"]

    with torch.no_grad():
        logits = _classifier(img_tensor.unsqueeze(0).to(DEVICE))
        probs = logits.softmax(dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_class = IDX_TO_CLASS[pred_idx]

    heatmap = generate_gradcam(_classifier, img_tensor, target_class=pred_idx)
    overlay = _overlay_heatmap(image_resized, heatmap)

    return {
        "predicted_class": pred_class,
        "confidence": float(probs[pred_idx]),
        "probabilities": {IDX_TO_CLASS[i]: float(p) for i, p in enumerate(probs)},
        "heatmap_overlay_base64": _encode_png_base64(overlay),
        "disclaimer": (
            "Kết quả chỉ mang tính tham khảo, KHÔNG thay thế chẩn đoán y khoa "
            "chính thức. Vui lòng tham vấn bác sĩ chuyên khoa."
        ),
    }
```

**`api/db.py`**

```python
# api/db.py
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/predictions.db")

def init_db():
    """Gọi một lần lúc server khởi động."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(predicted_class: str, confidence: float) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (timestamp, predicted_class, confidence) VALUES (?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), predicted_class, confidence),
    )
    conn.commit()
    conn.close()
```

**`api/main.py`**

```python
# api/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from io import BytesIO

from api.schemas import PredictResponse
from api.inference import load_models, predict_image
from api.db import init_db, log_prediction

app = FastAPI(title="Chest X-ray Diagnosis API")

@app.on_event("startup")
def startup():
    load_models()
    init_db()

@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh")

    image_bytes = file.file.read()
    try:
        pil_image = Image.open(BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh")

    result = predict_image(pil_image)
    log_prediction(result["predicted_class"], result["confidence"])
    return result
```

Chạy thử server local: `uvicorn api.main:app --reload --port 8000`, sau đó mở `http://localhost:8000/docs` để thử endpoint qua giao diện Swagger tự sinh.

### 12.4. Chi tiết implementation

**Vì sao `@app.on_event("startup")` thay vì load model ngay ở module level?** Cả hai cách đều hoạt động được. Dùng sự kiện `startup` tường minh hơn về mặt ý định (rõ ràng đây là hành động "khởi tạo trước khi phục vụ request"), và tránh việc model bị load ngay cả khi bạn chỉ `import api.main` để test một phần khác (ví dụ chạy `pytest` trên `schemas.py`) mà không cần model thật.

**Vì sao trả `probabilities` cho cả 3 lớp, không chỉ lớp dự đoán?** Giao diện Gradio (Giai đoạn 10) sẽ vẽ thanh xác suất cho cả 3 lớp — giúp người dùng (bác sĩ) thấy được mức độ "chắc chắn" hay "phân vân" của model, không chỉ một con số duy nhất. Đây cũng là thông tin hữu ích để đưa vào phần đánh giá định tính của báo cáo.

**Disclaimer y tế.** Luôn đính kèm ở mọi response — đây là yêu cầu đạo đức cơ bản với bất kỳ hệ thống AI hỗ trợ chẩn đoán nào, không phải chi tiết tùy chọn.

### 12.5. Gotchas

- Quên xử lý trường hợp `file` không phải ảnh hợp lệ (file rác, PDF đổi đuôi...) → server crash thay vì trả lỗi 400 rõ ràng cho client. Luôn validate `content_type` và bọc `Image.open()` trong `try/except`.
- Load model trên CPU khi deploy (Hugging Face Spaces free tier) nhưng test local trên máy có GPU mà quên đổi `DEVICE` → khi deploy thật sẽ lỗi hoặc chạy rất chậm nếu code có chỗ hardcode `.cuda()`. Luôn dùng biến `DEVICE` thống nhất, đọc từ `torch.cuda.is_available()` hoặc biến môi trường.
- `UploadFile` của FastAPI đọc dữ liệu dạng stream — nếu đọc hai lần (`file.file.read()` gọi 2 lần) lần thứ hai sẽ trả về rỗng vì con trỏ đã ở cuối file. Đọc một lần, lưu vào biến, dùng lại từ biến đó.
- SQLite không xử lý tốt ghi đồng thời (concurrent writes) ở mức độ cao — với quy mô demo/đồ án (không nhiều người dùng cùng lúc) không thành vấn đề, nhưng không phải lựa chọn phù hợp nếu sau này scale lên production thật.

### 12.6. Cách tự test

```bash
# Chạy server
uvicorn api.main:app --reload --port 8000

# Test bằng curl (terminal khác), thay đường dẫn ảnh thật
curl -X POST "http://localhost:8000/predict" \
     -F "file=@data/split/test/images/COVID-1.png"
```

Kỳ vọng: nhận JSON với `predicted_class`, `confidence` trong `[0,1]`, `probabilities` có đủ 3 khóa và tổng ≈ 1.0, `heatmap_overlay_base64` là một chuỗi base64 dài (không rỗng). Kiểm tra thêm: mở `data/predictions.db` bằng bất kỳ công cụ xem SQLite nào (ví dụ extension DB Browser, hoặc `sqlite3` CLI: `sqlite3 data/predictions.db "SELECT * FROM predictions;"`), xác nhận có dòng log mới ứng với lần gọi vừa rồi.

---

## 13. Giai đoạn 10 — `app.py`: giao diện Gradio

### 13.1. Mục đích

Xây một giao diện web đơn giản cho phép người dùng (bác sĩ, hoặc bạn khi demo) upload ảnh X-quang, bấm nút, và xem: nhãn chẩn đoán, % tin cậy từng lớp, ảnh overlay heatmap, và khuyến cáo y tế — không cần biết HTML/CSS/JavaScript.

### 13.2. Kiến thức nền tảng cần nắm

**Mô hình lập trình của Gradio.** Bạn viết một **hàm Python thuần** nhận input, trả output; Gradio tự sinh giao diện tương ứng dựa trên kiểu component bạn khai báo cho input/output (`gr.Image`, `gr.Label`, `gr.Textbox`...). Không cần quản lý state phía client thủ công như một SPA JavaScript thật.

**Gọi sang backend FastAPI hay import trực tiếp `predict_image`?** Có hai cách kiến trúc:

1. **Gọi HTTP sang API** (`requests.post("http://localhost:8000/predict", ...)`) — tách biệt hoàn toàn UI và backend, đúng với kiến trúc mô tả ở Phần 1.3 (Gradio UI → FastAPI backend), cho phép UI và API chạy trên hai tiến trình/container khác nhau, dễ scale riêng từng phần sau này.
2. **Import trực tiếp** `predict_image()` từ `api/inference.py` trong cùng một tiến trình Python — đơn giản hơn để chạy local, không cần khởi động 2 process, nhưng làm mờ ranh giới kiến trúc.

Khuyến nghị cho đồ án: **cách 1** khi deploy thật (đúng kiến trúc client-server đã thiết kế, và là cách cả `main.py` lẫn `app.py` có thể chạy độc lập trong cùng container Docker — xem Phần 14), nhưng khi phát triển local có thể tạm dùng cách 2 để không phải chạy 2 terminal song song.

### 13.3. Skeleton code (gọi qua HTTP — kiến trúc đúng theo thiết kế)

```python
# app.py
import base64
from io import BytesIO

import gradio as gr
import requests
from PIL import Image

API_URL = "http://localhost:8000/predict"


def diagnose(image: Image.Image):
    if image is None:
        return None, {}, "Vui lòng upload ảnh X-quang trước."

    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    response = requests.post(API_URL, files={"file": ("image.png", buf, "image/png")})
    response.raise_for_status()
    result = response.json()

    overlay_bytes = base64.b64decode(result["heatmap_overlay_base64"])
    overlay_image = Image.open(BytesIO(overlay_bytes))

    label_text = f"{result['predicted_class']} ({result['confidence']*100:.1f}%)"
    probs = result["probabilities"]

    return overlay_image, probs, f"{label_text}\n\n{result['disclaimer']}"


with gr.Blocks(title="Chest X-ray Diagnosis") as demo:
    gr.Markdown("# Chest X-ray Segmentation & Diagnosis of Pneumonia and COVID-19")
    gr.Markdown(
        "Công cụ hỗ trợ tham khảo, **không** thay thế chẩn đoán y khoa chính thức."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Ảnh X-quang ngực")
            submit_btn = gr.Button("Chẩn đoán", variant="primary")
        with gr.Column():
            output_overlay = gr.Image(label="Grad-CAM overlay")
            output_probs = gr.Label(label="Xác suất từng lớp", num_top_classes=3)
            output_text = gr.Textbox(label="Kết luận", lines=3)

    submit_btn.click(
        fn=diagnose,
        inputs=[input_image],
        outputs=[output_overlay, output_probs, output_text],
    )

if __name__ == "__main__":
    demo.launch()
```

### 13.4. Gotchas

- `gr.Label` kỳ vọng `dict[str, float]` với giá trị là xác suất — đúng định dạng `probabilities` mà `PredictResponse` (Phần 12.3) đã trả về, không cần convert thêm.
- Nếu chạy `app.py` và `api/main.py` trên hai container/tiến trình khác nhau khi deploy thật, `API_URL` **không thể** hardcode `localhost` — phải đọc từ biến môi trường (ví dụ `API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")`), vì trong Docker Compose hay Hugging Face Spaces, tên service khác `localhost`.
- `image.save(buf, format="PNG")` yêu cầu ảnh input ở chế độ tương thích PNG — nếu người dùng upload ảnh có kênh alpha lạ hoặc mode `CMYK`, nên `image.convert("RGB")` trước khi save để tránh lỗi ở phía server.
- Quên `buf.seek(0)` sau khi `save()` → gửi request với file rỗng (con trỏ đang ở cuối buffer).

### 13.5. Cách tự test

Chạy backend ở một terminal (`uvicorn api.main:app --port 8000`), chạy UI ở terminal khác (`python app.py`), Gradio sẽ in ra một URL local (thường `http://127.0.0.1:7860`). Mở trình duyệt, upload một ảnh bất kỳ từ `data/split/test/images/`, bấm "Chẩn đoán", xác nhận: ảnh overlay hiển thị đúng, thanh xác suất hợp lý (tổng ≈ 100%), và không có lỗi nào in ra ở cả hai terminal.

---

## 14. Giai đoạn 11 — `Dockerfile` & triển khai lên Hugging Face Spaces

### 14.1. Mục đích

Đóng gói backend + UI cùng mọi dependency vào một container, đảm bảo chạy giống hệt nhau trên máy bạn và trên server Hugging Face Spaces — không còn tình trạng "chạy được trên máy tôi".

### 14.2. Kiến thức nền tảng cần nắm

**Vì sao chỉ cần CPU khi deploy?** Việc train (tốn tài nguyên GPU nặng) đã hoàn tất ở Giai đoạn 5–6, trước khi deploy. Lúc deploy, hệ thống chỉ làm **inference** (một forward pass mỗi request) — nhẹ hơn train hàng trăm/nghìn lần, chạy được chấp nhận được trên CPU của free tier Hugging Face Spaces (thường vài giây/ảnh, đủ cho demo).

**Base image nên chọn.** Dùng `python:3.10-slim` (nhẹ) thay vì base image đầy đủ CUDA (nặng, không cần thiết khi chỉ chạy CPU inference) — giảm đáng kể thời gian build và dung lượng image.

**Chạy cả FastAPI lẫn Gradio trong cùng một container.** Hugging Face Spaces (loại "Docker Space") chỉ expose **một port** ra ngoài. Có hai cách giải quyết: (1) chạy Gradio làm entrypoint chính (expose port 7860 — port mặc định Spaces mong đợi), và cho `app.py` **import trực tiếp** `predict_image()` thay vì gọi HTTP sang một service `uvicorn` riêng chạy nội bộ container (đơn giản hơn, tránh phải quản lý 2 tiến trình trong 1 container); hoặc (2) dùng một script khởi động chạy cả `uvicorn` (nội bộ, không expose ra ngoài) lẫn `gradio` cùng lúc, Gradio gọi sang `localhost:8000` như lúc dev. Với quy mô đồ án, **cách 1** đơn giản và đủ dùng — chấp nhận đánh đổi là kiến trúc "tách UI/API" chỉ còn ý nghĩa khi phát triển local, không giữ nguyên 100% khi deploy.

### 14.3. Skeleton Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Cài dependency hệ thống cần cho opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY app.py .
COPY weights/ weights/

EXPOSE 7860

CMD ["python", "app.py"]
```

### 14.4. Chi tiết implementation

**Vì sao cài `libgl1`, `libglib2.0-0`?** `opencv-python-headless` (dùng để `cv2.applyColorMap` trong Phần 12) vẫn phụ thuộc một vài thư viện hệ thống ở tầng OS mà image `python:3.10-slim` không có sẵn — thiếu chúng sẽ báo lỗi `ImportError: libGL.so.1: cannot open shared object file` khi `import cv2`. Đây là lỗi rất phổ biến khi Docker hóa dự án có OpenCV, nên liệt kê trước ở đây thay vì để bạn tự mò.

**Thứ tự COPY trong Dockerfile.** `COPY requirements.txt` và `RUN pip install` được đặt **trước** khi copy code (`src/`, `api/`...) — tận dụng cơ chế cache layer của Docker: nếu bạn chỉ sửa code mà không đổi dependency, bước `pip install` (chậm) sẽ được lấy từ cache thay vì chạy lại từ đầu.

**Vì sao `weights/` cần được copy vào image?** Vì đây không phải dữ liệu người dùng tạo ra lúc runtime — nó là artifact cố định cần có sẵn để container hoạt động. Lưu ý: file `.pth` có thể khá lớn (vài chục MB) — Hugging Face Spaces hỗ trợ Git LFS cho file lớn trong repo, cần cấu hình nếu trọng số vượt giới hạn kích thước file thường của Git.

### 14.5. Các bước triển khai lên Hugging Face Spaces

1. Tạo một Space mới trên [huggingface.co/spaces](https://huggingface.co/spaces), chọn SDK là **Docker**.
2. Trong Space, cấu hình Git LFS cho các file `.pth` nếu chúng vượt quá giới hạn kích thước file thông thường của Git (Hugging Face Spaces hỗ trợ sẵn LFS, chỉ cần đánh dấu đúng loại file).
3. Push code (bao gồm `Dockerfile`, `weights/`, `requirements.txt`, `src/`, `api/`, `app.py`) lên remote của Space (Space hoạt động như một Git repo).
4. Hugging Face tự động build Docker image từ `Dockerfile` và chạy container — theo dõi log build trực tiếp trên giao diện web của Space để phát hiện lỗi (thường là thiếu dependency hệ thống, xem gotchas bên dưới).
5. Khi build xong, Space cấp một URL public — đây chính là bản demo bạn có thể chia sẻ trong báo cáo/thuyết trình.

### 14.6. Gotchas

- Quên `EXPOSE 7860` hoặc set sai port so với `demo.launch()` mặc định (`gr.Blocks.launch()` mặc định dùng `7860`) → Hugging Face Spaces không nhận diện được service đang chạy, Space báo "unhealthy" dù container thực sự chạy được.
- `demo.launch()` mặc định chỉ bind `127.0.0.1` trong một số phiên bản Gradio — bên trong Docker container cần bind `0.0.0.0` để traffic từ bên ngoài container tới được. Thêm tường minh: `demo.launch(server_name="0.0.0.0", server_port=7860)`.
- Build Docker image chạy được ở local (`docker build` + `docker run` trên máy bạn) là bước bắt buộc phải làm **trước khi** push lên Hugging Face Spaces — debug lỗi build ngay trên Space (qua log web, vòng lặp chậm) mất thời gian hơn nhiều so với debug local.
- File `.dockerignore` nên loại trừ `data/`, `COVID-19_Radiography_Dataset/`, `.git/`, `__pycache__/` — nếu không, Docker build context sẽ cực lớn (dataset gốc có thể vài GB), làm chậm hẳn quá trình build dù các thư mục đó không thực sự cần trong image.

### 14.7. Cách tự test trước khi deploy

```bash
# Build image local
docker build -t chest-xray-demo .

# Chạy container, map port 7860
docker run -p 7860:7860 chest-xray-demo

# Mở trình duyệt tại http://localhost:7860, thử upload ảnh và xem kết quả
```

Nếu container chạy được ở local với đúng luồng như lúc chạy `python app.py` trực tiếp (không qua Docker), coi như sẵn sàng push lên Hugging Face Spaces.

---

## 15. Debug playbook tổng hợp

### 15.1. Train loss không giảm sau vài epoch

- LR quá thấp — thử tăng ×10.
- Backbone đang đóng băng quá nhiều — kiểm tra lại bằng `count_trainable_params`.
- Dataset trả nhãn sai — in `Counter(labels)` từ batch đầu tiên của loader để xác nhận.

### 15.2. Train loss giảm nhưng val loss tăng (overfit)

- Tăng cường độ augmentation.
- Tăng `weight_decay` lên `5e-4`.
- Tăng dropout ở head (mặc định `torchvision.efficientnet_b3` dùng dropout=0.3).
- Rút ngắn số epoch, dùng early stopping nghiêm ngặt hơn (giảm `PATIENCE`).

### 15.3. Loss ra NaN

- LR quá cao ngay lúc vừa unfreeze backbone.
- Ảnh trong batch có giá trị bất thường (toàn 0, toàn NaN sau normalize) — in `min/max/mean` của tensor ngay trước forward để kiểm tra.
- Mixed precision: đảm bảo đang dùng `GradScaler`; nếu vẫn NaN, thử tắt AMP để cô lập nguyên nhân.

### 15.4. CUDA out of memory

- Giảm `batch_size` (đôi khi phải xuống 8).
- Bật mixed precision nếu chưa bật.
- Giảm `num_workers` (mỗi worker giữ một phần dữ liệu trong RAM/VRAM).
- Gọi `torch.cuda.empty_cache()` giữa các pha train.

### 15.5. Grad-CAM heatmap toàn 0 hoặc toàn 1

- Quên bỏ `torch.no_grad()` (xem Phần 10.4).
- Target layer sai — in `target_layer` ra để xác nhận nó là `Conv2d`, không phải `Linear`.
- Model đang bị bọc trong `DataParallel` — unwrap trước khi truyền vào `GradCAM`.

### 15.6. U-Net predict ra mask trống hoặc toàn ảnh

- Loss thiếu thành phần Dice — chỉ dùng BCE một mình dễ bị chi phối bởi lớp nền, model học cách dự đoán toàn 0 vẫn ra loss thấp.
- Sai ngưỡng lúc đánh giá — quên sigmoid, so sánh trực tiếp logit `> 0.5` (đúng phải là `sigmoid(logit) > 0.5`, tương đương `logit > 0`).
- Mask chưa chuẩn hóa — nếu mask còn giá trị `3` (từ nhãn COVID trong `preprocess.py`) thay vì đã nhị phân hóa về `{0,1}`, Dice/IoU sẽ tính sai hoàn toàn.

### 15.7. Backend FastAPI báo lỗi 500 khi gọi `/predict`

- Kiểm tra log server (terminal chạy `uvicorn`) — lỗi Python đầy đủ luôn in ra đó, kể cả khi client chỉ thấy "Internal Server Error".
- Ảnh upload sai định dạng hoặc hỏng — đảm bảo đã bọc `Image.open()` trong `try/except` (Phần 12.5).
- Model chưa load (quên gọi `load_models()` ở `startup`, hoặc đường dẫn `weights/best_classifier.pth` sai so với thư mục chạy `uvicorn`).

### 15.8. Gradio UI không nhận được response từ API

- Kiểm tra `API_URL` có đúng port đang chạy `uvicorn` không.
- Nếu chạy trong 2 terminal riêng, đảm bảo cả hai vẫn đang chạy (không bị Ctrl+C nhầm).
- Lỗi CORS thường không xảy ra khi gọi từ Python (`requests`) — nếu sau này đổi sang gọi từ JavaScript phía trình duyệt, cần cấu hình `CORSMiddleware` ở `main.py`.

### 15.9. Docker build thành công nhưng container crash ngay khi chạy

- Xem log bằng `docker logs <container_id>` — thường là thiếu thư viện hệ thống (`libgl1` cho OpenCV, xem Phần 14.4) hoặc sai đường dẫn `weights/` bên trong image so với lúc code chạy local.
- Kiểm tra image có thực sự copy đúng `weights/*.pth` vào — dễ quên nếu `.dockerignore` hoặc `.gitignore` cấu hình chặn nhầm.

---

## 16. Lộ trình đề xuất & checklist kiểm thử cuối

### 16.1. Lộ trình theo tuần (gợi ý, có thể co giãn)

| Tuần | Công việc | Đầu ra |
|---|---|---|
| 1 | Data pipeline (đã có) + `dataset.py` + `model.py` + `unet.py` | 3 file `.py` pass hết sanity check ở mỗi phần |
| 2 | `train_classifier.ipynb` + `train_unet.ipynb` | `weights/best_classifier.pth`, `weights/best_unet.pth`, biểu đồ loss/F1/Dice trong `figures/` |
| 3 | `gradcam.py` + `shortcut_iou.py` | Bộ heatmap mẫu + bảng/biểu đồ IoU shortcut analysis cho báo cáo |
| 4 | `api/` (FastAPI) + `app.py` (Gradio) chạy local | Demo chạy được trên máy, test `curl` + UI pass |
| 5 | `Dockerfile` + deploy Hugging Face Spaces | URL demo public, sẵn sàng đưa vào báo cáo/thuyết trình |

### 16.2. Checklist trước khi coi một giai đoạn là "xong"

- [ ] Đoạn "Cách tự test" của giai đoạn đó chạy không lỗi, kết quả đúng như mô tả.
- [ ] Đã đọc phần "Gotchas" tương ứng và tự kiểm tra code của mình không dính lỗi nào trong danh sách đó.
- [ ] Nếu giai đoạn có xuất ra file (`weights/*.pth`, `figures/*.png`, `data/predictions.db`) — đã xác nhận file thực sự tồn tại và mở được, không rỗng.
- [ ] Hằng số dùng chung (`CLASS_TO_IDX`, `IMAGE_SIZE`, `MEAN`, `STD`, `RANDOM_SEED`) không bị vô tình định nghĩa lại khác giá trị ở file mới.

### 16.3. Checklist kiểm thử toàn hệ thống (trước khi coi là "demo được")

- [ ] Chạy lại từ đầu trên máy sạch (hoặc container Docker) — không phụ thuộc file/biến còn sót lại từ REPL cũ.
- [ ] Upload lần lượt 1 ảnh mỗi lớp (Normal, Lung_Opacity, COVID) qua Gradio UI, xác nhận nhãn dự đoán hợp lý và heatmap không rơi ra ngoài vùng ảnh.
- [ ] Kiểm tra `data/predictions.db` có ghi log đúng sau mỗi lần dự đoán qua UI.
- [ ] Build và chạy thử `Dockerfile` local trước khi push lên Hugging Face Spaces.
- [ ] Đọc lại toàn bộ giá trị metric cuối cùng (Macro F1 classifier, Dice/IoU U-Net, mean IoU shortcut analysis) — đối chiếu với ngưỡng kỳ vọng đã nêu ở từng phần (F1 hợp lý theo bài toán 3 lớp, Dice > 0.90, IoU shortcut theo bảng ở Phần 11.4) trước khi đưa vào báo cáo.

---

## 17. Đánh giá mô hình (Evaluation) toàn diện

Phần 8–11 đã cho bạn các con số theo dõi *trong lúc* train (val loss/F1, val Dice/IoU, shortcut IoU) — những con số đó dùng để **chọn checkpoint tốt nhất và dừng train đúng lúc**, không phải con số cuối cùng đưa vào report. Phần này mô tả bộ đánh giá **đầy đủ, chạy một lần trên test set sau khi đã chốt model**, và cách diễn giải kết quả để viết report ở Phần 18.

### 17.1. Nguyên tắc chung — đọc trước khi chạy bất kỳ số liệu nào

- **Test set chỉ chạm đúng một lần**, sau khi đã chốt toàn bộ quyết định (kiến trúc, hyperparameter, số epoch mỗi pha) dựa trên val set. Nếu bạn nhìn số liệu test, thấy chưa ưng rồi quay lại tune thêm, sau đó chạy lại test — đó là một dạng rò rỉ dữ liệu (test set overfitting), số liệu cuối không còn đáng tin. Chỉ chạy test **một lần** khi đã thực sự xong.
- Mọi số liệu báo cáo phải kèm `n` (số mẫu tính trên đó) — một Macro F1 tính trên 10 ảnh và trên 1000 ảnh không đáng tin như nhau.
- Tách rõ hai loại bằng chứng trong report: **định lượng** (bảng số liệu, con số) và **định tính** (ảnh minh họa cụ thể) — người đọc/giảng viên cần cả hai, số liệu để tin, ảnh để hiểu vì sao.

### 17.2. Evaluation cho classifier (EfficientNet-B3)

Notebook train chỉ theo dõi Macro F1 mỗi epoch trên val. Trên **test set**, cần đầy đủ hơn:

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from src.model import build_classifier
from src.dataset import ChestXrayClassificationDataset, get_val_transforms, IDX_TO_CLASS
from torch.utils.data import DataLoader

model = build_classifier(num_classes=3, pretrained=False)
model.load_state_dict(torch.load("weights/best_classifier.pth", map_location="cpu"))
model.eval()

test_ds = ChestXrayClassificationDataset("data/split/test", get_val_transforms())
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

y_true, y_pred, y_probs = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        probs = model(x).softmax(dim=1)
        y_true.extend(y.tolist())
        y_pred.extend(probs.argmax(1).tolist())
        y_probs.extend(probs.tolist())

class_names = [IDX_TO_CLASS[i] for i in range(len(IDX_TO_CLASS))]

# 1. Bảng precision/recall/F1 đầy đủ theo từng lớp + macro/weighted avg
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

# 2. Confusion matrix TRÊN TEST SET — đây là con số cuối cùng đưa vào report,
#    không phải confusion matrix trên val đã vẽ lúc train
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("Ground truth")
plt.tight_layout()
plt.savefig("figures/confusion_matrix_test.png", dpi=120)

# 3. ROC-AUC one-vs-rest — đo khả năng phân tách của model độc lập với ngưỡng 0.5
y_true_onehot = np.eye(len(class_names))[y_true]
auc_macro = roc_auc_score(y_true_onehot, np.array(y_probs), multi_class="ovr", average="macro")
print("Macro ROC-AUC:", auc_macro)
```

**Vì sao Recall (Sensitivity) của lớp COVID quan trọng hơn Accuracy tổng?** Trong bài toán y tế, bỏ sót một ca bệnh thật (false negative) nguy hiểm hơn nhiều so với báo động nhầm một ca khỏe mạnh (false positive) — bệnh nhân bị bỏ sót không được điều trị kịp thời, trong khi báo động nhầm chỉ dẫn đến kiểm tra thêm. Vì vậy khi trình bày kết quả, **luôn tách riêng Recall theo từng lớp** thay vì chỉ đưa một con số Accuracy tổng — một model 90% accuracy nhưng chỉ bắt được 60% ca COVID thật (recall thấp) là một model tệ về mặt lâm sàng dù accuracy nhìn "đẹp".

`classification_report` với `digits=3` cho đủ 3 chữ số thập phân — cần thiết vì các con số F1/Recall giữa các cấu hình thử nghiệm thường chỉ chênh nhau ở chữ số thứ 2–3.

### 17.3. Evaluation cho U-Net (segmentation)

Tương tự, tính Dice + IoU trên **test set** (không phải chỉ val đã theo dõi lúc train):

```python
from src.unet import build_unet, dice_score, iou_score
from src.dataset import ChestXraySegmentationDataset, get_val_transforms
import numpy as np, torch
from torch.utils.data import DataLoader

unet = build_unet(pretrained=False)
unet.load_state_dict(torch.load("weights/best_unet.pth", map_location="cpu"))
unet.eval()

test_seg_ds = ChestXraySegmentationDataset("data/split/test", get_val_transforms())
test_seg_loader = DataLoader(test_seg_ds, batch_size=16, shuffle=False)

dices, ious = [], []
with torch.no_grad():
    for x, y in test_seg_loader:
        logits = unet(x)
        dices.append(dice_score(logits, y).item())
        ious.append(iou_score(logits, y).item())

print(f"Test Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
print(f"Test IoU : {np.mean(ious):.4f} ± {np.std(ious):.4f}")
```

Bổ sung phần **định tính bắt buộc**: lưu một lưới ảnh gồm 5 case Dice **cao nhất** và 5 case Dice **thấp nhất** trên test set, mỗi case hiển thị `[ảnh gốc | mask ground-truth | mask dự đoán]`. Case Dice thấp nhất thường lộ ra lỗi hệ thống (ví dụ mask bị thủng ở vùng có tổn thương nặng, hoặc lem ra ngoài lồng ngực ở ảnh chụp nghiêng) — đây là nội dung tốt cho phần Discussion của report.

### 17.4. Ablation study — nên làm ít nhất 1–2 cái

Ablation study (bỏ/đổi một thành phần rồi so sánh) chứng minh các lựa chọn thiết kế trong Phần 6–7 thực sự có tác dụng, không phải chọn tùy tiện. Với thời gian có hạn, ưu tiên các ablation **rẻ để chạy** (không cần train lại từ đầu nhiều lần):

| Ablation | Cách làm | Giả thuyết kỳ vọng |
|---|---|---|
| Có vs không 3-phase fine-tuning | So sánh pipeline hiện tại với một lần chạy `unfreeze_all()` + train thẳng từ đầu, cùng tổng số epoch | 3-phase cho Macro F1 cao hơn, tránh catastrophic forgetting (Phần 2.3) |
| Có vs không augmentation | Train lại classifier với `get_val_transforms()` thay `get_train_transforms()` cho tập train | Có augmentation cho val F1 cao hơn hoặc gap train/val nhỏ hơn (đỡ overfit) |
| BCE-only vs BCE+Dice cho U-Net | Đổi `BCEDiceLoss(bce_weight=1.0)` so với mặc định `0.5` | BCE-only cho Dice test thấp hơn, đặc biệt dễ thấy nếu ảnh có ít pixel phổi |

Trình bày kết quả dưới dạng một bảng duy nhất trong report:

```
| Cấu hình                  | Macro F1 (test) | Test Dice |
|----------------------------|-----------------|-----------|
| Full pipeline (đề xuất)    | 0.xx            | 0.xx      |
| Không 3-phase fine-tune    | 0.xx            | —         |
| Không augmentation         | 0.xx            | —         |
| BCE-only (không Dice loss) | —               | 0.xx      |
```

### 17.5. Error analysis

- Từ confusion matrix (Phần 17.2), xác định cặp lớp hay bị nhầm nhất — thường là **COVID ↔ Lung_Opacity** vì cả hai đều biểu hiện bất thường mô phổi trên X-quang, khó phân biệt hơn nhiều so với việc phân biệt với `Normal`.
- Lấy 5–10 ảnh bị model dự đoán sai với **confidence cao** (model "tự tin nhưng sai") trên test set, visualize kèm Grad-CAM — đây là các case đáng chú ý nhất, vì model sai mà vẫn tự tin thường phản ánh một lỗi hệ thống (data quality, shortcut) chứ không phải chỉ là ảnh khó.
- Đối chiếu chéo với kết quả shortcut analysis (Phần 11): case nào vừa bị misclassify vừa có IoU(Grad-CAM, lung mask) thấp — đó là bằng chứng khá thuyết phục cho việc model học sai đặc điểm ở đúng case đó.

### 17.6. Đưa kết quả shortcut/explainability (Phần 11) vào bộ evaluation

Không cần chạy lại gì mới — `shortcut_iou.py` (Giai đoạn 8) đã sinh đủ số liệu. Khi tổng hợp cho report, chuẩn bị:

- Bảng mean/median/std IoU theo từng lớp, với cả hai nguồn mask (`gt` và `unet`) — chứng minh U-Net đủ tốt để thay ground-truth khi deploy (không có mask thật cho ảnh mới).
- Biểu đồ IoU theo ngưỡng Grad-CAM (`[0.3, 0.4, 0.5, 0.6, 0.7]`) — chứng minh kết luận không nhạy với lựa chọn ngưỡng.
- 3–5 heatmap minh họa: ít nhất 1 case IoU cao (model tập trung đúng phổi) và 1 case IoU thấp (khả nghi shortcut), đặt cạnh nhau để so sánh trực quan.

### 17.7. Sanity check cuối trước khi chốt số liệu

- Xác nhận test set **chưa từng** được dùng để chọn hyperparameter hay quyết định early stopping trong bất kỳ notebook nào (chỉ val set được phép dùng cho việc đó — kiểm tra lại `train_classifier.ipynb`/`train_unet.ipynb` không có dòng nào vô tình đọc từ `data/split/test`).
- Chạy lại đúng đoạn code evaluation hai lần liên tiếp (cùng checkpoint, cùng seed) — phải ra **đúng cùng một con số**. Nếu không, khả năng cao là còn phép toán ngẫu nhiên chưa seed (augmentation vẫn đang bật nhầm ở transform dùng cho test — kiểm tra lại đang dùng `get_val_transforms()` chứ không phải `get_train_transforms()`).

## 18. Viết report

### 18.1. Cấu trúc đề xuất

Cấu trúc dưới đây theo chuẩn báo cáo đồ án/nghiên cứu CS phổ biến, đã ánh xạ trực tiếp vào những gì dự án này thực sự có (không có mục nào yêu cầu số liệu bạn chưa tạo ra ở các phần trước):

1. **Tóm tắt (Abstract)** — 150–250 từ: bài toán, phương pháp (2 model + Grad-CAM), kết quả chính (con số cụ thể lấy từ Phần 17.2/17.3), đóng góp nổi bật (kiểm định shortcut learning — không phải đồ án nào cũng làm phần này).
2. **1. Giới thiệu** — bối cảnh (vì sao cần công cụ hỗ trợ đọc X-quang), phát biểu bài toán rõ ràng (phân loại 3 lớp + phân đoạn phổi + giải thích được), liệt kê đóng góp dạng bullet.
3. **2. Công trình liên quan** — ngắn gọn, dựa trên danh sách ở Phần 19: EfficientNet, U-Net, Grad-CAM là nền tảng phương pháp; Geirhos et al. và Zech et al. là cơ sở cho phần shortcut learning — nói rõ dự án này *áp dụng* các phương pháp đã có (không tự nhận là đề xuất kiến trúc mới).
4. **3. Dữ liệu** — nguồn (Kaggle COVID-19 Radiography Database), số ảnh gốc mỗi lớp vs. số ảnh sau khi giới hạn `MAX_IMAGES_PER_CLASS` (Phần 4.2), bảng số lượng train/val/test thực tế (đếm lại, xem lưu ý ở Phần 4.3 về công thức chia có thể lệch nhẹ 15/15 danh nghĩa), mô tả tiền xử lý + augmentation (Phần 4–5).
5. **4. Phương pháp** — sơ đồ kiến trúc tổng thể (dùng lại sơ đồ ở Phần 1.3), mô tả classifier (EfficientNet-B3 + chiến lược fine-tune 3 pha, Phần 2.3/6), U-Net (SMP + encoder ResNet-34 + loss BCE+Dice, Phần 2.4/7), Grad-CAM (layer mục tiêu, công thức, Phần 2.5/10), phương pháp shortcut IoU (Phần 2.6/11).
6. **5. Thực nghiệm** — môi trường (GPU dùng, phiên bản PyTorch), bảng hyperparameter đầy đủ từng pha (LR, epoch, batch size, optimizer, weight decay — lấy thẳng từ Cell 2 của mỗi notebook), seed dùng để tái lập (Phần 3.3).
7. **6. Kết quả** — bảng kết quả classifier (Phần 17.2: accuracy, macro P/R/F1, ROC-AUC), confusion matrix test set, bảng kết quả U-Net (Phần 17.3: Dice, IoU), biểu đồ training curves (loss/F1 theo epoch từ Cell 6 notebook), bảng ablation nếu có (Phần 17.4).
8. **7. Explainability & Shortcut Learning** — heatmap minh họa, bảng/biểu đồ IoU theo lớp và theo ngưỡng (Phần 17.6), diễn giải kết quả theo thang ở Phần 11.4, liên hệ trực tiếp với case study của Zech et al.
9. **8. Thảo luận & hạn chế** — model nhầm lẫn ở đâu và vì sao (Phần 17.5), hạn chế của dataset (gộp từ nhiều nguồn — chính là rủi ro Zech et al. đã cảnh báo, dataset không đại diện đầy đủ cho mọi nhóm bệnh nhân/máy chụp), hạn chế phương pháp (Grad-CAM là một kỹ thuật XAI trong nhiều kỹ thuật, chưa được bác sĩ xác nhận lâm sàng), và **nhắc lại rõ ràng** hệ thống không phải công cụ chẩn đoán y khoa chính thức.
10. **9. Kết luận & hướng phát triển** — tóm tắt đóng góp, đề xuất mở rộng cụ thể (ví dụ: thêm lớp Viral Pneumonia hiện đang bị loại ở Phần 4.2, thử Grad-CAM++, thêm bootstrap confidence interval cho các metric).
11. **Tài liệu tham khảo** — dùng lại danh sách ở Phần 19, định dạng theo đúng chuẩn trích dẫn môn học yêu cầu (IEEE/APA...).
12. **Phụ lục (tùy chọn)** — bảng/hình chi tiết không cần thiết ở thân bài (ví dụ full config từng phase, thêm ảnh minh họa ngoài số lượng đã chọn cho phần 6–7).

### 18.2. Nguyên tắc viết

- **Mọi con số phải truy được nguồn** — biết chính xác nó lấy từ lệnh nào, output của Phần 17 nào; không gõ tay từ trí nhớ hay ước lượng. Nếu giảng viên hỏi "con số này từ đâu ra", bạn phải chỉ được ngay vào đoạn code sinh ra nó.
- **Không phóng đại kết quả** — ví dụ tránh những câu như "hệ thống sẵn sàng ứng dụng lâm sàng" khi mới chỉ test trên một dataset công khai, chưa qua kiểm định lâm sàng thật. Phần Discussion (mục 8) chính là nơi thể hiện sự nghiêm túc khoa học qua việc tự nêu hạn chế.
- Mọi hình/bảng đánh số rõ (Figure 1, Table 1...) và có caption mô tả đủ để hiểu độc lập với văn bản xung quanh — người đọc lướt qua phần hình trước khi đọc kỹ nên caption phải tự giải thích được.
- Ưu tiên trình bày số liệu bằng bảng/biểu đồ hơn liệt kê trong văn xuôi — một đoạn văn liệt kê 6 con số khó đọc hơn nhiều một bảng 6 dòng.

### 18.3. Checklist trước khi nộp report

- [ ] Mọi số liệu đã đối chiếu lại với output thực tế chạy từ Phần 17 (không có số nào gõ từ trí nhớ).
- [ ] Mọi hình trong `figures/` được dùng đều được nhắc và giải thích trong văn bản — không có hình "mồ côi".
- [ ] Trích dẫn đầy đủ các paper đã dùng (danh sách ở Phần 19), đúng định dạng citation môn học yêu cầu.
- [ ] Đọc lại toàn bộ phần Discussion (mục 8) một lượt riêng, chỉ để kiểm tra không có câu nào phóng đại kết quả so với những gì đã thực sự đo được.
- [ ] Kiểm tra định dạng theo đúng yêu cầu của môn học/đồ án (giới hạn số trang, font, citation style, có cần nộp kèm code/weights không).

## 19. Tài liệu tham khảo

**Kiến trúc & phương pháp:**

- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.

**Shortcut learning & độ tin cậy của model y tế:**

- Geirhos, R., et al. (2020). *Shortcut Learning in Deep Neural Networks.* Nature Machine Intelligence.
- Zech, J. R., et al. (2018). *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study.* PLoS Medicine.

**Công cụ & thư viện (tài liệu chính thức):**

- [PyTorch — Data Loading Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [PyTorch — cài đặt theo CUDA version](https://pytorch.org)
- [Albumentations documentation](https://albumentations.ai/docs/)
- [segmentation-models-pytorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Gradio documentation](https://www.gradio.app/docs)
- [Hugging Face Spaces — Docker SDK](https://huggingface.co/docs/hub/spaces-sdks-docker)

**Dataset:**

- [COVID-19 Radiography Database — Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

---

*Hết tài liệu. Đi đúng theo thứ tự 11 giai đoạn ở Phần 0 — mỗi giai đoạn xong, chạy "Cách tự test" trước khi qua giai đoạn tiếp theo. Chúc bạn xây dựng hệ thống thành công.*
