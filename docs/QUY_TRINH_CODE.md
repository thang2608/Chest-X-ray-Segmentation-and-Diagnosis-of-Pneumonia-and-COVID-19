# QUY TRÌNH CODE — Bản đồ Pipeline, Hàm, Biến, Input/Output

*Tài liệu "visualize" toàn bộ codebase: file nào gọi file nào, hàm nào nhận gì trả gì, biến/hằng số nào dùng chung xuyên suốt dự án — đi kèm sơ đồ luồng dữ liệu cụ thể ở từng bước.*

---

## Cách đọc tài liệu này

Đây là tài liệu **bản đồ code** — khác với `docs/LY_THUYET.md` (giải thích *vì sao* thuật toán hoạt động về mặt toán học) và khác với `docs/TUTORIAL.md` (hướng dẫn *cách gõ* từng dòng code kèm giải thích quyết định thiết kế). Tài liệu này trả lời một câu hỏi hẹp hơn nhưng rất hay bị rối khi nhìn cả một repo nhiều file: **"dữ liệu/đối tượng nào chảy từ file này sang file kia, qua hàm nào, shape/kiểu dữ liệu gì?"**

**Chú giải trạng thái dùng xuyên suốt tài liệu** (khớp với `CLAUDE.md` của repo):

| Ký hiệu | Ý nghĩa |
|---|---|
| ✅ **ĐÃ CODE** | File có logic thật, đã chạy được, tôi đọc trực tiếp từ source hiện tại trong repo |
| 📝 **SKELETON** | File hiện là placeholder (`# TODO: implement...`) — nội dung mô tả lấy từ khung code đề xuất trong `docs/TUTORIAL.md`, **chưa tồn tại thật trong repo**, dùng để bạn hình dung trước khi tự gõ |

**Quy ước ký hiệu shape/kiểu dữ liệu:**

| Ký hiệu | Ý nghĩa |
|---|---|
| `(N, C, H, W)` | Tensor 4 chiều PyTorch chuẩn: batch, channel, height, width |
| `(H, W)` | Mảng 2D (ảnh xám, mask, heatmap) |
| `Path` | Đối tượng `pathlib.Path` (đường dẫn file/thư mục) |
| `→` | "trả về" / "biến thành" khi mô tả một hàm |
| `⇒` | "chảy sang" khi mô tả luồng dữ liệu giữa các FILE (không phải trong 1 hàm) |

---

## Mục lục

- **Phần 0** — Sơ đồ tổng quan: toàn bộ pipeline ở mức file, một nhìn thấy hết
- **Phần 1** — Giai đoạn 1: Data pipeline (`preprocess.py`, `split_data.py`, `verify.py`, `visualize.py`) — ✅ ĐÃ CODE
- **Phần 2** — Giai đoạn 2: `src/dataset.py` — 📝 SKELETON
- **Phần 3** — Giai đoạn 3: `src/model.py` — 📝 SKELETON
- **Phần 4** — Giai đoạn 4: `src/unet.py` — 📝 SKELETON
- **Phần 5** — Giai đoạn 5–6: `notebooks/train_classifier.ipynb`, `train_unet.ipynb` — 📝 SKELETON
- **Phần 6** — Giai đoạn 7: `src/gradcam.py` — 📝 SKELETON
- **Phần 7** — Giai đoạn 8: `src/shortcut_iou.py` — 📝 SKELETON
- **Phần 8** — Giai đoạn 9: `api/` (`schemas.py`, `inference.py`, `db.py`, `main.py`) — 📝 SKELETON
- **Phần 9** — Giai đoạn 10: `app.py` (Gradio UI) — 📝 SKELETON
- **Phần 10** — Giai đoạn 11: `Dockerfile` — 📝 SKELETON
- **Phần 11** — Bảng tổng hợp: MỌI hàm trong dự án (1 bảng duy nhất, tra cứu nhanh)
- **Phần 12** — Bảng tổng hợp: MỌI hằng số/biến toàn cục dùng chung
- **Phần 13** — Sơ đồ luồng dữ liệu end-to-end với shape cụ thể tại mỗi mũi tên

---

# PHẦN 0 — SƠ ĐỒ TỔNG QUAN: TOÀN BỘ PIPELINE Ở MỨC FILE

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  DỮ LIỆU THÔ (tải từ Kaggle, KHÔNG commit — gitignored)                           │
│  COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset/{COVID,Lung_Opacity,   │
│  Normal, Viral Pneumonia}/{images,masks}/*.png                                    │
└──────────────────────────────────────┬───────────────────────────────────────────┘
                                        │
                                        ▼
                        ✅ src/preprocess.py  (Phần 1.1)
                        resize 224×224, gắn nhãn mask theo LABELS
                                        │
                                        ▼
                    data/processed/<class>/{images,masks}/*.png
                                        │
                                        ▼
                        ✅ src/split_data.py  (Phần 1.2)
                        shuffle + cắt 70/15/15
                                        │
                                        ▼
              data/split/{train,val,test}/{images,masks}/*.png
                     │                                    ▲
                     │ (sanity check bằng mắt/số liệu)     │
                     ▼                                    │
    ✅ src/verify.py, ✅ src/visualize.py ──────────────────┘
    (Phần 1.3 — không sinh ra file mới, chỉ in/vẽ để kiểm tra)
                     │
                     ▼
                📝 src/dataset.py  (Phần 2)
   ChestXrayClassificationDataset, ChestXraySegmentationDataset
   get_train_transforms(), get_val_transforms()
   → trả về TENSOR (image, label) hoặc (image, mask)
                     │
        ┌────────────┴─────────────┐
        ▼                           ▼
📝 src/model.py (Phần 3)     📝 src/unet.py (Phần 4)
build_classifier()           build_unet()
        │                           │
        ▼                           ▼
📝 notebooks/train_classifier.ipynb  📝 notebooks/train_unet.ipynb   (Phần 5)
        │                           │
        ▼                           ▼
weights/best_classifier.pth   weights/best_unet.pth
        │                           │
        └─────────────┬─────────────┘
                       ▼
              📝 src/gradcam.py  (Phần 6)
        generate_gradcam(model, img_tensor) → heatmap (H,W)
                       │
                       ▼
              📝 src/shortcut_iou.py  (Phần 7)
        run_shortcut_analysis() → IoU(heatmap, lung_mask) theo lớp
                       │
                       ▼
              📝 api/ (schemas.py, inference.py, db.py, main.py)  (Phần 8)
        POST /predict  →  JSON {predicted_class, confidence, probabilities,
                                  heatmap_overlay_base64, disclaimer}
                       │  (đồng thời ghi log)
                       ▼
              data/predictions.db  (SQLite)
                       │
                       ▼
              📝 app.py — Gradio UI  (Phần 9)
        gr.Blocks → gọi HTTP sang api/main.py → hiển thị cho người dùng
                       │
                       ▼
              📝 Dockerfile  (Phần 10)
        đóng gói app.py + api/ + src/ + weights/ → deploy Hugging Face Spaces
```

**Đọc sơ đồ này thế nào:** mỗi ô là một file/nhóm file; mũi tên `▼` là hướng dữ liệu chảy. Toàn bộ pipeline chia làm 3 "cụm" lớn: **cụm dữ liệu** (trên cùng, đã code xong), **cụm model** (giữa, cần code + train), **cụm serving** (dưới cùng, cần code sau khi có model). Không thể làm cụm sau trước cụm trước — mỗi cụm là input bắt buộc của cụm kế tiếp.

---

# PHẦN 1 — GIAI ĐOẠN 1: DATA PIPELINE (✅ ĐÃ CODE)

## 1.1. `src/preprocess.py`

**Vai trò:** đọc ảnh X-quang + mask thô từ dataset Kaggle, resize về kích thước chuẩn, gắn nhãn lớp vào giá trị pixel của mask, ghi ra `data/processed/`.

**Trạng thái:** ✅ đã code, không có hàm (`def`) nào — toàn bộ là script chạy tuần tự ở top-level (2 vòng `for` lồng nhau), không có `if __name__ == "__main__":`. **Điểm cần lưu ý:** vì không có main guard, nếu file này bị `import` (thay vì chạy trực tiếp bằng `python src\preprocess.py`) thì toàn bộ logic tiền xử lý sẽ **chạy ngay lập tức** lúc import — đây là hành vi có chủ đích cho một script chạy 1 lần, nhưng sẽ gây bất ngờ nếu ai đó cố `from src.preprocess import ...` từ notebook khác.

**Hằng số toàn cục (định nghĩa ở đầu file):**

| Biến | Giá trị | Ý nghĩa |
|---|---|---|
| `DATASET_DIR` | `Path("COVID-19_Radiography_Dataset")` | Thư mục dataset thô (gitignored) |
| `OUTPUT_DIR` | `Path("data/processed")` | Nơi ghi kết quả |
| `CLASSES` | `["COVID", "Lung_Opacity", "Normal"]` | **Chỉ 3 lớp** — `Viral Pneumonia` có trong dataset thô nhưng KHÔNG được xử lý |
| `IMAGE_SIZE` | `(224, 224)` | Kích thước resize — hằng số này lặp lại (định nghĩa riêng, không import chung) ở mọi file khác |
| `MAX_IMAGES_PER_CLASS` | `3000` | Giới hạn số ảnh lấy mỗi lớp (cân bằng dữ liệu) |
| `RANDOM_SEED` | `42` | Seed cho `random.sample` khi lớp có > 3000 ảnh |
| `LABELS` | `{"Normal": 1, "Lung_Opacity": 2, "COVID": 3}` | Giá trị pixel gắn vào mask — **không phải** nhị phân 0/1 |

**Luồng xử lý bên trong (không có hàm, đọc theo thứ tự vòng lặp):**

```
for class_name in CLASSES:                                    # 3 vòng lặp lớn: COVID, Lung_Opacity, Normal
    image_dir  = DATASET_DIR/class_name/"images"
    mask_dir   = DATASET_DIR/class_name/"masks"
    output_image_dir = OUTPUT_DIR/class_name/"images"          # tạo mới (mkdir parents=True)
    output_mask_dir  = OUTPUT_DIR/class_name/"masks"

    image_paths = list(image_dir.glob("*.png"))                 # LIỆT KÊ toàn bộ ảnh .png
    if len(image_paths) > MAX_IMAGES_PER_CLASS:
        image_paths = random.sample(image_paths, 3000)          # random.seed(42) đặt 1 LẦN DUY NHẤT
                                                                  # ở đầu file, trước cả vòng lặp for

    for image_path in image_paths:                              # với TỪNG ảnh đã chọn
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            print("WARNING: Missing mask"); continue             # BỎ QUA ảnh thiếu mask, không raise lỗi

        image = Image.open(image_path).convert("L")              # (H_gốc, W_gốc) ảnh xám 1 kênh
        mask  = Image.open(mask_path).convert("L")

        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)   # → (224,224), nội suy MƯỢT
        mask  = mask.resize(IMAGE_SIZE, Image.Resampling.NEAREST)     # → (224,224), nội suy GIỮ BIÊN SẮC

        mask_array  = np.array(mask)                              # (224,224) uint8, giá trị thô [0,255]
        mask_binary = mask_array > 127                            # (224,224) bool — nhị phân hoá TẠM THỜI
        mask_array  = np.zeros_like(mask_array, dtype=np.uint8)   # tạo mảng 0 cùng shape
        mask_array[mask_binary] = LABELS[class_name]              # GÁN LẠI: pixel phổi = 1/2/3 theo lớp

        mask = Image.fromarray(mask_array)                        # đóng gói lại thành ảnh PIL
        image.save(output_image_dir / image_path.name)            # ghi file, TÊN GIỮ NGUYÊN như gốc
        mask.save(output_mask_dir / image_path.name)              #   (đây là quy ước file ảnh/mask
                                                                    #    khớp tên nhau xuyên suốt cả repo)
```

**Input:** thư mục `COVID-19_Radiography_Dataset/<class>/{images,masks}/*.png` (ảnh RGB hoặc grayscale bất kỳ kích thước, do Kaggle cung cấp).

**Output:** thư mục `data/processed/<class>/{images,masks}/*.png` — ảnh `224×224` xám, mask `224×224` với giá trị pixel `∈ {0, 1, 2, 3}` (0=nền, còn lại tuỳ lớp).

**Ai dùng output này tiếp theo:** `split_data.py` (Phần 1.2), `verify.py`/`visualize.py` (Phần 1.3).

## 1.2. `src/split_data.py`

**Vai trò:** chia dữ liệu đã xử lý thành 3 tập train/val/test theo tỉ lệ 70/15/15, copy cặp ảnh+mask sang thư mục mới.

**Trạng thái:** ✅ đã code — có **đúng 1 hàm** duy nhất (`copy_pairs`), phần còn lại vẫn là script top-level. Cũng **không có** main guard — cùng lưu ý như `preprocess.py`.

**Hằng số toàn cục:**

| Biến | Giá trị | Ý nghĩa |
|---|---|---|
| `RANDOM_SEED` | `42` | Định nghĩa **lại** ở file này (không import từ `preprocess.py`) |
| `CLASSES` | `["COVID", "Lung_Opacity", "Normal"]` | Định nghĩa **lại** — trùng giá trị nhưng là biến độc lập |
| `PROCESS_DIR` | `Path("data/processed")` | Input của file này |
| `SPLIT_DIR` | `Path("data/split")` | Output của file này |
| `SPLITS` | `{"train": 0.7, "val": 0.15, "test": 0.15}` | Tỉ lệ chia |

**Hàm duy nhất trong file:**

```
copy_pairs(image_paths: list[Path], mask_dir: Path, split_names: str) -> None
```

| | |
|---|---|
| **Input** | `image_paths` — danh sách `Path` ảnh cần copy; `mask_dir` — thư mục chứa mask nguồn (`data/processed/<class>/masks`); `split_names` — tên tập đích (`"train"`/`"val"`/`"test"`) |
| **Output** | Không trả về gì (`None`) — **side-effect**: copy file vật lý vào `data/split/<split_names>/{images,masks}/` bằng `shutil.copy2` |
| **Hành vi đặc biệt** | Nếu `mask_path` tương ứng không tồn tại, in cảnh báo và **bỏ qua** ảnh đó — giống hệt logic ở `preprocess.py`, dù về lý thuyết ở bước này mọi ảnh trong `data/processed` đã chắc chắn có mask (vì `preprocess.py` chỉ ghi cặp đủ cả hai) |

**Luồng xử lý top-level (bên ngoài hàm):**

```
for class_name in CLASSES:                                    # lặp riêng biệt cho từng lớp — quan trọng:
    image_paths = list((PROCESS_DIR/class_name/"images").glob("*.png"))    # SHUFFLE + SPLIT TỪNG LỚP RIÊNG,
    random.shuffle(image_paths)                                # không trộn 3 lớp rồi mới chia — đảm bảo
                                                                 # tỉ lệ lớp giống nhau ở cả 3 tập (stratified
                                                                 # theo cách làm thủ công, không dùng
                                                                 # sklearn.train_test_split)
    n = len(image_paths)
    train_end = int(n * 0.7)
    val_end   = int(train_end + n * 0.15)      # ⚠️ xem cảnh báo công thức bên dưới
    train_path = image_paths[:train_end]
    val_path   = image_paths[train_end:val_end]
    test_path  = image_paths[val_end:]

    copy_pairs(train_path, mask_dir, "train")
    copy_pairs(val_path,   mask_dir, "val")
    copy_pairs(test_path,  mask_dir, "test")
```

**⚠️ Cảnh báo công thức `val_end` (đã ghi trong `CLAUDE.md`, nhắc lại cụ thể ở đây vì đây chính là dòng code gây ra nó — dòng 56):**

```python
val_end = int(train_end + n * SPLITS["val"])
```

`train_end` là số nguyên **đã làm tròn** (`int(n*0.7)`), còn `n * SPLITS["val"]` là số thực **chưa làm tròn** trên `n` gốc — cộng hai đại lượng này rồi làm tròn 1 lần nữa khiến tỉ lệ val/test thực tế **lệch nhẹ** so với 15/15 danh nghĩa (không sai về mặt "3 tập không chồng lấp, tổng = n", chỉ lệch về mặt tỉ lệ chính xác). Công thức đúng về tỉ lệ phải là `val_end = int(n * (SPLITS["train"] + SPLITS["val"]))`. File **có sẵn đoạn code đếm lại số ảnh** (dòng 66-77) nhưng đang bị **comment out** — nên bật lại (`# for split in [...]`) để tự kiểm tra số liệu thật cho báo cáo thay vì tin vào tỉ lệ danh nghĩa.

**Input:** `data/processed/<class>/{images,masks}/*.png` (output của Phần 1.1).

**Output:** `data/split/{train,val,test}/{images,masks}/*.png` — cùng định dạng ảnh/mask, chỉ khác cách tổ chức thư mục (theo split thay vì theo class — tên lớp giờ chỉ còn nằm trong **tên file**, ví dụ `COVID-123.png`, không còn nằm trong đường dẫn thư mục).

**Ai dùng output này tiếp theo:** `verify.py`, `visualize.py` (đọc `data/processed`, không phải `data/split` — xem gotcha ở mục 1.3), và toàn bộ `src/dataset.py` (Phần 2) sau này sẽ đọc từ `data/split/<split>/`.

## 1.3. `src/verify.py` và `src/visualize.py` — sanity-check bằng số liệu và bằng mắt

**Vai trò:** không sinh ra file mới — chỉ **đọc lại** dữ liệu đã xử lý và in/vẽ ra để con người tự kiểm tra bằng mắt trước khi tin tưởng đi tiếp.

**⚠️ Lưu ý quan trọng cả hai file đều có:** `DATA_DIR = Path("data/processed")` — nghĩa là hai script này kiểm tra **output của `preprocess.py`** (Phần 1.1), **KHÔNG** kiểm tra output của `split_data.py` (Phần 1.2). Nếu bạn vừa sửa `split_data.py` và muốn xác nhận tập train/val/test đúng, hai file này **không giúp được** — cần tự viết thêm đoạn đếm (gợi ý đã có sẵn dạng comment trong `split_data.py`, xem 1.2).

**`src/verify.py`** — không có hàm, vòng lặp `for class_name in CLASSES` (3 lần):

```
image_path = next(image_dir.glob("*.png"))    # LẤY 1 ẢNH DUY NHẤT mỗi lớp (ảnh đầu tiên glob trả về,
                                                 # không phải random — chạy lại nhiều lần ra cùng 1 ảnh)
mask_path  = mask_dir / image_path.name
image = Image.open(image_path); mask = Image.open(mask_path)
print("Image:", image.size)                    # kỳ vọng (224, 224)
print("Mask :", mask.size)                      # kỳ vọng (224, 224)
mask_array = np.array(mask)
print("Mask unique values:", np.unique(mask_array))   # kỳ vọng vd. [0 3] cho lớp COVID (0=nền, 3=phổi)
```

**Input:** `data/processed/<class>/{images,masks}/`. **Output:** chỉ in ra console (`stdout`), không ghi file.

**`src/visualize.py`** — không có hàm, dựng lưới `matplotlib` 3×3 (`plt.subplots(3, 3, figsize=(12,12))`):

```
for row, class_name in enumerate(CLASSES):                      # row 0,1,2 = COVID, Lung_Opacity, Normal
    image_path = random.choice(image_paths)                      # NGẪU NHIÊN mỗi lần chạy (không seed!)
    axes[row, 0].imshow(image, cmap="gray")        # cột 0: ảnh X-quang gốc
    axes[row, 1].imshow(mask,  cmap="gray")         # cột 1: mask thô (giá trị 0-3, matplotlib tự scale màu)
    axes[row, 2].imshow(image, cmap="gray")
    axes[row, 2].imshow(mask, alpha=0.4)            # cột 2: OVERLAY — chồng mask lên ảnh, alpha=0.4 (mờ)
plt.tight_layout(); plt.show()
```

**Lưu ý:** `visualize.py` **không** dùng `RANDOM_SEED` — mỗi lần chạy hiện ảnh khác nhau (khác hành vi với `preprocess.py`, vốn seed cố định). Đây là chủ đích hợp lý cho một script sanity-check thủ công (muốn xem nhiều mẫu khác nhau qua nhiều lần chạy), nhưng nếu cần tái lập chính xác hình đưa vào báo cáo, phải tự thêm `random.seed(...)` trước khi chạy.

**Input:** `data/processed/<class>/{images,masks}/`. **Output:** một cửa sổ `matplotlib` hiển thị trực tiếp (`plt.show()`), không tự động lưu file — muốn lưu cho báo cáo phải tự thêm `fig.savefig(...)` trước dòng `plt.show()`.

---

# PHẦN 2 — GIAI ĐOẠN 2: `src/dataset.py` (📝 SKELETON)

**Vai trò:** cầu nối duy nhất giữa "ảnh nằm trên đĩa" và "tensor PyTorch model có thể forward" — mọi file sau này (model, train, gradcam, api) đều import hằng số/class từ đây, **không tự định nghĩa lại**.

**Trạng thái:** hiện là `# TODO: implement`. Nội dung dưới đây là khung đề xuất trong `TUTORIAL.md` Phần 5.

## 2.1. Hằng số toàn cục (PHẢI export, không đổi tên sau khi file khác đã import)

| Biến | Giá trị dự kiến | Ý nghĩa |
|---|---|---|
| `CLASS_TO_IDX` | `{"Normal": 0, "Lung_Opacity": 1, "COVID": 2}` | Ánh xạ tên lớp → chỉ số dùng cho `CrossEntropyLoss`. **Lưu ý thứ tự khác `LABELS` ở `preprocess.py`** (Normal=1,Lung_Opacity=2,COVID=3) — đây là 2 hệ đánh số **độc lập**, dùng cho 2 mục đích khác nhau (mask pixel-value vs. classification index), dễ nhầm nếu không để ý |
| `IDX_TO_CLASS` | `{0:"Normal", 1:"Lung_Opacity", 2:"COVID"}` | Chiều ngược lại, dùng khi hiển thị kết quả cho người dùng |
| `NUM_CLASSES` | `3` | `len(CLASS_TO_IDX)` |
| `IMAGE_SIZE` | `(224, 224)` | Định nghĩa **lại** (độc lập với `preprocess.py`) |
| `MEAN` | `[0.485, 0.456, 0.406]` | Thống kê ImageNet — **bắt buộc** giữ nguyên vì dùng pretrained (Phần III `LY_THUYET.md`) |
| `STD` | `[0.229, 0.224, 0.225]` | nt |

## 2.2. Hàm dựng transform

```
get_train_transforms(image_size=IMAGE_SIZE)      -> A.Compose      (dùng cho ChestXrayClassificationDataset lúc train)
get_val_transforms(image_size=IMAGE_SIZE)        -> A.Compose      (dùng cho val/test/inference — KHÔNG augment)
get_train_transforms_seg(image_size=IMAGE_SIZE)  -> A.Compose      (dùng cho ChestXraySegmentationDataset lúc train
                                                                     — có additional_targets={"mask":"mask"} để
                                                                     augment ảnh+mask ĐỒNG BỘ)
```

Cả 3 hàm không nhận input dữ liệu thật — chỉ trả về một **đối tượng transform** (khai báo pipeline biến đổi), bản thân transform đó mới nhận ảnh khi được gọi bên trong `__getitem__` của Dataset.

## 2.3. Hàm nội bộ

```
_parse_label(filename: str) -> int
```
| | |
|---|---|
| **Input** | tên file, ví dụ `"COVID-123.png"` |
| **Output** | index lớp (`int`), tra bằng cách so khớp **tiền tố** tên file với từng key của `CLASS_TO_IDX` |
| **Raise** | `ValueError` nếu không khớp lớp nào — đây là lý do tên file gốc từ dataset Kaggle (`COVID-1.png`, `Normal-1.png`...) phải được giữ nguyên xuyên suốt `preprocess.py` → `split_data.py` → thư mục `data/split/` — đổi tên file ở bất kỳ bước nào sẽ làm hàm này crash |

## 2.4. Hai class Dataset

```
class ChestXrayClassificationDataset(Dataset):
    __init__(self, split_dir: str, transform=None)
    __len__(self)  -> int
    __getitem__(self, idx: int) -> (torch.Tensor, int)
```

| | |
|---|---|
| **`__init__` input** | `split_dir` — ví dụ `"data/split/train"`; `transform` — 1 trong 2 hàm ở mục 2.2 |
| **`__init__` side-effect** | glob toàn bộ `split_dir/images/*.png`, sort, lưu vào `self.image_paths` — **glob 1 lần duy nhất**, không lặp lại mỗi `__getitem__` (lý do hiệu năng, xem Gotchas) |
| **`__getitem__` output** | `image`: tensor `(3, 224, 224)` float32, đã normalize; `label`: `int` (từ `_parse_label`) |

```
class ChestXraySegmentationDataset(Dataset):
    __init__(self, split_dir: str, transform=None)
    __len__(self)  -> int
    __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor)
```

| | |
|---|---|
| **`__getitem__` output** | `image`: tensor `(3, 224, 224)` float32; `mask`: tensor `(1, 224, 224)` float32, giá trị **đã nhị phân hoá về `{0.0, 1.0}`** — chuyển từ mask gốc `{0,1,2,3}` (từ `preprocess.py` `LABELS`) bằng `(mask_array > 0).astype(np.float32)`, **mất thông tin lớp bệnh**, chỉ còn phổi/không-phổi |

## 2.5. Sơ đồ luồng bên trong `__getitem__` (Classification)

```
idx (int)
   │
   ▼
self.image_paths[idx]  ──►  Path("data/split/train/images/COVID-123.png")
   │                                       │
   │                                       ▼
   │                          np.array(Image.open(path).convert("RGB"))   # (224,224,3) uint8
   │                                       │
   │            _parse_label("COVID-123.png") = 2                        │
   │                                       │                              │
   ▼                                       ▼                              │
label=2                       transform(image=array)["image"]  ◄──────────┘
   │                                       │
   │                          [Resize→Flip→ShiftScaleRotate→Brightness→Normalize→ToTensorV2]
   │                                       ▼
   │                          tensor (3,224,224) float32, giá trị ~[-2.1, 2.6]
   │                                       │
   ▼                                       ▼
   └──────────────  return (image_tensor, label)  ◄─────────────────────┘
```

**Input của cả file:** thư mục `data/split/<split>/{images,masks}/` (output Phần 1.2). **Output của cả file:** hai class Dataset sẵn sàng bọc trong `torch.utils.data.DataLoader` để dùng ở Phần 5 (notebooks).

---

# PHẦN 3 — GIAI ĐOẠN 3: `src/model.py` (📝 SKELETON)

**Vai trò:** file duy nhất chịu trách nhiệm **dựng kiến trúc** classifier — cả lúc train (Phần 5) lẫn lúc serve (Phần 8, `api/inference.py`) đều gọi lại đúng hàm này để đảm bảo kiến trúc khớp nhau khi `load_state_dict`.

## 3.1. Bảng hàm

| Hàm | Input | Output | Vai trò |
|---|---|---|---|
| `build_classifier(num_classes=3, pretrained=True)` | 2 tham số có default | `nn.Module` (EfficientNet-B3 với `classifier[1]` đã thay bằng `Linear(1536, num_classes)`) | Hàm **chính**, chữ ký **không được đổi** sau khi các file khác đã import |
| `freeze_backbone(model)` | `nn.Module` | `None` (side-effect: set `requires_grad=False` cho mọi tham số trong `model.features`) | Dùng ở Pha 1 (warm-up head) |
| `unfreeze_last_blocks(model, num_blocks=2)` | `nn.Module`, `int` | `None` (side-effect: `requires_grad=True` cho `num_blocks` block cuối của `model.features`) | Dùng ở Pha 2 |
| `unfreeze_all(model)` | `nn.Module` | `None` (side-effect: `requires_grad=True` cho **mọi** tham số) | Dùng ở Pha 3 (tuỳ chọn) |
| `count_trainable_params(model)` | `nn.Module` | `int` — tổng `numel()` của mọi tham số có `requires_grad=True` | Dùng để **kiểm chứng** đang ở đúng pha (xem Phần III.5 `LY_THUYET.md`) |

## 3.2. Điểm dữ liệu quan trọng cần nhớ khi debug (không phải hàm, mà là thuộc tính kiến trúc)

```
model.features            # backbone convolution — nơi freeze/unfreeze tác động
model.features[-1]        # block conv CUỐI CÙNG — target layer bắt buộc dùng cho Grad-CAM (Phần 6)
model.avgpool             # Global Average Pooling — KHÔNG có tham số, không cần freeze/unfreeze
model.classifier           # Sequential(Dropout(p=0.3), Linear(1536, num_classes))
model.classifier[1].in_features   # = 1536 với B3 — dùng để dựng lại Linear đúng shape
```

## 3.3. Sơ đồ shape đi qua model

```
input (N,3,224,224)
   │  model.features(...)
   ▼
(N,1536,7,7)              ← feature map cuối, chính là A^k dùng trong Grad-CAM (Phần 6)
   │  model.avgpool(...)
   ▼
(N,1536,1,1)
   │  .flatten(1)   (bên trong forward() gốc của torchvision, không cần tự viết)
   ▼
(N,1536)
   │  model.classifier(...)  = Dropout → Linear(1536,3)
   ▼
(N,3)                      ← logits, CHƯA qua softmax — output cuối cùng của build_classifier()
```

**Input của cả file:** không có (chỉ dựng kiến trúc, chưa có dữ liệu thật đi qua cho tới khi được gọi ở Phần 5). **Output của cả file:** một `nn.Module` sẵn sàng nhận tensor `(N,3,224,224)` từ Dataset ở Phần 2.

---

# PHẦN 4 — GIAI ĐOẠN 4: `src/unet.py` (📝 SKELETON)

**Vai trò:** dựng kiến trúc U-Net (qua thư viện `segmentation_models_pytorch`), định nghĩa loss và metric riêng cho bài toán segmentation.

## 4.1. Bảng hàm/class

| Tên | Kiểu | Input | Output | Vai trò |
|---|---|---|---|---|
| `build_unet(in_channels=3, out_channels=1, pretrained=True, encoder_name="resnet34")` | hàm | 4 tham số có default | `nn.Module` (`smp.Unet(...)`) | Dựng kiến trúc, chữ ký cố định |
| `BCEDiceLoss(bce_weight=0.5, smooth=1.0)` | class kế thừa `nn.Module` | — | — | `.forward(logits, target)` → scalar loss |
| `dice_score(logits, target, thresh=0.5)` | hàm, decorator `@torch.no_grad()` | logits `(N,1,H,W)`, target cùng shape | `float` (Dice trung bình cả batch) | Metric, KHÔNG dùng để backward |
| `iou_score(logits, target, thresh=0.5)` | hàm, `@torch.no_grad()` | như trên | `float` | Metric |

## 4.2. Sơ đồ shape

```
input (N,3,224,224)
   │  build_unet() — encoder ResNet-34 (downsample dần) + decoder (upsample dần + skip concat)
   ▼
logits (N,1,224,224)        ← CHƯA sigmoid — dùng trực tiếp với BCEWithLogitsLoss bên trong BCEDiceLoss

logits ──► BCEDiceLoss(logits, target) ──► loss (scalar)                      [dùng lúc TRAIN]
logits ──► dice_score(logits, target)  ──► dice (float, 0..1)                 [dùng lúc EVAL — không backward]
logits ──► iou_score(logits, target)   ──► iou  (float, 0..1)                 [dùng lúc EVAL]
logits ──► torch.sigmoid(logits) > 0.5 ──► mask nhị phân (N,1,224,224) {0,1}  [dùng lúc INFERENCE thật —
                                                                                 api/inference.py hoặc
                                                                                 shortcut_iou.py]
```

**`BCEDiceLoss.forward(logits, target)` — chi tiết luồng bên trong:**

```
logits, target (N,1,H,W)
   │
   ├──► self.bce(logits, target)                       ──► bce_loss (scalar, BCEWithLogitsLoss built-in)
   │
   └──► probs = sigmoid(logits)
        p = probs.view(N,-1); t = target.view(N,-1)      ──► flatten mỗi ảnh thành vector
        inter = (p*t).sum(1)                              ──► (N,) — giao nhau MỀM (soft) mỗi ảnh
        dice  = (2*inter+smooth)/(p.sum(1)+t.sum(1)+smooth) ──► (N,)
        dice_loss = 1 - dice.mean()                        ──► scalar

return bce_weight*bce_loss + (1-bce_weight)*dice_loss      ──► scalar cuối cùng
```

**Input của cả file:** không có (giống `model.py`, chỉ dựng kiến trúc + hàm tiện ích). **Output của cả file:** `nn.Module` nhận `(N,3,224,224)` từ `ChestXraySegmentationDataset` (Phần 2), cộng 3 hàm loss/metric dùng ở Phần 5.

---

# PHẦN 5 — GIAI ĐOẠN 5–6: `notebooks/train_classifier.ipynb` & `train_unet.ipynb` (📝 SKELETON)

**Vai trò:** đây là nơi **duy nhất** trong dự án thực sự "huấn luyện" — mọi file trước đó chỉ định nghĩa kiến trúc/dữ liệu, chưa có bước cập nhật trọng số nào xảy ra. Notebook chỉ **orchestrate** (gọi hàm theo đúng thứ tự, lặp epoch) — không chứa logic kiến trúc (đã nằm ở `model.py`/`unet.py`).

## 5.1. `train_classifier.ipynb` — bảng biến cấu hình (Cell 2)

| Biến | Giá trị đề xuất | Ý nghĩa |
|---|---|---|
| `SPLIT_DIR` | `"data/split"` | Input dữ liệu |
| `BATCH_SIZE` | `32` | Theo VRAM (xem bảng gợi ý ở `LY_THUYET.md`/`TUTORIAL.md`) |
| `DEVICE` | `"cuda"` nếu có, else `"cpu"` | |
| `CKPT_PATH` | `"weights/best_classifier.pth"` | Nơi lưu checkpoint tốt nhất |
| `LR_HEAD_ONLY` | `1e-3` | LR Pha 1 |
| `LR_LAST_BLOCKS` | `1e-4` | LR Pha 2 |
| `LR_ALL` | `1e-5` | LR Pha 3 |
| `EPOCHS_P1/P2/P3` | `3 / 15 / 5` | Số epoch mỗi pha |
| `PATIENCE` | `5` | Early stopping — số epoch không cải thiện trước khi dừng |

## 5.2. Hàm định nghĩa trong notebook

```
run_epoch(loader, train: bool, optimizer=None) -> (float, float)
```

| | |
|---|---|
| **Input** | `loader` — `DataLoader` (train hoặc val); `train` — bật `model.train()`/`optimizer.step()` hay `model.eval()`/`no_grad()`; `optimizer` — chỉ cần nếu `train=True` |
| **Output** | `(mean_loss, macro_f1)` — cả hai tính trên **toàn bộ** loader đưa vào, không phải 1 batch |
| **Luồng bên trong** | với mỗi batch: `optimizer.zero_grad()` (nếu train) → `autocast()` (mixed precision) → `logits=model(x)` → `loss=criterion(logits,y)` → `scaler.scale(loss).backward()` + `scaler.step(optimizer)` (nếu train) → cộng dồn `loss.item()`, `y`, `argmax(logits)` vào 3 list → cuối cùng `f1_score(ys, ps, average="macro")` |

```
train_phase(phase_name: str, epochs: int, lr: float, best_f1: float) -> float
```

| | |
|---|---|
| **Input** | tên pha (chỉ để in log), số epoch, learning rate của pha đó, `best_f1` hiện tại (để tiếp tục so sánh xuyên suốt 3 pha, không reset về 0 mỗi pha) |
| **Output** | `best_f1` mới nhất sau khi chạy xong pha này (dùng làm input cho pha tiếp theo) |
| **Side-effect quan trọng** | (a) **tạo optimizer MỚI** mỗi lần gọi — bắt buộc vì Pha trước đó vừa đổi `requires_grad`; (b) ghi đè `weights/best_classifier.pth` mỗi khi `val_f1` cải thiện; (c) có thể `break` sớm nếu early-stop |

## 5.3. Sơ đồ luồng gọi 3 pha (đúng thứ tự bắt buộc, không hoán đổi)

```
best_f1 = 0.0
   │
   ▼
freeze_backbone(model)                    ◄── src/model.py
train_phase("head-only", 3, 1e-3, 0.0) ──► best_f1 (pha 1)
   │
   ▼
unfreeze_last_blocks(model, 2)            ◄── src/model.py
train_phase("last-2-blocks", 15, 1e-4, best_f1) ──► best_f1 (pha 2, KẾ THỪA từ pha 1)
   │
   ▼
unfreeze_all(model)                        ◄── src/model.py
train_phase("all", 5, 1e-5, best_f1) ──► best_f1 (pha 3, KẾ THỪA từ pha 2)
   │
   ▼
in "BEST VAL MACRO F1"  +  weights/best_classifier.pth đã ghi (từ epoch tốt nhất, KHÔNG PHẢI epoch cuối)
```

**Điểm hay bị hiểu nhầm:** `weights/best_classifier.pth` sau khi chạy xong **không phải** trọng số ở cuối Pha 3 — mà là trọng số tại **epoch có `val_f1` cao nhất từng thấy qua CẢ 3 pha** (vì `torch.save` chỉ chạy bên trong khối `if va_f1 > best_f1`). Nếu Pha 3 làm F1 giảm (dấu hiệu overfit khi full fine-tune), checkpoint cuối cùng vẫn là bản tốt của Pha 2, không phải bản tệ hơn của Pha 3 — đây chính là lý do 3-pha + early stopping + "chỉ lưu khi tốt hơn" phối hợp với nhau an toàn.

## 5.4. `train_unet.ipynb` — khác biệt so với trên (không lặp lại phần giống)

| | `train_classifier.ipynb` | `train_unet.ipynb` |
|---|---|---|
| Dataset | `ChestXrayClassificationDataset` | `ChestXraySegmentationDataset` |
| Model | `build_classifier()` | `build_unet()` |
| Loss | `nn.CrossEntropyLoss()` | `BCEDiceLoss()` |
| Metric theo dõi | Macro F1 | Dice + IoU (trung bình cộng dồn qua batch) |
| Chiến lược train | **3 pha** freeze/unfreeze | **1 pha duy nhất**, `LR=1e-4`, unfreeze toàn bộ ngay từ đầu (decoder luôn train from scratch nên không có catastrophic forgetting để lo) |
| `BATCH_SIZE` đề xuất | 32 | 16 (U-Net tốn VRAM hơn vì có decoder) |
| Ngưỡng chấp nhận | (tuỳ báo cáo) | **Dice > 0.90** trên val |

**Input của cả Phần 5:** `nn.Module` từ Phần 3/4, `Dataset` từ Phần 2. **Output:** `weights/best_classifier.pth`, `weights/best_unet.pth` — hai file `state_dict` độc lập, cộng biểu đồ loss/F1/Dice lưu vào `figures/`.

---

# PHẦN 6 — GIAI ĐOẠN 7: `src/gradcam.py` (📝 SKELETON)

**Vai trò:** file duy nhất chịu trách nhiệm sinh heatmap Grad-CAM — nhận model classifier **đã train xong** + 1 ảnh, trả về ma trận "mức độ quan trọng" cùng kích thước ảnh.

## 6.1. Bảng hàm

| Hàm | Input | Output |
|---|---|---|
| `_get_target_layer(model)` | `nn.Module` | `model.features[-1]` — layer conv cuối, hardcode cho kiến trúc EfficientNet-B3 |
| `generate_gradcam(model, img_tensor, target_class=None)` | `model`: đã `.eval()`; `img_tensor`: `(3,H,W)` hoặc `(1,3,H,W)`; `target_class`: `int` hoặc `None` | `np.ndarray` shape `(H,W)`, `float32`, giá trị `∈[0,1]` |
| `overlay_heatmap(image_rgb, heatmap, alpha=0.4)` | `image_rgb`: `(H,W,3)` uint8; `heatmap`: `(H,W)` float32 | `(H,W,3)` uint8 — ảnh đã chồng màu jet lên heatmap |

## 6.2. Luồng bên trong `generate_gradcam` — từng bước, khớp công thức Phần VII `LY_THUYET.md`

```
img_tensor (3,H,W) hoặc (1,3,H,W)
   │  if dim==3: unsqueeze(0)
   ▼
img_tensor (1,3,224,224)  ──► .to(device)
   │
   ▼
model.eval()          ⚠️ KHÔNG có torch.no_grad() bao quanh — Grad-CAM CẦN gradient
   │
   ├── nếu target_class is None:
   │      with torch.no_grad():           (chỉ no_grad ở NHÁNH NÀY, để lấy argmax — không cần gradient)
   │          logits = model(img_tensor)  ──► (1,3)
   │          target_class = argmax(logits, dim=1).item()   ──► int
   │
   ▼
target_layer = model.features[-1]
targets = [ClassifierOutputTarget(target_class)]
   │
   ▼
GradCAM(model, [target_layer])(input_tensor=img_tensor, targets=targets)
   │        (thư viện tự làm: forward → lấy A^k tại target_layer → backward logit[target_class]
   │         → GAP gradient ra α_k → Σα_k·A^k → ReLU → resize lên (H,W) → chuẩn hoá [0,1])
   ▼
grayscale_cam (1, H, W)
   │  [0]  — bỏ chiều batch
   ▼
return heatmap (H, W) float32 [0,1]
```

**Input của cả file:** `weights/best_classifier.pth` đã load vào 1 `nn.Module` (từ Phần 5), 1 ảnh tensor từ `Dataset` (Phần 2). **Output:** heatmap 1 kênh — dùng ở Phần 7 (`shortcut_iou.py`) và Phần 8 (`api/inference.py`).

---

# PHẦN 7 — GIAI ĐOẠN 8: `src/shortcut_iou.py` (📝 SKELETON)

**Vai trò:** chạy hàng loạt Grad-CAM trên toàn bộ test set, so khớp với mask phổi (ground-truth hoặc U-Net dự đoán), tổng hợp thống kê IoU theo từng lớp — công cụ kiểm định shortcut learning (Phần VIII `LY_THUYET.md`).

## 7.1. Bảng hàm

| Hàm | Input | Output |
|---|---|---|
| `binarize(x, thresh)` | mảng `float`, ngưỡng | mảng `uint8` `{0,1}` — `(x > thresh)` |
| `iou(a, b)` | 2 mảng cùng shape | `float` — `|a∩b|/|a∪b|`, trả `0.0` nếu union rỗng |
| `load_gt_mask(image_path, mask_dir)` | `Path` ảnh, `Path` thư mục mask | `(H,W)` uint8 `{0,1}` — đọc mask thật, nhị phân hoá `>0` |
| `predict_lung_mask(unet, img_tensor)` | U-Net đã train, ảnh `(3,H,W)` | `(H,W)` uint8 `{0,1}` — `sigmoid(unet(x)) > 0.5` |
| `run_shortcut_analysis(classifier_path, unet_path, test_split_dir, mask_source, gradcam_thresh, device)` | 6 tham số cấu hình | `dict[str, list[float]]` — IoU theo từng lớp, cộng side-effect: in bảng thống kê + lưu histogram vào `figures/` |

## 7.2. Sơ đồ luồng `run_shortcut_analysis` — vòng lặp chính

```
Load classifier (eval) + Load unet (eval)
   │
   ▼
ds = ChestXrayClassificationDataset(test_split_dir, get_val_transforms())
mask_dir = test_split_dir/"masks"
ious_per_class = {c: [] for c in CLASS_TO_IDX}          # {"Normal":[], "Lung_Opacity":[], "COVID":[]}
   │
   ▼
for i in range(len(ds)):                                 # LẶP QUA TOÀN BỘ TEST SET (~1350 ảnh)
     img, label = ds[i]
     path = ds.image_paths[i]
        │
        ▼
     heatmap = generate_gradcam(clf, img, target_class=label)   ⚠️ dùng NHÃN THẬT, không phải model dự đoán
        │                                                          — mục đích: "khi model ĐÚNG, nó nhìn đâu"
        ▼
     cam_bin = binarize(heatmap, gradcam_thresh)          # (224,224) {0,1}
        │
        ├── nếu mask_source=="gt":   lung_mask = load_gt_mask(path, mask_dir)
        └── nếu mask_source=="unet": lung_mask = predict_lung_mask(unet, img)
        │
        ▼
     score = iou(cam_bin, lung_mask)                       # float
     ious_per_class[IDX_TO_CLASS[label]].append(score)     # gom theo lớp
   │
   ▼
in bảng: mean/median/std IoU theo từng lớp
lưu figures/shortcut_iou_<mask_source>_t<thresh>.png (histogram 3 lớp chồng nhau)
return ious_per_class
```

**Input của cả file:** `weights/best_classifier.pth`, `weights/best_unet.pth` (Phần 5), `data/split/test/` (Phần 1.2), `generate_gradcam` (Phần 6). **Output:** số liệu thống kê in ra console + hình lưu `figures/` — dùng trực tiếp cho báo cáo (Phần 17-18 `TUTORIAL.md`), **không** có file nào khác trong pipeline import lại kết quả này (đây là điểm cuối của nhánh "kiểm định", tách khỏi nhánh "serving" ở Phần 8-9).

---

# PHẦN 8 — GIAI ĐOẠN 9: `api/` (📝 SKELETON)

**Vai trò:** expose model đã train qua HTTP — nhận ảnh, trả JSON. **Không train gì** — chỉ load `weights/*.pth` và forward.

## 8.1. Sơ đồ trách nhiệm 4 file

```
api/schemas.py     — CHỈ định nghĩa DỮ LIỆU (Pydantic model), không có logic
api/db.py          — CHỈ đọc/ghi SQLite, không biết gì về model AI
api/inference.py   — TOÀN BỘ logic AI (load model, forward, gradcam, encode ảnh) — "bộ não"
api/main.py        — CHỈ định nghĩa ROUTE HTTP, gọi sang inference.py + db.py — "người điều phối"
```

## 8.2. `api/schemas.py`

```
class PredictResponse(BaseModel):
    predicted_class: str                    # "Normal" | "Lung_Opacity" | "COVID"
    confidence: float                       # 0.0–1.0
    probabilities: dict[str, float]         # {"Normal":0.05, "Lung_Opacity":0.15, "COVID":0.80}
    heatmap_overlay_base64: str             # chuỗi base64 PNG, KHÔNG phải URL
    disclaimer: str                         # luôn cùng 1 câu cố định
```

Không có hàm — chỉ là **hợp đồng dữ liệu (data contract)** giữa `main.py` và client. FastAPI dùng class này để (a) tự validate response đúng shape trước khi trả về, (b) tự sinh docs OpenAPI tại `/docs`.

## 8.3. `api/inference.py` — bảng hàm

| Hàm | Input | Output | Khi nào gọi |
|---|---|---|---|
| `load_models()` | không | `None` (side-effect: gán vào biến module-level `_classifier`) | **1 lần duy nhất**, lúc server khởi động (`main.py` `startup` event) |
| `_overlay_heatmap(image_rgb, heatmap, alpha=0.4)` | `(H,W,3)` uint8, `(H,W)` float32 | `(H,W,3)` uint8 | Nội bộ, gọi bên trong `predict_image` |
| `_encode_png_base64(image_rgb)` | `(H,W,3)` uint8 | `str` (base64) | Nội bộ |
| `predict_image(pil_image)` | `PIL.Image` bất kỳ | `dict` khớp `PredictResponse` | **Mỗi request** — hàm trung tâm của toàn API |

**Biến module-level (trạng thái toàn cục, tồn tại suốt vòng đời server):**

| Biến | Khởi tạo | Vai trò |
|---|---|---|
| `DEVICE` | `"cpu"` (hardcode — vì Hugging Face Spaces free tier) | Thiết bị chạy inference |
| `_classifier` | `None` lúc import, gán thật trong `load_models()` | Model đã load, tái sử dụng mọi request — **không load lại** |
| `_transform` | `get_val_transforms()` (từ `src/dataset.py`) | Transform cố định, không augment |

## 8.4. Sơ đồ luồng `predict_image()` — hàm quan trọng nhất của cả `api/`

```
pil_image (PIL.Image, size bất kỳ, mode bất kỳ)
   │
   ▼
image_np = np.array(pil_image.convert("RGB"))          # (H_gốc, W_gốc, 3) uint8
image_resized = cv2.resize(image_np, (224,224))         # (224,224,3) uint8 — GIỮ LẠI để overlay sau,
   │                                                      #   KHÔNG dùng cho model (model dùng bản qua transform)
   ▼
img_tensor = _transform(image=image_np)["image"]         # (3,224,224) float32, normalize — từ ẢNH GỐC,
   │                                                       #   Albumentations tự resize bên trong transform
   ▼
with torch.no_grad():
    logits = _classifier(img_tensor.unsqueeze(0))         # (1,3)
    probs = logits.softmax(dim=1)[0].cpu().numpy()         # (3,) — vd [0.05, 0.15, 0.80]
   │
   ▼
pred_idx = argmax(probs)                                  # int, vd 2
pred_class = IDX_TO_CLASS[pred_idx]                        # "COVID"
   │
   ▼
heatmap = generate_gradcam(_classifier, img_tensor, target_class=pred_idx)   # (224,224) float32 [0,1]
   │                                        ⚠️ dùng pred_idx (dự đoán) — KHÁC shortcut_iou.py
   │                                          (dùng label thật) vì mục đích khác nhau: đây là
   │                                          "giải thích cho NGƯỜI DÙNG vì sao model kết luận vậy"
   ▼
overlay = _overlay_heatmap(image_resized, heatmap)         # (224,224,3) uint8
   │
   ▼
return {
    "predicted_class": pred_class,
    "confidence": float(probs[pred_idx]),
    "probabilities": {tên_lớp: float(p) cho từng lớp},
    "heatmap_overlay_base64": _encode_png_base64(overlay),
    "disclaimer": "...",
}
```

## 8.5. `api/db.py` — bảng hàm

| Hàm | Input | Output | Side-effect |
|---|---|---|---|
| `init_db()` | không | `None` | Tạo file `data/predictions.db` + bảng `predictions` nếu chưa có (`CREATE TABLE IF NOT EXISTS`) — gọi 1 lần lúc startup |
| `log_prediction(predicted_class, confidence)` | `str`, `float` | `None` | `INSERT` 1 dòng mới, kèm `timestamp` UTC tự sinh |

**Schema bảng `predictions`:** `id` (autoincrement) — `timestamp` (text, ISO 8601) — `predicted_class` (text) — `confidence` (real).

## 8.6. `api/main.py` — route duy nhất

```
FastAPI app
   │
   ├── @app.on_event("startup")
   │      def startup(): load_models(); init_db()          # chạy 1 LẦN khi `uvicorn` khởi động
   │
   └── @app.post("/predict", response_model=PredictResponse)
          def predict(file: UploadFile) -> PredictResponse
```

**Sơ đồ luồng 1 request `POST /predict`:**

```
Client gửi file ảnh (multipart/form-data)
   │
   ▼
if not file.content_type.startswith("image/"): raise HTTPException(400)
   │
   ▼
image_bytes = file.file.read()                    # đọc 1 LẦN — đọc 2 lần sẽ ra rỗng (con trỏ đã ở cuối)
   │
   ▼
try: pil_image = Image.open(BytesIO(image_bytes))
except: raise HTTPException(400, "Không đọc được ảnh")
   │
   ▼
result = predict_image(pil_image)                  # ◄── api/inference.py (mục 8.3-8.4)
   │
   ▼
log_prediction(result["predicted_class"], result["confidence"])   # ◄── api/db.py
   │
   ▼
return result                                       # FastAPI tự serialize theo PredictResponse
```

**Input của cả Phần 8:** ảnh upload qua HTTP (từ `app.py`, Phần 9), `weights/*.pth` (Phần 5). **Output:** JSON `PredictResponse` trả về client + 1 dòng mới trong `data/predictions.db`.

---

# PHẦN 9 — GIAI ĐOẠN 10: `app.py` — Gradio UI (📝 SKELETON)

**Vai trò:** giao diện web, gọi HTTP sang `api/main.py` (kiến trúc client-server tách biệt, xem Phần 1.3 `TUTORIAL.md`) — **không** tự chứa logic AI nào.

## 9.1. Bảng thành phần

| Tên | Kiểu | Vai trò |
|---|---|---|
| `API_URL` | biến toàn cục, `str` | `"http://localhost:8000/predict"` — địa chỉ backend |
| `diagnose(image)` | hàm | **Duy nhất** hàm logic của file — nhận `PIL.Image`, trả `(overlay_image, probs_dict, text)` |
| `demo` | `gr.Blocks` | Định nghĩa layout UI, không chứa logic |

## 9.2. Sơ đồ luồng `diagnose()`

```
image (PIL.Image từ gr.Image, hoặc None nếu chưa upload)
   │
   ├── if image is None: return (None, {}, "Vui lòng upload ảnh...")   ← early return, không gọi API
   │
   ▼
buf = BytesIO(); image.save(buf, format="PNG"); buf.seek(0)     # encode ảnh thành PNG bytes trong RAM
   │
   ▼
response = requests.post(API_URL, files={"file": (...)})        # ◄── HTTP call sang api/main.py (Phần 8.6)
response.raise_for_status()                                      # raise nếu status != 2xx
result = response.json()                                         # dict khớp PredictResponse
   │
   ▼
overlay_bytes = base64.b64decode(result["heatmap_overlay_base64"])
overlay_image = Image.open(BytesIO(overlay_bytes))                # PIL.Image — hiển thị trong gr.Image
   │
   ▼
label_text = f"{predicted_class} ({confidence*100:.1f}%)"
probs = result["probabilities"]                                   # dict[str,float] — đưa thẳng vào gr.Label
   │
   ▼
return (overlay_image, probs, f"{label_text}\n\n{disclaimer}")
```

## 9.3. Sơ đồ khai báo UI (`gr.Blocks`) — ánh xạ input/output component ↔ tham số hàm

```
gr.Blocks:
  ┌─────────────────────────────┬──────────────────────────────────┐
  │ input_image (gr.Image)       │ output_overlay (gr.Image)         │
  │ submit_btn  (gr.Button)      │ output_probs   (gr.Label, top=3)  │
  │                               │ output_text    (gr.Textbox)       │
  └─────────────────────────────┴──────────────────────────────────┘

submit_btn.click(
    fn=diagnose,
    inputs=[input_image],                              # ── tham số `image` của diagnose()
    outputs=[output_overlay, output_probs, output_text] # ── 3 giá trị return của diagnose(), ĐÚNG THỨ TỰ
)
```

**Input của cả file:** ảnh người dùng upload qua trình duyệt. **Output:** hiển thị trực tiếp trên UI — không ghi file, không gọi DB trực tiếp (việc log đã xảy ra ở `api/main.py`, phía backend).

---

# PHẦN 10 — GIAI ĐOẠN 11: `Dockerfile` (📝 SKELETON)

**Vai trò:** đóng gói toàn bộ hệ thống thành 1 image chạy được ở bất kỳ máy nào — không có "hàm"/"biến" theo nghĩa code Python, nhưng có **thứ tự lệnh** quan trọng tương đương.

## 10.1. Sơ đồ các bước build (thứ tự KHÔNG tuỳ ý — tận dụng Docker layer cache)

```
FROM python:3.10-slim                              # base image nhẹ, không cần CUDA (chỉ inference CPU)
   │
   ▼
RUN apt-get install libgl1 libglib2.0-0             # dependency HỆ THỐNG cho opencv-python-headless
   │                                                   (thiếu → ImportError: libGL.so.1 khi `import cv2`)
   ▼
COPY requirements.txt .
RUN pip install -r requirements.txt                 # ĐẶT TRƯỚC copy code — tận dụng cache: sửa code
   │                                                   không làm bước pip install (chậm) chạy lại
   ▼
COPY src/ api/ app.py weights/                       # copy code + trọng số đã train SAU CÙNG
   │
   ▼
EXPOSE 7860                                           # đúng port mặc định gr.Blocks.launch()
CMD ["python", "app.py"]                              # entrypoint — CHỈ chạy Gradio, KHÔNG chạy uvicorn
                                                        # riêng (app.py tự import predict_image trực tiếp
                                                        # trong container, không gọi HTTP nội bộ — khác
                                                        # với lúc dev local chạy 2 terminal)
```

**Input của cả Phần 10:** toàn bộ `src/`, `api/`, `app.py`, `weights/*.pth`, `requirements.txt` (mọi phần trước). **Output:** 1 Docker image, chạy `docker run` → container expose port 7860 → deploy lên Hugging Face Spaces (Phần 14 `TUTORIAL.md`).

---

# PHẦN 11 — BẢNG TỔNG HỢP: MỌI HÀM TRONG DỰ ÁN

Một bảng duy nhất để tra cứu nhanh — không cần lật lại từng phần. Sắp theo đúng thứ tự pipeline (Phần 0).

| # | File | Trạng thái | Hàm/Method | Input → Output |
|---|---|---|---|---|
| 1 | `src/preprocess.py` | ✅ | *(không có hàm — script top-level)* | `COVID-19_Radiography_Dataset/` → `data/processed/` |
| 2 | `src/split_data.py` | ✅ | `copy_pairs(image_paths, mask_dir, split_names)` | `list[Path], Path, str` → `None` (side-effect: copy file) |
| 3 | `src/verify.py` | ✅ | *(không có hàm)* | `data/processed/` → in console |
| 4 | `src/visualize.py` | ✅ | *(không có hàm)* | `data/processed/` → hiển thị matplotlib |
| 5 | `src/dataset.py` | 📝 | `get_train_transforms(image_size)` | `tuple` → `A.Compose` |
| 6 | | 📝 | `get_val_transforms(image_size)` | `tuple` → `A.Compose` |
| 7 | | 📝 | `get_train_transforms_seg(image_size)` | `tuple` → `A.Compose` (có `additional_targets`) |
| 8 | | 📝 | `_parse_label(filename)` | `str` → `int` |
| 9 | | 📝 | `ChestXrayClassificationDataset.__getitem__(idx)` | `int` → `(Tensor(3,224,224), int)` |
| 10 | | 📝 | `ChestXraySegmentationDataset.__getitem__(idx)` | `int` → `(Tensor(3,224,224), Tensor(1,224,224))` |
| 11 | `src/model.py` | 📝 | `build_classifier(num_classes, pretrained)` | `int, bool` → `nn.Module` |
| 12 | | 📝 | `freeze_backbone(model)` | `nn.Module` → `None` |
| 13 | | 📝 | `unfreeze_last_blocks(model, num_blocks)` | `nn.Module, int` → `None` |
| 14 | | 📝 | `unfreeze_all(model)` | `nn.Module` → `None` |
| 15 | | 📝 | `count_trainable_params(model)` | `nn.Module` → `int` |
| 16 | `src/unet.py` | 📝 | `build_unet(in_channels, out_channels, pretrained, encoder_name)` | 4 tham số → `nn.Module` |
| 17 | | 📝 | `BCEDiceLoss.forward(logits, target)` | `Tensor(N,1,H,W)` ×2 → `Tensor` scalar |
| 18 | | 📝 | `dice_score(logits, target, thresh)` | như trên → `float` |
| 19 | | 📝 | `iou_score(logits, target, thresh)` | như trên → `float` |
| 20 | `notebooks/train_classifier.ipynb` | 📝 | `run_epoch(loader, train, optimizer)` | `DataLoader, bool, Optimizer` → `(float, float)` |
| 21 | | 📝 | `train_phase(phase_name, epochs, lr, best_f1)` | `str, int, float, float` → `float` |
| 22 | `notebooks/train_unet.ipynb` | 📝 | *(vòng lặp tương tự `run_epoch`, không tách hàm riêng theo skeleton đề xuất)* | — |
| 23 | `src/gradcam.py` | 📝 | `_get_target_layer(model)` | `nn.Module` → `nn.Module` (`model.features[-1]`) |
| 24 | | 📝 | `generate_gradcam(model, img_tensor, target_class)` | `nn.Module, Tensor, int\|None` → `ndarray(H,W)` |
| 25 | | 📝 | `overlay_heatmap(image_rgb, heatmap, alpha)` | `ndarray(H,W,3), ndarray(H,W), float` → `ndarray(H,W,3)` |
| 26 | `src/shortcut_iou.py` | 📝 | `binarize(x, thresh)` | `ndarray, float` → `ndarray` uint8 |
| 27 | | 📝 | `iou(a, b)` | `ndarray ×2` → `float` |
| 28 | | 📝 | `load_gt_mask(image_path, mask_dir)` | `Path, Path` → `ndarray(H,W)` uint8 |
| 29 | | 📝 | `predict_lung_mask(unet, img_tensor)` | `nn.Module, Tensor` → `ndarray(H,W)` uint8 |
| 30 | | 📝 | `run_shortcut_analysis(...)` | 6 tham số → `dict[str, list[float]]` |
| 31 | `api/inference.py` | 📝 | `load_models()` | `None` → `None` (side-effect: gán `_classifier`) |
| 32 | | 📝 | `_overlay_heatmap(image_rgb, heatmap, alpha)` | như #25 | |
| 33 | | 📝 | `_encode_png_base64(image_rgb)` | `ndarray(H,W,3)` → `str` |
| 34 | | 📝 | `predict_image(pil_image)` | `PIL.Image` → `dict` (khớp `PredictResponse`) |
| 35 | `api/db.py` | 📝 | `init_db()` | `None` → `None` (side-effect: tạo bảng SQLite) |
| 36 | | 📝 | `log_prediction(predicted_class, confidence)` | `str, float` → `None` (side-effect: `INSERT`) |
| 37 | `api/main.py` | 📝 | `startup()` | `None` → `None` (gọi `load_models()` + `init_db()`) |
| 38 | | 📝 | `predict(file)` | `UploadFile` → `PredictResponse` |
| 39 | `app.py` | 📝 | `diagnose(image)` | `PIL.Image\|None` → `(PIL.Image, dict, str)` |

---

# PHẦN 12 — BẢNG TỔNG HỢP: MỌI HẰNG SỐ/BIẾN TOÀN CỤC DÙNG CHUNG

Đây là các giá trị **lặp lại** (định nghĩa độc lập, không import chung) qua nhiều file — điểm dễ gây bug nhất nếu một ngày bạn đổi giá trị ở 1 nơi mà quên đổi nơi khác.

| Tên biến | Giá trị | Xuất hiện ở | Lưu ý đồng bộ |
|---|---|---|---|
| `RANDOM_SEED` | `42` | `preprocess.py` ✅, `split_data.py` ✅ | Mỗi file `set.seed()` **độc lập** — không có module hằng số chung. Notebook train cũng cần tự set lại (`set_seed(42)`, `TUTORIAL.md` Phần 3.3) |
| `CLASSES` | `["COVID", "Lung_Opacity", "Normal"]` | `preprocess.py` ✅, `split_data.py` ✅, `verify.py` ✅, `visualize.py` ✅ | Định nghĩa **lặp lại y hệt** ở 4 file — đổi thứ tự/thêm lớp phải sửa đủ cả 4 |
| `IMAGE_SIZE` | `(224, 224)` | `preprocess.py` ✅, `dataset.py` 📝 | 2 nơi độc lập — `preprocess.py` dùng để resize khi ghi file, `dataset.py` dùng để resize lại trong transform (Albumentations `A.Resize`) dù ảnh trên đĩa đã đúng 224×224 sẵn |
| `CLASS_TO_IDX` / `LABELS` | `{"Normal":0,"Lung_Opacity":1,"COVID":2}` (dataset.py) **≠** `{"Normal":1,"Lung_Opacity":2,"COVID":3}` (preprocess.py) | `dataset.py` 📝 vs `preprocess.py` ✅ | **HAI HỆ ĐÁNH SỐ KHÁC NHAU, CÓ CHỦ ĐÍCH** — `LABELS` mã hoá giá trị pixel mask (0=nền luôn giữ nguyên), `CLASS_TO_IDX` mã hoá index classification (0-based liên tục). Không được nhầm lẫn/gộp hai bảng này |
| `MEAN`, `STD` | ImageNet stats | `dataset.py` 📝 (dùng ở cả train và `api/inference.py` qua `get_val_transforms()`) | Chỉ định nghĩa 1 nơi (`dataset.py`) — mọi file khác **import**, không tự định nghĩa lại (khác với `RANDOM_SEED`/`CLASSES`) |
| `SPLITS` | `{"train":0.7,"val":0.15,"test":0.15}` | `split_data.py` ✅ | Chỉ 1 nơi — không lặp lại |
| `DEVICE` | `"cuda"` nếu có else `"cpu"` (train) / hardcode `"cpu"` (serve) | notebooks 📝, `api/inference.py` 📝 | **Cố ý khác nhau**: lúc train ưu tiên GPU nếu có; lúc serve hardcode CPU vì Hugging Face Spaces free tier không có GPU |
| `CKPT_PATH` | `"weights/best_classifier.pth"` / `"weights/best_unet.pth"` | notebooks 📝 (nơi ghi) → `gradcam.py`/`shortcut_iou.py`/`api/inference.py` 📝 (nơi đọc) | Đường dẫn string lặp lại ở nhiều nơi — không có hằng số chung, dễ gõ sai nếu đổi tên file |

---

# PHẦN 13 — SƠ ĐỒ LUỒNG DỮ LIỆU END-TO-END VỚI SHAPE CỤ THỂ

Toàn bộ hành trình một bức ảnh, từ file PNG thô trên Kaggle tới JSON trả về trình duyệt — mỗi mũi tên ghi rõ **shape/kiểu dữ liệu** và **file/hàm** chịu trách nhiệm.

```
[Kaggle] ảnh PNG bất kỳ size, RGB/L      mask PNG bất kỳ size
   │                                          │
   │ src/preprocess.py (✅, top-level loop)    │
   ▼                                          ▼
(224,224) L, uint8                    (224,224) uint8 {0,1,2,3}
   │  data/processed/<class>/images/*.png     │  data/processed/<class>/masks/*.png
   │                                          │
   │ src/split_data.py :: copy_pairs() (✅)     │
   ▼                                          ▼
data/split/<train|val|test>/images/*.png    data/split/<train|val|test>/masks/*.png
   │                                          │
   │ src/dataset.py :: Dataset.__getitem__() (📝)
   ▼                                          ▼
Tensor (3,224,224) float32, normalized    Tensor (1,224,224) float32 {0.0,1.0}
   │  [Classification path]                   │  [Segmentation path]
   │                                          │
   │ src/model.py :: build_classifier()(📝)    │ src/unet.py :: build_unet()(📝)
   ▼                                          ▼
logits (N,3) float32                       logits (N,1,224,224) float32
   │                                          │
   │ notebooks/train_classifier.ipynb(📝)      │ notebooks/train_unet.ipynb(📝)
   ▼                                          ▼
weights/best_classifier.pth (state_dict)   weights/best_unet.pth (state_dict)
   │                                          │
   └──────────────┬───────────────────────────┘
                   │
                   │ src/gradcam.py :: generate_gradcam(clf, img) (📝)
                   ▼
              heatmap (224,224) float32 [0,1]
                   │
        ┌──────────┴───────────┐
        │                       │
        ▼                       ▼
src/shortcut_iou.py(📝)   api/inference.py :: predict_image() (📝)
IoU(heatmap, lung_mask)    │
→ số liệu báo cáo           │  overlay_heatmap() + encode base64
                             ▼
                    dict {predicted_class, confidence,
                          probabilities, heatmap_overlay_base64,
                          disclaimer}
                             │
                             │ api/main.py :: predict() (📝) — POST /predict
                             ▼
                    JSON response (HTTP)  ──────────────► ghi song song vào
                             │                              data/predictions.db
                             │                              (api/db.py :: log_prediction, 📝)
                             │ app.py :: diagnose() (📝) — requests.post(...)
                             ▼
                    gr.Image(overlay) + gr.Label(probs) + gr.Textbox(text)
                             │
                             ▼
                    NGƯỜI DÙNG nhìn thấy trên trình duyệt (Gradio, port 7860)
```

**Cách dùng sơ đồ này khi debug:** nếu kết quả cuối (UI) sai, lần theo mũi tên **ngược lên** — kiểm tra `dict` trả về từ `predict_image()` đúng chưa (in ra bằng `print`/breakpoint), nếu đúng thì lỗi nằm ở `app.py`; nếu `dict` đã sai, lùi tiếp lên `heatmap`/`logits` — mỗi ô trong sơ đồ là một điểm có thể chèn `print(x.shape, x.dtype, x.min(), x.max())` để cô lập chính xác bước nào bắt đầu sai, đúng tinh thần "sanity check từng bước" mà `TUTORIAL.md` khuyến khích xuyên suốt.

---

*Hết tài liệu. Dùng song song với `docs/LY_THUYET.md` (vì sao thuật toán hoạt động) và `docs/TUTORIAL.md` (cách gõ code chi tiết + giải thích quyết định thiết kế) — ba tài liệu trả lời ba câu hỏi khác nhau cho cùng một dự án.*
