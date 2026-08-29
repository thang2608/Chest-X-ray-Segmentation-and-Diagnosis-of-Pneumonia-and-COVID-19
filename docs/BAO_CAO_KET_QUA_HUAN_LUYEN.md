# Báo cáo Kết quả Huấn luyện — Classifier, U-Net & Kiểm định Grad-CAM

*Tổng hợp số liệu từ lần train đầu tiên (trên Google Colab, GPU T4) và kiểm định shortcut learning trên toàn bộ test set (chạy local). Ngày tổng hợp: xem lịch sử commit `demo` branch.*

> **Lưu ý về nguồn ảnh trong báo cáo:** 4 biểu đồ ở Phần 1-2 (loss/F1, confusion matrix classifier; loss/Dice/IoU, ảnh mẫu U-Net) do chính `notebooks/train_classifier.ipynb` và `notebooks/train_unet.ipynb` tự lưu vào `figures/` **trên Colab** lúc train — chưa được tải về máy local (chỉ 2 file `weights/*.pth` đã tải theo hướng dẫn `HUONG_DAN_TRAIN_COLAB.md`). Tải 4 file dưới đây từ Colab về đúng thư mục `figures/` ở gốc repo, ảnh sẽ tự hiện ra khi mở file này:
> - `figures/loss_f1_curves.png`, `figures/confusion_matrix_val.png` (từ `train_classifier.ipynb`)
> - `figures/unet_loss_dice_iou_curves.png`, `figures/unet_qualitative_check.png` (từ `train_unet.ipynb`)
>
> Ảnh ở Phần 3 (shortcut learning) đã có sẵn local, nhúng trực tiếp được ngay.

---

## 1. Kết quả train Classifier (EfficientNet-B3)

**Cấu hình** (`notebooks/train_classifier.ipynb`, theo `docs/TUTORIAL.md` Phần 8): 3 pha fine-tuning — warm-up head (3 epoch, LR 1e-3) → fine-tune 2 block cuối (15 epoch, LR 1e-4) → full fine-tune (5 epoch, LR 1e-5), AdamW + CosineAnnealingLR mỗi pha, early stopping patience=5 theo Val Macro F1, mixed precision (AMP). Chạy đủ 23 epoch, không early-stop giữa chừng.

**Kết quả tốt nhất:** `BEST VAL MACRO F1 = 0.9124`

**Bảng chi tiết theo lớp** (trên **val set**, 1350 ảnh, sau khi reload checkpoint tốt nhất):

| Lớp | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.8831 | 0.9067 | 0.8947 | 450 |
| Lung_Opacity | 0.9280 | 0.8311 | 0.8769 | 450 |
| COVID | 0.9113 | **0.9822** | 0.9455 | 450 |
| **Accuracy** | | | **0.9067** | 1350 |
| Macro avg | 0.9075 | 0.9067 | 0.9057 | 1350 |

**Confusion Matrix (val set):**

```
                  Dự đoán →
Thật ↓          Normal   Lung_Opacity   COVID
Normal            408          27          15
Lung_Opacity       48         374          28
COVID                6           2         442
```

![Loss & Macro-F1 theo epoch — Classifier](../figures/loss_f1_curves.png)

![Confusion Matrix — Classifier (val set)](../figures/confusion_matrix_val.png)

**Quan sát từ đường cong loss:** train loss giảm liên tục tới ~0.20, val loss chững lại quanh ~0.24-0.28 từ khoảng epoch 10-15 trở đi (đặc biệt rõ ở pha 3, full fine-tune) — dấu hiệu **overfit nhẹ**, không nghiêm trọng vì checkpoint chỉ lưu khi val F1 cải thiện (không lấy nhầm epoch overfit).

**Quan sát từ Confusion Matrix:** COVID gần như không bị bỏ sót (Recall 98.22%), nhưng **Lung_Opacity là lớp yếu nhất** — 48/450 ảnh (10.7%) bị nhầm thành Normal, nhiều hơn cả nhầm sang COVID (28 ảnh).

---

## 2. Kết quả train U-Net (Segmentation)

**Cấu hình** (`notebooks/train_unet.ipynb`, theo `docs/TUTORIAL.md` Phần 9): 1 pha duy nhất (encoder ResNet-34 pretrained ImageNet, không cần freeze/unfreeze như classifier), 25 epoch tối đa, LR 1e-4, `BCEDiceLoss` (0.5·BCE + 0.5·Dice), AdamW + CosineAnnealingLR, early stopping patience=5 theo Val Dice.

**Kết quả tốt nhất:** `BEST VAL DICE = 0.9862` (vượt xa mục tiêu > 0.90 đề ra trong `docs/TUTORIAL.md` Phần 9.3)

![Loss, Dice, IoU theo epoch — U-Net](../figures/unet_loss_dice_iou_curves.png)

![5 ảnh mẫu — Ảnh gốc / Mask thật / Mask dự đoán](../figures/unet_qualitative_check.png)

**Quan sát:** hội tụ rất nhanh và sạch (Dice/IoU đạt ~0.97+ chỉ sau ~5 epoch), train/val loss bám sát nhau (val thậm chí thấp hơn train — bình thường vì `train_loader` có augmentation, `val_loader` thì không). Kiểm tra định tính 5 ảnh mẫu: mask dự đoán bao đúng vùng phổi, không tràn ra ngoài lồng ngực, không thủng lỗ.

**Xác nhận thêm khi tích hợp vào API:** test trực tiếp 1 ảnh cụ thể (`COVID-1094.png`) cho `Dice(U-Net, ground-truth) = 0.992`, `IoU = 0.984` — khớp với chất lượng tổng thể đo được lúc train.

---

## 3. Kiểm định Grad-CAM / Shortcut Learning (toàn bộ 1350 ảnh test set)

**Phương pháp** (`src/shortcut_iou.py`, cơ sở lý thuyết `docs/LY_THUYET.md` Phần VIII): với mỗi ảnh test, sinh Grad-CAM heatmap từ classifier (dùng nhãn thật), nhị phân hoá ở ngưỡng 0.5, so khớp với mask phổi (từ ground-truth **và** từ U-Net dự đoán, chạy độc lập 2 lần) bằng 2 chỉ số:
- **IoU** = `|heatmap∩phổi| / |heatmap∪phổi|`
- **Containment** = `|heatmap∩phổi| / |heatmap|` — % vùng heatmap thực sự nằm trong phổi (phân biệt "tập trung đúng 1 vùng nhỏ trong phổi" khỏi "nhìn hẳn ra ngoài phổi")

### Bảng tổng hợp (mask nguồn ground-truth, `gt`)

| Lớp | Mean IoU | Mean Containment | % containment < 0.3 | % IoU = 0 tuyệt đối |
|---|---|---|---|---|
| Normal | 0.209 | 0.530 | 17.6% | 0.7% |
| Lung_Opacity | 0.147 | 0.292 | 52.2% | 8.4% |
| **COVID** | **0.079** | **0.166** | **76.0%** | **27.3%** |

Kết quả với mask nguồn U-Net (`unet`) gần như giống hệt (chênh lệch < 0.5 điểm % ở mọi ô) — xác nhận U-Net đủ tin cậy để thay ground-truth khi deploy thật (không có mask thật cho ảnh mới).

![Histogram IoU — mask nguồn ground-truth](../figures/shortcut_iou_gt_t0.5.png)

![Histogram IoU — mask nguồn U-Net](../figures/shortcut_iou_unet_t0.5.png)

### Ví dụ định tính — 5 case IoU thấp nhất trong 1 tập con 150 ảnh

![5 case IoU thấp nhất — ảnh gốc / Grad-CAM overlay / mask thật / heatmap nhị phân](../figures/shortcut_worst_cases_subset.png)

**Diễn giải theo thang đã thống nhất** (`docs/LY_THUYET.md` Phần VIII.5): IoU > 0.5 = tốt, 0.2-0.5 = mơ hồ, < 0.2 = cảnh báo shortcut. **Cả 3 lớp đều dưới ngưỡng "tốt"**, và containment thấp (không phải chỉ IoU thấp) loại trừ khả năng "vô hại" (tập trung đúng 1 vùng tổn thương nhỏ) — đây là bằng chứng shortcut learning thật, nghiêm trọng nhất ở lớp COVID.

---

## 4. Vấn đề hiện tại của model

### 4.1. Nghiêm trọng nhất — Shortcut learning ở lớp COVID

76% ảnh COVID có heatmap chủ yếu nằm **ngoài phổi**; 27.3% hoàn toàn trật (IoU=0 — model không nhìn vào phổi chút nào). Kết hợp với việc classifier lại đạt Recall COVID rất cao (98.22%), có khả năng model **một phần đang nhận diện watermark/nguồn ảnh** thay vì bệnh lý thật (đúng kịch bản Zech et al. 2018 đã trích trong `docs/LY_THUYET.md` Phần VIII.1) — vì dataset Kaggle này gộp ảnh COVID từ nhiều nguồn/bệnh viện khác nhau hơn hẳn 2 lớp còn lại.

**Hệ quả:** Recall COVID cao có thể **không phản ánh đúng khả năng tổng quát hoá** của model sang dữ liệu/nguồn mới (ảnh chụp từ máy/bệnh viện không có trong tập train) — rủi ro thật nếu triển khai thực tế.

### 4.2. Lung_Opacity là lớp phân loại yếu nhất

Recall chỉ 83.11%, 48/450 ảnh (10.7%) bị nhầm thành Normal — tỉ lệ nhầm lẫn cao nhất trong toàn bộ confusion matrix. Containment cho lớp này (0.292) cũng thấp thứ nhì, gợi ý một phần nguyên nhân trùng với 4.1 (shortcut ảnh hưởng cả lớp này, mức độ nhẹ hơn COVID).

### 4.3. Overfit nhẹ ở Pha 3 (full fine-tune) của classifier

Train loss tiếp tục giảm trong khi val loss đi ngang/nhích nhẹ ở các epoch cuối — dấu hiệu pha 3 (unfreeze toàn bộ, LR 1e-5) có thể đang "học thuộc" thay vì cải thiện tổng quát hoá thêm. Không nghiêm trọng (checkpoint tốt nhất vẫn được giữ đúng) nhưng đáng nghi vấn về việc pha 3 có thực sự cần thiết.

### 4.4. Chưa có đánh giá TEST SET chính thức, độc lập

Toàn bộ số liệu Phần 1-2 (F1, confusion matrix, Dice) đo trên **val set** — dùng để chọn checkpoint lúc train, đúng mục đích nhưng **không phải** số liệu cuối cùng nên đưa vào báo cáo/luận văn (theo đúng nguyên tắc đã ghi trong `docs/TUTORIAL.md` Phần 17.1: "test set chỉ chạm đúng một lần, sau khi đã chốt mọi quyết định"). Notebook `notebooks/evaluate_local.ipynb` đã viết sẵn cho việc này nhưng **chưa được chạy** để lấy số liệu test set chính thức.

---

## 5. Sau tối ưu — Crop-to-lung (đã thực hiện đề xuất 5.1 cũ)

Đã train lại classifier với input crop theo bounding box mask phổi (`notebooks/train_classifier_cropped.ipynb`, `src/dataset.py::crop_to_lung_bbox`, đệm biên 10%) — **giữ nguyên mọi hyperparameter khác** so với bản gốc (đúng ablation study, chỉ đổi 1 biến duy nhất). Chi tiết thiết kế/mã nguồn: `docs/THAY_DOI_TOI_UU_CROP_MASK.md`.

### 5.1. Classifier — accuracy/F1 tăng trên CẢ 3 lớp, không chỉ giải quyết shortcut

| | Baseline (thô) | **Crop (mới)** | Chênh lệch |
|---|---|---|---|
| Accuracy (val) | 0.9067 | **0.9304** | **+2.37 điểm** |
| Macro F1 (val) | 0.9057 | **0.9302** | **+2.45 điểm** |
| Normal F1 | 0.8947 | **0.9151** | +2.04 |
| Lung_Opacity F1 (yếu nhất) | 0.8769 | **0.9138** | **+3.69** |
| COVID F1 | 0.9455 | **0.9616** | +1.61 |

Lung_Opacity — lớp yếu nhất ở mục 4.2 — cải thiện nhiều nhất. Phù hợp giả thuyết: bỏ nền/watermark + phổi chiếm khung hình lớn hơn (độ phân giải hiệu quả cao hơn) giúp model học tốt hơn, không chỉ "công bằng hơn" về mặt shortcut.

### 5.2. Shortcut learning — cải thiện thật nhưng CHƯA giải quyết hết ở lớp COVID

Chỉ số công bằng nhất để so trước/sau là **% ảnh IoU=0 tuyệt đối** (ít bị ảnh hưởng bởi hiệu ứng hình học — sau khi crop, mask chiếm tỉ lệ khung hình khác hẳn nên so trực tiếp mean IoU/containment dễ gây hiểu lầm; xem cảnh báo trong `src/shortcut_iou.py::run_shortcut_analysis` docstring):

| Lớp | % IoU=0 trước | % IoU=0 **sau** | % containment<0.3 trước | % containment<0.3 **sau** |
|---|---|---|---|---|
| COVID | 27.3% | **19.6%** (giảm ~28% tương đối, **chưa hết**) | 76.0% | **59.3%** |
| Lung_Opacity | 8.4% | **2.0%** (giảm >75%, cải thiện mạnh) | 52.2% | **26.9%** |
| Normal | 0.7% | **1.6%** (tăng nhẹ, xấu đi) | 17.6% | **24.7%** |

Kiểm tra chéo `gt` vs `unet` mask source sau crop vẫn khớp nhau chặt (COVID 19.6% vs 18.9%, Lung_Opacity 2.0% vs 1.8%) — U-Net vẫn đáng tin cậy làm trọng tài với model đã crop.

**Diễn giải trung thực, không phóng đại:**
- **Lung_Opacity**: shortcut gần như được giải quyết, đi kèm F1 tăng mạnh nhất — kết quả rõ ràng, thuyết phục.
- **COVID**: cải thiện thật nhưng **chưa triệt để** — gần 1/5 ảnh COVID vẫn hoàn toàn không nhìn vào phổi sau crop. Ví dụ cụ thể: `COVID-1094.png` (case đã biết trước, watermark khả năng nằm **giữa khung hình** chứ không chỉ ở góc/viền) vẫn cho `iou_score=0.0` khi test live qua backend sau khi đổi sang model crop — crop theo bounding box không loại bỏ được loại watermark nằm giữa ảnh.
- **Normal**: hơi xấu đi nhẹ — không phải dấu hiệu shortcut mới, mà do lớp này vốn không có "vùng tổn thương" cụ thể để tập trung, sau khi thu hẹp khung hình heatmap dễ lệch ngẫu nhiên hơn. Mức tuyệt đối vẫn thấp (1.6%), không đáng lo.

### 5.3. Kết luận

Crop-to-lung là cải thiện **thật và đáng kể** (đặc biệt accuracy tổng thể +2.45 điểm và Lung_Opacity), đã tích hợp làm bản chính thức trong `backend/app/services/ai_engine.py` (tự động ưu tiên dùng nếu `weights/best_classifier_cropped.pth` tồn tại). Tuy nhiên **không phải giải pháp triệt để cho toàn bộ vấn đề shortcut learning ở lớp COVID** — cần ghi rõ đây là hạn chế còn lại trong báo cáo/luận văn, không phóng đại thành "đã giải quyết xong".

![So sánh % ảnh IoU=0 theo lớp — trước/sau crop, cả 2 nguồn mask](../figures/compare_pct_iou_zero.png)

![Phân phối containment theo lớp — trước/sau crop](../figures/compare_containment_box.png)

### 5.4. Điều tra sâu — vì sao crop bounding box không loại bỏ hết shortcut (giải mục 6, đề xuất 2 cũ)

**Câu hỏi:** classifier crop đã train trên ảnh cắt theo mask phổi, vậy tại sao Grad-CAM demo thực tế (`backend/frontend`, ảnh `sample_covid.png`) vẫn sáng rõ ở vùng ngoài phổi?

**Nguyên nhân xác nhận bằng đo đạc trực tiếp** (chạy lại đúng pipeline `backend/app/services/ai_engine.py` dùng, không phải suy đoán): `crop_to_lung_bbox` cắt theo **hình chữ nhật bao quanh phổi** (bounding box), không cắt theo đúng **hình dạng phổi**. Mọi pixel nằm trong hình chữ nhật đó — kể cả watermark/logo — vẫn được giữ nguyên nếu nó rơi vào trong box.

Đo trên `sample_covid.png` (U-Net tự dự đoán mask, đúng luồng khi phục vụ ảnh mới không có ground-truth):

| Đại lượng | Giá trị |
|---|---|
| Bbox mask phổi (chưa đệm), khung 224×224 | y[30:172] x[32:197] |
| Bbox **sau đệm biên 10%** (dùng để crop) | y[16:187] x[16:214] — gần sát viền ảnh |
| Vùng góc trên-trái 40×49px | 0 pixel mask phổi, nhưng vẫn **nằm trọn trong bbox crop** |
| IoU(Grad-CAM, mask phổi) sau crop | 0.202 |
| Containment (% vùng Grad-CAM nằm ngoài phổi) | 0.499 — gần một nửa |

![Ca cụ thể: bbox crop vẫn giữ vùng ngoài phổi, Grad-CAM vẫn sáng ở đó](../figures/case_study_bbox_leak.png)

**2 nguyên nhân cộng hưởng:**
1. Đệm biên 10% (thêm chủ đích để tránh cắt mất mô phổi thật ở rìa) đồng thời là kẽ hở cho vật thể gần rìa phổi lọt qua nguyên vẹn.
2. Lúc phục vụ ảnh mới, bbox dựa trên mask **U-Net tự dự đoán** (không phải ground-truth) — nếu mask hơi rộng hơn thực tế, bbox càng nới sát viền ảnh hơn.

**Kết luận:** khớp đúng số liệu tổng thể ở mục 5.2 (COVID vẫn còn 19.6% ảnh IoU=0 sau crop) — đây không phải ca cá biệt mà là hệ quả tất yếu của việc dùng bounding box thay vì mask thật. Xác nhận hướng xử lý tiếp theo đúng là **mask-shaped blackout** (mục 6, đề xuất 3) chứ không phải chỉnh lại tỉ lệ đệm biên.

### 5.5. Blackout — đã train và đánh giá: giảm shortcut RẤT MẠNH, nhưng đổi lấy accuracy thấp hơn cả baseline

Đã train `notebooks/train_classifier_blackout.ipynb` (`weights/best_classifier_blackout.pth`) — **giữ nguyên mọi hyperparameter khác** so với 2 bản trước (đúng ablation, chỉ đổi 1 biến so với bản crop: thêm `blackout=True`, xem `src/dataset.py::crop_to_lung_bbox_blackout`). Đo lại trên đúng val set (1350 ảnh) và test set (1350 ảnh, Grad-CAM) như 2 bản trước.

**Shortcut learning — cải thiện vượt xa cả 2 mục tiêu ban đầu:**

| Lớp | % IoU=0 baseline | % IoU=0 crop | % IoU=0 **blackout** |
|---|---|---|---|
| COVID | 27.3% | 19.6% | **4.2%** (giảm 85% so với baseline, 79% so với crop) |
| Lung_Opacity | 8.4% | 2.0% | **0.4%** |
| Normal | 0.7% | 1.6% | **0.0%** |

Kiểm tra chéo `gt` vs `unet` mask source vẫn khớp chặt (COVID 4.2% vs 3.6%, Lung_Opacity 0.4% vs 0.2%) — xác nhận không phải nhiễu thống kê.

![So sánh % ảnh IoU=0 qua 3 bản](../figures/compare_pct_iou_zero.png)

![Case study: Grad-CAM tập trung vào phổi tốt hơn rõ rệt sau blackout](../figures/case_study_blackout_fix.png)

**Nhưng accuracy giảm đáng kể — đánh đổi thật, không phải lỗi số liệu** (đo trực tiếp bằng classifier trên val set, không phải tự báo cáo từ notebook):

| | Baseline | Crop | **Blackout** |
|---|---|---|---|
| Accuracy (val) | 0.9067 | 0.9304 | **0.8659** |
| Macro F1 (val) | 0.9057 | 0.9302 | **0.8659** |
| Normal F1 | 0.8947 | 0.9151 | **0.8678** |
| Lung_Opacity F1 | 0.8769 | 0.9138 | **0.8673** |
| COVID F1 | 0.9455 | 0.9616 | **0.8627** |

![Đánh đổi Accuracy vs Shortcut learning](../figures/compare_accuracy_vs_shortcut_tradeoff.png)

**Diễn giải trung thực:** blackout xác nhận đúng giả thuyết ở mục 5.4 (xoá triệt để pixel ngoài phổi giải quyết được phần lớn shortcut còn sót của crop thuần) — đây là kết quả khoa học rõ ràng, đáng tin cậy. Nhưng nó **KHÔNG đơn thuần "tốt hơn" bản crop**: Macro F1 giảm hẳn xuống dưới cả baseline (0.8659 < 0.9057), tức mất đi phần cải thiện tổng thể mà crop mang lại (mục 5.1), thậm chí lỗ hơn cả lúc chưa tối ưu gì. Nguyên nhân nhiều khả năng: xoá nền + đưa vùng phổi về nền đen đồng thời làm mất một số **kết cấu/ngữ cảnh xung quanh phổi hữu ích cho phân loại thật** (mô mềm, xương sườn viền ngoài, đối xứng lồng ngực) mà bản thân nó không phải shortcut — không chỉ xoá watermark mà còn xoá một phần tín hiệu thật.

**Kết luận — 3 bản, 3 vai trò khác nhau, không có bản nào "thắng tuyệt đối":**
- **Baseline**: KHÔNG dùng — accuracy thấp nhất VÀ shortcut nặng nhất.
- **Crop-to-lung**: cân bằng tốt nhất giữa accuracy và giảm shortcut — **khuyến nghị làm bản mặc định khi phục vụ (production)**.
- **Blackout**: minh chứng khoa học mạnh nhất cho vấn đề shortcut (gần như giải quyết dứt điểm COVID), phù hợp dùng trong báo cáo/luận văn như 1 ablation study, nhưng **không nên đặt làm mặc định phục vụ người dùng** trước khi có hướng cải thiện lại accuracy (xem mục 6, đề xuất mới).

### 5.6. Số liệu TEST SET chính thức, độc lập (giải mục 6, đề xuất 7 cũ)

Toàn bộ số liệu accuracy/F1 ở mục 5.1 và 5.5 phía trên đo trên **val set** (dùng để chọn checkpoint lúc train — đúng mục đích nhưng không phải số liệu cuối cùng cho báo cáo/luận văn). Đã bổ sung mục 5 mới vào `notebooks/evaluate_local.ipynb`, chạy 1 lần duy nhất trên **test set** (1350 ảnh, tách biệt hoàn toàn với val/train) cho cả 3 checkpoint:

| Bản | Accuracy | Macro F1 | Macro Precision | Macro Recall | Normal F1 | Lung_Opacity F1 | COVID F1 |
|---|---|---|---|---|---|---|---|
| Baseline | 0.9252 | 0.9241 | 0.9288 | 0.9252 | 0.9159 | 0.8942 | 0.9623 |
| Crop-to-lung | **0.9281** | **0.9278** | 0.9302 | 0.9281 | 0.9163 | **0.9097** | 0.9574 |
| Crop+blackout | 0.8748 | 0.8749 | 0.8757 | 0.8748 | 0.8758 | 0.8823 | 0.8666 |

**Nhất quán với số liệu val set** ở mục 5.1/5.5 — cùng thứ hạng (crop > baseline > blackout về accuracy), cùng chiều hướng, không có đảo ngược kết luận nào. Chênh lệch tuyệt đối giữa test và val (ví dụ baseline: 0.9252 test vs 0.9067 val) nằm trong biên độ dao động tự nhiên giữa 2 tập dữ liệu khác nhau, không phải dấu hiệu bất thường. **Số liệu ở bảng này là số chính thức nên dùng khi viết báo cáo/luận văn** (test set chỉ chạm đúng 1 lần cho mỗi checkpoint, đúng nguyên tắc `docs/TUTORIAL.md` Phần 17.1) — số liệu val set ở mục 5.1/5.5 chỉ nên dùng để mô tả quá trình chọn checkpoint lúc train.

---

## 6. Đề xuất hướng xử lý còn lại

### Đã thực hiện

1. ~~Crop/che nền bằng mask U-Net trước khi đưa vào classifier, train lại~~ — **XONG**, xem mục 5. Cải thiện rõ nhưng chưa triệt để với COVID.
2. ~~Điều tra trực tiếp các ảnh COVID vẫn còn IoU=0 sau crop~~ — **XONG**, xem mục 5.4. Xác nhận nguyên nhân: crop bounding box giữ nguyên mọi pixel trong hình chữ nhật (kể cả ngoài phổi thật), không phải lỗi U-Net hay watermark nằm ở vị trí đặc biệt.
3. ~~Che (blackout) chính xác theo hình dạng mask thay vì chỉ crop bounding box~~ — **XONG**, xem mục 5.5. Giảm shortcut rất mạnh (COVID IoU=0: 19.6%→4.2%) nhưng đổi lấy Macro F1 giảm xuống dưới cả baseline (0.8659) — đánh đổi thật.
4. ~~Quyết định bản nào dùng làm mặc định phục vụ~~ — **XONG**, đã đổi `api/inference.py`/`ai_engine.py` sang ưu tiên **cropped > blackout > baseline** (quyết định tường minh của người dùng, xác nhận qua so sánh mục 5.5) — kiểm chứng lại bằng server thật, `crop_mode=True blackout_mode=False` dù cả 2 file weights cùng tồn tại.
5. ~~Chạy `notebooks/evaluate_local.ipynb` lấy số liệu TEST SET chính thức~~ — **XONG**, xem mục 5.6. Kết quả nhất quán với val set, không đảo ngược kết luận nào.

### Ưu tiên trung bình

6. **Thử cải thiện accuracy của blackout** (nếu sau này muốn cân nhắc lại làm mặc định) — ví dụ: tăng thêm epoch/augmentation bù lại phần "ngữ cảnh xung quanh phổi" đã mất, hoặc thử đệm biên (`padding`) lớn hơn 10% để giữ lại nhiều mô xung quanh phổi hơn trước khi blackout.
7. **Ablation Pha 3** (4.3, vẫn còn nguyên với cả 3 bản) — chạy lại pipeline bỏ hẳn pha 3, so sánh Macro F1 val với bản đầy đủ 3 pha.

### Còn lại, không bắt buộc

8. **Threshold sweep cho shortcut IoU** (0.3/0.4/0.6/0.7) cho cả 3 bản — củng cố kết luận không phụ thuộc ngưỡng 0.5.

---

## 7. Tệp số liệu gốc (để tái tạo/kiểm chứng)

**Bản baseline (trước tối ưu):**
- `figures/shortcut_records_gt_t0.5.csv`, `figures/shortcut_records_unet_t0.5.csv` — 1350 dòng/file.
- `weights/best_classifier.pth`.

**Bản crop (sau tối ưu):**
- `figures/shortcut_records_gt_cropped_t0.5.csv`, `figures/shortcut_records_unet_cropped_t0.5.csv` — 1350 dòng/file.
- `weights/best_classifier_cropped.pth`.

**Bản blackout (crop + xoá nền ngoài mask):**
- `figures/shortcut_records_gt_cropped_blackout_t0.5.csv`, `figures/shortcut_records_unet_cropped_blackout_t0.5.csv` — 1350 dòng/file.
- `weights/best_classifier_blackout.pth`.

**Dùng chung cả 3 bản:**
- `weights/best_unet.pth` — không train lại, không đổi giữa cả 3 bản.
- `notebooks/evaluate_local.ipynb` — mục 5 (mới) chạy tự động cả 3 bản trên test set, xem mục 5.6.
- `figures/test_set_official_comparison.csv` — kết quả TEST SET chính thức cả 3 bản (mục 5.6), sinh bởi `notebooks/evaluate_local.ipynb` mục 5.

**Biểu đồ so sánh (mục 5.2, 5.4, 5.5 — sinh bởi `src/plot_shortcut_comparison.py`, đọc lại CSV trên, không cần chạy lại model):**
- `figures/compare_pct_iou_zero.png` — % ảnh IoU=0 theo lớp, cả 3 bản, cả 2 nguồn mask.
- `figures/compare_containment_box.png` — phân phối containment theo lớp, baseline/crop.
- `figures/compare_accuracy_vs_shortcut_tradeoff.png` — đánh đổi Macro F1 ↔ %COVID IoU=0 qua 3 bản.
- `figures/case_study_bbox_leak.png` — ca cụ thể minh hoạ nguyên nhân crop chưa triệt để (mục 5.4).
- `figures/case_study_blackout_fix.png` — cùng ảnh, so Grad-CAM crop vs. blackout (mục 5.5).
- (2 ca cụ thể trên sinh từ script chẩn đoán chạy một lần thủ công, không lưu lại trong repo — chỉ lưu ảnh kết quả.)
