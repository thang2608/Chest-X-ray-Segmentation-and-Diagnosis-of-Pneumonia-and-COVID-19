# LÝ THUYẾT & TOÁN HỌC NỀN TẢNG — Chest X-ray Segmentation & Diagnosis

*Tài liệu giải thích cặn kẽ bản chất toán học của toàn bộ hệ thống: từ neuron đơn lẻ, qua CNN, Transfer Learning, EfficientNet-B3, U-Net, các hàm mất mát, Grad-CAM, đến các chỉ số đánh giá — kèm ví dụ số cụ thể ở mỗi bước.*

---

## Cách đọc tài liệu này

Đây **không phải** tài liệu hướng dẫn code (đã có `TUTORIAL.md` và `SoTay_ModelLead.md` cho việc đó). Đây là tài liệu **lý thuyết + toán học**, trả lời câu hỏi "vì sao nó hoạt động", "con số này từ đâu ra", "gradient này ảnh hưởng gì".

Với **mọi khái niệm mô hình** (CNN, EfficientNet, U-Net, Grad-CAM...), tài liệu luôn trình bày theo đúng 5 mục cố định mà bạn yêu cầu:

1. **Nó là gì** — định nghĩa, tổng quan, đặt trong bối cảnh dự án.
2. **Nó hoạt động trên nguyên tắc nào** — cơ chế toán học/thuật toán bên trong.
3. **Input là gì** — hình dạng (shape), kiểu dữ liệu, ý nghĩa từng chiều.
4. **Output là gì** — hình dạng, kiểu dữ liệu, ý nghĩa từng chiều.
5. **Output có ý nghĩa gì, ứng dụng ra sao, test như thế nào** — cách diễn giải con số, cách kiểm chứng nó đúng.

Song song với 5 mục trên, mỗi phần đều có:

- **Sơ đồ trực quan (ASCII)** — vì đây là file `.md` thuần văn bản, "visualize" ở đây nghĩa là vẽ bằng ký tự — ma trận, luồng dữ liệu, kiến trúc — được căn chỉnh để bạn hình dung được hình dạng thật của tensor đi qua từng bước.
- **Công thức toán học tường minh** — viết dạng ký hiệu chuẩn (Σ, ∂, ×) kèm giải thích từng ký hiệu là gì.
- **Ví dụ số cụ thể** — không chỉ công thức trừu tượng, mà thay số thật vào, tính ra kết quả thật, để bạn thấy "con số này chạy qua công thức kia ra con số kia".

**Quy ước ký hiệu dùng xuyên suốt tài liệu:**

| Ký hiệu | Ý nghĩa |
|---|---|
| `x` | input (ảnh, vector, tensor) |
| `w`, `W` | trọng số (weight) — scalar hoặc ma trận |
| `b` | bias |
| `z` | tổng có trọng số trước khi qua activation: `z = Wx + b` |
| `a` | activation — output sau khi qua hàm kích hoạt: `a = f(z)` |
| `ŷ` (y-hat) | giá trị model dự đoán |
| `y` | nhãn thật (ground truth) |
| `L` | loss (hàm mất mát) — một số thực đo "model sai bao nhiêu" |
| `∂L/∂w` | đạo hàm riêng của loss theo w — "nếu tăng w một chút, L thay đổi bao nhiêu" |
| `η` (eta) | learning rate — tốc độ học |
| `⊙` | phép nhân element-wise (Hadamard product) |
| `*` | phép convolution (tích chập) |
| `N, C, H, W` | shape chuẩn của tensor ảnh trong PyTorch: Batch size, Channels, Height, Width |

---

## Mục lục

- **Phần 0** — Cách đọc tài liệu này *(ở trên)*
- **Phần I** — Nền tảng toán học của mạng nơ-ron: neuron, MLP, forward pass, loss, backpropagation, gradient descent, optimizer
- **Phần II** — Convolutional Neural Network (CNN): convolution, pooling, feature map, receptive field
- **Phần III** — Transfer Learning & Fine-tuning theo pha
- **Phần IV** — EfficientNet-B3 & Compound Scaling
- **Phần V** — U-Net & bài toán Segmentation
- **Phần VI** — Các hàm mất mát: Cross-Entropy, BCE, Dice Loss
- **Phần VII** — Grad-CAM — giải thích quyết định của model
- **Phần VIII** — Shortcut Learning & kiểm định bằng IoU
- **Phần IX** — Các chỉ số đánh giá mô hình (Evaluation Metrics)
- **Phần X** — Ví dụ số end-to-end xuyên suốt toàn bộ pipeline
- **Phần XI** — Kết nối lý thuyết với pipeline thực tế của dự án
- **Phần XII** — Tổng kết & tài liệu tham khảo

---

# PHẦN I — NỀN TẢNG TOÁN HỌC CỦA MẠNG NƠ-RON

CNN, EfficientNet, U-Net đều chỉ là các cách sắp xếp phức tạp hơn của cùng một viên gạch cơ bản: **neuron nhân tạo**, học bằng cùng một thuật toán: **gradient descent qua backpropagation**. Nếu hiểu chắc phần này, mọi phần sau chỉ là "áp dụng lại nguyên lý cũ cho cấu trúc dữ liệu mới" (ảnh 2D thay vì vector).

## I.1. Neuron nhân tạo là gì

### 1. Nó là gì

Một neuron nhân tạo là một hàm số rất đơn giản: nhận nhiều số đầu vào, nhân mỗi số với một trọng số, cộng lại, cộng thêm một hằng số (bias), rồi đưa qua một hàm phi tuyến gọi là **hàm kích hoạt (activation function)**.

### 2. Nguyên tắc hoạt động

Với input là vector `x = [x₁, x₂, ..., xₙ]`, trọng số `w = [w₁, w₂, ..., wₙ]`, bias `b`:

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b   =   w·x + b     (tích vô hướng + bias)
a = f(z)                                            (activation)
```

`f` thường là một trong các hàm sau:

| Hàm | Công thức | Miền giá trị | Dùng ở đâu trong dự án |
|---|---|---|---|
| ReLU | `f(z) = max(0, z)` | `[0, +∞)` | Hầu hết layer ẩn trong CNN/EfficientNet/U-Net |
| Sigmoid | `f(z) = 1 / (1 + e^(-z))` | `(0, 1)` | Output U-Net (xác suất pixel là phổi), gate trong Squeeze-Excitation |
| Softmax | `f(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)` | `(0,1)`, tổng = 1 | Output EfficientNet-B3 (xác suất 3 lớp bệnh) |
| Swish/SiLU | `f(z) = z · sigmoid(z)` | `(-0.28, +∞)` | Activation mặc định bên trong EfficientNet (MBConv block) |

**Vì sao cần phi tuyến (non-linear)?** Nếu `f` là hàm tuyến tính (ví dụ `f(z)=z`), thì xếp chồng nhiều lớp neuron chỉ tương đương với **một** phép biến đổi tuyến tính duy nhất (tích của nhiều ma trận vẫn là một ma trận) — mạng sâu bao nhiêu lớp cũng không "mạnh" hơn một lớp. Hàm phi tuyến (ReLU, Sigmoid...) là thứ cho phép mạng học được các ranh giới quyết định (decision boundary) cong, phức tạp — cần thiết để phân biệt COVID/Lung Opacity/Normal, vốn không thể tách bằng một đường thẳng trong không gian pixel.

### Ví dụ số

Cho 1 neuron nhận 3 input, `x = [0.5, -0.2, 1.0]`, trọng số `w = [0.8, 0.3, -0.5]`, `b = 0.1`.

```
z = (0.8)(0.5) + (0.3)(-0.2) + (-0.5)(1.0) + 0.1
  = 0.40 - 0.06 - 0.50 + 0.1
  = -0.06

ReLU(z)    = max(0, -0.06)              = 0
Sigmoid(z) = 1 / (1 + e^(0.06))         = 1 / (1 + 1.0618) = 0.4850
```

Ý nghĩa: với ReLU, neuron này "tắt" (output 0) vì tổng có trọng số âm — neuron chỉ "kích hoạt" (truyền tín hiệu tiếp) khi tổ hợp input×weight của nó dương. Đây chính là ý nghĩa trực quan của "một neuron học được một đặc trưng (feature) cụ thể": nó chỉ "sáng" khi input khớp với pattern mà trọng số của nó mã hóa.

## I.2. Mạng nhiều lớp (MLP) — forward pass

### 1. Nó là gì

Multi-Layer Perceptron (MLP) là nhiều neuron xếp thành **lớp (layer)**, nhiều lớp xếp nối tiếp nhau. Mỗi lớp nhận output của lớp trước làm input.

### 2. Nguyên tắc hoạt động — forward pass bằng ma trận

Với một lớp có `n` input và `m` neuron (output), thay vì viết `m` phương trình riêng lẻ, ta gộp thành phép nhân ma trận:

```
z = W x + b

trong đó:
  x  ∈ ℝ^(n×1)   — vector input
  W  ∈ ℝ^(m×n)   — ma trận trọng số, mỗi HÀNG là trọng số của 1 neuron
  b  ∈ ℝ^(m×1)   — vector bias
  z  ∈ ℝ^(m×1)   — vector tổng có trọng số, đưa qua f(.) ra a ∈ ℝ^(m×1)
```

Sơ đồ trực quan cho input 3 chiều → lớp ẩn 2 neuron → lớp output 1 neuron:

```
x₁ ──┬────w11────┐
     │            ▼
x₂ ──┼────w12──►[Σ + b1]──f──► a1 ──┐
     │            ▲                  │
x₃ ──┴────w13────┘                  ├──w21──►[Σ + b]──f──► ŷ
                                      │
x₁ ──────w21'────┐                   │
x₂ ──────w22'────┼──►[Σ + b2]──f──► a2 ──w22──┘
x₃ ──────w23'────┘
```

Mạng gồm `L` lớp thì forward pass là một chuỗi phép biến đổi lặp lại:

```
a⁽⁰⁾ = x                                (input là "activation lớp 0")
z⁽ˡ⁾ = W⁽ˡ⁾ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾     với l = 1..L
a⁽ˡ⁾ = f⁽ˡ⁾(z⁽ˡ⁾)
ŷ    = a⁽ᴸ⁾
```

### 3. Input là gì

Một vector (hoặc tensor được "duỗi phẳng" — flatten — thành vector) chứa các đặc trưng số hóa của mẫu dữ liệu. Trong CNN sau này, input của các lớp fully-connected cuối cùng là vector đặc trưng đã được trích xuất qua các lớp convolution, **không phải** ảnh thô.

### 4. Output là gì

Một vector số thực, kích thước bằng số neuron ở lớp cuối. Ý nghĩa của vector này phụ thuộc bài toán:

- Bài toán hồi quy: output là chính giá trị cần dự đoán.
- Bài toán phân loại: output là **logits** (điểm số chưa chuẩn hóa), sau đó đưa qua Softmax để thành xác suất.

### 5. Ý nghĩa, ứng dụng, cách test

Chạy forward pass với input đã biết trước kết quả tay (như ví dụ số dưới) rồi so sánh với code — đây chính là cách "unit test" một layer bằng tay trước khi tin tưởng cả mạng lớn.

### Ví dụ số — forward pass 2 lớp

Input: `x = [1.0, 2.0]`. Lớp ẩn 2 neuron, ReLU. Lớp output 1 neuron, Sigmoid (bài toán phân loại nhị phân).

```
W⁽¹⁾ = [[0.1, 0.2],      b⁽¹⁾ = [0.0, 0.1]
        [0.3, -0.1]]

z⁽¹⁾₁ = 0.1(1.0) + 0.2(2.0) + 0.0 = 0.5     → a⁽¹⁾₁ = ReLU(0.5)  = 0.5
z⁽¹⁾₂ = 0.3(1.0) + (-0.1)(2.0) + 0.1 = 0.2  → a⁽¹⁾₂ = ReLU(0.2)  = 0.2

W⁽²⁾ = [0.5, -0.4],   b⁽²⁾ = 0.05

z⁽²⁾ = 0.5(0.5) + (-0.4)(0.2) + 0.05 = 0.25 - 0.08 + 0.05 = 0.22
ŷ = Sigmoid(0.22) = 1/(1+e^-0.22) = 1/(1+0.8025) = 0.5548
```

Nếu nhãn thật `y = 1` (lớp dương), model đang dự đoán xác suất 55.48% — chưa tệ nhưng chưa tự tin. Phần tiếp theo (Loss + Backpropagation) sẽ dùng chính con số `0.5548` này để tính "sai bao nhiêu" và cập nhật `W⁽¹⁾, W⁽²⁾` theo hướng làm giảm sai số.

## I.3. Hàm mất mát (Loss Function) — đo "model sai bao nhiêu"

### 1. Nó là gì

Loss là một hàm số nhận `(ŷ, y)` — dự đoán và nhãn thật — trả về **một số thực không âm**, càng lớn nghĩa là model dự đoán càng sai. Toàn bộ quá trình "học" của mạng nơ-ron chỉ là bài toán tối ưu: tìm `W, b` sao cho `L` trung bình trên tập train là nhỏ nhất.

### 2. Nguyên tắc — Mean Squared Error (ví dụ đơn giản nhất để xây trực giác)

```
L = (1/N) Σᵢ (ŷᵢ - yᵢ)²
```

Bình phương đảm bảo lỗi luôn dương (âm hay dương đều bị phạt) và phạt nặng hơn khi lỗi lớn (lỗi gấp đôi → phạt gấp bốn). Với bài toán phân loại của dự án, hàm dùng thật là **Cross-Entropy** (trình bày chi tiết ở Phần VI) — nhưng vai trò của nó y hệt: một con số duy nhất tóm tắt "cả mạng đang sai bao nhiêu, tính trung bình trên batch".

### 3. Input là gì

Cặp `(ŷ, y)`: output model (thường là xác suất hoặc logits, shape phụ thuộc bài toán) và nhãn thật (số nguyên class-index cho phân loại, hoặc mask nhị phân cho segmentation).

### 4. Output là gì

Một **scalar** (một số thực duy nhất) — dù batch có hàng trăm ảnh, loss cuối cùng luôn được rút gọn (thường lấy trung bình — `reduction='mean'`) về một con số để có thể lấy đạo hàm và tối ưu.

### 5. Ý nghĩa, ứng dụng, cách test

Loss giảm dần qua các epoch là dấu hiệu model đang học. Loss trên tập train giảm nhưng loss trên tập validation tăng lên là dấu hiệu **overfitting** (model học thuộc lòng train set thay vì học pattern tổng quát) — đây là lý do mọi vòng lặp train trong dự án đều theo dõi song song `train_loss` và `val_loss`.

## I.4. Gradient Descent & Backpropagation — trái tim của việc "học"

### 1. Nó là gì

**Gradient Descent** là thuật toán tối ưu: để giảm `L`, ta cập nhật từng trọng số theo hướng ngược với đạo hàm riêng của `L` theo trọng số đó. **Backpropagation** là thuật toán *tính* các đạo hàm riêng đó một cách hiệu quả cho toàn bộ mạng nhiều lớp, dựa trên **quy tắc chuỗi (chain rule)** của giải tích.

### 2. Nguyên tắc hoạt động

**Bước 1 — Trực giác đạo hàm.** `∂L/∂w` cho biết: nếu tăng `w` một lượng cực nhỏ, `L` thay đổi bao nhiêu (và theo hướng nào). Nếu `∂L/∂w > 0` (tăng w làm L tăng) → ta nên **giảm** w. Nếu `∂L/∂w < 0` → ta nên **tăng** w. Công thức cập nhật:

```
w ← w - η · (∂L/∂w)
```

`η` (learning rate) là bước nhảy — quá lớn thì "nhảy" qua điểm tối ưu (loss dao động, không hội tụ, có thể NaN); quá nhỏ thì học rất chậm.

Sơ đồ trực quan (mặt cắt 1 chiều của bề mặt loss theo 1 trọng số):

```
 L(w)
  │      ╲                                    ╱
  │       ╲                                  ╱
  │        ╲          gradient âm ──►       ╱
  │         ╲       (dốc xuống bên phải)  ╱
  │          ╲                          ╱
  │           ╲                       ╱
  │            ╲                    ╱
  │             ╲___minimum___╱
  └──────────────────┼──────────────────► w
                    w* (điểm tối ưu, ∂L/∂w = 0)
```

Nếu ta đang ở điểm bên trái đáy: đạo hàm âm (hàm đang giảm khi w tăng) → công thức `w ← w - η·(âm)` = `w + η·|đạo hàm|` → w tăng, tiến về đáy. Đúng hướng.

**Bước 2 — Vì sao cần backpropagation (không tính đạo hàm trực tiếp)?** Mạng có hàng triệu trọng số qua hàng chục lớp lồng nhau (`L = loss(f_L(f_{L-1}(...f_1(x)...))`). Tính trực tiếp `∂L/∂w` cho một trọng số ở lớp đầu tiên đòi hỏi đi qua đạo hàm của **toàn bộ** các lớp phía sau nó. Backpropagation giải quyết việc này bằng cách áp dụng **chain rule** và tính toán từ lớp cuối lùi về lớp đầu, **tái sử dụng** kết quả trung gian — mỗi trọng số chỉ cần tính đạo hàm cục bộ một lần, ghép lại bằng phép nhân.

**Chain rule cho một mạng 2 lớp** (khớp với ví dụ số ở mục I.2):

```
L phụ thuộc ŷ phụ thuộc z⁽²⁾ phụ thuộc a⁽¹⁾ phụ thuộc z⁽¹⁾ phụ thuộc W⁽¹⁾

∂L/∂W⁽¹⁾  =  (∂L/∂ŷ) · (∂ŷ/∂z⁽²⁾) · (∂z⁽²⁾/∂a⁽¹⁾) · (∂a⁽¹⁾/∂z⁽¹⁾) · (∂z⁽¹⁾/∂W⁽¹⁾)
              └──────────────┬──────────────┘
               "gradient lan truyền ngược từ loss"
```

Mỗi số hạng trong tích trên là một đạo hàm **cục bộ**, dễ tính:

| Đạo hàm cục bộ | Công thức |
|---|---|
| `∂L/∂ŷ` (với L = MSE) | `2(ŷ - y)` |
| `∂ŷ/∂z⁽²⁾` (với ŷ = Sigmoid(z)) | `ŷ(1-ŷ)` |
| `∂z⁽²⁾/∂a⁽¹⁾` | `W⁽²⁾` (chính là trọng số lớp sau) |
| `∂a⁽¹⁾/∂z⁽¹⁾` (với a = ReLU(z)) | `1` nếu `z>0`, else `0` |
| `∂z⁽¹⁾/∂W⁽¹⁾` | `x` (input của lớp đó) |

### Ví dụ số — backward pass đầy đủ, nối tiếp ví dụ forward ở mục I.2

Nhắc lại forward: `x=[1,2]`, `a⁽¹⁾=[0.5, 0.2]`, `z⁽²⁾=0.22`, `ŷ=0.5548`. Giả sử nhãn thật `y=1`, dùng MSE cho dễ tính tay: `L = (ŷ-y)²`.

**Bước ngược 1 — gradient tại output:**

```
∂L/∂ŷ = 2(ŷ - y) = 2(0.5548 - 1) = 2(-0.4452) = -0.8904
```

**Bước ngược 2 — qua Sigmoid:**

```
∂ŷ/∂z⁽²⁾ = ŷ(1-ŷ) = 0.5548 × 0.4452 = 0.2470
δ⁽²⁾ = (∂L/∂ŷ)(∂ŷ/∂z⁽²⁾) = -0.8904 × 0.2470 = -0.2199
```

`δ⁽²⁾` ("delta" lớp 2) là gradient của L theo `z⁽²⁾` — đại lượng trung tâm mà backprop lan truyền ngược qua từng lớp.

**Bước ngược 3 — gradient cho trọng số lớp 2** (`z⁽²⁾ = W⁽²⁾·a⁽¹⁾ + b⁽²⁾` nên `∂z⁽²⁾/∂W⁽²⁾ᵢ = a⁽¹⁾ᵢ`):

```
∂L/∂W⁽²⁾₁ = δ⁽²⁾ × a⁽¹⁾₁ = -0.2199 × 0.5 = -0.1100
∂L/∂W⁽²⁾₂ = δ⁽²⁾ × a⁽¹⁾₂ = -0.2199 × 0.2 = -0.0440
∂L/∂b⁽²⁾  = δ⁽²⁾ × 1     = -0.2199
```

**Bước ngược 4 — lan gradient về lớp 1** (nhân với trọng số lớp 2, vì `z⁽²⁾` phụ thuộc `a⁽¹⁾` qua `W⁽²⁾`):

```
∂L/∂a⁽¹⁾₁ = δ⁽²⁾ × W⁽²⁾₁ = -0.2199 × 0.5  = -0.1100
∂L/∂a⁽¹⁾₂ = δ⁽²⁾ × W⁽²⁾₂ = -0.2199 × (-0.4) = 0.0880
```

**Bước ngược 5 — qua ReLU** (đạo hàm = 1 nếu z>0, cả hai `z⁽¹⁾₁=0.5>0` và `z⁽¹⁾₂=0.2>0` nên đạo hàm đều = 1, gradient đi qua nguyên vẹn):

```
δ⁽¹⁾₁ = -0.1100 × 1 = -0.1100
δ⁽¹⁾₂ = 0.0880 × 1  = 0.0880
```

**Bước ngược 6 — gradient cho trọng số lớp 1** (`∂z⁽¹⁾/∂W⁽¹⁾ᵢⱼ = xⱼ`):

```
∂L/∂W⁽¹⁾₁₁ = δ⁽¹⁾₁ × x₁ = -0.1100 × 1.0 = -0.1100
∂L/∂W⁽¹⁾₁₂ = δ⁽¹⁾₁ × x₂ = -0.1100 × 2.0 = -0.2200
∂L/∂W⁽¹⁾₂₁ = δ⁽¹⁾₂ × x₁ =  0.0880 × 1.0 =  0.0880
∂L/∂W⁽¹⁾₂₂ = δ⁽¹⁾₂ × x₂ =  0.0880 × 2.0 =  0.1760
```

**Bước cập nhật trọng số** (gradient descent, `η = 0.1`):

```
W⁽²⁾₁_mới = 0.5  - 0.1×(-0.1100) = 0.5110
W⁽²⁾₂_mới = -0.4 - 0.1×(-0.0440) = -0.3956
W⁽¹⁾₁₁_mới = 0.1 - 0.1×(-0.1100) = 0.1110
W⁽¹⁾₁₂_mới = 0.2 - 0.1×(-0.2200) = 0.2220
```

Ý nghĩa: sau **một bước** cập nhật, mọi trọng số nhích một chút theo hướng làm `L` giảm. Lặp lại quy trình forward → loss → backward → update này **hàng nghìn lần** trên hàng nghìn ảnh (mỗi lần một batch) chính là toàn bộ nội dung của "huấn luyện mô hình" trong `train_classifier.ipynb` và `train_unet.ipynb` — không có gì bí ẩn hơn phép tính tay ở trên, chỉ là quy mô lớn hơn hàng triệu lần và được PyTorch tự động hoá qua `loss.backward()`.

**Điều `loss.backward()` của PyTorch làm chính xác:** xây một đồ thị tính toán (computational graph) ghi lại mọi phép toán từ input đến loss lúc forward, sau đó duyệt ngược đồ thị này áp dụng chain rule tự động (autograd) — đúng như 6 bước tay ở trên nhưng cho hàng triệu tham số, kết quả lưu vào thuộc tính `.grad` của từng tensor có `requires_grad=True`.

## I.5. Optimizer — SGD, Momentum, Adam/AdamW

### 1. Nó là gì

Optimizer là thuật toán quyết định **cách** dùng gradient đã tính được để cập nhật trọng số — không chỉ đơn thuần `w ← w - η·∇L` (đó là dạng cơ bản nhất, gọi là **SGD thuần**).

### 2. Nguyên tắc hoạt động

**SGD (Stochastic Gradient Descent).** "Stochastic" vì gradient được tính trên một **batch** nhỏ (ngẫu nhiên lấy mẫu từ tập train), không phải trên toàn bộ dữ liệu — ước lượng nhiễu nhưng nhanh hơn rất nhiều so với tính gradient chính xác trên cả triệu ảnh mỗi bước.

```
w ← w - η · ∇L
```

**Vấn đề của SGD thuần:** dao động mạnh khi bề mặt loss có hình "thung lũng hẹp" (gradient theo một chiều lớn hơn nhiều chiều kia) — bước nhảy zig-zag, hội tụ chậm.

**Momentum** — thêm "quán tính": tích lũy trung bình động (exponential moving average) của các gradient trước đó, giúp dao động bị triệt tiêu (các gradient ngược chiều nhau qua các bước tự khử) còn hướng đi nhất quán được khuếch đại.

```
v ← β·v + (1-β)·∇L         (β thường = 0.9)
w ← w - η·v
```

**Adam (Adaptive Moment Estimation)** — kết hợp Momentum (moment bậc 1 — trung bình gradient) **và** một cơ chế **tự động điều chỉnh learning rate cho từng tham số riêng lẻ** dựa trên moment bậc 2 (trung bình bình phương gradient — ước lượng "độ nhiễu"/"độ lớn" của gradient tham số đó):

```
m ← β₁·m + (1-β₁)·∇L                    (trung bình động của gradient — "hướng đi trung bình")
v ← β₂·v + (1-β₂)·∇L²                   (trung bình động của gradient² — "mức độ biến thiên")
m̂ = m / (1-β₁ᵗ)                          (hiệu chỉnh bias — vì m,v khởi tạo 0 nên lúc đầu bị lệch thấp)
v̂ = v / (1-β₂ᵗ)
w ← w - η · m̂ / (√v̂ + ε)                (ε ~ 1e-8, tránh chia 0)
```

Trực giác: nếu một tham số có gradient **luôn lớn và ổn định hướng** (ví dụ luôn dương) → `m̂` lớn, hội tụ nhanh theo hướng đó. Nếu gradient của một tham số **dao động thất thường** (lúc dương lúc âm, biên độ lớn) → `v̂` lớn → mẫu số `√v̂` lớn → bước cập nhật bị "phanh lại" tự động, tránh làm mất ổn định các trọng số nhạy cảm.

**AdamW** (dùng trong dự án, xem `TUTORIAL.md` Phần 6) khác Adam ở cách áp dụng **weight decay** (L2 regularization — phạt trọng số lớn để chống overfit): Adam gốc cộng weight decay *vào gradient* trước khi tính `m, v` (làm nó bị ảnh hưởng bởi cơ chế điều chỉnh thích ứng theo tham số — không nhất quán về mặt toán học); AdamW áp dụng weight decay **tách biệt, trực tiếp lên trọng số** ở bước cập nhật cuối:

```
w ← w - η·(m̂/(√v̂+ε) + λ·w)         (λ = weight decay coefficient)
```

Đây là lý do các paper hiện đại (kể cả EfficientNet, U-Net fine-tune) đều khuyến nghị AdamW thay vì Adam.

### 3-4. Input/Output

Input của optimizer: gradient hiện tại (`p.grad` — PyTorch tự tính qua `.backward()`) và trạng thái nội bộ (`m, v` được optimizer tự lưu giữa các bước — đây là lý do `optimizer.zero_grad()` phải gọi mỗi bước để xoá gradient cũ, nhưng **không** xoá `m, v`, vì chính `m, v` là "trí nhớ" các bước trước mà optimizer cố tình giữ lại).

Output: giá trị trọng số mới, ghi đè trực tiếp lên tensor trọng số (in-place update).

### 5. Ý nghĩa, ứng dụng, cách test

**Vì sao pha 1 (warm-up head) trong `TUTORIAL.md` dùng LR = 1e-3 còn pha 2 (fine-tune backbone) dùng LR = 1e-4?** Vì trọng số backbone đã pretrained trên ImageNet — đã ở gần một điểm tốt trong không gian trọng số; bước nhảy lớn (LR cao) có nguy cơ nhảy ra khỏi vùng tốt đó (catastrophic forgetting, xem Phần III). Trọng số head (Linear cuối) đang khởi tạo ngẫu nhiên, ở rất xa điểm tối ưu, cần bước nhảy lớn để tiến nhanh.

Cách kiểm chứng optimizer đang hoạt động đúng: theo dõi `train_loss` — nếu giảm dần đều là bình thường; nếu **NaN** ngay từ epoch đầu, gần như luôn là do LR quá cao (bước nhảy quá lớn làm trọng số "văng" ra vùng số học không ổn định, thường xảy ra ngay khi vừa unfreeze pha 2/3 mà quên hạ LR — đúng như gotcha đã ghi trong `TUTORIAL.md` Phần 6.5).

## I.6. Learning Rate Scheduling — vì sao `η` không nên cố định suốt quá trình train

### 1. Nó là gì

Learning Rate Scheduler là một cơ chế **thay đổi `η` theo thời gian** (theo epoch hoặc theo step) thay vì giữ nguyên một giá trị từ đầu tới cuối. Optimizer (Phần I.5) quyết định *cách dùng* gradient; scheduler quyết định *độ lớn bước nhảy* đó thay đổi ra sao qua thời gian.

### 2. Nguyên tắc hoạt động

**Vì sao cần đổi LR theo thời gian?** Đầu quá trình train, trọng số còn xa điểm tối ưu — cần bước nhảy lớn để tiến nhanh. Càng gần cuối, trọng số đã gần điểm tối ưu (đáy "thung lũng" loss, Phần I.4) — bước nhảy lớn lúc này dễ khiến trọng số "nhảy qua" đáy rồi dao động qua lại mãi không hội tụ chính xác. Giảm dần `η` theo thời gian giải quyết mâu thuẫn "cần nhanh lúc đầu, cần chính xác lúc cuối" này.

```
 L(w)                                          η lớn ban đầu: bước nhảy dài,
  │      ╲                    ╱                tiến nhanh nhưng dễ vọt qua đáy
  │       ╲    ★1            ╱
  │        ╲  (bước nhảy dài) ╱
  │         ╲★2         ★3  ╱     η nhỏ dần: bước nhảy ngắn lại,
  │          ╲   ●───●──●══╱      hội tụ SÁT đáy hơn thay vì dao động quanh nó
  │           ╲__★4_★5__╱
  └──────────────────┼──────────────────► w
                    w* (điểm tối ưu)
```

**Ba lược đồ phổ biến (dùng được cho `train_classifier.ipynb`/`train_unet.ipynb`):**

- **Step Decay** — nhân `η` với một hệ số `γ<1` (ví dụ 0.1) sau mỗi `step_size` epoch cố định:
  ```
  η(epoch) = η₀ × γ^⌊epoch/step_size⌋
  ```
- **Cosine Annealing** — giảm mượt theo hình nửa đường cong cosine từ `η_max` xuống `η_min` qua `T` epoch, không giật cục như step decay:
  ```
  η(t) = η_min + 0.5×(η_max - η_min)×(1 + cos(t·π/T))
  ```
- **ReduceLROnPlateau** — không theo lịch cố định, mà **theo dõi** `val_loss`: nếu `val_loss` không cải thiện sau `patience` epoch liên tiếp, tự động nhân `η` với một hệ số (ví dụ 0.5). Phù hợp nhất khi không biết trước train bao nhiêu epoch là đủ — để chính dữ liệu val "quyết định" khi nào cần giảm tốc.
- **Warm-up** — ngược lại, **tăng dần** `η` từ 0 lên giá trị mục tiêu trong vài trăm step đầu tiên, trước khi áp dụng lược đồ giảm dần. Hữu ích khi bắt đầu unfreeze một pha mới (Phần III.2): gradient ở bước đầu tiên ngay sau unfreeze thường nhiễu/lớn bất thường vì optimizer (AdamW) chưa kịp tích lũy đủ `m, v` ổn định (Phần I.5) — bắt đầu bằng `η` rất nhỏ rồi tăng dần tránh cú sốc này.

### Ví dụ số — Cosine Annealing qua 10 epoch

`η_max = 1×10⁻³`, `η_min = 1×10⁻⁶`, `T = 10`. Tính `η` tại epoch `t=5` (giữa lịch trình):

```
η(5) = 1e-6 + 0.5×(1e-3 - 1e-6)×(1 + cos(5×π/10))
     = 1e-6 + 0.5×(0.000999)×(1 + cos(π/2))
     = 1e-6 + 0.0004995×(1 + 0)
     = 1e-6 + 0.0004995
     = 0.0005005          (≈ 5.0×10⁻⁴ — đúng bằng nửa η_max, vì cos(π/2)=0, hợp lý vì t=5 là điểm giữa)
```

Tại `t=0`: `cos(0)=1` → `η(0) = 1e-6 + 0.0004995×2 = 0.001` (≈`η_max`, đúng điểm xuất phát). Tại `t=10`: `cos(π)=-1` → `η(10) = 1e-6 + 0.0004995×0 = 1e-6` (≈`η_min`, đúng điểm kết thúc).

### 3-4-5. Input/Output/Ứng dụng

**Input:** epoch/step hiện tại (và với `ReduceLROnPlateau`: giá trị `val_loss` mới nhất). **Output:** giá trị `η` mới, ghi vào `optimizer.param_groups[i]['lr']`. **Ứng dụng:** kết hợp trực tiếp với chiến lược 3 pha ở Phần III.2 — mỗi pha có thể có scheduler riêng (ví dụ Cosine trong mỗi pha, reset lại khi chuyển pha) — **cách test:** in `optimizer.param_groups[0]['lr']` sau mỗi epoch, vẽ đường LR theo epoch, phải khớp hình dạng lược đồ đã chọn (đường thẳng bậc thang cho Step Decay, đường cong mượt cho Cosine).

## I.7. Vanishing/Exploding Gradient — vì sao mạng sâu khó train hơn mạng nông

### 1. Nó là gì

Đây là hiện tượng gradient lan truyền ngược qua backpropagation (Phần I.4) trở nên **cực nhỏ** (vanishing — gần 0, trọng số ở các lớp đầu gần như không cập nhật, mạng "ngừng học" ở phần nông) hoặc **cực lớn** (exploding — trọng số nhảy vọt hỗn loạn, loss thành NaN) khi mạng có **nhiều lớp**.

### 2. Nguyên tắc hoạt động — vì sao nó xảy ra

Nhắc lại chain rule (Phần I.4): gradient tại lớp `1` của mạng `L` lớp là **tích** của `L` số hạng liên tiếp:

```
∂L/∂W⁽¹⁾  =  δ⁽ᴸ⁾ · W⁽ᴸ⁾ · f'⁽ᴸ⁻¹⁾ · W⁽ᴸ⁻¹⁾ · f'⁽ᴸ⁻²⁾ · ... · W⁽²⁾ · f'⁽¹⁾ · x
```

Nếu **mỗi** số hạng trong tích này có độ lớn trung bình `< 1`, tích của hàng chục số hạng như vậy tiến rất nhanh về **0** (vanishing). Nếu mỗi số hạng `> 1`, tích tiến rất nhanh về **vô cực** (exploding) — đây thuần túy là hệ quả của phép nhân lặp lại nhiều lần một số `<1` hoặc `>1` (giống `0.5¹⁰ ≈ 0.001` so với `1.5¹⁰ ≈ 57.7`).

**Vì sao Sigmoid gây vanishing gradient nghiêm trọng:** đạo hàm `Sigmoid'(z) = ŷ(1-ŷ)` đạt giá trị **lớn nhất** `0.25` tại `z=0`, và tiến về `0` khi `|z|` lớn (hai đầu hàm Sigmoid gần như phẳng — xem Phần I.1). Một mạng `Sigmoid` sâu 5 lớp, dù ở điều kiện tốt nhất (`z=0` mọi lớp), gradient vẫn bị nhân với `0.25` năm lần:

```
0.25⁵ = 0.0009765625   (gradient ở lớp 1 chỉ còn ~0.1% so với gradient gốc ở lớp 5)
```

Đây là lý do lịch sử vì sao mạng sâu dùng Sigmoid/Tanh rất khó train trước khi ReLU trở thành lựa chọn mặc định.

**Vì sao ReLU giảm nhẹ vấn đề (nhưng không loại bỏ hoàn toàn):** `ReLU'(z) = 1` nếu `z>0` (không làm suy giảm gradient qua lớp đó) hoặc `= 0` nếu `z≤0` (chặn hoàn toàn, không phải suy giảm dần) — không có vùng "gần phẳng nhưng khác 0" như Sigmoid. Đánh đổi: một neuron ReLU "chết" (luôn `z≤0` với mọi input) sẽ vĩnh viễn có gradient `0`, không bao giờ cập nhật lại được (**dying ReLU**) — đây là lý do Swish/SiLU (Phần I.1, dùng trong EfficientNet) được ưa chuộng hơn ReLU thuần ở kiến trúc hiện đại: `Swish(z)=z·Sigmoid(z)` mượt và không có vùng "chết cứng" tuyệt đối như ReLU.

**Vì sao Skip Connection (residual, dùng trong cả ResNet-34 encoder của U-Net lẫn cấu trúc MBConv của EfficientNet, Phần IV.2/V.2) là giải pháp kiến trúc quan trọng nhất:** với `output = x + F(x)` (cộng trực tiếp input vào output của một khối, thay vì chỉ `output = F(x)`):

```
∂output/∂x = ∂(x + F(x))/∂x = I + ∂F(x)/∂x
             └┬┘   └────┬────┘
          luôn = 1   có thể vanishing, nhưng không quan trọng nữa
```

Vì luôn có số hạng `I` (identity, đạo hàm = 1) cộng thêm, gradient có **một con đường tắt trực tiếp** để chảy ngược qua toàn mạng mà không bị nhân suy giảm qua `F(x)` — dù `∂F(x)/∂x` có nhỏ tới đâu, tổng vẫn tối thiểu bằng `1` chứ không tiến về 0. Đây là lý do kiến trúc có residual connection có thể train sâu hàng trăm lớp mà kiến trúc không có residual (như VGG thuần) không làm được.

**BatchNorm** (trình bày đầy đủ ở Phần II.6) cũng góp phần chống vanishing/exploding bằng cách giữ `z` ở mỗi lớp luôn trong một khoảng ổn định (trung bình 0, phương sai 1 trước khi scale/shift) — tránh `z` trôi dạt tới vùng bão hoà của activation (nơi đạo hàm gần 0) hoặc vùng giá trị cực lớn (dễ gây tràn số/NaN).

### 3-4-5. Input/Output/Ứng dụng

Đây là một **hiện tượng** xảy ra trong quá trình backward, không phải một layer/hàm cụ thể có input/output riêng — "input" ở đây là kiến trúc mạng (độ sâu, loại activation, có/không có skip connection, có/không có BatchNorm), "output" là hành vi gradient quan sát được qua các epoch train.

**Cách test/phát hiện:** in `p.grad.norm()` (độ lớn L2-norm của gradient) cho từng lớp sau `loss.backward()`, so sánh giữa lớp đầu và lớp cuối — nếu gradient lớp đầu nhỏ hơn lớp cuối hàng nghìn lần, đó là dấu hiệu vanishing gradient thực sự đang xảy ra, cần kiểm tra lại activation function/kiến trúc thay vì chỉ nghi ngờ learning rate.

## I.8. Khởi tạo trọng số (Weight Initialization)

### 1. Nó là gì

Là cách gán giá trị **ban đầu** (trước khi train bước nào) cho mỗi trọng số trong mạng — trước khi có bất kỳ gradient nào để cập nhật, trọng số phải bắt đầu từ đâu đó.

### 2. Nguyên tắc hoạt động

**Vì sao không thể khởi tạo mọi trọng số bằng 0?** Nếu mọi trọng số trong một lớp bằng nhau (kể cả bằng 0), mọi neuron trong lớp đó nhận **cùng một input**, tính ra **cùng một `z`**, cho **cùng một gradient** khi backprop — chúng cập nhật **giống hệt nhau** ở mọi bước, mãi mãi không bao giờ "phân hoá" để học các đặc trưng khác nhau (gọi là **symmetry problem**). Về bản chất, một lớp `m` neuron bị khởi tạo đối xứng như vậy chỉ có sức biểu diễn tương đương **1** neuron duy nhất, dù có bao nhiêu neuron đi nữa.

**Giải pháp: khởi tạo ngẫu nhiên nhỏ**, nhưng "ngẫu nhiên nhỏ" cỡ nào cũng cần tính toán cẩn thận — quá nhỏ thì `z` mọi lớp gần 0, gradient các lớp sâu bị vanishing (Phần I.7); quá lớn thì `z` bùng nổ qua nhiều lớp, exploding. Hai lược đồ chuẩn:

- **Xavier/Glorot Init** (phù hợp Sigmoid/Tanh — activation đối xứng quanh 0):
  ```
  W ~ Normal(0, 2/(n_in + n_out))
  ```
  Thiết kế để giữ **phương sai của activation** (khi forward) và **phương sai của gradient** (khi backward) xấp xỉ không đổi qua các lớp — cân bằng cả hai chiều truyền tín hiệu.

- **He/Kaiming Init** (phù hợp ReLU — dùng cho hầu hết layer trong EfficientNet/U-Net của dự án):
  ```
  W ~ Normal(0, 2/n_in)
  ```
  Vì ReLU triệt tiêu khoảng **một nửa** giá trị âm về 0 (Phần I.1), phương sai output sau ReLU chỉ còn một nửa so với input — He Init nhân đôi phương sai khởi tạo (`2/n_in` thay vì `1/n_in`) để bù lại đúng phần bị ReLU "cắt mất", giữ phương sai tín hiệu ổn định qua các lớp ReLU liên tiếp.

### Ví dụ số

Một lớp Linear có `n_in=100` neuron input:

```
Xavier (n_in=100, n_out=100):  std = √(2/(100+100)) = √0.01     = 0.1000
He     (n_in=100):              std = √(2/100)        = √0.02   = 0.1414
```

He init cho `std` lớn hơn Xavier ~41% với cùng `n_in` — đúng như kỳ vọng "bù lại phần ReLU cắt mất".

### 3-4-5. Input/Output/Ứng dụng

**Ứng dụng trực tiếp trong dự án:** backbone EfficientNet-B3/ResNet-34 **không** dùng khởi tạo ngẫu nhiên — chúng dùng trọng số **đã học** từ ImageNet (chính là lý do Transfer Learning hiệu quả, Phần III.2). **Chỉ** lớp `Linear` mới thay thế cuối cùng (`model.classifier[1] = nn.Linear(1536, 3)` trong `model.py`) mới được khởi tạo ngẫu nhiên — mặc định PyTorch dùng biến thể Kaiming Uniform cho `nn.Linear`. Đây là lý do Pha 1 (warm-up head, Phần III.2) cần LR cao hơn hẳn các pha sau: trọng số head đang ở trạng thái "khởi tạo ngẫu nhiên nhỏ" kinh điển vừa mô tả ở trên, cần nhiều bước cập nhật lớn để rời xa điểm xuất phát ngẫu nhiên đó.

**Cách test:** `model.classifier[1].weight.std().item()` ngay sau khi khởi tạo (trước khi train) phải xấp xỉ giá trị `std` lý thuyết tính theo công thức Kaiming ở trên (sai số do lấy mẫu ngẫu nhiên hữu hạn) — nếu lệch quá xa, có thể lớp đó đã bị load nhầm trọng số cũ thay vì khởi tạo mới.

---

# PHẦN II — CONVOLUTIONAL NEURAL NETWORK (CNN)

## II.1. Nó là gì

CNN (Mạng nơ-ron tích chập) là một kiến trúc mạng nơ-ron chuyên biệt cho dữ liệu có **cấu trúc lưới không gian** — điển hình là ảnh (lưới 2D điểm ảnh). Thay vì mỗi neuron kết nối tới **toàn bộ** input (như MLP ở Phần I — gọi là *fully-connected*), một neuron trong CNN chỉ nhìn vào một **vùng nhỏ cục bộ** của ảnh, và **cùng một bộ trọng số** đó được trượt (slide) qua toàn bộ ảnh để tạo ra một bản đồ đặc trưng.

Trong dự án này, CNN là "xương sống" của cả hai model: EfficientNet-B3 (Phần IV) là một CNN cho phân loại, U-Net (Phần V) dùng CNN làm cả encoder lẫn decoder cho phân đoạn.

**Vì sao không dùng MLP thuần cho ảnh?** Một ảnh 224×224×3 có `224×224×3 = 150,528` giá trị pixel. Một lớp fully-connected đầu tiên nối input này tới, ví dụ, 1000 neuron sẽ cần `150,528 × 1000 ≈ 150 triệu` trọng số — cho **một** lớp duy nhất. Với dataset chỉ ~9.000 ảnh của dự án, số tham số này chắc chắn overfit nặng (học thuộc lòng, không tổng quát hoá). Ngoài ra, MLP không có khái niệm "vị trí tương đối" — nếu dịch một vật thể sang phải 5 pixel, MLP phải học lại từ đầu pattern đó ở vị trí mới, vì mỗi input pixel nối tới một trọng số **khác nhau**. CNN giải quyết cả hai vấn đề bằng hai nguyên lý: **chia sẻ trọng số (weight sharing)** và **kết nối cục bộ (local connectivity)**.

## II.2. Nguyên tắc hoạt động

### Phép tích chập (Convolution)

Một **kernel** (còn gọi filter) là một ma trận trọng số nhỏ, ví dụ 3×3. Phép convolution trượt kernel này qua từng vị trí của ảnh input, tại mỗi vị trí tính **tích vô hướng** giữa kernel và vùng ảnh nó đang phủ lên, ghi kết quả vào một pixel của **feature map** output.

Công thức toán học (convolution 2D rời rạc, thực chất trong deep learning là **cross-correlation**, không lật kernel như định nghĩa convolution toán học thuần túy, nhưng cộng đồng vẫn quen gọi là "convolution"):

```
S(i,j) = Σₘ Σₙ  I(i+m, j+n) · K(m,n)  + b
```

trong đó `I` là ảnh input, `K` là kernel kích thước `k×k`, `S` là feature map output, `b` là bias (một số duy nhất, cộng vào mọi vị trí).

### Ví dụ số — convolution tay trên ma trận 5×5 với kernel 3×3

Input (một "ảnh" xám 5×5, giả lập một cạnh dọc sáng ở giữa):

```
I = ⎡0  0  1  0  0⎤
    ⎢0  0  1  0  0⎥
    ⎢0  0  1  0  0⎥
    ⎢0  0  1  0  0⎥
    ⎣0  0  1  0  0⎦
```

Kernel dò cạnh dọc (Sobel-like), `K`:

```
K = ⎡-1  0  1⎤
    ⎢-1  0  1⎥
    ⎣-1  0  1⎦
```

Tính giá trị output tại vị trí `(1,1)` (0-indexed, vùng ảnh phủ là hàng 0-2, cột 0-2):

```
Vùng ảnh:  ⎡0 0 1⎤
           ⎢0 0 1⎥
           ⎣0 0 1⎦

S(1,1) = (0×-1)+(0×0)+(1×1) + (0×-1)+(0×0)+(1×1) + (0×-1)+(0×0)+(1×1)
       = 1 + 1 + 1 = 3
```

Tính tương tự cho vị trí `(1,2)` (vùng ảnh phủ hàng 0-2, cột 1-3 — đúng lúc cạnh sáng nằm ở giữa vùng phủ, kernel bên trái toàn 0, bên phải toàn 1):

```
Vùng ảnh: ⎡0 1 0⎤
          ⎢0 1 0⎥
          ⎣0 1 0⎦

S(1,2) = (0×-1)+(1×0)+(0×1) ×3 hàng = 0
```

Trượt kernel qua toàn bộ ảnh (với `padding=0`, ảnh 5×5 và kernel 3×3 cho output 3×3 — công thức kích thước ở dưới), ta được feature map:

```
S = ⎡3  0  -3⎤
    ⎢3  0  -3⎥
    ⎣3  0  -3⎦
```

**Ý nghĩa:** giá trị `3` (dương, lớn) ở cột trái nghĩa là kernel "khớp mạnh" với pattern nó đang tìm (cạnh sáng bên phải, tối bên trái) tại vị trí đó; giá trị `-3` ở cột phải nghĩa là pattern **ngược lại** (tối bên phải, sáng bên trái). Đây chính xác là cách một kernel dò cạnh hoạt động — và trong CNN thật, kernel **không được thiết kế tay** như ví dụ trên mà **tự học** qua backpropagation (Phần I.4) để tối ưu hoá loss, kết quả là mạng tự khám phá ra những kernel hữu ích (không nhất thiết trực quan như dò-cạnh, nhưng các lớp nông đầu tiên trong thực nghiệm thường học ra các kernel dò cạnh/góc/màu tương tự ví dụ trên).

### Padding, Stride — các tham số điều khiển hình dạng output

- **Padding**: thêm viền số 0 quanh ảnh trước khi convolve, để (a) kiểm soát kích thước output, (b) tránh mất thông tin ở viền ảnh (nếu không padding, pixel góc chỉ được "nhìn" bởi 1 vị trí kernel duy nhất, trong khi pixel giữa được nhìn bởi nhiều vị trí — thông tin viền bị "yếu thế" một cách hệ thống).
- **Stride**: bước nhảy của kernel mỗi lần trượt. Stride=1 trượt từng pixel một; stride=2 nhảy 2 pixel — feature map output nhỏ đi một nửa, dùng để **downsample** (giảm kích thước không gian) thay cho pooling ở một số kiến trúc.

Công thức tính kích thước output:

```
H_out = ⌊(H_in + 2×padding − kernel_size) / stride⌋ + 1
```

Kiểm tra lại ví dụ trên: `H_in=5, padding=0, kernel=3, stride=1` → `H_out = ⌊(5+0-3)/1⌋+1 = 2+1 = 3`. Khớp với feature map 3×3 tính tay ở trên.

### Nhiều kênh (channels) — convolution thật trong CNN sâu

Ảnh input có 3 kênh (RGB). Một kernel thực tế **không phải** ma trận 2D mà là tensor 3D `k×k×C_in` — trượt qua ảnh và cộng dồn theo cả 3 kênh cùng lúc ra **một** giá trị số cho mỗi vị trí không gian. Một lớp conv có `C_out` kernel như vậy (độc lập, mỗi kernel học một pattern khác nhau) sẽ cho ra feature map có `C_out` kênh. Vì vậy trọng số một lớp conv có shape chuẩn PyTorch: `(C_out, C_in, k, k)`.

Ví dụ: lớp conv đầu tiên của EfficientNet nhận ảnh 3 kênh, tạo ra 40 kênh feature map với kernel 3×3 → số tham số (chưa tính bias): `40 × 3 × 3 × 3 = 1,080` — so với MLP fully-connected cần hàng trăm triệu tham số cho cùng lượng thông tin, đây là mức tiết kiệm tham số khổng lồ nhờ *weight sharing* (cùng một kernel 3×3×3 dùng lại ở **mọi** vị trí trên ảnh, không phải một bộ trọng số riêng cho từng vị trí).

### Pooling — giảm chiều không gian có kiểm soát

Sau vài lớp conv, một lớp **pooling** (thường Max Pooling) trượt một cửa sổ (ví dụ 2×2) qua feature map, chỉ giữ lại giá trị **lớn nhất** trong mỗi cửa sổ:

```
Feature map 4×4:              Max Pool 2×2, stride 2:
⎡1  3  2  4⎤                  ⎡3  4⎤
⎢5  6  1  2⎥          ──►     ⎣9  6⎦
⎢8  9  0  1⎥
⎣2  3  4  6⎦
```

(Giải thích: cửa sổ trên-trái phủ `[[1,3],[5,6]]` → max = 6... — thực tế cần khớp đúng ô, ở đây minh hoạ nguyên lý: mỗi vùng 2×2 rút gọn còn 1 giá trị lớn nhất.)

**Vì sao Max Pooling hữu ích:** (a) giảm kích thước không gian (và do đó giảm chi phí tính toán ở các lớp sau) theo cấp số nhân; (b) tạo tính **bất biến cục bộ với dịch chuyển nhỏ** (small translation invariance) — nếu vật thể dịch 1 pixel, giá trị max trong hầu hết các cửa sổ không đổi; (c) chỉ giữ lại tín hiệu "mạnh nhất" — phù hợp trực giác "có tồn tại pattern này trong vùng hay không" hơn là giá trị trung bình.

### Receptive Field — vì sao CNN sâu "nhìn" được toàn cảnh

Một neuron ở lớp conv đầu tiên chỉ "nhìn thấy" một vùng 3×3 pixel gốc. Nhưng một neuron ở lớp conv **thứ hai** (nhận input là feature map của lớp một, cũng qua kernel 3×3) gián tiếp "nhìn thấy" một vùng 5×5 trên ảnh gốc — vì mỗi trong 9 pixel nó nhìn ở lớp 1 lại là kết quả tổng hợp từ một vùng 3×3 khác ở ảnh gốc. Vùng ảnh gốc mà một neuron ở lớp sâu `l` có thể "nhìn thấy" gọi là **receptive field**, tăng dần theo độ sâu mạng (và tăng nhanh hơn nữa mỗi khi qua một lớp pooling/stride giảm kích thước không gian). Đây là lý do các lớp sâu của CNN học được pattern ở mức **toàn cục** hơn (hình dạng cơ quan, texture mô phổi) trong khi lớp nông chỉ học pattern **cục bộ** (cạnh, góc, gradient màu).

```
Ảnh gốc                Lớp 1 (receptive field 3×3)     Lớp 2 (receptive field 5×5)
┌───────────┐          ┌───────────┐                    ┌───────────┐
│ · · · · · │          │ · · · · · │                     │ ▓ ▓ ▓ ▓ · │
│ · ▓ ▓ ▓ · │   conv    │ · · ● · · │      conv           │ ▓ ▓ ▓ ▓ · │
│ · ▓ ▓ ▓ · │  ─────►   │ · · · · · │     ─────►           │ ▓ ▓ ● ▓ · │
│ · ▓ ▓ ▓ · │           │ · · · · · │                     │ ▓ ▓ ▓ ▓ · │
│ · · · · · │           │ · · · · · │                     │ · · · · · │
└───────────┘           └───────────┘                    └───────────┘
 (một neuron ● ở lớp sâu hơn "nhìn thấy" gián tiếp vùng ▓ càng lúc càng rộng trên ảnh gốc)
```

## II.3. Input của CNN là gì

Một tensor 4 chiều theo quy ước PyTorch: `(N, C, H, W)`:

- `N` — batch size (số ảnh xử lý cùng lúc, ví dụ 32).
- `C` — số kênh (3 cho RGB; ảnh X-quang xám được lặp kênh thành 3 như `TUTORIAL.md` Phần 5.2 đã nêu).
- `H, W` — chiều cao, chiều rộng (224×224 trong dự án — `IMAGE_SIZE` ở `dataset.py`).

Giá trị pixel **đã chuẩn hoá** (Phần I không nói tới, nhưng quan trọng ở đây): trừ mean, chia std theo thống kê ImageNet (`MEAN=[0.485,0.456,0.406]`, `STD=[0.229,0.224,0.225]`) — để phân phối input khớp với phân phối lúc mạng pretrained được huấn luyện (chi tiết ở Phần III).

## II.4. Output của CNN là gì

Phụ thuộc CNN đó đóng vai trò gì trong kiến trúc lớn hơn:

- Nếu là **backbone trích đặc trưng** (như phần `features` của EfficientNet-B3): output là một tensor feature map, ví dụ `(N, 1536, 7, 7)` — 1536 kênh, không gian đã bị thu nhỏ từ 224×224 xuống 7×7 qua nhiều lớp stride/pooling.
- Nếu là **classifier hoàn chỉnh** (backbone + global average pooling + fully-connected): output là vector logits `(N, num_classes)` = `(N, 3)` trong dự án.
- Nếu là **encoder của U-Net**: output là **nhiều** feature map ở nhiều độ phân giải khác nhau (giữ lại để dùng làm skip connection — Phần V).

## II.5. Ý nghĩa Output, ứng dụng, cách test

**Global Average Pooling (GAP)** — cầu nối giữa feature map không gian và vector phân loại: thay vì flatten toàn bộ `(1536, 7, 7)` thành vector 75,264 chiều (tốn tham số, dễ overfit, và phá vỡ tính bất biến vị trí), GAP lấy **trung bình** mọi giá trị trong mỗi kênh, cho ra vector `(1536,)` — mỗi số đại diện "mức độ hiện diện trung bình của đặc trưng đó trên toàn ảnh", bất kể đặc trưng đó nằm ở đâu trong ảnh.

```
GAP: feature map kênh k, kích thước 7×7  ──►  1 số = trung bình cộng 49 giá trị
```

**Cách test một CNN backbone đang hoạt động đúng:** kiểm tra shape ở từng điểm nối (`torchinfo.summary`, như `TUTORIAL.md` Phần 6.6 gợi ý) — nếu shape khớp kỳ vọng ở mọi lớp, và forward pass không lỗi/NaN với input ngẫu nhiên, đó là bước sanity-check tối thiểu trước khi bắt đầu train thật.

## II.6. Batch Normalization

### 1. Nó là gì

BatchNorm là một lớp (không có "ý nghĩa đặc trưng" như conv, mà mang tính **kỹ thuật ổn định số học**) chèn giữa các lớp conv/linear, chuẩn hoá lại phân phối của `z` (giá trị trước activation) ngay trong lúc train, đồng thời có **2 tham số học được riêng** cho mỗi kênh: hệ số scale `γ` và hệ số dịch `β`. Có mặt dày đặc bên trong cả EfficientNet lẫn ResNet-34 (encoder U-Net) — mỗi khối MBConv/residual đều có ít nhất một lớp BatchNorm.

### 2. Nguyên tắc hoạt động

Với một batch `m` giá trị `z` của **cùng một kênh** (tính trên toàn bộ batch, mọi vị trí không gian với ảnh):

```
μ_B   = (1/m) Σᵢ zᵢ                              (trung bình của batch)
σ_B²  = (1/m) Σᵢ (zᵢ - μ_B)²                      (phương sai của batch)
ẑᵢ    = (zᵢ - μ_B) / √(σ_B² + ε)                  (chuẩn hoá: trung bình 0, phương sai 1)
yᵢ    = γ·ẑᵢ + β                                  (scale + shift — HỌC ĐƯỢC qua backprop)
```

`ε` (~1e-5) tránh chia 0 khi `σ_B²` rất nhỏ. **Vì sao cần `γ, β` học lại** sau khi vừa chuẩn hoá về `(0,1)`? Nếu chỉ chuẩn hoá cứng, mạng bị **ép buộc** mọi `z` phải có phân phối `(0,1)` — nhưng đôi khi phân phối tối ưu cho một kênh cụ thể không phải vậy (ví dụ một activation cần luôn dương). `γ, β` cho phép mạng **tự học lại** scale/shift phù hợp nếu chuẩn hoá cứng không tối ưu — trường hợp xấu nhất, mạng học `γ=√(σ_B²+ε), β=μ_B` để "hoàn tác" hoàn toàn phép chuẩn hoá, tức BatchNorm không bao giờ làm mạng **kém đi** so với không có nó (về mặt biểu diễn lý thuyết).

**Lợi ích thực tế:** (a) giảm hiện tượng *internal covariate shift* (phân phối input của một lớp thay đổi liên tục qua các bước train vì trọng số lớp trước đó thay đổi) — mỗi lớp "thấy" input ổn định hơn, học nhanh hơn; (b) cho phép dùng LR cao hơn mà không bị exploding (Phần I.7); (c) có hiệu ứng regularize nhẹ (vì `μ_B, σ_B²` tính trên một batch ngẫu nhiên, khác nhau giữa các batch, tạo ra nhiễu nhỏ tương tự Dropout).

### Ví dụ số

4 giá trị `z` trong một batch (cùng 1 kênh): `z = [2.0, 4.0, 4.0, 6.0]`. Giả sử `γ=1.5, β=0.5` (đã học được từ trước), `ε≈0`:

```
μ_B  = (2+4+4+6)/4 = 16/4 = 4.0
σ_B² = [(2-4)²+(4-4)²+(4-4)²+(6-4)²]/4 = [4+0+0+4]/4 = 8/4 = 2.0

ẑ₁ = (2.0-4.0)/√2.0 = -2.0/1.4142 = -1.4142
ẑ₂ = (4.0-4.0)/√2.0 = 0.0
ẑ₃ = (4.0-4.0)/√2.0 = 0.0
ẑ₄ = (6.0-4.0)/√2.0 = 1.4142

y₁ = 1.5×(-1.4142) + 0.5 = -2.1213 + 0.5 = -1.6213
y₄ = 1.5×(1.4142) + 0.5  =  2.1213 + 0.5 =  2.6213
```

### 3-4-5. Input/Output/Ứng dụng — điểm mấu chốt nối sang `model.eval()`

**Input:** `z` của cả batch (lúc train) — đây là điểm quan trọng nhất cần nhớ: BatchNorm **cần nhiều hơn 1 ảnh trong batch** để tính `μ_B, σ_B²` có ý nghĩa thống kê. **Output:** `y` cùng shape với `z`, cùng `γ, β` được lưu trong `state_dict` (Phần VI, "Vì sao lưu `state_dict`") như mọi trọng số khác.

**Vấn đề lúc suy luận (inference, chỉ 1 ảnh mỗi lần — đúng tình huống `POST /predict` của `api/inference.py`):** không thể tính `μ_B, σ_B²` có ý nghĩa từ batch chỉ có 1 ảnh. Giải pháp: trong lúc **train**, BatchNorm âm thầm duy trì thêm 2 giá trị **running average** (`running_mean`, `running_var`) — trung bình động cộng dồn qua mọi batch đã thấy:

```
running_mean ← (1-momentum)·running_mean + momentum·μ_B      (momentum thường 0.1)
running_var  ← (1-momentum)·running_var  + momentum·σ_B²
```

Lúc **inference**, BatchNorm dùng `running_mean, running_var` (thống kê tích luỹ từ toàn bộ quá trình train) **thay vì** `μ_B, σ_B²` của batch hiện tại — đây chính là lý do bắt buộc phải gọi `model.eval()` trước khi dự đoán (chi tiết cơ chế chuyển đổi này ở Phần II.8 ngay dưới).

**Cách test:** kiểm tra `model.bn_layer.running_mean`/`running_var` không phải giá trị mặc định (`0`/`1`) sau vài epoch train — nếu vẫn là mặc định, BatchNorm chưa thực sự "học" thống kê nào, có thể do quên gọi `model.train()` trong vòng lặp train.

## II.7. Dropout

### 1. Nó là gì

Dropout là một kỹ thuật **regularization** (chống overfitting): trong lúc train, ngẫu nhiên "tắt" (đặt về 0) một tỉ lệ `p` neuron ở mỗi bước forward, buộc mạng không được phụ thuộc quá nhiều vào bất kỳ neuron đơn lẻ nào. Xuất hiện tường minh trong dự án ở `classifier` head của EfficientNet-B3 (`Sequential(Dropout, Linear)`, Phần VI mapping).

### 2. Nguyên tắc hoạt động

Với activation `a` và xác suất tắt `p` (ví dụ `p=0.3` — tắt 30% neuron mỗi lần):

```
mask ~ Bernoulli(1-p)     (mỗi phần tử độc lập, = 1 với xác suất (1-p), = 0 với xác suất p)
a_dropout = (a ⊙ mask) / (1-p)         ("inverted dropout" — chia cho (1-p) NGAY LÚC TRAIN)
```

**Vì sao chia cho `(1-p)`?** Nếu không chia, kỳ vọng (giá trị trung bình) của `a_dropout` sẽ thấp hơn `a` gốc một hệ số `(1-p)` (vì trung bình có `p` tỉ lệ phần tử bị tắt về 0) — làm lệch phân phối activation giữa lúc train (bị tắt bớt) và lúc inference (không tắt gì). Chia cho `(1-p)` ngay lúc train "bù" trước sự chênh lệch này, để lúc **inference** (Dropout tắt hoàn toàn, dùng nguyên `a`) không cần điều chỉnh gì thêm — đây là lý do kỹ thuật này gọi là "inverted" dropout, là cách PyTorch (`nn.Dropout`) triển khai mặc định.

**Vì sao Dropout chống overfitting:** một neuron không thể "ỷ lại" vào việc luôn có một neuron khác cụ thể đi kèm để cùng nhận diện một pattern (vì neuron đó có thể ngẫu nhiên bị tắt ở bước bất kỳ) — buộc mỗi neuron phải học một đặc trưng **đủ hữu ích một cách độc lập tương đối**, không "đồng phụ thuộc" (co-adaptation) quá mức vào một tổ hợp neuron cụ thể. Về hình thức, có thể xem Dropout như đang huấn luyện đồng thời một tập hợp cực lớn các mạng con (mỗi lần forward là một "mạng con" khác nhau do mask ngẫu nhiên khác nhau) rồi ngầm lấy trung bình chúng lúc inference — một dạng ensemble ẩn, không tốn thêm chi phí lưu trữ nhiều model.

### Ví dụ số

`a = [1.0, 2.0, 3.0, 4.0]`, `p=0.5` (tắt 50%). Giả sử lần lấy mẫu ngẫu nhiên cho `mask = [1, 0, 1, 0]` (neuron 2 và 4 bị tắt):

```
a ⊙ mask = [1.0, 0.0, 3.0, 0.0]
a_dropout = [1.0, 0.0, 3.0, 0.0] / (1-0.5) = [2.0, 0.0, 6.0, 0.0]

Kiểm tra kỳ vọng: E[a_dropout] xấp xỉ a gốc khi lấy trung bình qua NHIỀU lần lấy mẫu mask khác nhau
(mỗi phần tử có 50% cơ hội = 2×giá trị gốc, 50% cơ hội = 0 → trung bình = giá trị gốc)
```

### 3-4-5. Input/Output/Ứng dụng

**Input/Output:** cùng shape (`Dropout` không đổi shape, chỉ đổi giá trị một số phần tử về 0). **Ứng dụng:** đặt ngay trước lớp `Linear` cuối cùng — vị trí có nhiều tham số nhất trên mỗi neuron đầu vào (`1536 → 3`, dễ overfit nhất vì học trực tiếp trên tập train nhỏ) nên cần regularize mạnh nhất tại đây; các lớp conv trong backbone thường **không** cần Dropout riêng vì đã có BatchNorm + weight sharing (Phần II.1) làm regularize một phần.

**Giống BatchNorm, Dropout đổi hành vi hoàn toàn giữa train/eval** — xem Phần II.8.

## II.8. `model.train()` vs `model.eval()` — vì sao gọi sai chế độ gây lỗi khó phát hiện

### 1. Nó là gì

Đây không phải một layer, mà là **hai trạng thái toàn cục** của một `nn.Module` trong PyTorch, quyết định các lớp "nhạy cảm với chế độ" (BatchNorm, Dropout) hoạt động theo công thức nào.

### 2. Nguyên tắc hoạt động

```
                    model.train()                          model.eval()
                    (đang HUẤN LUYỆN)                       (đang SUY LUẬN/ĐÁNH GIÁ)
─────────────────  ───────────────────────────────────    ───────────────────────────────────
BatchNorm           dùng μ_B, σ_B² của BATCH HIỆN TẠI      dùng running_mean, running_var
                     (Phần II.6) — VÀ cập nhật running_*    TÍCH LUỸ từ train — KHÔNG cập nhật
                     mean/var                                running_* nữa

Dropout              tắt ngẫu nhiên p% neuron, chia         KHÔNG tắt gì cả — dùng nguyên
                     cho (1-p) (Phần II.7)                   100% neuron, không chia gì thêm
```

**Vì sao lỗi này "khó phát hiện":** không có exception/crash nào xảy ra nếu quên gọi `model.eval()` trước validation/inference — code vẫn chạy, vẫn ra một con số dự đoán, chỉ là **con số đó sai/không ổn định**: (a) nếu batch validation nhỏ (đặc biệt `batch_size=1` khi serve qua `api/inference.py`), `μ_B, σ_B²` của BatchNorm tính trên chỉ 1 ảnh gần như vô nghĩa về thống kê → output nhiễu loạn thất thường giữa các lần gọi cùng một ảnh; (b) Dropout vẫn ngẫu nhiên tắt neuron → dự đoán cho **cùng một ảnh** ra **kết quả khác nhau** ở mỗi lần gọi — vi phạm trực tiếp yêu cầu cơ bản nhất của một hệ thống chẩn đoán: cùng input phải cho cùng output.

Đây chính là gốc rễ kỹ thuật của gotcha đã nêu ngắn gọn ở `TUTORIAL.md` Phần 6.5 ("hoặc quên gọi `.eval()` khiến BatchNorm học sai thống kê từ batch nhỏ") và ở Phần III.5 tài liệu này — giờ đã có đầy đủ cơ chế toán học phía sau lời cảnh báo đó.

### 3-4-5. Input/Output/Ứng dụng — quy tắc bắt buộc trong dự án

```python
# Lúc train (mỗi epoch, trước vòng lặp batch train):
model.train()
for images, labels in train_loader:
    ...

# Lúc validate/test/serve (BẮT BUỘC trước khi forward):
model.eval()
with torch.no_grad():          # thêm bonus: tắt luôn việc build computational graph
    for images, labels in val_loader:      # (Phần I.4) — tiết kiệm bộ nhớ vì không
        ...                                  # cần gradient khi không train
```

**Cách test:** gọi `model.eval()`, forward **cùng một ảnh 2 lần liên tiếp** — hai lần phải cho **kết quả giống hệt nhau** (logits khớp tới nhiều chữ số thập phân). Nếu khác nhau, gần như chắc chắn quên `model.eval()` (còn ở chế độ `train()`, Dropout đang tắt ngẫu nhiên neuron khác nhau mỗi lần) — đây là bài test tự động hoá được, nên đưa vào sanity-check trước khi deploy `api/inference.py`.

## II.9. CLAHE — tiền xử lý tăng tương phản (kỹ thuật nâng cao, tham khảo cho tương lai)

### 1. Nó là gì

CLAHE (Contrast Limited Adaptive Histogram Equalization) là một kỹ thuật xử lý ảnh **cổ điển** (không dùng mạng nơ-ron) để tăng độ tương phản cục bộ. `description.md`/`pipeline.md` liệt kê CLAHE trong bước tiền xử lý dự kiến của hệ thống; **lưu ý theo CLAUDE.md của repo: `preprocess.py` hiện tại chưa cài đặt bước này** — phần dưới là nền tảng lý thuyết chuẩn bị cho việc thêm nó sau, không mô tả code đã có sẵn.

### 2. Nguyên tắc hoạt động

**Histogram Equalization (HE) — nền tảng của CLAHE.** Với ảnh xám có `L=256` mức sáng, gọi `n_k` là số pixel có giá trị sáng `k`, `N` là tổng số pixel. Hàm phân phối tích luỹ (CDF — Cumulative Distribution Function):

```
CDF(k) = Σ_{i=0}^{k} (n_i / N)
```

Ánh xạ lại mỗi mức sáng theo CDF của chính nó:

```
k_mới = round[ (CDF(k) - CDF_min) / (1 - CDF_min) × (L-1) ]
```

**Trực giác:** nếu một mức sáng `k` chiếm tỉ lệ lớn trong ảnh (ảnh bị "dồn cụm" quanh mức sáng đó — ảnh X-quang thường có nhiều pixel tối ở nền và nhiều pixel sáng ở xương, ít pixel "trung tính" ở mô phổi), HE sẽ **kéo giãn** khoảng cách giữa các mức sáng lân cận đó ra (vì CDF tăng dốc ở vùng có nhiều pixel), làm chi tiết ở vùng đó **dễ phân biệt bằng mắt hơn** — đúng mục tiêu "tăng tương phản".

**Vì sao cần "Adaptive" (áp dụng cục bộ theo từng ô nhỏ, ví dụ 8×8) thay vì tính CDF một lần cho toàn ảnh?** Một ảnh X-quang có vùng rất sáng (xương sườn) và vùng rất tối (ngoài viền phổi) cùng lúc — CDF tính trên toàn ảnh sẽ "trung bình hoá" và làm tương phản trong vùng mô phổi (nơi thực sự cần nhìn rõ để chẩn đoán) cải thiện rất ít. Chia ảnh thành các ô nhỏ (tile) và tính CDF **riêng cho từng ô** giúp mỗi vùng cục bộ được tăng tương phản theo đúng phân phối sáng của chính nó.

**Vì sao cần "Contrast Limited" (giới hạn độ tương phản)?** HE thuần áp dụng cục bộ (Adaptive HE không giới hạn) có nhược điểm: ở những ô gần như đồng nhất (ví dụ vùng nền đen tuyền), một biến động nhiễu nhỏ cũng có thể bị khuếch đại quá mức thành tương phản giả (khuếch đại nhiễu cảm biến thành các đốm sáng-tối giả trông giống tổn thương). CLAHE khắc phục bằng cách **cắt (clip)** histogram tại một ngưỡng cố định trước khi tính CDF — phần vượt ngưỡng ở mỗi bin được **phân phối đều lại** cho các bin khác — giới hạn mức khuếch đại tối đa, tránh tạo giả-tổn thương từ nhiễu.

### Ví dụ số — Histogram Equalization (chưa áp clip) trên ảnh nhỏ giả định

Ảnh `4×4=16` pixel, chỉ 4 mức sáng có thể (`L=4` để đơn giản hoá, thay vì 256), phân phối: mức `0` xuất hiện 6 lần, mức `1` xuất hiện 4 lần, mức `2` xuất hiện 4 lần, mức `3` xuất hiện 2 lần.

```
CDF(0) = 6/16                    = 0.375
CDF(1) = (6+4)/16                = 0.625
CDF(2) = (6+4+4)/16              = 0.875
CDF(3) = (6+4+4+2)/16            = 1.000

CDF_min = CDF(0) = 0.375

k_mới(0) = round[(0.375-0.375)/(1-0.375) × 3] = round[0]                = 0
k_mới(1) = round[(0.625-0.375)/(1-0.375) × 3] = round[0.250/0.625×3]    = round[1.2]  = 1
k_mới(2) = round[(0.875-0.375)/(1-0.375) × 3] = round[0.500/0.625×3]    = round[2.4]  = 2
k_mới(3) = round[(1.000-0.375)/(1-0.375) × 3] = round[0.625/0.625×3]    = round[3.0]  = 3
```

Với ảnh nhỏ và ít mức sáng này, ánh xạ gần như giữ nguyên (vì phân phối khá đều) — hiệu ứng "kéo giãn tương phản" của HE chỉ thể hiện rõ khi phân phối gốc **lệch mạnh** (dồn cụm nhiều pixel vào ít mức sáng), đúng tình huống thực tế của ảnh X-quang.

### 3-4-5. Input/Output/Ứng dụng

**Input:** ảnh xám gốc (`np.uint8`, `[0,255]`). **Output:** ảnh xám cùng kích thước, cùng khoảng giá trị, nhưng phân phối histogram đã "dàn đều" hơn — dùng thư viện `cv2.createCLAHE(clipLimit=..., tileGridSize=(8,8))` trong OpenCV thay vì tự cài đặt tay. **Vị trí trong pipeline nếu triển khai:** áp dụng trong `preprocess.py`, **trước** bước resize (Phần II.3) và **trước** khi lưu vào `data/processed/` — vì đây là biến đổi cường độ pixel thuần túy, không phụ thuộc kích thước ảnh. **Cách test:** so sánh histogram ảnh trước/sau bằng `matplotlib.pyplot.hist()` — histogram sau CLAHE phải "trải rộng" hơn (ít đỉnh nhọn tập trung, nhiều mức sáng được sử dụng hơn) so với trước.

---

# PHẦN III — TRANSFER LEARNING & FINE-TUNING THEO PHA

## III.1. Nó là gì

Transfer Learning là kỹ thuật tái sử dụng một mạng đã được huấn luyện trên một bài toán/dataset lớn (ImageNet — 1.28 triệu ảnh, 1000 lớp vật thể đời thường) để làm điểm khởi đầu cho một bài toán khác (ở đây: phân loại 3 lớp bệnh lý X-quang), thay vì khởi tạo trọng số ngẫu nhiên và học từ đầu (*train from scratch*).

## III.2. Nguyên tắc hoạt động

**Vì sao pretrained feature lại "chuyển giao" được sang bài toán khác?** Các lớp nông của bất kỳ CNN nào huấn luyện trên ảnh tự nhiên đều học ra các đặc trưng **tổng quát** — dò cạnh, góc, gradient màu, texture cơ bản — những pattern này xuất hiện trong **mọi** loại ảnh, không riêng gì "chó mèo xe cộ" của ImageNet. Ảnh X-quang cũng có cạnh (viền xương sườn), texture (mô phổi), gradient sáng-tối (vùng mờ đục do viêm) — các lớp nông pretrained vẫn hữu ích. Chỉ có các lớp **sâu nhất** (gần lớp phân loại cuối) mới học các pattern đặc thù ImageNet (hình dạng tai chó, bánh xe...) — không liên quan X-quang, và chính lớp phân loại cuối cùng (1000 lớp vật thể) chắc chắn phải thay bằng lớp mới (3 lớp bệnh lý).

```
ImageNet-pretrained EfficientNet-B3:

[Lớp nông: cạnh/góc/màu] → [Lớp giữa: texture/pattern] → [Lớp sâu: hình dạng vật thể ImageNet] → [Linear 1000 lớp]
        │ dùng lại được            │ dùng lại được               │ cần fine-tune                    │ THAY MỚI
        │ hầu như nguyên vẹn        │ gần như nguyên vẹn           │ (đặc thù ImageNet → X-quang)      │ Linear(1536→3)
```

**Vì sao cần dataset lớn để pretrained hoạt động tốt, và vì sao dataset nhỏ của dự án (≤3000 ảnh/lớp) không đủ để train from scratch?** Số tham số EfficientNet-B3 (~10.7M theo `TUTORIAL.md` Phần 6.6) lớn hơn nhiều lần số ảnh train (~6.300 ảnh sau split 70%). Nếu khởi tạo ngẫu nhiên và train từ đầu, mạng có đủ "sức chứa" (capacity) để **học thuộc lòng** từng ảnh train thay vì học pattern tổng quát — overfitting nghiêm trọng, accuracy trên val/test sẽ thấp dù train accuracy gần 100%. Bắt đầu từ trọng số pretrained tương đương với việc mạng đã "biết cách nhìn ảnh" từ 1.28 triệu ảnh khác trước đó — chỉ còn phải học phần khác biệt, cần ít dữ liệu hơn nhiều để hội tụ đúng hướng.

### Fine-tuning theo 3 pha (progressive unfreezing) — vì sao không mở khoá hết ngay từ đầu

Nếu unfreeze toàn bộ mạng (`requires_grad=True` cho mọi tham số) và train ngay với LR bình thường: lớp Linear cuối đang khởi tạo **ngẫu nhiên** → dự đoán ban đầu gần như đoán mò → loss rất cao → gradient rất lớn lan ngược qua backprop (Phần I.4) → gradient lớn này chạm tới các lớp backbone đã pretrained tốt → **phá huỷ** trọng số tốt đó chỉ để "chiều theo" một head đang random — hiện tượng gọi là **catastrophic forgetting**.

Giải pháp 3 pha đã nêu trong `TUTORIAL.md` Phần 2.3, nhắc lại dưới góc độ toán học của gradient:

```
Pha 1 (Warm-up head)         Pha 2 (Fine-tune vài block cuối)      Pha 3 (Full fine-tune, tuỳ chọn)
────────────────────         ─────────────────────────────         ────────────────────────────────
requires_grad=False cho       requires_grad=True cho 2-3 block       requires_grad=True cho TOÀN BỘ
mọi tham số backbone           cuối backbone                          
requires_grad=True chỉ         LR = 1e-4 (trung bình)                  LR = 1e-5 (rất thấp)
cho Linear cuối
LR = 1e-3 (cao)
                              
Gradient KHÔNG chạm vào       Gradient chạm nhẹ vào block cuối       Gradient chạm mọi nơi, nhưng
backbone (∂L/∂W_backbone      (đã "sẵn sàng" tiếp nhận vì head       bước nhảy cực nhỏ nên không phá
không được tính vì             hết random) — bước nhảy nhỏ, không     vỡ trọng số đã tốt
requires_grad=False)           phá vỡ pretrained feature
```

Về mặt kỹ thuật PyTorch: `requires_grad=False` khiến autograd **không tính** `∂L/∂w` cho tham số đó khi gọi `.backward()` (tiết kiệm tính toán) và optimizer (nếu khởi tạo đúng cách, lọc theo `requires_grad`) sẽ **không cập nhật** nó — trọng số giữ nguyên y hệt giá trị pretrained.

## III.3. Input là gì

Giống Phần II.3 — ảnh đã chuẩn hoá `(N, 3, 224, 224)`. Điểm khác biệt chỉ nằm ở **trạng thái trọng số ban đầu** (pretrained thay vì random) và **cờ `requires_grad`** trên từng nhóm tham số theo từng pha.

## III.4. Output là gì

Cũng giống Phần II.4 — logits `(N, 3)`. Transfer learning không đổi hình dạng input/output của mạng, chỉ đổi **điểm khởi đầu** và **chiến lược cập nhật trọng số** trong quá trình train.

## III.5. Ý nghĩa, ứng dụng, cách test

**Cách kiểm chứng đang ở đúng pha:** gọi `count_trainable_params(model)` (đã có trong `TUTORIAL.md` Phần 6.3/6.6) trước và sau mỗi lần freeze/unfreeze — số tham số "trainable" phải khớp kỳ vọng (pha 1: chỉ vài nghìn tham số của Linear cuối; pha 2: thêm vài triệu của 2-3 block cuối; pha 3: toàn bộ ~10.7M). Nếu số này sai (ví dụ pha 1 mà vẫn thấy hàng triệu tham số trainable) nghĩa là quên gọi `freeze_backbone()` hoặc quên tạo lại optimizer sau khi đổi `requires_grad` (gotcha đã nêu ở `TUTORIAL.md` Phần 6.5 — optimizer chỉ "biết" các tham số nó được truyền vào **lúc khởi tạo**, đổi `requires_grad` sau đó không tự động cập nhật optimizer).

**Dấu hiệu fine-tune đúng hướng:** val loss/F1 cải thiện dần qua các pha, không có bước nhảy đột ngột (loss tăng vọt) khi chuyển pha — nếu có, gần như chắc chắn LR của pha mới đang quá cao so với mức độ "đã sẵn sàng" của trọng số ở pha đó.

---

# PHẦN IV — EFFICIENTNET-B3 & COMPOUND SCALING

## IV.1. Nó là gì

EfficientNet là họ kiến trúc CNN (Tan & Le, ICML 2019) được thiết kế để đạt độ chính xác cao nhất **với số tham số/FLOPs ít nhất**, bằng cách scale (phóng to) một kiến trúc gốc (EfficientNet-B0) theo một công thức tối ưu, thay vì scale ngẫu nhiên/thủ công như các CNN trước đó (VGG, ResNet scale bằng cách thêm layer một cách trực giác). EfficientNet-B3 là bản scale thứ 3 trong họ B0→B7, được dự án chọn làm bộ phân loại chính (`build_classifier` trong `model.py`).

## IV.2. Nguyên tắc hoạt động

### Ba trục scale một CNN

Có 3 cách độc lập để làm một CNN "mạnh hơn":

| Trục | Ý nghĩa | Ví dụ tăng |
|---|---|---|
| **Depth (độ sâu, `d`)** | Số lớp conv xếp chồng | Nhiều lớp hơn → receptive field lớn hơn, học pattern phức tạp hơn |
| **Width (độ rộng, `w`)** | Số kênh (channel) mỗi lớp | Nhiều kênh hơn → mỗi lớp học được nhiều loại đặc trưng song song hơn |
| **Resolution (độ phân giải, `r`)** | Kích thước ảnh input | Ảnh lớn hơn → nhìn được chi tiết nhỏ hơn |

Trước EfficientNet, các nghiên cứu thường chỉ tăng **một** trục (ResNet tăng depth: ResNet-18→50→101; WideResNet tăng width). Tan & Le chỉ ra bằng thực nghiệm: tăng riêng lẻ một trục nhanh chóng **bão hoà** (đạt lợi ích giảm dần — accuracy tăng chậm dần dù tham số tăng nhanh), trong khi tăng **cân đối cả ba trục cùng lúc** cho hiệu quả tốt hơn hẳn ở cùng một ngân sách tính toán (FLOPs).

### Compound Scaling — công thức

```
depth:      d = α^φ
width:      w = β^φ
resolution: r = γ^φ

với ràng buộc:  α · β² · γ² ≈ 2   (α≥1, β≥1, γ≥1)
```

`φ` (phi) là **một** hệ số người dùng chọn (compound coefficient) — quyết định "scale lên bao nhiêu", B0 ứng với `φ=0`, B3 ứng với `φ=3`... `α, β, γ` là hằng số tìm được bằng grid search nhỏ trên B0 (paper gốc: `α=1.2, β=1.1, γ=1.15`), sau đó **cố định** và chỉ thay đổi `φ` để sinh ra cả họ B0→B7. Ràng buộc `α·β²·γ² ≈ 2` đảm bảo: mỗi khi `φ` tăng 1, tổng FLOPs của mạng tăng xấp xỉ `2¹ = 2` lần một cách có kiểm soát (vì FLOPs của conv tỉ lệ với `d × w² × r²` — depth tuyến tính, còn width và resolution ảnh hưởng bậc hai vì chúng nhân đôi lượng tính toán ở **cả hai** chiều: số kênh input và output cho width, chiều cao và chiều rộng ảnh cho resolution).

### Ví dụ số — tính scale factor cho B3

Với `φ=3`, `α=1.2, β=1.1, γ=1.15`:

```
d = 1.2³ = 1.728    → depth gấp ~1.73 lần B0
w = 1.1³ = 1.331    → width gấp ~1.33 lần B0
r = 1.15³ = 1.521   → resolution gấp ~1.52 lần B0 (B0 dùng ảnh 224 → thực ra B3 paper gốc dùng 300,
                       nhưng dự án này cố định 224×224 cho mọi model để đồng bộ pipeline — đây là
                       một lựa chọn thiết kế có đánh đổi, xem ghi chú dưới)

Kiểm tra ràng buộc: α·β²·γ² = 1.2 × 1.1² × 1.15² = 1.2 × 1.21 × 1.3225 ≈ 1.920 ≈ 2  ✓ (khớp gần đúng)
```

**Ghi chú quan trọng cho báo cáo:** paper gốc EfficientNet-B3 dùng input resolution 300×300 (không phải 224×224). Dự án này cố định `IMAGE_SIZE=(224,224)` cho mọi ảnh (đồng bộ với U-Net, với `dataset.py`) — nghĩa là đang dùng **kiến trúc** B3 (số lớp, số kênh theo đúng scale) nhưng **không dùng đúng resolution khuyến nghị** của B3. Đây không phải lỗi (torchvision cho phép input bất kỳ resolution nhờ `AdaptiveAvgPool` cuối backbone), nhưng là điểm cần biết: hiệu năng thực nghiệm có thể khác nhẹ so với con số benchmark gốc trên ImageNet ở 300×300 — nên nêu rõ trong report thay vì mặc định copy số liệu paper.

### Khối xây dựng: MBConv (Mobile Inverted Bottleneck Convolution)

EfficientNet không dùng conv 3×3 thông thường xuyên suốt mà dùng khối **MBConv**, gồm 3 ý tưởng kết hợp:

1. **Depthwise Separable Convolution** — tách một conv thông thường thành 2 bước: (a) **depthwise conv**: mỗi kênh input được convolve **riêng biệt** bởi một kernel không gian (không trộn thông tin giữa các kênh), (b) **pointwise conv** (kernel 1×1): trộn thông tin giữa các kênh. Tách như vậy giảm số phép tính đáng kể so với conv thường (vốn convolve **đồng thời** cả không gian lẫn kênh trong một bước) — cụ thể, conv thường tốn `k²·C_in·C_out` phép nhân mỗi vị trí, còn depthwise-separable chỉ tốn `k²·C_in + C_in·C_out` — với `k=3, C_in=C_out=64`: conv thường `9×64×64=36,864`, depthwise-separable `9×64 + 64×64 = 576+4096=4,672` — giảm gần **8 lần**.
2. **Inverted Bottleneck** — thay vì "thu nhỏ rồi mở rộng" số kênh (bottleneck kiểu ResNet), MBConv **mở rộng** số kênh trước (bằng conv 1×1, ví dụ ×6) rồi mới depthwise conv, rồi **thu nhỏ lại**. Mở rộng trước giúp depthwise conv (vốn không trộn kênh) có nhiều kênh hơn để biểu diễn phong phú hơn trước khi bị nén lại.
3. **Squeeze-and-Excitation (SE)** — một cơ chế "attention theo kênh": nén feature map mỗi kênh về 1 số (giống Global Average Pooling ở Phần II.5), đưa qua 2 lớp fully-connected nhỏ + Sigmoid ra một trọng số `∈(0,1)` cho **mỗi kênh**, rồi nhân trở lại vào feature map gốc — cho phép mạng tự học "kênh nào quan trọng hơn trong ảnh này" và khuếch đại/giảm nhẹ theo ngữ cảnh, thay vì đối xử mọi kênh như nhau.

```
MBConv block:

input (C kênh)
   │
   ▼
[Conv 1×1, mở rộng ×6] ──► (6C kênh)
   │
   ▼
[Depthwise Conv k×k]  ──► (6C kênh, mỗi kênh convolve riêng)
   │
   ▼
[Squeeze-Excitation]  ──► nhân trọng số theo kênh (attention)
   │
   ▼
[Conv 1×1, nén lại]   ──► (C' kênh)
   │
   ▼
[+ input] (residual connection, nếu C'==C và stride=1)
   │
   ▼
output
```

## IV.3. Input là gì

`(N, 3, 224, 224)` — giống Phần II.3, đã qua chuẩn hoá ImageNet.

## IV.4. Output là gì

Sau backbone (`model.features`): feature map `(N, 1536, 7, 7)` — 1536 là số kênh ở lớp conv cuối cùng của B3 (kết quả của compound scaling áp lên width). Sau `avgpool` (Global Average Pooling): `(N, 1536, 1, 1)` → flatten `(N, 1536)`. Sau `classifier` (Dropout + Linear đã thay bằng `Linear(1536, 3)`): **logits** `(N, 3)` — 3 số thực chưa chuẩn hoá, mỗi số ứng với một lớp `{Normal, Lung_Opacity, COVID}` theo `CLASS_TO_IDX`.

## IV.5. Ý nghĩa Output, ứng dụng, cách test

**Từ logits sang xác suất — Softmax:**

```
P(class=i) = e^(logit_i) / Σⱼ e^(logit_j)
```

### Ví dụ số

Giả sử logits cho một ảnh: `[Normal=1.2, Lung_Opacity=0.3, COVID=2.5]`.

```
e^1.2 = 3.3201
e^0.3 = 1.3499
e^2.5 = 12.1825

Tổng = 3.3201 + 1.3499 + 12.1825 = 16.8525

P(Normal)       = 3.3201 / 16.8525 = 0.1970  (19.70%)
P(Lung_Opacity) = 1.3499 / 16.8525 = 0.0801  (8.01%)
P(COVID)        = 12.1825 / 16.8525 = 0.7228 (72.28%)

Kiểm tra: 0.1970 + 0.0801 + 0.7228 = 0.9999 ≈ 1.0  ✓
```

Model dự đoán lớp **COVID** với độ tin cậy 72.28% — đây chính là con số "% tin cậy" hiển thị trên Gradio UI (`app.py`) theo `pipeline.md`.

**Vì sao Softmax (không phải chọn max logit trực tiếp)?** Softmax không chỉ chọn lớp có logit cao nhất mà còn cho biết **mức độ chênh lệch** giữa các lớp — logits `[5,1,1]` và `[2,1,1]` đều chọn lớp đầu, nhưng Softmax cho xác suất khác hẳn nhau (lớp đầu rất chắc chắn vs. khá mơ hồ), thông tin quan trọng cho bài toán y tế nơi độ tự tin của model cần hiển thị cho bác sĩ tham khảo, không chỉ nhãn cuối cùng.

**Cách test:** input ảnh giả (`torch.randn`) qua model, kiểm tra `out.softmax(1).sum(dim=1)` phải xấp xỉ 1.0 cho mọi ảnh trong batch (đúng như `TUTORIAL.md` Phần 6.6 hướng dẫn) — đây là bất biến toán học của Softmax, nếu sai nghĩa là có lỗi implementation (ví dụ áp Softmax nhầm chiều `dim`).

---

# PHẦN V — U-NET & BÀI TOÁN SEGMENTATION

## V.1. Nó là gì

U-Net (Ronneberger et al., MICCAI 2015) là một kiến trúc CNN cho bài toán **segmentation** (phân đoạn ngữ nghĩa ở mức pixel) — thay vì trả về 1 nhãn cho cả ảnh (như classifier ở Phần IV), nó trả về **1 nhãn cho từng pixel**. Trong dự án, U-Net đóng vai trò "định vị phổi" (`build_unet` trong `unet.py`) — nhận ảnh X-quang, trả về mask nhị phân đánh dấu vùng nào là phổi.

## V.2. Nguyên tắc hoạt động

### Kiến trúc Encoder–Decoder

U-Net gồm hai nửa đối xứng hình chữ **U**:

- **Encoder (nửa trái, đi xuống — "contracting path"):** giống hệt một CNN phân loại thông thường (thực tế dự án dùng ResNet-34 pretrained làm encoder qua thư viện `segmentation_models_pytorch`, xem `TUTORIAL.md` Phần 7.2) — càng đi sâu, không gian càng bị thu nhỏ (qua pooling/stride) nhưng số kênh càng tăng, trích xuất đặc trưng ngày càng trừu tượng, giống hệt nguyên lý ở Phần II.2.
- **Decoder (nửa phải, đi lên — "expansive path"):** làm ngược lại — từ feature map nhỏ, trừu tượng, **phóng to dần** (upsampling) trở lại kích thước ảnh gốc, đồng thời giảm dần số kênh, để cuối cùng ra mask có cùng `H×W` với ảnh input.

### Vì sao cần Skip Connection

Nếu chỉ nối encoder → decoder tuần tự (không skip connection): thông tin **vị trí chính xác** của pixel (ranh giới phổi nằm chính xác ở đâu) bị mất dần qua các lớp pooling ở encoder — pooling giữ lại "có đặc trưng này hay không" (Phần II.2) nhưng đánh đổi bằng việc làm mờ **vị trí chính xác**. Decoder một mình, chỉ dựa vào feature map đã bị nén ở đáy chữ U, **không thể khôi phục lại** độ chính xác pixel-level đó — kết quả là mask output bị mờ biên, không sắc nét.

**Skip connection** giải quyết bằng cách nối trực tiếp feature map ở encoder **cùng độ phân giải** sang decoder (ghép nối theo chiều kênh — `concatenate`, không phải cộng), mang theo thông tin không gian chi tiết chưa bị nén, giúp decoder "tham khảo" lại vị trí chính xác khi tái tạo mask.

```
Encoder (đi xuống)                              Decoder (đi lên)
─────────────────────                           ─────────────────────
Input 224×224×3
   │ conv+pool
   ▼
224×224×64  ──────────────skip───────────────►  224×224×(64+64)  → conv → Output 224×224×1
   │ conv+pool                                                  ▲
   ▼                                                             │ upsample+concat
112×112×128 ─────────────skip───────────────►  112×112×(128+128)
   │ conv+pool                                                  ▲
   ▼                                                             │ upsample+concat
56×56×256 ───────────────skip───────────────►  56×56×(256+256)
   │ conv+pool                                                  ▲
   ▼                                                             │ upsample+concat
28×28×512 ───────────────skip───────────────►  28×28×(512+512)
   │ conv+pool                                                  ▲
   ▼                                                             │ upsample
14×14×1024  ═══════════ đáy chữ U (bottleneck) ══════════════════
```

Số kênh tăng gấp đôi (64→128→256→512→1024) trong khi độ phân giải giảm một nửa mỗi tầng — đây chính xác là sự đánh đổi "trừu tượng hoá đặc trưng" đổi lấy "mất chi tiết không gian" đã nói ở Phần II.2, và skip connection là cơ chế "trả lại" phần đã mất đó.

### Upsampling — ConvTranspose vs Upsample+Conv

Để tăng kích thước feature map (ngược lại pooling), có hai cách chính:

- **Transposed Convolution (deconvolution):** một phép conv "ngược" có trọng số học được — mỗi input pixel "rải" giá trị của nó ra một vùng output lớn hơn theo kernel, các vùng chồng lấn được cộng lại. Nhược điểm: nếu `stride` không chia hết `kernel_size`, một số vị trí output nhận đóng góp từ nhiều pixel input hơn vị trí khác → tạo vệt caro (**checkerboard artifact**) — hoạ tiết lặp không tự nhiên trên mask.
- **Upsample (nearest/bilinear) + Conv thường:** phóng to bằng nội suy đơn giản trước (không có trọng số học được ở bước này), sau đó một lớp conv thường học cách "làm mịn/tinh chỉnh" kết quả đã phóng to. Không có checkerboard artifact, ít tham số hơn — đây là lựa chọn mặc định của `segmentation_models_pytorch` mà dự án dùng.

### Loss cho segmentation nhị phân — nhắc lại kết nối với Phần VI

U-Net output là **logits** `(N, 1, H, W)` (chưa qua sigmoid — lý do kỹ thuật: dùng `BCEWithLogitsLoss` ổn định số học hơn tách rời sigmoid+BCE, xem Phần VI.2). Kết hợp `0.5·BCE + 0.5·Dice` là loss mặc định của dự án (`BCEDiceLoss` trong `unet.py`).

## V.3. Input là gì

`(N, 3, 224, 224)` — cùng shape và chuẩn hoá như classifier (Phần IV.3). `in_channels=3` dù ảnh gốc là ảnh xám, vì encoder ResNet-34 pretrained kỳ vọng input 3 kênh (`TUTORIAL.md` Phần 7.4) — giữ nguyên lợi ích pretrained từ lớp conv đầu tiên.

## V.4. Output là gì

**Logits** `(N, 1, H, W) = (N, 1, 224, 224)` — một số thực cho **mỗi pixel**, chưa qua sigmoid. Sau khi áp `sigmoid`, mỗi giá trị nằm trong `(0,1)`, diễn giải là **xác suất pixel đó thuộc vùng phổi**. Nhị phân hoá bằng ngưỡng (`threshold=0.5` mặc định) cho ra mask nhị phân cuối cùng `{0, 1}` — khớp với dạng nhị phân hoá mask ground truth ở `dataset.py` (`mask = (mask > 0).astype(np.float32)`, gộp mọi giá trị lớp `{1,2,3}` từ `preprocess.py` thành nhãn nhị phân "có phổi/không").

## V.5. Ý nghĩa Output, ứng dụng, cách test

**Dice coefficient và IoU (Jaccard Index)** — hai chỉ số đo "mức chồng lấp" giữa mask dự đoán `P` và mask thật `G`:

```
Dice(P,G) = 2|P∩G| / (|P|+|G|)          IoU(P,G) = |P∩G| / |P∪G|

Quan hệ:  Dice = 2·IoU / (1+IoU)   (Dice luôn ≥ IoU với cùng một cặp mask)
```

### Ví dụ số

Giả sử trên một ảnh 10×10=100 pixel (đơn giản hoá để tính tay): mask thật `G` có 40 pixel phổi, mask dự đoán `P` có 36 pixel được gán "phổi", trong đó 30 pixel trùng đúng với `G` (phần giao `|P∩G|=30`).

```
|P∩G| = 30
|P| = 36,  |G| = 40
|P∪G| = |P| + |G| - |P∩G| = 36 + 40 - 30 = 46

Dice = 2×30 / (36+40) = 60/76 = 0.7895
IoU  = 30/46 = 0.6522

Kiểm tra quan hệ: 2×IoU/(1+IoU) = 2×0.6522/1.6522 = 1.3044/1.6522 = 0.7895  ✓ khớp Dice tính trực tiếp
```

**Ý nghĩa thực tế:** Dice=0.79 nghĩa là ~79% của "tổng diện tích trung bình 2 mask" là phần trùng khớp — mức chấp nhận được cho bài toán segmentation y tế nhưng chưa xuất sắc (mô hình tốt trên bài toán lung segmentation thường đạt Dice > 0.90). IoU luôn khắt khe hơn Dice cùng một cặp mask (0.65 < 0.79) vì mẫu số của IoU (`|P∪G|`) không nhân đôi phần giao như Dice — đây là lý do quy ước báo cáo cả hai chỉ số (Phần 2.4 `TUTORIAL.md`), tránh chỉ chọn chỉ số "đẹp số" hơn.

**Vì sao +1 (smoothing) trong `dice_score`/`iou_score`:** nếu cả `P` và `G` đều rỗng (ảnh không có phổi — hiếm nhưng có thể xảy ra ở ảnh lỗi), công thức gốc `0/0` (undefined). Cộng `1` vào cả tử và mẫu tránh chia 0 và cho kết quả `≈1` (đúng — "trùng khớp hoàn hảo" khi cả hai đều rỗng) mà không làm lệch đáng kể kết quả khi `|P|,|G|` đủ lớn.

**Cách test:** chạy `dice_score`/`iou_score` trên cặp mask **tự tạo tay** (như ví dụ số trên, viết thành ma trận numpy nhỏ) trước khi tin dùng trên dataset thật — nếu hàm cho đúng `0.7895` và `0.6522` như tính tay, hàm đã đúng.

---

# PHẦN VI — CÁC HÀM MẤT MÁT: CROSS-ENTROPY, BCE, DICE LOSS

## VI.1. Nó là gì

Đây là phần mở rộng cụ thể của Phần I.3 (Loss Function tổng quát) cho hai bài toán chính của dự án: phân loại đa lớp (Cross-Entropy, cho EfficientNet-B3) và phân đoạn nhị phân (BCE + Dice, cho U-Net).

## VI.2. Nguyên tắc hoạt động

### Cross-Entropy Loss — cho phân loại 3 lớp

Cross-Entropy đo "khoảng cách" giữa phân phối xác suất model dự đoán (`ŷ`, sau Softmax — Phần IV.5) và phân phối thật (`y`, dạng one-hot: `1` ở đúng lớp, `0` ở các lớp còn lại):

```
L_CE = - Σᵢ yᵢ · log(ŷᵢ)
```

Vì `y` là one-hot (chỉ một `yᵢ=1`, còn lại `=0`), tổng này thực chất chỉ còn **một số hạng** — trọng số của lớp đúng:

```
L_CE = -log(ŷ_c)     với c là index của lớp đúng
```

**Trực giác:** nếu model dự đoán xác suất lớp đúng gần 1 (rất tự tin và đúng) → `log(ŷ_c)` gần `log(1)=0` → loss gần 0. Nếu model dự đoán xác suất lớp đúng gần 0 (rất tự tin nhưng **sai**) → `log(ŷ_c)` tiến tới `-∞` → loss tiến tới `+∞` — Cross-Entropy phạt **cực nặng** các dự đoán sai mà tự tin cao, đây chính xác là hành vi mong muốn cho bài toán y tế: một model tự tin 99% "Normal" cho một ca thật sự là COVID phải bị phạt nặng hơn nhiều so với một model do dự 55/45.

### Ví dụ số

Từ ví dụ Phần IV.5: `ŷ = [P(Normal)=0.1970, P(Lung_Opacity)=0.0801, P(COVID)=0.7228]`.

**Trường hợp A — nhãn thật là COVID (đúng):**

```
L_CE = -log(0.7228) = -(-0.3245) = 0.3245
```

**Trường hợp B — giả sử nhãn thật là Normal (model sai, vì đã đoán COVID với 72%):**

```
L_CE = -log(0.1970) = -(-1.6243) = 1.6243
```

So sánh: loss ở trường hợp B (model sai) gấp **5 lần** loss ở trường hợp A (model đúng) — minh hoạ trực tiếp cơ chế "phạt nặng khi tự tin mà sai".

**Vì sao dùng `log` (không phải khoảng cách tuyến tính `1-ŷ_c`)?** Hàm `-log(x)` có đạo hàm `-1/x` — càng gần 0, đạo hàm càng lớn (tiến tới `-∞`) — nghĩa là gradient (tín hiệu học) càng **mạnh** khi model càng sai, thúc đẩy sửa lỗi nhanh hơn ở chính những trường hợp cần sửa nhất; ngược lại khi `ŷ_c` đã gần 1, đạo hàm nhỏ, tránh "học quá đà" khi đã gần đúng.

### Binary Cross-Entropy (BCE) — trường hợp đặc biệt 2 lớp, dùng pixel-wise cho U-Net

BCE là Cross-Entropy cho bài toán 2 lớp (ở đây: "là phổi" / "không phải phổi"), áp dụng **độc lập cho từng pixel**:

```
L_BCE = -(1/HW) Σ_{i,j} [ y_{ij}·log(ŷ_{ij}) + (1-y_{ij})·log(1-ŷ_{ij}) ]
```

`ŷ_{ij} = sigmoid(z_{ij})` — xác suất pixel `(i,j)` là phổi. Số hạng thứ nhất chỉ "kích hoạt" khi `y_{ij}=1` (pixel thật sự là phổi — phạt nếu `ŷ` thấp), số hạng thứ hai chỉ kích hoạt khi `y_{ij}=0` (phạt nếu `ŷ` cao nhầm).

**`BCEWithLogitsLoss` vì sao ổn định hơn `Sigmoid` + `BCELoss` tách rời?** Nếu `z` rất âm (ví dụ `z=-50`), `sigmoid(z)` gần `0` — tính `log(sigmoid(z))` riêng có thể gây lỗi số học `log(0) = -∞` hoặc mất độ chính xác dấu phẩy động. `BCEWithLogitsLoss` dùng công thức toán học tương đương nhưng viết lại để tránh tính `sigmoid` tường minh (dùng "log-sum-exp trick"), ổn định số học hơn với các giá trị logit cực trị — đây là lý do `unet.py` (Phần V.4/VI) luôn giữ output ở dạng logits, không tự áp sigmoid trước khi tính loss.

**Vấn đề mất cân bằng lớp (class imbalance) của BCE thuần.** Nếu vùng phổi chỉ chiếm 30% diện tích ảnh (nền chiếm 70%), một model "lười" luôn dự đoán "không phải phổi" cho mọi pixel vẫn đạt BCE tương đối thấp (đúng 70% pixel một cách "miễn phí", không thực sự học ranh giới phổi) — đây là động lực cần kết hợp thêm Dice Loss.

### Dice Loss — bù đắp nhược điểm mất cân bằng của BCE

Từ Dice coefficient (Phần V.5):

```
Dice(P,G) = 2|P∩G| / (|P|+|G|)
L_Dice = 1 - Dice
```

Ở dạng khả vi (differentiable) để dùng cho backpropagation, thay vì đếm pixel rời rạc (`|P∩G|` cần nhị phân hoá — không khả vi), dùng trực tiếp xác suất liên tục `ŷ_{ij} ∈ (0,1)`:

```
Dice_soft = 2·Σᵢⱼ(ŷᵢⱼ·yᵢⱼ) / [Σᵢⱼŷᵢⱼ + Σᵢⱼyᵢⱼ]
L_Dice = 1 - Dice_soft
```

**Vì sao Dice Loss không bị chi phối bởi lớp đa số như BCE?** Dice đo tỉ lệ **chồng lấp tương đối** giữa 2 vùng, không quan tâm tổng số pixel nền (background) đúng bao nhiêu — một model dự đoán toàn bộ ảnh là "nền" sẽ có `Σŷᵢⱼ yᵢⱼ ≈ 0` (không có phần giao với vùng phổi thật) → `Dice_soft ≈ 0` → `L_Dice ≈ 1` (loss tối đa) — bị phạt nặng ngay lập tức, khác hẳn hành vi "lách luật" mà BCE thuần cho phép.

### Ví dụ số — tính `BCEDiceLoss` trên một batch giả định nhỏ

Giả sử 4 pixel, logits `z = [2.0, -1.0, 0.5, -3.0]`, nhãn thật `y = [1, 0, 1, 0]`.

```
Bước 1 — sigmoid:
ŷ₁ = 1/(1+e^-2.0) = 0.8808
ŷ₂ = 1/(1+e^1.0)  = 0.2689
ŷ₃ = 1/(1+e^-0.5) = 0.6225
ŷ₄ = 1/(1+e^3.0)  = 0.0474

Bước 2 — BCE từng pixel: -[y·log(ŷ) + (1-y)·log(1-ŷ)]
pixel1 (y=1): -log(0.8808)          = 0.1269
pixel2 (y=0): -log(1-0.2689)        = -log(0.7311) = 0.3133
pixel3 (y=1): -log(0.6225)          = 0.4741
pixel4 (y=0): -log(1-0.0474)        = -log(0.9526) = 0.0486

L_BCE = trung bình = (0.1269+0.3133+0.4741+0.0486)/4 = 0.9629/4 = 0.2407

Bước 3 — Dice (soft), smooth=1:
Σ(ŷ·y) = 0.8808×1 + 0.2689×0 + 0.6225×1 + 0.0474×0 = 0.8808+0.6225 = 1.5033
Σŷ = 0.8808+0.2689+0.6225+0.0474 = 1.8196
Σy = 1+0+1+0 = 2

Dice_soft = (2×1.5033 + 1) / (1.8196 + 2 + 1) = (3.0066+1)/(4.8196) = 4.0066/4.8196 = 0.8313
L_Dice = 1 - 0.8313 = 0.1687

Bước 4 — kết hợp 0.5·BCE + 0.5·Dice:
L_total = 0.5×0.2407 + 0.5×0.1687 = 0.1204 + 0.0844 = 0.2047
```

Đây là con số cụ thể (`0.2047`) mà `BCEDiceLoss.forward()` trong `unet.py` trả về cho batch 4-pixel giả định trên — chính xác quy trình mà PyTorch thực hiện (trên quy mô `224×224×N` pixel thay vì 4).

## VI.3-4. Input/Output

**Input** của mọi loss ở trên: `(logits, target)`. Với Cross-Entropy: `logits` shape `(N, num_classes)`, `target` shape `(N,)` chứa index lớp đúng (PyTorch `CrossEntropyLoss` tự áp Softmax + log bên trong, không cần tự làm tay). Với BCE/Dice: `logits` và `target` cùng shape `(N, 1, H, W)`.

**Output**: một scalar (Phần I.3 đã giải thích lý do — cần một số duy nhất để lấy đạo hàm và tối ưu).

## VI.5. Ý nghĩa, ứng dụng, cách test

**`class_weight` trong CrossEntropyLoss** (Phần 5.4 `TUTORIAL.md`) nhân thêm một hệ số vào từng số hạng theo lớp, để lớp hiếm bị phạt nặng hơn khi model dự đoán sai — công thức trở thành `L = -w_c · log(ŷ_c)`. Với dataset đã cân bằng khá tốt (≤3000 ảnh/lớp) tác dụng này nhỏ, nhưng cần thiết nếu một lớp trong thực tế có ít ảnh hơn giới hạn.

**Cách test:** viết đúng ví dụ số ở trên (4 pixel) thành tensor PyTorch, gọi `BCEDiceLoss()(logits, target)`, kiểm tra kết quả có khớp `0.2047` (sai số làm tròn) — đây là unit test tối thiểu trước khi tin loss function không có bug trước khi bắt đầu train hàng giờ.

---

# PHẦN VII — GRAD-CAM: GIẢI THÍCH QUYẾT ĐỊNH CỦA MODEL

## VII.1. Nó là gì

Grad-CAM (Gradient-weighted Class Activation Mapping — Selvaraju et al., ICCV 2017) là một kỹ thuật **XAI (eXplainable AI)** trả lời câu hỏi: *"Model nhìn vào pixel nào của ảnh input để đưa ra quyết định lớp `c`?"* — không cần train thêm gì, không sửa kiến trúc model, chỉ cần một model đã train xong (ở đây: `EfficientNet-B3` đã fine-tune) và một lượt forward + backward.

Đây là thành phần bắt buộc của một "hệ thống có thể giải thích được" (explainable) như mục tiêu dự án đặt ra — bác sĩ không chỉ cần biết "COVID 72%" mà cần biết model dựa vào vùng nào của phổi để kết luận vậy, để có thể tự đánh giá tính hợp lý của gợi ý AI.

## VII.2. Nguyên tắc hoạt động

### Trực giác trước khi vào công thức

Nhắc lại Phần II.5: feature map ở lớp conv cuối cùng (trước GAP) có shape `(C, H', W') = (1536, 7, 7)` với B3 — mỗi trong 1536 "lát" 7×7 này là bản đồ không gian cho biết "đặc trưng thứ `k` xuất hiện mạnh ở đâu trên ảnh". Vấn đề: có 1536 lát như vậy, và **không phải lát nào cũng liên quan tới quyết định "đây là COVID"** — nhiều lát có thể mã hoá đặc trưng chung chung (độ sáng tổng thể, texture nền...). Grad-CAM dùng **gradient** để tìm ra: lát kênh nào (và mức độ bao nhiêu) thực sự ảnh hưởng tới điểm số của lớp `c` — chính là `∂y^c/∂A^k` (đạo hàm điểm số lớp c theo activation map kênh k) — sau đó cộng có trọng số các lát theo mức độ quan trọng đó.

### 5 bước toán học

**Bước 1 — Forward pass, lấy activation map mục tiêu.** Chạy ảnh qua model, lưu lại `A^k` — feature map tại lớp conv mục tiêu (kênh `k = 1..1536`, layer `model.features[-1]` cho B3), shape `(1536, 7, 7)`.

**Bước 2 — Lấy logit của lớp cần giải thích.** `y^c` — giá trị logit (trước Softmax) tại vị trí lớp `c` (ví dụ `c=COVID`), **không phải** xác suất sau Softmax — dùng logit vì Softmax làm các lớp "cạnh tranh" lẫn nhau (logit của một lớp giảm không hẳn vì đặc trưng lớp đó yếu đi, có thể vì lớp khác mạnh lên), làm nhiễu tín hiệu gradient nếu dùng xác suất.

**Bước 3 — Backward pass, tính gradient và pooling để ra trọng số kênh.**

```
α_k^c = (1/Z) Σᵢ Σⱼ  ∂y^c / ∂A^k_{ij}
```

`Z = H'×W' = 49` (số vị trí không gian trong feature map 7×7) — công thức này là **Global Average Pooling áp lên gradient**: với mỗi kênh `k`, lấy trung bình gradient trên toàn bộ 49 vị trí không gian, ra **một số duy nhất** `α_k^c` — "mức độ quan trọng trung bình của kênh `k` đối với lớp `c`".

**Bước 4 — Tổ hợp có trọng số + ReLU.**

```
L_Grad-CAM^c = ReLU( Σ_k α_k^c · A^k )
```

Nhân mỗi lát activation map `A^k` (7×7) với trọng số quan trọng `α_k^c` (một số) rồi **cộng dồn qua 1536 kênh** → ra **một** bản đồ 7×7 duy nhất. `ReLU` cắt bỏ mọi giá trị âm — chỉ giữ lại vùng có ảnh hưởng **dương** tới lớp `c` (vùng có ảnh hưởng âm — tức "bằng chứng chống lại lớp c" — bị loại khỏi heatmap, vì mục tiêu là "model nhìn vào đâu ĐỂ KẾT LUẬN lớp c", không phải "model nhìn vào đâu để loại trừ lớp c").

**Bước 5 — Resize và chuẩn hoá.** Bản đồ 7×7 được resize (nội suy bilinear) lên `224×224` (khớp ảnh gốc), rồi chuẩn hoá tuyến tính về `[0,1]` để có thể tô màu (colormap, thường "jet": xanh=thấp, đỏ=cao) và chồng (overlay) lên ảnh X-quang gốc.

### Ví dụ số — Grad-CAM tính tay trên feature map cực nhỏ (2 kênh, 2×2)

Giả sử (để tính tay được) lớp conv mục tiêu chỉ có **2 kênh**, mỗi kênh là feature map `2×2`:

```
A¹ = ⎡0.8  0.2⎤       A² = ⎡0.1  0.9⎤
     ⎣0.5  0.1⎦            ⎣0.3  0.6⎦
```

Giả sử backward pass (autograd) cho ra gradient của logit lớp COVID theo từng vị trí của mỗi kênh:

```
∂y^c/∂A¹ = ⎡0.4  0.4⎤       ∂y^c/∂A² = ⎡-0.2  -0.1⎤
           ⎣0.4  0.4⎦                  ⎣-0.3  -0.2⎦
```

**Bước 3 — GAP gradient ra trọng số kênh** (`Z=4`):

```
α₁^c = (0.4+0.4+0.4+0.4)/4 = 1.6/4 = 0.40
α₂^c = (-0.2-0.1-0.3-0.2)/4 = -0.8/4 = -0.20
```

Ý nghĩa: kênh 1 có ảnh hưởng **dương** (ủng hộ lớp COVID), kênh 2 có ảnh hưởng **âm** (bằng chứng chống lại lớp COVID — có thể là một đặc trưng đặc trưng cho "phổi khoẻ mạnh").

**Bước 4 — tổ hợp có trọng số:**

```
Σ_k α_k^c·A^k = 0.40×A¹ + (-0.20)×A²

= 0.40×⎡0.8  0.2⎤ + (-0.20)×⎡0.1  0.9⎤
       ⎣0.5  0.1⎦            ⎣0.3  0.6⎦

= ⎡0.32  0.08⎤ + ⎡-0.02  -0.18⎤
  ⎣0.20  0.04⎦   ⎣-0.06  -0.12⎦

= ⎡0.30  -0.10⎤
  ⎣0.14  -0.08⎦

ReLU(...) = ⎡0.30  0.00⎤
            ⎣0.14  0.00⎦
```

**Kết quả heatmap thô (trước resize):** góc trên-trái (`0.30`) là vùng ảnh hưởng mạnh nhất tới quyết định "COVID", góc dưới-trái (`0.14`) ảnh hưởng vừa, cột phải bị ReLU triệt tiêu hoàn toàn (dù `A²` có giá trị cao ở đó — `0.9` — nhưng vì gradient ở đó **âm**, đóng góp ròng bị loại bỏ). Đây minh hoạ điểm cốt lõi của Grad-CAM: heatmap **không phải** "nơi activation lớn nhất" (nếu vậy, đơn giản chỉ cần nhìn `A^k` mà không cần gradient) mà là "nơi activation lớn **và** có đóng góp dương thật sự vào quyết định lớp c" — chính vì kết hợp cả activation **lẫn** gradient mà kỹ thuật này khác với các phương pháp visualize CNN cũ hơn (như CAM gốc, chỉ dùng trọng số lớp Linear cuối, không tổng quát được cho kiến trúc có nhiều lớp FC hoặc GAP phức tạp).

### Vì sao chọn đúng layer mục tiêu quan trọng (Phần 2.5 `TUTORIAL.md`)

```
Lớp quá nông (conv đầu)          Lớp conv cuối (model.features[-1])       Sau GAP
────────────────────────         ──────────────────────────────           ─────────
Heatmap = bộ dò cạnh thô          Heatmap semantic — phản ánh              Không còn chiều không
(không phản ánh "bệnh lý",        pattern cấp cao model liên hệ            gian (đã average-pool
chỉ phản ánh viền/gradient        với quyết định bệnh lý                   thành 1 số) — KHÔNG
sáng-tối vật lý)                  ✓ LỰA CHỌN CHUẨN                         vẽ được heatmap 2D
```

## VII.3. Input là gì

Ảnh đã qua model (`(1,3,224,224)`, đã chuẩn hoá — Phần II.3), một model đã train (`weights/best_classifier.pth`), và **chỉ số lớp cần giải thích** `c` (thường lấy `c = argmax` của Softmax — giải thích cho chính lớp model vừa dự đoán, nhưng về nguyên tắc có thể chọn `c` bất kỳ để hỏi "nếu là lớp khác thì model sẽ nhìn vào đâu").

## VII.4. Output là gì

Một ma trận 2D `(H, W) = (224, 224)`, giá trị `∈[0,1]` sau chuẩn hoá — mỗi giá trị là "mức độ quan trọng" của pixel tương ứng đối với quyết định lớp `c`. Về mặt hình ảnh: một heatmap xám (hoặc sau khi tô colormap, một ảnh RGB) cùng kích thước ảnh gốc, sẵn sàng chồng (`overlay`, thường bằng `cv2.addWeighted`) lên ảnh X-quang gốc.

## VII.5. Ý nghĩa Output, ứng dụng, cách test

**Ứng dụng trực tiếp:** hiển thị trên Gradio UI song song với nhãn dự đoán và % tin cậy (Phần 1.3 `TUTORIAL.md`, `pipeline.md` bước 3) — bác sĩ nhìn heatmap để tự đánh giá "model có đang nhìn đúng chỗ không" trước khi tin vào kết quả.

**Cách test cơ bản:** heatmap trên một ảnh phổi bình thường thường trải đều/mờ nhạt (không có "điểm nóng" rõ rệt vì không có bất thường để model bám vào); heatmap trên ảnh COVID/Lung Opacity rõ ràng thường có "điểm nóng" tập trung ở vùng mờ đục (opacity) do bác sĩ đánh dấu trên phim gốc — nếu heatmap tập trung ở góc ảnh, viền phim, hoặc watermark thay vì vùng phổi, đây là **dấu hiệu cảnh báo shortcut learning** (Phần VIII) chứ không phải Grad-CAM tính sai — Grad-CAM chỉ "báo cáo trung thực" model đang nhìn vào đâu, kể cả khi đó là chỗ sai.

**Kiểm định định lượng (không chỉ nhìn bằng mắt):** đây chính là động lực của `shortcut_iou.py` (Phần VIII) — nhị phân hoá heatmap (ngưỡng, ví dụ giữ top 25% giá trị cao nhất) rồi tính IoU (công thức Phần V.5) giữa vùng heatmap "nóng" và mask phổi thật từ U-Net — IoU cao nhất quán qua nhiều ảnh là bằng chứng định lượng model học đúng đặc trưng bệnh lý trong phổi, không phải shortcut ngoài phổi.

---

# PHẦN VIII — SHORTCUT LEARNING & KIỂM ĐỊNH BẰNG IOU

## VIII.1. Nó là gì

Shortcut Learning (Geirhos et al., *Nature Machine Intelligence*, 2020) là hiện tượng một mạng nơ-ron đạt độ chính xác cao trên tập test **không phải vì** nó học đúng đặc điểm bản chất của bài toán, mà vì nó tìm ra một "đường tắt" — một đặc trưng dễ học hơn, tình cờ **tương quan** với nhãn trong dữ liệu train, nhưng không mang ý nghĩa nhân quả thật (không tổng quát hoá sang dữ liệu mới có phân phối hơi khác).

Ví dụ kinh điển được trích trong `TUTORIAL.md` Phần 2.6: Zech et al. (*PLoS Medicine*, 2018) — một CNN train trên X-quang bệnh viện A đạt accuracy cao trên test set **cùng bệnh viện A**, nhưng accuracy sụt mạnh khi test trên bệnh viện B. Nguyên nhân: máy chụp ở mỗi bệnh viện in một dấu hiệu (watermark/token) hơi khác nhau ở góc ảnh, và tình cờ bệnh viện A có tỉ lệ ca bệnh khác bệnh viện B — CNN học được "dấu bệnh viện A → nhiều khả năng là bệnh X" thay vì học đặc điểm bệnh lý thật trên mô phổi.

## VIII.2. Nguyên tắc hoạt động — vì sao shortcut xảy ra và cách kiểm định

**Vì sao mạng "thích" học shortcut hơn đặc trưng thật?** Gradient descent (Phần I.4) chỉ tối ưu **một mục tiêu duy nhất**: giảm loss trên tập train nhanh nhất có thể — nó không có khái niệm "đặc trưng nào đúng về mặt y khoa". Nếu một pixel watermark góc ảnh tương quan hoàn hảo với nhãn trong tập train (ví dụ do quy trình thu thập dữ liệu vô tình để lộ), gradient sẽ "phát hiện" tương quan này y hệt như phát hiện một đặc trưng bệnh lý thật — và vì watermark là pattern **đơn giản, cố định, dễ học** hơn nhiều so với texture mô phổi phức tạp, mạng có xu hướng ưu tiên học nó trước/nhiều hơn nếu không có ràng buộc nào ngăn cản.

**Cách kiểm định trong dự án — dùng chính U-Net làm "trọng tài":**

```
                    ┌─────────────────┐
Ảnh X-quang ───────►│  U-Net (đã train) │────► Mask phổi thật (ground-truth vị trí giải phẫu)
      │              └─────────────────┘                    │
      │                                                       │
      ▼                                                       ▼
┌──────────────────┐                                  ┌───────────────┐
│ EfficientNet-B3   │──logit lớp dự đoán──► Grad-CAM ──►│ Heatmap nhị     │
│ (đã train)         │                                  │ phân hoá        │
└──────────────────┘                                  └───────┬───────┘
                                                                 │
                                                                 ▼
                                                   IoU(Heatmap_nhị_phân, Mask_phổi)
                                                                 │
                                    IoU cao & nhất quán  ◄───────┴───────►  IoU thấp/thất thường
                                    → model nhìn đúng phổi              → cảnh báo shortcut
```

Đây chính là lý do kiến trúc tổng thể (Phần 1.2 `TUTORIAL.md`) cần **hai** model độc lập: U-Net không dùng để cải thiện accuracy phân loại trực tiếp (không crop ảnh trước khi đưa vào EfficientNet trong thiết kế mặc định), mà đóng vai trò **trọng tài độc lập** — nó được train trên một mục tiêu hoàn toàn khác (định vị giải phẫu, không liên quan nhãn bệnh), nên không thể "đồng loã" học chung một shortcut với classifier.

## VIII.3-4. Input/Output

**Input** của bước kiểm định: heatmap Grad-CAM (`(224,224)`, giá trị `[0,1]`, Phần VII.4) và mask phổi nhị phân từ U-Net (`(224,224)`, `{0,1}`, Phần V.4).

**Output**: một số IoU `∈[0,1]` (công thức Phần V.5) cho **mỗi ảnh**, và khi tổng hợp trên toàn bộ test set: phân phối (trung bình, độ lệch chuẩn, histogram) của các IoU đó.

## VIII.5. Ý nghĩa, ứng dụng, cách test

**Diễn giải ngưỡng (không có ngưỡng "đúng" tuyệt đối, nhưng theo thông lệ):**

| IoU trung bình trên test set | Diễn giải |
|---|---|
| > 0.5, ổn định qua các lớp | Model nhiều khả năng dựa vào vùng phổi để quyết định — dấu hiệu tốt |
| 0.2 – 0.5 | Mơ hồ — cần xem thêm heatmap từng ca cụ thể, không kết luận vội |
| < 0.2, hoặc lệch hẳn theo từng lớp bệnh | Cảnh báo shortcut — cần điều tra dữ liệu (watermark, viền phim khác nhau giữa các nguồn) trước khi tin dùng model |

**Vì sao đây không phải một "test pass/fail" đơn giản mà là báo cáo phân tích:** khác với accuracy (một con số duy nhất, so sánh trực tiếp với ngưỡng), shortcut learning cần nhìn **cả phân phối** IoU qua nhiều ảnh, và **đối chiếu qua từng lớp bệnh riêng** — một model có thể học đúng đặc trưng cho lớp Normal (dễ, phổi "sạch") nhưng học shortcut cho lớp COVID (khó phân biệt trực quan với Lung Opacity) — trung bình chung có thể che giấu vấn đề cục bộ này, đây là lý do report cuối (Phần 18 `TUTORIAL.md`) cần breakdown IoU theo từng lớp, không chỉ báo một con số tổng.

---

# PHẦN IX — CÁC CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH (EVALUATION METRICS)

## IX.1. Nó là gì

Đây là bộ công cụ đo lường **định lượng** chất lượng model sau khi train xong, dùng trên **tập test** (ảnh model chưa từng thấy trong lúc train lẫn lúc chọn hyperparameter ở tập validation) — trả lời câu hỏi khách quan "model này tốt tới đâu", tách biệt hẳn khỏi quá trình tối ưu (loss chỉ dùng để *train*, metrics dùng để *đánh giá* — hai vai trò không nên nhầm lẫn, dù đôi khi liên quan toán học gần gũi như Dice Loss/Dice score).

## IX.2. Nguyên tắc hoạt động

### Confusion Matrix — nền tảng của mọi metric phân loại

Với bài toán 3 lớp, Confusion Matrix là bảng `3×3`, hàng là nhãn thật, cột là nhãn model dự đoán:

```
                  Dự đoán:
                  Normal   Lung_Opacity   COVID
Thật: Normal        45          3           2      ← 50 ảnh Normal thật
      Lung_Opacity    4         41           5      ← 50 ảnh Lung_Opacity thật
      COVID            1          6          43      ← 50 ảnh COVID thật
```

Đường chéo chính (45, 41, 43) là số ảnh dự đoán **đúng**; mọi ô ngoài đường chéo là **nhầm lẫn** — ví dụ `4` ở hàng Lung_Opacity, cột Normal nghĩa là 4 ảnh Lung_Opacity thật bị model đoán nhầm thành Normal.

### Accuracy, Precision, Recall, F1 — định nghĩa qua TP/FP/FN/TN

Quy về bài toán nhị phân "một lớp `c` so với phần còn lại" (one-vs-rest) để định nghĩa 4 đại lượng nền tảng — lấy lớp **COVID** làm ví dụ từ bảng trên:

```
TP (True Positive)  = 43   — thật COVID, đoán đúng COVID
FN (False Negative) = 1+6 = 7    — thật COVID, đoán sai thành lớp khác (bỏ sót)
FP (False Positive) = 2+5 = 7    — thật không phải COVID, đoán nhầm thành COVID
TN (True Negative)  = 45+3+4+41 = 93   — thật không phải COVID, đoán đúng không phải COVID
```

```
Accuracy  = (TP+TN) / Tổng                  — tỉ lệ đoán đúng trên TOÀN BỘ (mọi lớp gộp lại)
Precision = TP / (TP+FP)                    — "trong các ca model bảo là COVID, bao nhiêu % thật sự là COVID"
Recall    = TP / (TP+FN)                    — "trong các ca thật sự COVID, model bắt được bao nhiêu %"
F1        = 2·Precision·Recall / (Precision+Recall)   — trung bình điều hoà (harmonic mean) của 2 số trên
```

### Ví dụ số — tính đầy đủ cho lớp COVID từ bảng Confusion Matrix trên

```
Precision_COVID = 43 / (43+7) = 43/50 = 0.8600  (86.00%)
Recall_COVID    = 43 / (43+7) = 43/50 = 0.8600  (86.00%)   [trùng ngẫu nhiên do FP=FN=7 trong ví dụ này]
F1_COVID        = 2×0.86×0.86 / (0.86+0.86) = 1.4792/1.72 = 0.8600

Accuracy (toàn bộ 3 lớp) = (45+41+43) / 150 = 129/150 = 0.8600  (86.00%)
```

**Vì sao Recall đặc biệt quan trọng trong y tế (nhắc lại từ `TUTORIAL.md`/`SoTay_ModelLead.md`):** `FN` (False Negative) trong bài toán này nghĩa là "bệnh nhân thật sự có COVID nhưng model báo không có" — bỏ sót một ca bệnh nguy hiểm hơn nhiều so với `FP` (báo nhầm một ca khoẻ mạnh là có bệnh, gây thêm một bước kiểm tra nhưng không nguy hiểm tính mạng). Vì vậy khi hai model có Accuracy ngang nhau, model có **Recall cao hơn cho lớp bệnh** thường được ưu tiên trong bối cảnh sàng lọc y tế, dù có thể đánh đổi Precision thấp hơn (nhiều báo động giả hơn).

### Macro F1 — vì sao dùng "Macro" thay vì "Micro"/"Weighted"

```
Macro F1 = (1/C) Σ_c F1_c              — trung bình CỘNG đơn giản qua từng lớp, KHÔNG tính theo trọng số số lượng ảnh mỗi lớp
```

Với dataset đã cân bằng khá tốt (`MAX_IMAGES_PER_CLASS=3000`, Phần 4.2 `TUTORIAL.md`), Macro F1 và Weighted F1 (trung bình có trọng số theo số ảnh mỗi lớp) sẽ gần nhau. Nhưng Macro F1 vẫn là lựa chọn chuẩn cho báo cáo học thuật vì nó đối xử **mọi lớp bình đẳng** — nếu một lớp ít ảnh hơn có F1 thấp, Macro F1 vẫn phản ánh đầy đủ (không bị "pha loãng" bởi lớp nhiều ảnh có F1 cao, như Weighted/Micro có thể gây ra).

### ROC-AUC (mở rộng, tham khảo)

Với bài toán đa lớp, có thể vẽ đường ROC (Receiver Operating Characteristic) cho từng lớp theo kiểu one-vs-rest: trục hoành là False Positive Rate (`FP/(FP+TN)`), trục tung là True Positive Rate (chính là Recall) tại các ngưỡng xác suất khác nhau (không chỉ ngưỡng 0.5 mặc định). Diện tích dưới đường này (AUC — Area Under Curve) `∈[0,1]`: `0.5` = model đoán ngẫu nhiên (không tốt hơn tung đồng xu), `1.0` = phân loại hoàn hảo. Đây là chỉ số bổ sung hữu ích khi cần đánh giá model **độc lập với việc chọn ngưỡng quyết định** cụ thể.

## IX.3. Input là gì

Một danh sách cặp `(nhãn thật, nhãn/xác suất dự đoán)` cho toàn bộ tập test — với phân loại: `(y_true, y_pred)` là các số nguyên chỉ lớp (hoặc `y_prob` cho ROC-AUC); với segmentation: cặp mask `(mask_true, mask_pred)` cho từng ảnh.

## IX.4. Output là gì

Một hoặc nhiều số vô hướng (Accuracy, Macro F1...) và/hoặc một ma trận (Confusion Matrix `C×C`) — luôn ở dạng **tổng hợp trên toàn tập test**, khác với loss (Phần I.3) vốn tính trên từng batch trong lúc train.

## IX.5. Ý nghĩa, ứng dụng, cách test

**Ứng dụng trong report (Phần 17-18 `TUTORIAL.md`):** báo cáo phải trình bày đủ cả 4 chỉ số (Accuracy, Precision, Recall, Macro F1) **theo từng lớp** cộng với Confusion Matrix trực quan (heatmap `seaborn`), không chỉ một con số Accuracy tổng — vì Accuracy tổng có thể "che giấu" model kém ở một lớp cụ thể nếu lớp đó ít ảnh hoặc dễ nhầm với lớp khác (như ví dụ COVID/Lung_Opacity dễ nhầm đã nêu ở Phần 2.2 `TUTORIAL.md`).

**Cách test hàm tính metric của chính mình:** dùng `sklearn.metrics.classification_report`/`confusion_matrix` trên chính ví dụ số ở trên (viết tay `y_true`, `y_pred` tương ứng đúng bảng Confusion Matrix `3×3` đã cho) — nếu `sklearn` trả về `Precision_COVID=0.86, Recall_COVID=0.86, F1_COVID=0.86, Accuracy=0.86` khớp với tính tay, đó là bằng chứng code đánh giá không có bug trước khi chạy trên tập test thật (~900+ ảnh).

---

# PHẦN X — VÍ DỤ SỐ END-TO-END XUYÊN SUỐT TOÀN BỘ PIPELINE

Phần này dựng một phiên bản **thu nhỏ cực độ** của toàn bộ hệ thống — một "ảnh X-quang" giả định `6×6` pixel, một CNN mini (1 lớp conv + pool + FC) đóng vai trò EfficientNet-B3 thu nhỏ, một nhánh decoder mini đóng vai trò U-Net thu nhỏ — và đi **một vòng đầy đủ**: tiền xử lý → forward → loss → backward → cập nhật trọng số → Grad-CAM → metrics. Mọi công thức dùng lại nguyên vẹn từ các phần trước; mục tiêu của phần này là cho thấy chúng **nối với nhau** thành một pipeline liên tục như thế nào, đúng như yêu cầu "nhìn rõ đường đi của dữ liệu qua các bước biến đổi và huấn luyện".

## X.1. Dữ liệu đầu vào — "ảnh X-quang" giả định

Một ảnh xám `6×6`, giá trị pixel gốc trong khoảng `[0,255]` (mô phỏng một vùng sáng bất thường ở góc trên-trái — giả lập tổn thương):

```
I_raw = ⎡200 210 190  40  35  30⎤
        ⎢205 220 195  38  42  33⎥
        ⎢195 200 180  45  40  36⎥
        ⎢ 50  45  48  60  55  58⎥
        ⎢ 42  40  44  58  60  62⎥
        ⎣ 38  36  40  55  57  60⎦
```

Nhãn thật cho ảnh này (bài toán phân loại nhị phân thu nhỏ để dễ tính tay — thay vì 3 lớp): `y=1` ("Bất thường"), `y=0` ("Bình thường"). Mask phân đoạn thật (vùng "bất thường", ứng vai trò mask phổi trong U-Net) — góc trên-trái `3×3`:

```
G = ⎡1 1 1 0 0 0⎤
    ⎢1 1 1 0 0 0⎥
    ⎢1 1 1 0 0 0⎥
    ⎢0 0 0 0 0 0⎥
    ⎢0 0 0 0 0 0⎥
    ⎣0 0 0 0 0 0⎦
```

## X.2. Tiền xử lý — chuẩn hoá (tương ứng Phần II.3 / `dataset.py`)

Thay vì dùng thống kê ImageNet (chỉ đúng khi có pretrained thật), với ví dụ tối giản này ta chuẩn hoá bằng min-max về `[0,1]` để phép tính gọn: `x = I_raw / 255`.

```
x = ⎡0.784 0.824 0.745 0.157 0.137 0.118⎤
    ⎢0.804 0.863 0.765 0.149 0.165 0.129⎥
    ⎢0.765 0.784 0.706 0.176 0.157 0.141⎥
    ⎢0.196 0.176 0.188 0.235 0.216 0.227⎥
    ⎢0.165 0.157 0.173 0.227 0.235 0.243⎥
    ⎣0.149 0.141 0.157 0.216 0.224 0.235⎦
```

(Trong pipeline thật, bước này là `A.Normalize(mean=MEAN, std=STD)` trong `dataset.py` — cùng vai trò "đưa pixel về thang đo mà mạng học ổn định", chỉ khác công thức chuẩn hoá cụ thể.)

## X.3. Forward pass qua "CNN mini" (đóng vai trò EfficientNet-B3 — Phần II & IV)

**Lớp Conv** — 1 kernel `3×3` dò "vùng sáng" (giống ví dụ Phần II.2, nhưng kernel này là trọng số **đã học** ở một bước train trước đó, không phải thiết kế tay):

```
K = ⎡0.5  0.5  0.5⎤     b_conv = -1.0
    ⎢0.5  0.5  0.5⎥
    ⎣0.5  0.5  0.5⎦
```

Trượt `K` qua `x` với `stride=2` (để feature map nhỏ nhanh, giống stride dùng để downsample trong EfficientNet), cho ra feature map `2×2` (áp dụng công thức `H_out=⌊(6-3)/2⌋+1=2`, Phần II.2):

```
Vị trí (0,0) — vùng x[0:3,0:3]:
Σ = 0.5×(0.784+0.824+0.745+0.804+0.863+0.765+0.765+0.784+0.706) = 0.5×7.040 = 3.520
z(0,0) = 3.520 + (-1.0) = 2.520

Vị trí (0,1) — vùng x[0:3,3:6] (vùng tối, đối chứng):
Σ = 0.5×(0.157+0.137+0.118+0.149+0.165+0.129+0.176+0.157+0.141) = 0.5×1.329 = 0.6645
z(0,1) = 0.6645 - 1.0 = -0.3355

Vị trí (1,0) — vùng x[2:5,0:3] (nửa sáng nửa tối, vì hàng 2 sáng còn hàng 3-4 tối):
Σ = 0.5×(0.765+0.784+0.706+0.196+0.176+0.188+0.165+0.157+0.173) = 0.5×3.310 = 1.655
z(1,0) = 1.655 - 1.0 = 0.655

Vị trí (1,1) — vùng x[2:5,3:6]:
Σ = 0.5×(0.176+0.157+0.141+0.235+0.216+0.227+0.227+0.235+0.243) = 0.5×1.857 = 0.9285
z(1,1) = 0.9285 - 1.0 = -0.0715
```

```
z_conv = ⎡ 2.520  -0.3355⎤
         ⎣ 0.655  -0.0715⎦

ReLU:  A = ⎡2.520  0.000⎤     (đúng như Phần II.2 — vùng khớp mạnh với kernel "còn sống",
           ⎣0.655  0.000⎦      vùng không khớp bị dập về 0)
```

**Global Average Pooling** (Phần II.5, cầu nối feature map → vector phân loại — ở đây chỉ có 1 kênh nên GAP ra 1 số):

```
GAP(A) = (2.520+0.000+0.655+0.000)/4 = 3.175/4 = 0.79375
```

**Lớp Fully-Connected (2 lớp output, thay Softmax 2 lớp cho "Bất thường/Bình thường")** — trọng số đã học từ trước:

```
W_fc = [1.2, -0.8]     b_fc = [0.1, 0.3]      (hàng 1 ứng lớp "Bất thường", hàng 2 ứng "Bình thường")

logit_bấtthường  = 1.2 × 0.79375 + 0.1 = 0.9525 + 0.1 = 1.0525
logit_bìnhthường = -0.8 × 0.79375 + 0.3 = -0.635 + 0.3 = -0.335
```

**Softmax** (Phần IV.5):

```
e^1.0525 = 2.8649,  e^-0.335 = 0.7154
Tổng = 3.5803

P(Bất thường)  = 2.8649/3.5803 = 0.8002  (80.02%)
P(Bình thường) = 0.7154/3.5803 = 0.1998  (19.98%)
```

Model dự đoán **"Bất thường"** với 80.02% tin cậy — đúng với nhãn thật `y=1`. Đây là **output cuối cùng** mà API trả về (`{"label": "Bất thường", "confidence": 0.8002}`), tương đương JSON `POST /predict` trả về theo `pipeline.md`.

## X.4. Tính Loss (Cross-Entropy — Phần VI.2)

```
L_CE = -log(P(lớp đúng)) = -log(0.8002) = 0.2231
```

Loss thấp (gần 0) khớp với việc model dự đoán đúng và khá tự tin — nhất quán với ví dụ Trường hợp A ở Phần VI.2.

## X.5. Backward pass — lan gradient ngược về kernel conv (Phần I.4)

Đặt lớp "Bất thường" là index 0. Với Cross-Entropy + Softmax kết hợp, đạo hàm rút gọn đẹp (một tính chất toán học nổi tiếng, tránh phải nhân riêng đạo hàm Softmax và đạo hàm `-log`):

```
∂L/∂logit_i = P(i) - y_i      (y one-hot: y=[1,0] vì nhãn đúng là "Bất thường")

∂L/∂logit_bấtthường  = 0.8002 - 1 = -0.1998
∂L/∂logit_bìnhthường = 0.1998 - 0 =  0.1998
```

**Gradient cho `W_fc`** (`∂logit_i/∂W_fc_i = GAP(A) = 0.79375`):

```
∂L/∂W_fc[bấtthường]  = -0.1998 × 0.79375 = -0.1586
∂L/∂W_fc[bìnhthường] =  0.1998 × 0.79375 =  0.1586
```

**Gradient lan về GAP output** (tổng theo cả 2 nhánh logit, vì GAP output đi vào cả hai neuron output):

```
∂L/∂GAP = (∂L/∂logit_bt)×W_fc[bt] + (∂L/∂logit_bth)×W_fc[bth]
        = (-0.1998)×1.2 + (0.1998)×(-0.8)
        = -0.2398 + (-0.1598)
        = -0.3996
```

**Gradient qua GAP** (mỗi vị trí trong `A` nhận đều `1/4` gradient của GAP output — vì GAP là trung bình cộng, đạo hàm phân bố đều, giống công thức Phần VII.2 bước 3 nhưng theo chiều ngược lại):

```
∂L/∂A_{ij} = ∂L/∂GAP × (1/4) = -0.3996/4 = -0.0999   (bằng nhau tại cả 4 vị trí)
```

**Gradient qua ReLU** (chỉ đi qua ở vị trí `z>0`, tức `(0,0)` và `(1,0)`; bị chặn ở `(0,1)` và `(1,1)` vì `z<0` ở đó):

```
∂L/∂z(0,0) = -0.0999 × 1 = -0.0999
∂L/∂z(0,1) = -0.0999 × 0 =  0.0000
∂L/∂z(1,0) = -0.0999 × 1 = -0.0999
∂L/∂z(1,1) = -0.0999 × 0 =  0.0000
```

**Gradient cho kernel `K`** — mỗi trọng số kernel nhận tổng đóng góp từ mọi vị trí nó tham gia (ở đây, để đơn giản, minh hoạ đóng góp từ vị trí `(0,0)` — vùng ảnh `x[0:3,0:3]`, mọi trọng số kernel đều = 0.5 nên đóng góp giống nhau về công thức, khác nhau về giá trị `x` nhân vào):

```
∂L/∂K[0,0] += ∂L/∂z(0,0) × x[0,0] = -0.0999 × 0.784 = -0.0783   (từ vị trí (0,0))
                                    + tương tự cộng dồn từ vị trí (1,0) với x[2,0]=0.765:
                                    -0.0999 × 0.765 = -0.0764
∂L/∂K[0,0]_tổng = -0.0783 + (-0.0764) = -0.1547
```

(Các phần tử khác của kernel tính hoàn toàn tương tự — cộng dồn gradient từ mọi vị trí trượt kernel qua, đúng nguyên lý *weight sharing* nêu ở Phần II.1: một trọng số dùng lại nhiều nơi thì nhận gradient **cộng dồn** từ tất cả các nơi đó.)

## X.6. Cập nhật trọng số — AdamW bước đơn giản hoá (Phần I.5)

Với learning rate `η=0.1`, dùng gradient descent thuần (bỏ qua động lượng để tính tay đơn giản — minh hoạ nguyên lý, không phải Adam đầy đủ):

```
W_fc[bấtthường]_mới  = 1.2 - 0.1×(-0.1586) = 1.2 + 0.01586 = 1.2159
W_fc[bìnhthường]_mới = -0.8 - 0.1×(0.1586) = -0.8 - 0.01586 = -0.8159
K[0,0]_mới            = 0.5 - 0.1×(-0.1547) = 0.5 + 0.01547 = 0.5155
```

**Ý nghĩa của bước cập nhật này:** `W_fc[bấtthường]` tăng nhẹ (từ 1.2 → 1.2159) — logic hợp lý: model đã đoán đúng nhưng loss vẫn dương (`0.2231`, chưa = 0), gradient đẩy trọng số theo hướng làm model **tự tin hơn nữa** ở lần dự đoán tiếp theo cho input tương tự. `K[0,0]` cũng tăng nhẹ — kernel "học" khớp mạnh hơn với đúng vùng sáng đã giúp nó dự đoán đúng. Lặp lại pha forward→loss→backward→update này hàng nghìn lần (một epoch = một lượt qua hết tập train, chia thành nhiều batch) là toàn bộ nội dung của `train_classifier.ipynb`.

## X.7. Grad-CAM trên chính mạng mini này (Phần VII.2)

Dùng `A` (activation sau ReLU, **trước** khi cập nhật trọng số — Grad-CAM luôn chạy trên model ở trạng thái hiện tại) và gradient đã tính:

```
α^c (chỉ 1 kênh, GAP gradient trên A):
α = (∂L/∂z(0,0) + ∂L/∂z(0,1) + ∂L/∂z(1,0) + ∂L/∂z(1,1)) / 4
  = (-0.0999 + 0 + -0.0999 + 0)/4 = -0.1998/4 = -0.04995
```

Lưu ý: Grad-CAM chuẩn dùng gradient của **logit lớp cần giải thích** (`y^c`, Phần VII.2 bước 2), không dùng gradient của **loss** như bước X.5 vừa tính (loss đã lẫn cả thông tin về nhãn thật `y`, còn Grad-CAM chỉ muốn hỏi "vì sao model nghĩ đây là lớp c", không quan tâm nhãn thật) — nhưng vì `∂L/∂logit = P(i)-y_i`, và ta đang giải thích đúng lớp mà `y_i=1`, dấu **âm** ở trên chỉ là do quy ước đạo hàm loss (giảm loss = tăng logit); nếu dùng đúng `∂logit/∂A` (không qua loss) thay vì `∂L/∂A`, dấu sẽ dương, phản ánh đúng "activation này *ủng hộ* lớp Bất thường". Điểm mấu chốt cần nhớ khi code thật (đúng như cách thư viện `pytorch-grad-cam` làm): `target = logit[c]`, gọi `target.backward()`, **không phải** `loss.backward()`.

Tính lại đúng chuẩn: `∂logit_bấtthường/∂A_{ij}` truyền qua `W_fc[bấtthường]=1.2` (trước khi qua GAP) rồi qua ReLU:

```
∂logit_bt/∂GAP = 1.2
∂logit_bt/∂A_{ij} = 1.2/4 = 0.30  (mọi vị trí, trước ReLU-mask)
Qua ReLU-mask (chỉ giữ (0,0) và (1,0)): α = (0.30+0+0.30+0)/4 = 0.15

Heatmap = ReLU(α × A) = ReLU(0.15 × A) = 0.15 × ⎡2.520 0.000⎤ = ⎡0.378 0.000⎤
                                                  ⎣0.655 0.000⎦   ⎣0.098 0.000⎦
```

**Kết quả:** vùng `(0,0)` (góc trên-trái — đúng vùng có mask bất thường thật `G`) có giá trị heatmap cao nhất (`0.378`) — Grad-CAM "chỉ đúng" vào vùng tổn thương giả định, khớp trực giác mong đợi.

## X.8. Kiểm định bằng IoU với mask thật (Phần VIII)

Nhị phân hoá heatmap ở ngưỡng ví dụ `>0.05` (sau khi resize lên `6×6` bằng nội suy — ở đây đơn giản hoá, coi mỗi ô `2×2` heatmap tương ứng khối `3×3` trên ảnh gốc `6×6`):

```
Heatmap_resize (6×6, mỗi giá trị lặp lại cho khối 3×3 tương ứng):
⎡0.378 0.378 0.378 0.000 0.000 0.000⎤
⎢0.378 0.378 0.378 0.000 0.000 0.000⎥
⎢0.378 0.378 0.378 0.000 0.000 0.000⎥
⎢0.098 0.098 0.098 0.000 0.000 0.000⎥
⎢0.098 0.098 0.098 0.000 0.000 0.000⎥
⎣0.098 0.098 0.098 0.000 0.000 0.000⎦

Nhị phân hoá (>0.05) → P = ⎡1 1 1 0 0 0⎤     (khối trên: qua ngưỡng)
                            ⎢1 1 1 0 0 0⎥     (khối dưới: 0.098>0.05, CŨNG qua ngưỡng
                            ⎢1 1 1 0 0 0⎥      trong ví dụ số này — minh hoạ P có thể
                            ⎢1 1 1 0 0 0⎥      RỘNG HƠN vùng thật G)
                            ⎢1 1 1 0 0 0⎥
                            ⎣1 1 1 0 0 0⎦
```

So với `G` (chỉ 3×3 trên cùng, `|G|=9`): `|P|=18`, `|P∩G|=9` (toàn bộ G nằm trong P), `|P∪G|=18`.

```
IoU = 9/18 = 0.50
Dice = 2×9/(18+9) = 18/27 = 0.667
```

**Ý nghĩa:** IoU=0.50 cho thấy Grad-CAM "chỉ đúng hướng" (toàn bộ vùng tổn thương thật nằm trong vùng model chú ý) nhưng **không chụm** (model còn chú ý cả một vùng thừa bên dưới, có thể do kernel/ngưỡng nhị phân hoá trong ví dụ này quá rộng) — minh hoạ đúng ý nghĩa thực tế của chỉ số IoU đã bàn ở Phần VIII.5: không phải "đúng/sai" nhị phân mà là một **thang đo mức độ** cần diễn giải có ngữ cảnh.

## X.9. Tổng kết một vòng dữ liệu hoàn chỉnh

```
I_raw (pixel thô)
   │  chuẩn hoá (Phần II.3)
   ▼
x (tensor chuẩn hoá)
   │  conv + ReLU (Phần II.2)
   ▼
A (feature map, activation)
   │  GAP (Phần II.5)          │
   ▼                            │ (giữ lại A cho Grad-CAM)
GAP(A)                          │
   │  Linear (Phần I.2)         │
   ▼                            │
logits                          │
   │  Softmax (Phần IV.5)       │
   ▼                            │
xác suất → dự đoán + %tin cậy   │
   │  Cross-Entropy (Phần VI.2) │
   ▼                            │
Loss (scalar)                   │
   │  backward() (Phần I.4)     │
   ▼                            │
gradient mọi trọng số           │
   │  AdamW update (Phần I.5)   ▼
   ▼                       target=logit[c] → backward() (KHÔNG dùng Loss)
Trọng số mới                    │
                                 ▼
                            Grad-CAM heatmap (Phần VII.2)
                                 │
                                 ▼
                     IoU với mask U-Net thật (Phần VIII) → báo cáo shortcut learning
```

Đây chính là "đường đi của dữ liệu" mà toàn bộ Phần I–IX mô tả rời rạc — trong pipeline thật của dự án (Phần XI), mỗi mũi tên trên tương ứng với một hàm/file cụ thể trong `src/` và `notebooks/`.

---

# PHẦN XI — KẾT NỐI LÝ THUYẾT VỚI PIPELINE THỰC TẾ CỦA DỰ ÁN

Bảng dưới đây ánh xạ từng khái niệm lý thuyết đã trình bày (Phần I–IX) sang đúng file/hàm/biến trong repo — để khi đọc code thật, bạn biết ngay "đoạn này đang thực hiện công thức nào ở phần nào của tài liệu".

| Khái niệm lý thuyết | Công thức chính | File / hàm trong dự án | Trạng thái hiện tại |
|---|---|---|---|
| Chuẩn hoá ảnh (Phần II.3) | `(x-mean)/std` theo ImageNet stats | `src/dataset.py` → `MEAN`, `STD`, `get_train_transforms()` | Đã có sẵn resize+label ở `preprocess.py`; `dataset.py` (Albumentations transform) là việc cần code theo Giai đoạn 2 của `TUTORIAL.md` |
| Convolution, Pooling, Receptive Field (Phần II.2) | `S(i,j)=ΣΣ I(i+m,j+n)K(m,n)+b` | Bên trong `torchvision.models.efficientnet_b3` (đã cài sẵn kiến trúc, không tự viết layer) | Dùng qua `build_classifier()` trong `src/model.py` (Giai đoạn 3) |
| MBConv, Compound Scaling (Phần IV.2) | `d=α^φ, w=β^φ, r=γ^φ` | Kiến trúc nội bộ `efficientnet_b3` (torchvision) | Không tự cài đặt — dùng qua pretrained weights `EfficientNet_B3_Weights.IMAGENET1K_V1` |
| Transfer Learning theo pha (Phần III.2) | `requires_grad=False/True` theo từng nhóm tham số | `src/model.py` → `freeze_backbone()`, `unfreeze_last_blocks()`, `unfreeze_all()` | Skeleton đã có trong `TUTORIAL.md` Giai đoạn 3 — cần code + gọi đúng thứ tự trong `train_classifier.ipynb` |
| Softmax, Cross-Entropy (Phần IV.5, VI.2) | `P(i)=e^zi/Σe^zj`, `L=-log(ŷ_c)` | `nn.CrossEntropyLoss()` (PyTorch built-in, tự gộp Softmax+log) trong vòng lặp train của `train_classifier.ipynb` | Chưa code — thuộc Giai đoạn 5 |
| Backpropagation, Gradient Descent, AdamW (Phần I.4, I.5) | `w ← w - η·m̂/(√v̂+ε)` | `loss.backward()` + `torch.optim.AdamW` trong vòng lặp train | Chưa code — Giai đoạn 5/6 |
| Encoder-Decoder, Skip Connection (Phần V.2) | Concatenate feature map cùng độ phân giải | `src/unet.py` → `build_unet()` qua `segmentation_models_pytorch.Unet` | Skeleton có sẵn trong `TUTORIAL.md` Giai đoạn 4 |
| BCE + Dice Loss (Phần VI.2) | `0.5·BCE + 0.5·Dice` | `src/unet.py` → class `BCEDiceLoss` | Skeleton có sẵn — cần dùng trong `train_unet.ipynb` |
| Dice score, IoU score (Phần V.5) | `2|P∩G|/(|P|+|G|)`, `|P∩G|/|P∪G|` | `src/unet.py` → `dice_score()`, `iou_score()` | Skeleton có sẵn |
| Grad-CAM (Phần VII.2) | `L^c = ReLU(Σ_k α_k^c·A^k)` | `src/gradcam.py` (dự kiến dùng thư viện `pytorch-grad-cam`, target layer `model.features[-1]`) | Chưa code — Giai đoạn 7, cần classifier train xong trước |
| Shortcut learning / kiểm định IoU (Phần VIII) | IoU(heatmap nhị phân, mask U-Net) | `src/shortcut_iou.py` | Chưa code — Giai đoạn 8, cần cả classifier + U-Net + gradcam |
| Confusion Matrix, Precision/Recall/F1 (Phần IX) | Công thức TP/FP/FN/TN | `sklearn.metrics.classification_report`, `confusion_matrix` trong notebook train + `docs`/report cuối | Thuộc Giai đoạn 5/6 (đánh giá sau khi train) và Phần 17 `TUTORIAL.md` |
| Toàn bộ pipeline dữ liệu thô → tensor | `LABELS`, `CLASS_TO_IDX`, `RANDOM_SEED=42`, `IMAGE_SIZE=(224,224)` | `src/preprocess.py`, `src/split_data.py` | **Đã code và chạy được** — nền tảng cho mọi phần lý thuyết ở trên |
| BatchNorm (Phần II.6) | `ŷ=(z-μ_B)/√(σ_B²+ε); y=γŷ+β` | Nội bộ `efficientnet_b3`/`resnet34` (torchvision/SMP), không tự cài | Tự động có sẵn qua pretrained — chỉ cần hiểu để debug train/eval mode |
| Dropout (Phần II.7) | `a_dropout=(a⊙mask)/(1-p)` | `model.classifier[0]` (`nn.Dropout`) trong `src/model.py` | Có sẵn trong kiến trúc `efficientnet_b3` gốc |
| `model.train()`/`model.eval()` (Phần II.8) | Chuyển chế độ BatchNorm/Dropout | Đầu mỗi vòng lặp train/val trong notebook, đầu `api/inference.py` | Chưa code — Giai đoạn 5/6/9, lỗi hay gặp nếu quên |
| Learning Rate Scheduling (Phần I.6) | Step Decay, Cosine Annealing, ReduceLROnPlateau | `torch.optim.lr_scheduler.*` trong `train_classifier.ipynb`/`train_unet.ipynb` | Chưa code — Giai đoạn 5/6, tuỳ chọn nhưng khuyến nghị |
| Vanishing/Exploding Gradient, Weight Init (Phần I.7, I.8) | Tích chain rule qua nhiều lớp; He/Xavier init | Đã giải quyết sẵn nhờ pretrained weights + kiến trúc residual của `efficientnet_b3`/`resnet34` | Chỉ áp dụng tường minh cho lớp `Linear` mới thay ở head |
| CLAHE (Phần II.9) | Cân bằng histogram theo CDF, có clip | `cv2.createCLAHE()` — dự kiến thêm vào `src/preprocess.py` | **Chưa implement** — có trong `pipeline.md`/`description.md` nhưng chưa có trong code hiện tại |

**Vì sao nắm được bảng này quan trọng:** mỗi hàng trên là một "điểm nối" giữa lý thuyết bạn đã học (Phần I–IX) và một dòng code cụ thể bạn sẽ viết hoặc đã có sẵn trong repo. Khi debug (ví dụ `train_classifier.ipynb` cho loss NaN), việc biết chính xác công thức toán học nào đang chạy ở dòng code nào giúp thu hẹp nguyên nhân nhanh hơn nhiều so với thử-sai không định hướng (ví dụ: NaN sau khi unfreeze → tra lại Phần III.2/I.5 về catastrophic forgetting và learning rate theo pha, thay vì đoán mò).

---

# PHẦN XII — TỔNG KẾT & TÀI LIỆU THAM KHẢO

## XII.1. Tổng kết các nguyên lý cốt lõi

1. **Mọi thứ trong dự án đều xây trên một cơ chế học duy nhất:** gradient descent qua backpropagation (Phần I.4) — CNN, EfficientNet, U-Net chỉ là các cách sắp xếp neuron khác nhau, không phải các thuật toán học khác nhau.
2. **CNN "thắng" MLP trên ảnh nhờ 2 nguyên lý:** kết nối cục bộ + chia sẻ trọng số (Phần II.1) — giảm tham số theo cấp số nhân, tạo tính bất biến dịch chuyển.
3. **Transfer Learning là bắt buộc, không phải tuỳ chọn**, với dataset cỡ ~9.000 ảnh so với ~10.7M tham số EfficientNet-B3 (Phần III.2) — và phải fine-tune **theo pha** để tránh catastrophic forgetting.
4. **U-Net cần skip connection** vì encoder/pooling đánh đổi độ chính xác không gian lấy tính trừu tượng hoá — decoder một mình không phục hồi lại được (Phần V.2).
5. **Loss không phải là metric** — BCE/Dice/Cross-Entropy dùng để *tối ưu* (cần khả vi), Accuracy/F1/IoU dùng để *đánh giá* (không cần khả vi, chỉ cần đúng ý nghĩa nghiệp vụ) — hai vai trò tách biệt dù công thức có thể gần gũi (Dice Loss vs Dice score).
6. **Grad-CAM giải thích bằng cách kết hợp activation VÀ gradient**, không phải chỉ activation — đây là lý do nó phản ánh đúng "model dùng gì để quyết định", không chỉ "model thấy gì" (Phần VII.2).
7. **Accuracy cao không đồng nghĩa model học đúng** — shortcut learning là rủi ro thật, cần kiểm định độc lập bằng một model khác (U-Net) không chia sẻ mục tiêu học với classifier (Phần VIII).
8. **Mạng sâu train được là nhờ một chuỗi kỹ thuật ổn định gradient phối hợp với nhau**, không phải một yếu tố đơn lẻ: khởi tạo trọng số đúng cách (He/Xavier — Phần I.8) chống bùng/nổ gradient ngay từ đầu, skip connection (Phần I.7/V.2) cho gradient một "đường tắt" xuyên suốt mạng, BatchNorm (Phần II.6) giữ phân phối activation ổn định qua từng lớp, và learning rate scheduler (Phần I.6) điều chỉnh tốc độ học phù hợp theo từng giai đoạn train — thiếu bất kỳ mắt xích nào cũng có thể khiến mạng ~10-30 lớp như EfficientNet-B3/ResNet-34 không hội tụ.
9. **`model.train()`/`model.eval()` không phải thủ tục hình thức** — BatchNorm và Dropout thực sự tính toán khác nhau giữa hai chế độ (Phần II.8); quên chuyển chế độ là nguồn lỗi phổ biến nhất khiến một model "chạy được nhưng cho kết quả sai/không ổn định" mà không có exception nào báo hiệu.

## XII.2. Gợi ý sử dụng tài liệu này khi làm việc thực tế

- Trước khi code một file `.py`/notebook nào trong `TUTORIAL.md`, đọc lại phần lý thuyết tương ứng trong bảng ánh xạ ở Phần XI.
- Khi loss/metric ra kết quả bất thường (NaN, không giảm, quá cao/thấp bất thường), quay lại đúng công thức trong Phần I/VI/IX, thay số thật vào để kiểm tra tay xem có khớp kỳ vọng không — kỹ thuật này chính là cách các ví dụ số trong tài liệu được thiết kế để bạn luyện tập.
- Khi viết report/luận văn, các công thức và ví dụ số trong tài liệu này có thể dùng trực tiếp làm phần "Cơ sở lý thuyết" — miễn trích dẫn đúng nguồn gốc paper (danh sách dưới).

## XII.3. Tài liệu tham khảo

- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV 2017.
- Geirhos, R., et al. (2020). *Shortcut Learning in Deep Neural Networks*. Nature Machine Intelligence.
- Zech, J. R., et al. (2018). *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study*. PLoS Medicine.
- Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR 2015.
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization* (AdamW). ICLR 2019.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors*. Nature.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press — chương 6-9 cho nền tảng MLP/backprop/CNN.
- Tài liệu nội bộ dự án: `docs/TUTORIAL.md`, `SoTay_ModelLead.md`, `pipeline.md`, `description.md` — dùng song song với tài liệu này: các file đó trả lời "code như thế nào", tài liệu này trả lời "vì sao nó hoạt động".

---

*Hết tài liệu. Mọi công thức, ví dụ số trong tài liệu này được thiết kế để tính lại bằng tay hoặc bằng vài dòng Python/NumPy — khuyến khích tự làm lại ít nhất một ví dụ số ở mỗi phần trước khi coi là đã hiểu, đúng tinh thần "tự gõ lại, không copy-paste mù" mà `TUTORIAL.md` đã nêu.*
