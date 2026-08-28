"""Load model 1 lần lúc startup, cung cấp predict_image() dùng chung cho api/main.py.

Nếu weights/best_classifier.pth hoặc weights/best_unet.pth chưa tồn tại (chưa train),
tự động fallback sang backbone pretrained ImageNet (đầu ra ngẫu nhiên với classifier,
mask vô nghĩa với U-Net), để backend vẫn khởi động và phục vụ được — dự đoán khi đó
không có ý nghĩa nhưng đúng shape/HTTP contract, đủ để test luồng API/UI trước khi có
model thật. Xem docs/QUY_TRINH_CODE.md Phần 8.3-8.4.

Ngoài classifier, còn load U-Net để tính "độ tin cậy giải thích": so khớp Grad-CAM
heatmap với mask phổi U-Net dự đoán bằng IoU/containment (cùng công thức dùng trong
src/shortcut_iou.py để kiểm định shortcut learning toàn tập test) — trả về NGAY trong
mỗi response, biến việc kiểm định thành một chỉ số minh bạch cho người dùng cuối, không
chỉ nằm trong báo cáo. Xem docs/LY_THUYET.md Phần VIII.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from src.dataset import (
    IDX_TO_CLASS,
    IMAGE_SIZE,
    NUM_CLASSES,
    crop_to_lung_bbox,
    crop_to_lung_bbox_blackout,
    get_val_transforms,
)
from src.gradcam import generate_gradcam, overlay_heatmap
from src.model import build_classifier, load_classifier
from src.shortcut_iou import binarize, containment, dice, iou, predict_lung_mask
from src.unet import build_unet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = Path("weights/best_classifier.pth")
CROPPED_WEIGHTS_PATH = Path("weights/best_classifier_cropped.pth")  # bản "đã tối ưu" — xem
# notebooks/train_classifier_cropped.ipynb, docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md Phần 5.
BLACKOUT_WEIGHTS_PATH = Path("weights/best_classifier_blackout.pth")  # bản "tối ưu hơn nữa" —
# xoá hẳn pixel ngoài hình dạng phổi (không chỉ crop bounding box), xem
# notebooks/train_classifier_blackout.ipynb, docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md Phần 5.4/6.
UNET_WEIGHTS_PATH = Path("weights/best_unet.pth")
DATA_PROCESSED_DIR = Path("data/processed")
DATASET_CLASSES = ("COVID", "Lung_Opacity", "Normal")
GRADCAM_THRESH = 0.5
CROP_PADDING = 0.1  # PHẢI khớp giá trị dùng lúc train notebooks/train_classifier_cropped.ipynb
LOW_TRUST_CONTAINMENT = 0.3  # dưới ngưỡng này: cảnh báo model có thể đang nhìn ngoài phổi

_classifier = None
_unet = None
_transform = get_val_transforms()
_model_is_trained: bool = False
_unet_is_trained: bool = False
_crop_mode: bool = False  # True nếu đang dùng classifier bản train trên ảnh crop theo mask
                          # (cropped HOẶC blackout — cả 2 đều cần luồng "U-Net trước, crop sau")
_blackout_mode: bool = False  # True CHỈ khi đang dùng bản blackout cụ thể (thêm bước xoá
                               # pixel ngoài mask, không chỉ crop bounding box — xem _crop_mode)

BASE_DISCLAIMER = (
    "Kết quả chỉ mang tính tham khảo, KHÔNG thay thế chẩn đoán y khoa chính thức. "
    "Vui lòng tham vấn bác sĩ chuyên khoa."
)
UNTRAINED_WARNING = (
    f"⚠️ CẢNH BÁO: không tìm thấy {WEIGHTS_PATH} — đang dùng backbone EfficientNet-B3 "
    "pretrained ImageNet với classification head KHỞI TẠO NGẪU NHIÊN (chưa huấn luyện). "
    "Dự đoán này KHÔNG có ý nghĩa chẩn đoán, chỉ phục vụ kiểm tra luồng kỹ thuật."
)


def load_models() -> None:
    """Gọi 1 lần lúc server khởi động (xem api/main.py::lifespan).

    Ưu tiên tự động theo thứ tự: BLACKOUT_WEIGHTS_PATH > CROPPED_WEIGHTS_PATH >
    WEIGHTS_PATH (baseline) — dùng bản "tối ưu nhất" đang có, fallback dần xuống nếu
    file không tồn tại hoặc load lỗi. Nếu chưa có bản nào ngoài baseline (đúng trạng
    thái trước khi train blackout), hành vi giữ NGUYÊN VẸN như trước. Code này AN TOÀN
    để merge ngay bây giờ — không đổi hành vi demo hiện tại cho tới khi
    weights/best_classifier_blackout.pth thực sự xuất hiện.
    """
    global _classifier, _model_is_trained, _unet, _unet_is_trained, _crop_mode, _blackout_mode

    if BLACKOUT_WEIGHTS_PATH.exists():
        try:
            _classifier = load_classifier(
                str(BLACKOUT_WEIGHTS_PATH), num_classes=NUM_CLASSES, device=DEVICE, verbose=True
            )
            _model_is_trained = True
            _crop_mode = True
            _blackout_mode = True
            print(f"[inference] OK: loaded OPTIMIZED (blackout) classifier from {BLACKOUT_WEIGHTS_PATH} on device={DEVICE}")
        except Exception as exc:
            print(
                f"[inference] WARNING: found {BLACKOUT_WEIGHTS_PATH} but failed to load it "
                f"({type(exc).__name__}: {exc}) — falling back to cropped/baseline classifier."
            )

    if _classifier is None and CROPPED_WEIGHTS_PATH.exists():
        try:
            _classifier = load_classifier(
                str(CROPPED_WEIGHTS_PATH), num_classes=NUM_CLASSES, device=DEVICE, verbose=True
            )
            _model_is_trained = True
            _crop_mode = True
            print(f"[inference] OK: loaded OPTIMIZED (cropped) classifier from {CROPPED_WEIGHTS_PATH} on device={DEVICE}")
        except Exception as exc:
            print(
                f"[inference] WARNING: found {CROPPED_WEIGHTS_PATH} but failed to load it "
                f"({type(exc).__name__}: {exc}) — falling back to baseline classifier."
            )

    if _classifier is None and WEIGHTS_PATH.exists():
        try:
            _classifier = load_classifier(
                str(WEIGHTS_PATH), num_classes=NUM_CLASSES, device=DEVICE, verbose=True
            )
            _model_is_trained = True
            _crop_mode = False
            _blackout_mode = False
            print(f"[inference] OK: loaded trained classifier from {WEIGHTS_PATH} on device={DEVICE}")
        except Exception as exc:
            # File tồn tại nhưng không load được (hỏng / sai kiến trúc / mid-write) —
            # KHÔNG để lỗi này làm sập server, rơi xuống nhánh fallback bên dưới.
            print(
                f"[inference] WARNING: found {WEIGHTS_PATH} but failed to load it "
                f"({type(exc).__name__}: {exc}) — falling back to UNTRAINED classifier."
            )
    elif _classifier is None:
        print(
            f"[inference] WARNING: {WEIGHTS_PATH} does not exist — falling back to "
            "UNTRAINED classifier (ImageNet backbone + random head). Predictions will "
            "be garbage; use only to verify API/UI wiring."
        )

    if _classifier is None:
        _classifier = build_classifier(num_classes=NUM_CLASSES, pretrained=True, verbose=True)
        _classifier.to(DEVICE).eval()
        _model_is_trained = False
        _crop_mode = False
        _blackout_mode = False

    if UNET_WEIGHTS_PATH.exists():
        try:
            _unet = build_unet(pretrained=False, verbose=False)
            _unet.load_state_dict(torch.load(UNET_WEIGHTS_PATH, map_location=DEVICE))
            _unet.to(DEVICE).eval()
            _unet_is_trained = True
            print(f"[inference] OK: loaded trained U-Net from {UNET_WEIGHTS_PATH} on device={DEVICE}")
        except Exception as exc:
            print(
                f"[inference] WARNING: found {UNET_WEIGHTS_PATH} but failed to load it "
                f"({type(exc).__name__}: {exc}) — falling back to UNTRAINED U-Net."
            )
    else:
        print(
            f"[inference] WARNING: {UNET_WEIGHTS_PATH} does not exist — falling back to "
            "UNTRAINED U-Net. lung_overlap_* trong response sẽ không đáng tin cậy."
        )

    if _unet is None:
        _unet = build_unet(pretrained=True, verbose=False)
        _unet.to(DEVICE).eval()
        _unet_is_trained = False


def _find_ground_truth_mask(filename: Optional[str]) -> Optional[np.ndarray]:
    """Tìm mask thật trong data/processed/<class>/masks/<filename> — chỉ có kết quả
    khi ảnh upload TRÙNG TÊN với 1 ảnh trong dataset gốc (đúng tình huống demo: upload
    lại ảnh mẫu từ data/split/test/images/ để so sánh U-Net với ground truth). Với ảnh
    thật sự mới (không có trong dataset), luôn trả None — đây là hành vi bình thường,
    không phải lỗi.
    """
    if not filename:
        return None
    for cls in DATASET_CLASSES:
        mask_path = DATA_PROCESSED_DIR / cls / "masks" / filename
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))
            return (mask > 0).astype(np.uint8)
    return None


def _encode_png_base64(image_rgb: np.ndarray) -> str:
    pil_img = Image.fromarray(image_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def predict_image(pil_image: Image.Image, filename: Optional[str] = None) -> dict:
    """Hàm dùng chung — api/main.py gọi hàm này cho mỗi request POST /predict.

    filename: tên file gốc lúc upload (từ UploadFile.filename) — dùng để tìm mask
        ground-truth nếu ảnh trùng tên với 1 ảnh trong dataset gốc (xem
        _find_ground_truth_mask). Không bắt buộc — None vẫn chạy bình thường, chỉ là
        sẽ không có số liệu so sánh U-Net vs ground truth.
    """
    if _classifier is None:
        raise RuntimeError("Model chưa được load — gọi load_models() lúc startup")

    image_np = np.array(pil_image.convert("RGB"))
    image_resized = cv2.resize(image_np, IMAGE_SIZE)

    if _crop_mode:
        # Bản ĐÃ TỐI ƯU: U-Net chạy TRƯỚC để lấy mask, dùng mask đó crop ảnh trước khi
        # đưa vào classifier (classifier này được train trên ảnh đã crop — xem
        # notebooks/train_classifier_cropped.ipynb). Grad-CAM sau đó cũng chạy trên
        # ảnh ĐÃ CROP, nên overlay hiển thị trên khung ảnh đã crop, không phải ảnh gốc.
        # _blackout_mode=True: dùng thêm crop_to_lung_bbox_blackout — xoá pixel NGOÀI
        # hình dạng phổi (không chỉ ngoài bounding box) — xem
        # notebooks/train_classifier_blackout.ipynb, docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md Phần 5.4.
        unet_input = _transform(image=image_np)["image"]
        lung_mask = predict_lung_mask(_unet, unet_input)
        crop_fn = crop_to_lung_bbox_blackout if _blackout_mode else crop_to_lung_bbox
        classifier_input_np = crop_fn(image_resized, lung_mask, padding=CROP_PADDING)
        img_tensor = _transform(image=classifier_input_np)["image"]
        overlay_base = cv2.resize(classifier_input_np, IMAGE_SIZE)
    else:
        # Bản BASELINE (chưa tối ưu) — GIỮ NGUYÊN y hệt logic gốc, không đổi hành vi
        # hiện tại (an toàn khi chưa có weights/best_classifier_cropped.pth).
        img_tensor = _transform(image=image_np)["image"]
        overlay_base = image_resized
        lung_mask = None  # tính sau khối no_grad, dùng lại logic chung với nhánh trên

    with torch.no_grad():
        logits = _classifier(img_tensor.unsqueeze(0).to(DEVICE))
        probs = logits.softmax(dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_class = IDX_TO_CLASS[pred_idx]

    # NGOÀI khối no_grad ở trên — Grad-CAM cần backward pass, không được bọc no_grad().
    heatmap = generate_gradcam(_classifier, img_tensor, target_class=pred_idx)
    overlay = overlay_heatmap(overlay_base, heatmap)

    if lung_mask is None:
        # Nhánh baseline: giữ đúng vị trí tính như code gốc (sau Grad-CAM, dùng chung
        # img_tensor với classifier) — không đổi giá trị/hành vi so với trước.
        lung_mask = predict_lung_mask(_unet, img_tensor)

    # Chỉ số tin cậy giải thích — so khớp heatmap với mask phổi U-Net dự đoán
    # (cùng công thức src/shortcut_iou.py dùng để kiểm định shortcut learning).
    cam_bin = binarize(heatmap, GRADCAM_THRESH)
    if _crop_mode:
        # So khớp trong ĐÚNG hệ toạ độ đã crop — cắt mask theo cùng box rồi resize
        # khớp kích thước heatmap (luôn 224×224 sau transform, bất kể box to nhỏ).
        # LƯU Ý: containment sau crop tự nhiên cao hơn vì hiệu ứng hình học (mask
        # chiếm tỉ lệ khung hình lớn hơn), không hẳn vì model "học tốt hơn" — xem
        # cảnh báo tương tự trong src/shortcut_iou.py::run_shortcut_analysis.
        compare_mask = crop_to_lung_bbox(lung_mask, lung_mask, padding=CROP_PADDING)
        if compare_mask.shape != cam_bin.shape:
            compare_mask = cv2.resize(
                compare_mask, (cam_bin.shape[1], cam_bin.shape[0]), interpolation=cv2.INTER_NEAREST
            )
    else:
        compare_mask = lung_mask
    overlap_iou = iou(cam_bin, compare_mask)
    overlap_containment = containment(cam_bin, compare_mask)
    # Luôn hiển thị mask trên ẢNH GỐC (chưa crop) — dễ hiểu cho người dùng hơn là
    # mask đã crop (nhìn gần như trắng xoá toàn khung).
    unet_mask_base64 = _encode_png_base64((lung_mask * 255).astype(np.uint8))

    # So sánh U-Net vs ground truth — chỉ có nếu ảnh upload trùng tên ảnh trong dataset.
    gt_mask = _find_ground_truth_mask(filename)
    gt_mask_found = gt_mask is not None
    unet_vs_gt_dice: Optional[float] = None
    unet_vs_gt_iou: Optional[float] = None
    if gt_mask_found:
        unet_vs_gt_dice = dice(lung_mask, gt_mask)
        unet_vs_gt_iou = iou(lung_mask, gt_mask)

    disclaimer = BASE_DISCLAIMER
    if not _model_is_trained:
        disclaimer = f"{disclaimer} {UNTRAINED_WARNING}"
    if not _unet_is_trained:
        disclaimer = (
            f"{disclaimer} ⚠️ U-Net chưa được huấn luyện — chỉ số lung_overlap_* "
            "bên dưới KHÔNG đáng tin cậy."
        )
    elif overlap_containment < LOW_TRUST_CONTAINMENT:
        pct_outside = (1 - overlap_containment) * 100
        disclaimer = (
            f"{disclaimer} ⚠️ LƯU Ý: model đang tập trung khoảng {pct_outside:.0f}% vào "
            "vùng NGOÀI phổi cho ảnh này (theo kiểm định Grad-CAM so với mask U-Net) — "
            "kết quả có thể không đáng tin cậy, xem docs/LY_THUYET.md Phần VIII."
        )

    return {
        "predicted_class": pred_class,
        "confidence": float(probs[pred_idx]),
        "probabilities": {IDX_TO_CLASS[i]: float(p) for i, p in enumerate(probs)},
        "heatmap_overlay_base64": _encode_png_base64(overlay),
        "disclaimer": disclaimer,
        "lung_overlap_iou": overlap_iou,
        "lung_overlap_containment": overlap_containment,
        "unet_mask_base64": unet_mask_base64,
        "gt_mask_found": gt_mask_found,
        "unet_vs_gt_dice": unet_vs_gt_dice,
        "unet_vs_gt_iou": unet_vs_gt_iou,
    }
