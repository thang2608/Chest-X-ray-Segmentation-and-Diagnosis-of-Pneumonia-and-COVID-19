# "Cong gac" phat hien anh KHONG PHAI X-quang phoi truoc khi dua vao classifier
# benh ly (Normal/Lung_Opacity/COVID) — giai quyet diem mentor #2: hien tai he
# thong van chan doan benh ngay ca khi dua vao anh khong lien quan (chu khi, xe
# hoi, X-quang bo phan khac...).
#
# 2 tang kiem tra doc lap, KHONG can train model moi — tai su dung U-Net co san
# + thong ke anh don gian:
#   Tang 1 (gan nhu mien phi, ~0ms): anh co PHAI la anh XAM khong. X-quang y te
#       luon la anh xam luu 3 kenh RGB GIONG HET NHAU (da kiem chung thuc te:
#       30 anh test that co channel-diff = 0.0 tuyet doi, xem
#       validate_ood_detector.py). Anh mau tu nhien (dong vat, xe...) co do lech
#       mau giua cac kenh RGB ro ret, gan nhu khong bao gio = 0.
#   Tang 2 (dung U-Net co san): mask phoi du doan phai co dien tich hop ly, so
#       vung lien thong hop ly, va doi xung trai-phai tuong doi — dac trung hinh
#       hoc rieng cua long nguc nguoi. Anh khong phai X-quang phoi se cho mask
#       vo nghia (rong/vun/khong doi xung) vi U-Net chi hoc dac trung phoi.
#
# LUU Y: day la HEURISTIC, khong phai model phan loai OOD "chuan" (vd train rieng
# 1 binary classifier voi du lieu negative that) — danh doi hop ly trong khung
# thoi gian ngan (khong can thu thap them du lieu/train them model), nhung van
# co the bi qua mat boi anh X-quang bo phan khac co hinh dang tinh co giong phoi,
# hoac anh xam nhan tao duoc thiet ke co chu y de danh lua. Xem
# validate_ood_detector.py de biet do chinh xac thuc te tren du lieu that + gia lap.
import cv2
import numpy as np


def channel_consistency_score(image_rgb: np.ndarray) -> float:
    """Do lech mau trung binh giua 3 kenh RGB. 0.0 = hoan toan xam (dung cho anh
    y te that). Anh mau tu nhien thuong > 10."""
    img = image_rgb.astype(np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return float((np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(r - b))) / 3.0)


def lung_mask_shape_score(mask: np.ndarray) -> dict:
    """Phan tich hinh hoc mask phoi du doan boi U-Net: % dien tich, so vung lien
    thong lon (>1% dien tich anh), do doi xung trai-phai. Tra ve dict (khong
    phai True/False) de log/debug ro nguyen nhan khi tu choi."""
    mask_bin = (mask > 0).astype(np.uint8)
    h, w = mask_bin.shape
    area_pct = 100.0 * mask_bin.sum() / (h * w)

    n_components, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n_components > 1 else np.array([])
    big_components = int((areas > 0.01 * h * w).sum())

    left_area = float(mask_bin[:, : w // 2].sum())
    right_area = float(mask_bin[:, w // 2:].sum())
    total = left_area + right_area
    symmetry = 1.0 - abs(left_area - right_area) / total if total > 0 else 0.0

    return {"area_pct": area_pct, "big_components": big_components, "symmetry": symmetry}


def is_valid_chest_xray(
    image_rgb: np.ndarray,
    lung_mask: np.ndarray,
    gray_threshold: float = 2.0,
    min_area_pct: float = 4.0,  # hieu chinh tu du lieu that: min quan sat = 4.19% tren 1350 anh test
    max_area_pct: float = 70.0,
    min_symmetry: float = 0.45,
    min_components: int = 1,
    max_components: int = 4,
) -> tuple[bool, dict]:
    """Cong 2 tang kiem tra. Tra ve (hop_le, chi_tiet) — chi_tiet['reasons'] liet
    ke chinh xac ly do tu choi (rong neu hop le) de UI/log giai thich duoc voi
    nguoi dung, khong chi bao 'anh khong hop le' chung chung."""
    reasons = []
    gray_score = channel_consistency_score(image_rgb)
    if gray_score >= gray_threshold:
        reasons.append(f"Ảnh có màu, không giống ảnh X-quang (độ lệch màu={gray_score:.1f})")

    shape = lung_mask_shape_score(lung_mask)
    if not (min_area_pct <= shape["area_pct"] <= max_area_pct):
        reasons.append(f"Diện tích vùng nghi là phổi bất thường ({shape['area_pct']:.1f}% khung hình)")
    if not (min_components <= shape["big_components"] <= max_components):
        reasons.append(f"Số vùng liên thông bất thường ({shape['big_components']})")
    if shape["symmetry"] < min_symmetry:
        reasons.append(f"Vùng nghi là phổi không đối xứng trái-phải (symmetry={shape['symmetry']:.2f})")

    detail = {"gray_score": gray_score, **shape, "reasons": reasons}
    return len(reasons) == 0, detail
