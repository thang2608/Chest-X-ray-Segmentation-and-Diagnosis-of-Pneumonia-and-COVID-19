# "Cong gac" phat hien anh KHONG PHAI X-quang phoi truoc khi dua vao classifier
# benh ly (Normal/Lung_Opacity/COVID) — giai quyet diem mentor #2: hien tai he
# thong van chan doan benh ngay ca khi dua vao anh khong lien quan (chu khi, xe
# hoi, X-quang bo phan khac...).
#
# 2 tang kiem tra doc lap, KHONG can train model moi — tai su dung U-Net co san
# + thong ke anh don gian:
#   Tang 1 (gan nhu mien phi, ~0ms): anh co phai la anh ĐƠN SẮC khong (mot mau
#       duy nhat, chi bien thien do sang — xam hoac co tint deu nhu sepia/xanh
#       lam, KHONG PHAI can moi kenh RGB giong het nhau tuyet doi). Mot so may
#       xuat/PACS to mau X-quang bang 1 tint co dinh (van la anh y te hop le) —
#       kiem tra "hieu TRUNG BINH giua cac kenh co = 0 khong" se tu choi NHAM
#       cac anh nay. Thay vao do do BIEN THIEN KHONG GIAN cua hieu giua cac kenh
#       (std, khong phai mean): anh don sac (co hoac khong tint deu) co hieu
#       R-G/G-B/R-B gan nhu HANG SO tren toan anh -> std thap; anh mau tu nhien
#       (dong vat, xe...) co hieu kenh mau THAY DOI mach theo tung vung (troi
#       xanh, co xanh la, da vang...) -> std cao. Da kiem chung: anh X-quang
#       that (khong tint) std=0.0 tuyet doi; anh mo phong bi tint deu (sepia)
#       std~8.5; anh mau tu nhien nhieu vung std~58 — tach biet ro rang, xem
#       validate_ood_detector.py.
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


def normalize_for_model(image_rgb: np.ndarray) -> np.ndarray:
    """Chuan hoa anh ve XAM/luminance TRUOC KHI dua vao U-Net (va classifier) —
    ca 2 model CHI duoc train tren anh xam nguon (R=G=B), KHONG tung thay anh
    tint. Da kiem chung thuc te: U-Net chay truc tiep tren anh tint xanh la cho
    mask RONG hoan toan (dien tich=0.0%); sau khi chuan hoa qua ham nay, dien
    tich=24.6%, khop gan tuyet doi anh KHONG tint cung ca (24.67%). BAT BUOC goi
    ham nay truoc khi predict_lung_mask() neu anh dau vao co the bi tint — xem
    channel_consistency_score() ben duoi de biet vi sao KHONG dung ham nay cho
    phan kiem tra mau (kiem tra mau phai dung anh GOC, chua chuan hoa, moi phan
    biet duoc tint hop le vs. anh mau tu nhien). Khong anh huong anh von di da
    la xam (chuyen xam -> xam = y nguyen)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return np.stack([gray] * 3, axis=-1)


def channel_consistency_score(image_rgb: np.ndarray) -> float:
    """Do BIEN THIEN KHONG GIAN cua hieu giua 3 kenh RGB (std, khong phai mean
    abs) — cho phep anh bi tint 1 mau duy nhat DEU toan bo khung hinh (van la
    anh don sac hop le, vd sepia/xanh lam), chi tu choi khi mau THAY DOI khac
    nhau giua cac VUNG trong anh (dau hieu anh mau tu nhien nhieu doi tuong).
    0.0 = hoan toan xam hoac tint deu tuyet doi. Anh mau tu nhien thuong > 20."""
    img = image_rgb.astype(np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    diff_rg, diff_gb, diff_rb = r - g, g - b, r - b
    return float((diff_rg.std() + diff_gb.std() + diff_rb.std()) / 3.0)


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
    # Hieu chinh tu phan bo thuc te: 150 anh test that bi tint (sepia/xanh lam/xanh
    # la) co max=22.7 (p99=18.6); 20 anh mau tu nhien gia lap co min=29.4 — khoang
    # trong ro rang giua 2 phia, chon 26.0 de co bien an toan ca 2 huong.
    gray_threshold: float = 26.0,
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
        reasons.append(f"Ảnh có nhiều màu sắc khác nhau theo từng vùng, không giống ảnh X-quang (độ biến thiên màu={gray_score:.1f})")

    shape = lung_mask_shape_score(lung_mask)
    if not (min_area_pct <= shape["area_pct"] <= max_area_pct):
        reasons.append(f"Diện tích vùng nghi là phổi bất thường ({shape['area_pct']:.1f}% khung hình)")
    if not (min_components <= shape["big_components"] <= max_components):
        reasons.append(f"Số vùng liên thông bất thường ({shape['big_components']})")
    if shape["symmetry"] < min_symmetry:
        reasons.append(f"Vùng nghi là phổi không đối xứng trái-phải (symmetry={shape['symmetry']:.2f})")

    detail = {"gray_score": gray_score, **shape, "reasons": reasons}
    return len(reasons) == 0, detail
