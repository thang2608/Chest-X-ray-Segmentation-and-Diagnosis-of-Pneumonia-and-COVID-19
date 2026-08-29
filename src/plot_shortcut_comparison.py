"""Tổng hợp biểu đồ so sánh shortcut learning TRƯỚC/SAU crop-to-lung cho báo cáo.

Đọc lại 4 file CSV đã có sẵn trong figures/ (do src/shortcut_iou.py::run_shortcut_analysis
sinh ra ở 2 lần chạy — baseline và crop, mỗi lần 2 nguồn mask gt/unet) — KHÔNG chạy lại
model (không cần GPU, chạy vài giây). Xem docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md Phần 5 cho
diễn giải số liệu.

Input  : figures/shortcut_records_{gt,unet}{,_cropped}_t0.5.csv
Output : figures/compare_pct_iou_zero.png       — % ảnh IoU=0 theo lớp, trước/sau (gt & unet)
         figures/compare_containment_box.png    — phân phối containment theo lớp, trước/sau (gt)
         figures/case_study_bbox_leak.png       — 1 ca cụ thể minh hoạ vì sao logo lọt qua bbox crop

Chạy: python -m src.plot_shortcut_comparison
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path("figures")
CLASSES = ["Normal", "Lung_Opacity", "COVID"]  # thứ tự hiển thị cố định, khớp CLASS_TO_IDX


def _read_records(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pct_iou_zero(records: list[dict], cls: str) -> float:
    vals = [float(r["iou"]) for r in records if r["class"] == cls]
    if not vals:
        return 0.0
    return 100.0 * sum(1 for v in vals if v == 0.0) / len(vals)


def _containments(records: list[dict], cls: str) -> np.ndarray:
    return np.array([float(r["containment"]) for r in records if r["class"] == cls])


def plot_pct_iou_zero():
    """Biểu đồ cột nhóm: % ảnh có IoU(Grad-CAM, phổi) = 0 tuyệt đối theo từng lớp,
    so sánh 3 bản baseline / cropped / blackout — CHỈ SỐ ƯU TIÊN khi so trước/sau vì
    không bị ảnh hưởng bởi hiệu ứng hình học (mask chiếm tỉ lệ khung hình khác nhau,
    xem docstring trong src/shortcut_iou.py::run_shortcut_analysis)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    for ax, source in zip(axes, ["gt", "unet"]):
        base = _read_records(FIGURES_DIR / f"shortcut_records_{source}_t0.5.csv")
        crop = _read_records(FIGURES_DIR / f"shortcut_records_{source}_cropped_t0.5.csv")
        black = _read_records(FIGURES_DIR / f"shortcut_records_{source}_cropped_blackout_t0.5.csv")

        pct_base = [_pct_iou_zero(base, c) for c in CLASSES]
        pct_crop = [_pct_iou_zero(crop, c) for c in CLASSES]
        pct_black = [_pct_iou_zero(black, c) for c in CLASSES]

        x = np.arange(len(CLASSES))
        w = 0.26
        b1 = ax.bar(x - w, pct_base, w, label="Baseline (thô)", color="#d9534f")
        b2 = ax.bar(x, pct_crop, w, label="Crop-to-lung", color="#f0ad4e")
        b3 = ax.bar(x + w, pct_black, w, label="Crop + blackout", color="#5cb85c")
        for bars in (b1, b2, b3):
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.1f}%", (bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES)
        ax.set_title(f"Mask nguồn: {source.upper()}")
        ax.set_ylabel("% ảnh có IoU = 0 (Grad-CAM không chạm phổi)")
        ax.legend()

    fig.suptitle("Shortcut learning qua 3 bản — % ảnh Grad-CAM hoàn toàn ngoài phổi")
    fig.tight_layout()
    out = FIGURES_DIR / "compare_pct_iou_zero.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Đã lưu {out}")


def plot_accuracy_vs_shortcut_tradeoff():
    """Biểu đồ đánh đổi: Macro F1 (val set) vs. % ảnh COVID IoU=0 (test set, mask gt),
    3 bản — minh hoạ trade-off blackout giảm shortcut MẠNH nhưng đổi lấy accuracy thấp
    hơn cả baseline. Số liệu F1 lấy từ docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md Phần 5 (đo
    trực tiếp bằng eval script trên weights/best_classifier_{}.pth, không phải suy đoán."""
    variants = ["Baseline", "Crop-to-lung", "Crop + blackout"]
    macro_f1 = [0.9057, 0.9302, 0.8659]
    covid_iou_zero_gt = [27.3, 19.6, 4.2]
    colors = ["#d9534f", "#f0ad4e", "#5cb85c"]

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(variants))
    bars = ax1.bar(x - 0.18, macro_f1, 0.32, color="#337ab7", label="Macro F1 (val, thang trái)")
    line = ax2.plot(x + 0.18, covid_iou_zero_gt, "o-", color="#d9534f", linewidth=2, markersize=9,
                     label="% COVID IoU=0 (test, thang phải)")

    for bx, v in zip(x - 0.18, macro_f1):
        ax1.annotate(f"{v:.4f}", (bx, v), xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=9, color="#337ab7")
    for lx, v in zip(x + 0.18, covid_iou_zero_gt):
        ax2.annotate(f"{v:.1f}%", (lx, v), xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=9, color="#d9534f")

    ax1.set_xticks(x)
    ax1.set_xticklabels(variants)
    ax1.set_ylabel("Macro F1 (val set)", color="#337ab7")
    ax1.set_ylim(0.8, 1.0)
    ax2.set_ylabel("% ảnh COVID có IoU=0 (test set, càng thấp càng ít shortcut)", color="#d9534f")
    ax2.set_ylim(0, 30)
    ax1.set_title("Đánh đổi Accuracy ↔ Shortcut learning qua 3 bản")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center", fontsize=8.5)

    fig.tight_layout()
    out = FIGURES_DIR / "compare_accuracy_vs_shortcut_tradeoff.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Đã lưu {out}")


def plot_containment_box():
    """Boxplot containment (|Grad-CAM ∩ phổi| / |Grad-CAM|) theo lớp, trước/sau crop
    (mask nguồn ground-truth). LƯU Ý: containment sau crop tự nhiên cao hơn một phần
    do hiệu ứng hình học (mask chiếm % khung hình lớn hơn sau khi crop nền đi) — biểu
    đồ này minh hoạ xu hướng, KHÔNG dùng để kết luận "model học tốt hơn X%"."""
    base = _read_records(FIGURES_DIR / "shortcut_records_gt_t0.5.csv")
    crop = _read_records(FIGURES_DIR / "shortcut_records_gt_cropped_t0.5.csv")

    data = []
    labels = []
    colors = []
    for c in CLASSES:
        data.append(_containments(base, c))
        labels.append(f"{c}\n(trước)")
        colors.append("#d9534f")
        data.append(_containments(crop, c))
        labels.append(f"{c}\n(sau)")
        colors.append("#5cb85c")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Containment = |Grad-CAM ∩ phổi| / |Grad-CAM|")
    ax.set_title("Phân phối containment theo lớp — trước/sau crop (mask ground-truth)\n"
                  "(containment sau crop cao hơn một phần do hiệu ứng hình học — xem báo cáo)")
    ax.axhline(0.3, color="gray", linestyle="--", linewidth=1, label="Ngưỡng cảnh báo low-trust (0.3)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = FIGURES_DIR / "compare_containment_box.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Đã lưu {out}")


def plot_case_study(diag_images: dict[str, Path]):
    """Ghép 3 ảnh chẩn đoán (full-frame+bbox+mask, ảnh đã crop, Grad-CAM trên ảnh crop)
    thành 1 hình duy nhất cho báo cáo, kèm số liệu đo được trên đúng ca này.

    diag_images: {"full": path, "cropped": path, "gradcam": path}
    """
    from PIL import Image

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    titles = [
        "1. Ảnh gốc: mask phổi (xanh) + bbox crop (vàng)\nGóc trên-trái: 0 pixel phổi nhưng VẪN trong bbox",
        "2. Ảnh sau crop-to-lung\n(đưa vào classifier)",
        "3. Grad-CAM trên ảnh đã crop\nIoU=0.202  Containment=0.499",
    ]
    for ax, key, title in zip(axes, ["full", "cropped", "gradcam"], titles):
        img = Image.open(diag_images[key])
        ax.imshow(img)
        ax.set_title(title, fontsize=9.5)
        ax.axis("off")

    fig.suptitle(
        "Ca cụ thể: sample_covid.png — vì sao Grad-CAM vẫn sáng ở góc ảnh sau khi train trên ảnh crop\n"
        "(crop-to-lung cắt theo HÌNH CHỮ NHẬT bao phổi, không theo đúng HÌNH DẠNG phổi)",
        fontsize=11,
        y=1.04,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    out = FIGURES_DIR / "case_study_bbox_leak.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Đã lưu {out}")


def plot_case_study_blackout_fix(diag_images: dict[str, Path]):
    """So sánh trực tiếp cropped vs. blackout trên CÙNG 1 ảnh (sample_covid.png) —
    minh hoạ blackout xoá sạch vùng ngoài phổi (kể cả logo/vật thể trong bbox) mà
    crop thuần không làm được. Xem docs/BAO_CAO_KET_QUA_HUAN_LUYEN.md Phần 5.5.

    diag_images: {"cropped_input": path, "cropped_gradcam": path,
                  "blackout_input": path, "blackout_gradcam": path}
    """
    from PIL import Image

    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    cells = [
        ("cropped_input", "1. Input — crop-to-lung\n(vẫn còn nền/vật thể ngoài phổi)"),
        ("cropped_gradcam", "2. Grad-CAM trên ảnh crop\nIoU=0.202  Containment=0.499"),
        ("blackout_input", "3. Input — crop + blackout\n(xoá sạch mọi pixel ngoài phổi)"),
        ("blackout_gradcam", "4. Grad-CAM trên ảnh blackout\nIoU=0.238  Containment=0.641"),
    ]
    for ax, (key, title) in zip(axes.flat, cells):
        ax.imshow(Image.open(diag_images[key]))
        ax.set_title(title, fontsize=9.5)
        ax.axis("off")

    fig.suptitle(
        "sample_covid.png — Grad-CAM tập trung vào phổi tốt hơn rõ rệt sau blackout",
        fontsize=11.5, y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGURES_DIR / "case_study_blackout_fix.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Đã lưu {out}")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_pct_iou_zero()
    plot_containment_box()
    plot_accuracy_vs_shortcut_tradeoff()
