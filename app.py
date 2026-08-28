import base64
import os
from io import BytesIO
from pathlib import Path

import gradio as gr
import requests
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")


def diagnose(image_path: str):
    if image_path is None:
        return None, {}, "Vui lòng upload ảnh X-quang trước.", "", None, ""

    # Giữ nguyên TÊN FILE gốc (không phải "image.png" cố định) — cần để backend tìm
    # được mask ground-truth trùng tên trong dataset (chỉ khi upload lại ảnh mẫu).
    filename = Path(image_path).name
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    try:
        response = requests.post(
            API_URL, files={"file": (filename, buf, "image/png")}, timeout=60
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        return None, {}, f"Lỗi khi gọi API ({API_URL}): {exc}", "", None, ""

    result = response.json()

    overlay_bytes = base64.b64decode(result["heatmap_overlay_base64"])
    overlay_image = Image.open(BytesIO(overlay_bytes))

    unet_mask_bytes = base64.b64decode(result["unet_mask_base64"])
    unet_mask_image = Image.open(BytesIO(unet_mask_bytes))

    label_text = f"{result['predicted_class']} ({result['confidence']*100:.1f}%)"
    probs = result["probabilities"]

    trust_iou = result.get("lung_overlap_iou")
    trust_containment = result.get("lung_overlap_containment")
    if trust_iou is not None:
        trust_text = (
            f"IoU(Grad-CAM, mask phổi) = {trust_iou:.3f}\n"
            f"Containment (% heatmap nằm trong phổi) = {trust_containment*100:.1f}%"
        )
        if trust_containment < 0.3:
            trust_text += "\n⚠️ Model có thể đang nhìn phần lớn NGOÀI vùng phổi cho ảnh này."
    else:
        trust_text = "Không có dữ liệu độ tin cậy."

    if result.get("gt_mask_found"):
        mask_compare_text = (
            f"Dice(U-Net, ground truth) = {result['unet_vs_gt_dice']:.3f}\n"
            f"IoU(U-Net, ground truth)  = {result['unet_vs_gt_iou']:.3f}"
        )
    else:
        mask_compare_text = (
            "Không tìm thấy mask ground-truth cho ảnh này — chỉ có khi upload lại "
            "đúng ảnh mẫu từ data/split/*/images/ (ảnh mới hoàn toàn sẽ không có)."
        )

    return (
        overlay_image,
        probs,
        f"{label_text}\n\n{result['disclaimer']}",
        trust_text,
        unet_mask_image,
        mask_compare_text,
    )


with gr.Blocks(title="Chest X-ray Diagnosis") as demo:
    gr.Markdown("# Chest X-ray Segmentation & Diagnosis of Pneumonia and COVID-19")
    gr.Markdown(
        "Công cụ hỗ trợ tham khảo, **không** thay thế chẩn đoán y khoa chính thức."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label="Ảnh X-quang ngực")
            submit_btn = gr.Button("Chẩn đoán", variant="primary")
        with gr.Column():
            output_overlay = gr.Image(label="Grad-CAM overlay")
            output_unet_mask = gr.Image(label="Mask phổi (U-Net)")
            output_probs = gr.Label(label="Xác suất từng lớp", num_top_classes=3)
            output_text = gr.Textbox(label="Kết luận", lines=4)
            output_trust = gr.Textbox(
                label="Độ tin cậy giải thích (Grad-CAM so với mask phổi U-Net)", lines=3
            )
            output_mask_compare = gr.Textbox(
                label="So sánh mask U-Net vs Ground Truth (nếu có)", lines=3
            )

    submit_btn.click(
        fn=diagnose,
        inputs=[input_image],
        outputs=[
            output_overlay,
            output_probs,
            output_text,
            output_trust,
            output_unet_mask,
            output_mask_compare,
        ],
    )

if __name__ == "__main__":
    demo.launch()
