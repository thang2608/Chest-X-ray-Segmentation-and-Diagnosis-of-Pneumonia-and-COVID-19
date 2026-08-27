import base64
import os
from io import BytesIO

import gradio as gr
import requests
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")


def diagnose(image: Image.Image):
    if image is None:
        return None, {}, "Vui lòng upload ảnh X-quang trước."

    if image.mode != "RGB":
        image = image.convert("RGB")

    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    try:
        response = requests.post(
            API_URL, files={"file": ("image.png", buf, "image/png")}, timeout=60
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        return None, {}, f"Lỗi khi gọi API ({API_URL}): {exc}"

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
            output_text = gr.Textbox(label="Kết luận", lines=4)

    submit_btn.click(
        fn=diagnose,
        inputs=[input_image],
        outputs=[output_overlay, output_probs, output_text],
    )

if __name__ == "__main__":
    demo.launch()
