import base64
import io
import os
import time
import traceback

import gradio as gr
import pandas as pd
import requests
from PIL import Image

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{BACKEND_URL}/predict"
METRICS_ENDPOINT = f"{BACKEND_URL}/metrics"
REQUEST_TIMEOUT_SEC = 15  # tránh treo UI khi backend đơ / timeout

CLASS_NAMES = ["Normal", "Pneumonia", "COVID-19"]

DISEASE_ADVICE = {
    "Normal": "Không phát hiện dấu hiệu bất thường rõ rệt trên ảnh X-quang. "
              "Vẫn nên duy trì khám sức khỏe định kỳ.",
    "Pneumonia": "Phát hiện dấu hiệu nghi ngờ **viêm phổi**. Khuyến cáo đến cơ sở y tế "
                 "để được bác sĩ chuyên khoa chẩn đoán và xét nghiệm bổ sung.",
    "COVID-19": "Phát hiện dấu hiệu nghi ngờ tổn thương liên quan **COVID-19**. "
                "Khuyến cáo cách ly, xét nghiệm PCR/RT-PCR xác nhận và liên hệ y tế ngay.",
}

# disclaimer
DISCLAIMER_HTML = """
<div style="
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 12px 20px;
    margin-bottom: 4px;
    font-size: 14px;
">
     <strong>MEDICAL DISCLAIMER:</strong>
  Hệ thống này chỉ được phát triển nhằm phục vụ <strong>mục đích nghiên cứu và giáo dục</strong>.
  Kết quả do hệ thống cung cấp <strong>không thay thế cho chẩn đoán, tư vấn hoặc điều trị y tế chuyên môn</strong>.
  Không sử dụng kết quả của hệ thống để tự chẩn đoán hoặc đưa ra quyết định điều trị.
  Vui lòng tham khảo ý kiến của <strong>bác sĩ hoặc chuyên gia y tế có chuyên môn</strong>
  để được đánh giá và tư vấn chính xác.
</div>
"""

# custom CSS
CUSTOM_CSS = """
/* ẩn footer của gradio */
footer { display: none !important; }

/* font */
body, .gradio-container {
    font-family: 'Times New Roman', serif !important;
}

/* panel XAI */
.xai-row { background: #f8f9fa; border-radius: 12px; padding: 8px; }

/* nút analyze */
#btn-analyze { font-size: 16px !important; height: 48px !important; }
"""

def decode_base64_to_pil(b64_str): # Giải mã base64 (có/không kèm data URI prefix) -> PIL Image. Trả None nếu lỗi
    if not b64_str:
        return None
    try:
        if isinstance(b64_str, str) and b64_str.strip().startswith("data:") and "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None


def safe_get(d: dict, path: list, default=None): #tránh KeyError nếu backend trả thiếu field
    cur = d
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def phan_tram(x):
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "N/A"


def recommend_md(disease, confidence, metrics):
    advice = DISEASE_ADVICE.get(disease, "Không có khuyến cáo tương ứng.")
    area = safe_get(metrics, ["affected_lung_area"])
    area_txt = f"\n- **Diện tích phổi bị ảnh hưởng:** {area}%" if area is not None else ""
    return (
        f"- **Độ tin cậy:** {phan_tram(confidence)}"
        f"{area_txt}\n\n"
        f"> {advice}"
    )


# gọi api
def call_predict_api(image: Image.Image):
    if image is None:
        return None, "empty_image"

    try:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        buf.seek(0)
        files = {"file": ("xray.png", buf, "image/png")}

        resp = requests.post(PREDICT_ENDPOINT, files=files, timeout=REQUEST_TIMEOUT_SEC)

        if resp.status_code != 200:
            try:
                msg = resp.json().get("message", f"HTTP {resp.status_code}")
            except Exception:
                msg = f"HTTP {resp.status_code}"
            return None, f"server_error:{msg}"

        data = resp.json()
        if data.get("status") == "error":
            return None, f"server_error:{data.get('message', 'Unknown error')}"

        return data, None

    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.ConnectionError:
        return None, "connection_error"
    except Exception as e:
        traceback.print_exc()
        return None, f"unexpected:{str(e)}"


def call_metrics_api():
    try:
        resp = requests.get(METRICS_ENDPOINT, timeout=REQUEST_TIMEOUT_SEC)
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        data = resp.json()
        if data.get("status") == "error":
            return None, data.get("message", "Unknown error")
        return data, None
    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.ConnectionError:
        return None, "connection_error"
    except Exception as e:
        return None, str(e)


# predict
def real_predict_fn(image):
    """
    0 output_label (dict), 1 status_time (str), 2 output_recommendation (md),
    3 xai_original, 4 xai_lungmask, 5 xai_gradcam, 6 xai_overlay (PIL/None),
    7 iou_number (float/None), 8 iou_interpretation (md)
    """
    t0 = time.time()

    if image is None:
        gr.Warning("Vui lòng upload ảnh X-quang trước khi bấm Analyze.")
        return (
            None,
            "—",
            "> Chưa có ảnh nào được tải lên.",
            None, None, None, None,
            None,
            "_IoU score sẽ được hiển thị sau khi chẩn đoán._",
        )

    data, err = call_predict_api(image)
    elapsed = time.time() - t0

    if err is not None:
        if err == "timeout":
            msg = "Server phản hồi quá lâu (timeout). Vui lòng thử lại sau."
        elif err == "connection_error":
            msg = f"Không thể kết nối tới backend tại `{PREDICT_ENDPOINT}`. Kiểm tra server FastAPI đã chạy chưa."
        elif err.startswith("server_error"):
            msg = f"Lỗi từ server: {err.split(':', 1)[-1]}"
        else:
            msg = f"Lỗi không xác định: {err}"
        gr.Error(msg)
        return (
            None,
            f"⏱ {elapsed:.3f}s (lỗi)",
            f"### Dự đoán thất bại\n\n{msg}",
            None, None, None, None,
            None,
            "_Không có dữ liệu IoU do lỗi dự đoán._",
        )

    # parse dl
    disease = safe_get(data, ["disease"], "Không xác định")
    confidence = safe_get(data, ["confidence"], 0.0)
    probabilities = safe_get(data, ["probabilities"], None)
    metrics = safe_get(data, ["metrics"], {}) or {}

    # gr.Label cần dict {class: prob}; fallback nếu backend không trả probabilities đầy đủ
    if isinstance(probabilities, dict) and probabilities:
        label_dict = {cls: float(probabilities.get(cls, 0.0)) for cls in CLASS_NAMES}
    else:
        label_dict = {cls: (float(confidence) if cls == disease else 0.0) for cls in CLASS_NAMES}

    original_img = decode_base64_to_pil(safe_get(data, ["images", "original_b64"]))
    lung_mask_img = decode_base64_to_pil(safe_get(data, ["images", "lung_mask_b64"]))
    gradcam_img = decode_base64_to_pil(safe_get(data, ["images", "gradcam_b64"]))
    overlay_img = decode_base64_to_pil(safe_get(data, ["images", "heatmap_overlay_b64"]))

    # Fallback ảnh gốc: nếu backend không trả, dùng ảnh người dùng upload
    if original_img is None:
        try:
            original_img = image.convert("RGB")
        except Exception:
            original_img = None

    recommendation_md = recommend_md(disease, confidence, metrics)

    iou_val = safe_get(metrics, ["iou_score"])
    iou_number = None
    iou_text = "_IoU score không có sẵn từ backend._"
    if iou_val is not None:
        try:
            iou_number = float(iou_val)
            if iou_number >= 0.5:
                iou_text = (
                    f"**IoU = {iou_number:.3f}** — Grad-CAM bám sát vùng phổi "
                    f"(không có dấu hiệu shortcut learning)."
                )
            else:
                iou_text = (
                    f"**IoU = {iou_number:.3f}** — Vùng Grad-CAM lệch khỏi phổi, "
                    f"cần kiểm tra khả năng shortcut learning."
                )
        except Exception:
            pass

    gr.Info(f"Dự đoán: {disease} ({phan_tram(confidence)})")

    return (
        label_dict,
        f"{elapsed:.3f}s",
        recommendation_md,
        original_img, lung_mask_img, gradcam_img, overlay_img,
        iou_number,
        iou_text,
    )


# metrics

def real_metrics_fn():
    data, err = call_metrics_api()

    if data is not None:
        cm_img = decode_base64_to_pil(data.get("confusion_matrix_b64"))
        roc_img = decode_base64_to_pil(data.get("roc_curve_b64"))
        table = data.get("performance_table")
        df = pd.DataFrame(table) if table else pd.DataFrame(
            [{"Metric": "N/A", "Full X-ray": "N/A", "Cropped Lung": "N/A"}]
        )
        return cm_img, roc_img, df

    df = pd.DataFrame([
        {"Metric": "Accuracy", "Full X-ray": "N/A", "Cropped Lung": "N/A"},
        {"Metric": "F1-Macro", "Full X-ray": "N/A", "Cropped Lung": "N/A"},
        {"Metric": "Precision", "Full X-ray": "N/A", "Cropped Lung": "N/A"},
        {"Metric": "Recall", "Full X-ray": "N/A", "Cropped Lung": "N/A"},
    ])
    return None, None, df


# ui

def build_interface(predict_fn=None, metrics_fn=None) -> gr.Blocks:
    _predict = predict_fn or real_predict_fn
    _metrics = metrics_fn or real_metrics_fn

    # layout
    with gr.Blocks(
            title="Chest X-ray AI Diagnostic System",
    ) as demo:
        # header
        gr.Markdown(
            "# Chest X-ray AI Diagnostic System\n"
            "**EfficientNet-B3 + U-Net + Grad-CAM XAI** | "
            "COVID-19 · Pneumonia · Normal"
        )
        gr.HTML(DISCLAIMER_HTML)
        gr.Markdown("---")

        # tabs
        with gr.Tabs():
            # tab diagnosis
            with gr.Tab("Diagnosis", id="tab-diagnosis"):
                with gr.Row(equal_height=True):
                    # input
                    with gr.Column(scale=4, min_width=260):
                        gr.Markdown("### Input Image")

                        img_input = gr.Image(
                            type="pil",
                            label="Upload Chest X-ray",
                            sources=["upload"],
                            image_mode="RGB",
                            height=300,
                        )

                        with gr.Row():
                            btn_predict = gr.Button(
                                value="Chẩn đoán",
                                variant="primary",
                                scale=3,
                                elem_id="btn-analyze",
                            )
                            btn_clear = gr.ClearButton(
                                value="Reset",
                                scale=1,
                            )

                        status_time = gr.Textbox(
                            label="Response Time",
                            value="—",
                            interactive=False,
                            max_lines=1,
                        )

                    # kết quả
                    with gr.Column(scale=5, min_width=320):
                        gr.Markdown("### Diagnosis Result")

                        output_label = gr.Label(
                            label="Classification (Top-3 Confidence)",
                            num_top_classes=3,
                            value=None,
                        )

                        output_recommendation = gr.Markdown(
                            value=(
                                "> Upload ảnh X-ray và click "
                                "**Chẩn đoán** để bắt đầu."
                            )
                        )

                # XAI panel
                gr.Markdown("---")
                gr.Markdown(
                    "### Explainable AI — Grad-CAM Visualization\n"
                    "**Heatmap** làm nổi bật các vùng mà mô hình AI tập trung khi đưa ra dự đoán. **Chỉ số IoU** đo mức độ chồng lấp giữa bản đồ kích hoạt Grad-CAM và mặt nạ phân vùng phổi (Lung Mask) từ U-Net. **IoU càng cao → mô hình càng tập trung vào các đặc trưng mô phổi có liên quan về mặt lâm sàng**."
                )

                with gr.Row(elem_classes=["xai-row"]):
                    xai_original = gr.Image(
                        label="Original X-ray", interactive=False, height=220,
                    )
                    xai_lungmask = gr.Image(
                        label="Lung Mask (U-Net)", interactive=False, height=220,
                    )
                    xai_gradcam = gr.Image(
                        label="Grad-CAM Heatmap", interactive=False, height=220,
                    )
                    xai_overlay = gr.Image(
                        label="Grad-CAM Overlay", interactive=False, height=220,
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        iou_number = gr.Number(
                            label="IoU Score  (Grad-CAM ∩ Lung Mask)",
                            value=None,
                            precision=3,
                            interactive=False,
                        )
                    with gr.Column(scale=3):
                        iou_interpretation = gr.Markdown(
                            value="_Chỉ số IoU sẽ xuất hiện sau khi chẩn đoán._"
                        )

            # tab 2 model evalution
            with gr.Tab("Model Evaluation", id="tab-eval"):
                with gr.Row():
                    eval_confusion = gr.Image(
                        label="Confusion Matrix", interactive=False, height=420,
                    )
                    eval_roc = gr.Image(
                        label="ROC Curve", interactive=False, height=420,
                    )

                eval_table = gr.DataFrame(
                    label="Performance: Full X-ray vs Lung-Cropped Input",
                    interactive=False,
                )

                gr.Markdown(
                    "> **Note**: **Ảnh đầu vào được cắt theo vùng phổi sử dụng U-Net để cô lập vùng phổi trước khi phân loại, giúp giảm ảnh hưởng của các đặc trưng nền (background bias).**"
                )

        # Danh sách tất cả outputs
        all_outputs = [
            output_label,  # 0: dict → gr.Label
            status_time,  # 1: str  → gr.Textbox
            output_recommendation,  # 2: str  → gr.Markdown
            xai_original,  # 3: PIL  → gr.Image
            xai_lungmask,  # 4: PIL  → gr.Image
            xai_gradcam,  # 5: PIL  → gr.Image
            xai_overlay,  # 6: PIL  → gr.Image
            iou_number,  # 7: float→ gr.Number
            iou_interpretation,  # 8: str  → gr.Markdown
        ]

        # btn
        btn_predict.click(
            fn=_predict,
            inputs=[img_input],
            outputs=all_outputs,
            api_name="predict",
        )

        # ClearButton tự clear input; thêm outputs vào để reset toàn bộ kết quả
        btn_clear.add(all_outputs)

        # Load metrics khi app khởi động (không chặn UI nếu backend chưa sẵn sàng)
        demo.load(
            fn=_metrics,
            inputs=None,
            outputs=[eval_confusion, eval_roc, eval_table],
        )

    return demo

demo = build_interface()

if __name__ == "__main__":
    demo.queue(max_size=20)  # tránh nghẽn khi nhiều người demo cùng lúc
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("GRADIO_PORT", 7860)),
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    )
