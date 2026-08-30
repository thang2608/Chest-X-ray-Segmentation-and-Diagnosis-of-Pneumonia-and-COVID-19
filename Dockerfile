# 1. Base image Python gọn nhẹ
FROM python:3.11-slim

# 2. Cài đặt các thư viện hệ thống cần thiết cho OpenCV (libgl1, libglib2.0-0)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Thư mục làm việc trong container
WORKDIR /app

# 4. Cài đặt toàn bộ thư viện với PyTorch CPU siêu nhẹ (~180MB thay vì 4.5GB CUDA)
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 6. Copy toàn bộ mã nguồn và trọng số model đã huấn luyện
COPY src/ /app/src/
COPY weights/ /app/weights/
COPY backend/ /app/backend/

# 7. Chuyển thư mục làm việc vào backend để uvicorn nạp package app
WORKDIR /app/backend

# 8. Mở cổng 8000
EXPOSE 8000

# 9. Lệnh khởi chạy server FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
