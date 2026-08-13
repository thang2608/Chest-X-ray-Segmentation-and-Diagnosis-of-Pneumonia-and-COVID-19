# Chest-X-ray-Segmentation-and-Diagnosis-of-Pneumonia-and-COVID-19

1. Download dataset về từ link Kaggle này : https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data?select=COVID-19_Radiography_Dataset

2. Extract file tải về vào clone repo 

3. Check lại cấu trúc thư mục

COVID-19_Radiography_Dataset/
├── COVID/
├── Lung_Opacity/
├── Normal/
└── Viral Pneumonia/

4. Chạy file preprocess.py để tạo ra thư mục processed_data bằng câu lệnh sau 

python src\preprocess.py

5. Chạy file split_data.py để chia tập train/val/test bằng câu lệnh

python src\split_data.py