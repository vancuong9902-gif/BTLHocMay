# Bài Tập Lớn Học Máy - Dự đoán bệnh tim mạch (Framingham Heart Study)

## 📋 Mô tả dự án
Dự án sử dụng bộ dữ liệu **Framingham Heart Study** để xây dựng mô hình học máy dự đoán nguy cơ mắc bệnh tim mạch vành (Coronary Heart Disease - CHD) trong 10 năm.

## 📁 Cấu trúc thư mục
```
BTLHocMay/
├── 01_Report/
│   └── Bao_cao_BTL_[Tên_Nhom].pdf       # Báo cáo kỹ thuật (15-25 trang)
├── 02_Slide/
│   └── Slide_Thuyet_trinh_[Tên_Nhom].pdf # Slide thuyết trình (12-15 trang)
├── 03_SourceCode/
│   ├── data/
│   │   └── framingham.csv                # Dữ liệu gốc Framingham
│   ├── src/
│   │   ├── __init__.py
│   │   └── features.py                   # Tạo đặc trưng mới & Pipeline chống Data Leakage
│   ├── notebooks/
│   │   ├── 01_EDA_Preprocessing.ipynb    # Phân tích dữ liệu & Tiền xử lý
│   │   └── 02_Modeling_Evaluation.ipynb  # Huấn luyện, tuning & so sánh mô hình
│   ├── models/
│   │   ├── preprocessor_pipeline.joblib  # Pipeline tiền xử lý đã đóng gói
│   │   └── best_model.joblib             # Mô hình tối ưu (sau khi huấn luyện)
│   ├── predict.py                        # Script suy luận mẫu dữ liệu mới
│   ├── requirements.txt                  # Danh sách thư viện & phiên bản
│   └── README.md                         # Hướng dẫn cài đặt và chạy mã nguồn (file này)
└── 04_Video_Demo/                        # Video ngắn 3 phút demo (khuyến khích)
```

## 📊 Thông tin bộ dữ liệu
- **Nguồn:** Framingham Heart Study
- **Số mẫu:** 4,240 bản ghi
- **Số đặc trưng:** 16 cột
- **Biến mục tiêu:** `TenYearCHD` (0: Không mắc bệnh, 1: Mắc bệnh trong 10 năm)

### Danh sách các đặc trưng (Features):
| STT | Tên cột | Mô tả | Kiểu dữ liệu |
|-----|---------|--------|---------------|
| 1 | `male` | Giới tính (1: Nam, 0: Nữ) | int |
| 2 | `age` | Tuổi | int |
| 3 | `education` | Trình độ học vấn (1-4) | float |
| 4 | `currentSmoker` | Đang hút thuốc (1: Có, 0: Không) | int |
| 5 | `cigsPerDay` | Số điếu thuốc/ngày | float |
| 6 | `BPMeds` | Đang dùng thuốc huyết áp (1: Có, 0: Không) | float |
| 7 | `prevalentStroke` | Tiền sử đột quỵ (1: Có, 0: Không) | int |
| 8 | `prevalentHyp` | Tiền sử tăng huyết áp (1: Có, 0: Không) | int |
| 9 | `diabetes` | Tiểu đường (1: Có, 0: Không) | int |
| 10 | `totChol` | Tổng cholesterol (mg/dL) | float |
| 11 | `sysBP` | Huyết áp tâm thu (mmHg) | float |
| 12 | `diaBP` | Huyết áp tâm trương (mmHg) | float |
| 13 | `BMI` | Chỉ số khối cơ thể | float |
| 14 | `heartRate` | Nhịp tim (nhịp/phút) | float |
| 15 | `glucose` | Đường huyết (mg/dL) | float |
| 16 | `TenYearCHD` | **Biến mục tiêu** - Nguy cơ CHD 10 năm | int |

### 3 Đặc trưng mới được tạo (STT 5 - Feature Engineering):
| Tên đặc trưng | Công thức y khoa | Ý nghĩa lâm sàng |
|---|---|---|
| `pulse_pressure` | `sysBP - diaBP` | Áp lực mạch: Phản ánh độ cứng thành động mạch, nguy cơ xơ vữa |
| `MAP` | `diaBP + (sysBP - diaBP) / 3` | Huyết áp động mạch trung bình: Đánh giá áp lực tưới máu cơ quan |
| `age_cigs_interaction` | `age * cigsPerDay` | Mức độ phơi nhiễm độc chất thuốc lá tích lũy theo độ tuổi |

### Dữ liệu thiếu (Missing Values):
| Cột | Số lượng thiếu | Tỷ lệ (%) |
|-----|----------------|-----------|
| `glucose` | 388 | 9.15% |
| `education` | 105 | 2.48% |
| `BPMeds` | 53 | 1.25% |
| `totChol` | 50 | 1.18% |
| `cigsPerDay` | 29 | 0.68% |
| `BMI` | 19 | 0.45% |
| `heartRate` | 1 | 0.02% |

## 🚀 Hướng dẫn cài đặt và chạy thử

### 1. Yêu cầu hệ thống
- Python >= 3.9

### 2. Cài đặt thư viện
```bash
cd 03_SourceCode
pip install -r requirements.txt
```

### 3. Chạy phân tích dữ liệu (EDA)
Mở và chạy notebook:
```bash
jupyter notebook notebooks/01_EDA_Preprocessing.ipynb
```

### 4. Huấn luyện mô hình
Mở và chạy notebook:
```bash
jupyter notebook notebooks/02_Modeling_Evaluation.ipynb
```

### 5. Dự đoán mẫu mới

**Cách 1: Nhập từ bàn phím**
```bash
python predict.py
```

**Cách 2: Dự đoán từ file CSV**
```bash
python predict.py --csv data/new_patients.csv
```

## 👥 Thành viên nhóm
- Thành viên 1 (Nhóm trưởng)
- Thành viên 2
- Thành viên 3

## 📝 Lưu ý quan trọng
- **Đường dẫn file:** Sử dụng đường dẫn tương đối `data/framingham.csv`, **KHÔNG** dùng `C:\Users\...`
- **Thư viện:** Đảm bảo cài đúng phiên bản trong `requirements.txt` để cả 3 máy thống nhất
- **Mô hình:** File `best_model.joblib` sẽ được tạo sau khi chạy notebook `02_Modeling_Evaluation.ipynb`
