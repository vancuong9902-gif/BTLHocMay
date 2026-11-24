"""
predict.py - Script nhận diện / suy luận mẫu dữ liệu mới
Dự đoán nguy cơ mắc bệnh tim mạch vành (CHD) trong 10 năm
Sử dụng mô hình đã huấn luyện từ bộ dữ liệu Framingham Heart Study
"""

import argparse
import numpy as np
import pandas as pd
import joblib
import os
import sys


# Đường dẫn mặc định đến mô hình và scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")

# Danh sách các đặc trưng (features) theo thứ tự
FEATURE_NAMES = [
    "male", "age", "education", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]


def load_model():
    """Tải mô hình và scaler đã lưu."""
    if not os.path.exists(MODEL_PATH):
        print(f"[LỖI] Không tìm thấy mô hình tại: {MODEL_PATH}")
        print("Vui lòng chạy notebook 02_Modeling_Evaluation.ipynb trước để huấn luyện và lưu mô hình.")
        sys.exit(1)

    if not os.path.exists(SCALER_PATH):
        print(f"[LỖI] Không tìm thấy scaler tại: {SCALER_PATH}")
        print("Vui lòng chạy notebook 02_Modeling_Evaluation.ipynb trước để huấn luyện và lưu scaler.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[OK] Đã tải mô hình và scaler thành công!")
    return model, scaler


def predict_single(model, scaler):
    """Nhập thông tin bệnh nhân từ bàn phím và dự đoán."""
    print("\n" + "=" * 50)
    print("NHẬP THÔNG TIN BỆNH NHÂN")
    print("=" * 50)

    try:
        data = {}
        data["male"] = int(input("Giới tính (1: Nam, 0: Nữ): "))
        data["age"] = int(input("Tuổi: "))
        data["education"] = float(input("Trình độ học vấn (1-4): "))
        data["currentSmoker"] = int(input("Đang hút thuốc (1: Có, 0: Không): "))
        data["cigsPerDay"] = float(input("Số điếu thuốc/ngày: "))
        data["BPMeds"] = float(input("Dùng thuốc huyết áp (1: Có, 0: Không): "))
        data["prevalentStroke"] = int(input("Tiền sử đột quỵ (1: Có, 0: Không): "))
        data["prevalentHyp"] = int(input("Tiền sử tăng huyết áp (1: Có, 0: Không): "))
        data["diabetes"] = int(input("Tiểu đường (1: Có, 0: Không): "))
        data["totChol"] = float(input("Tổng cholesterol (mg/dL): "))
        data["sysBP"] = float(input("Huyết áp tâm thu (mmHg): "))
        data["diaBP"] = float(input("Huyết áp tâm trương (mmHg): "))
        data["BMI"] = float(input("Chỉ số BMI: "))
        data["heartRate"] = float(input("Nhịp tim (nhịp/phút): "))
        data["glucose"] = float(input("Đường huyết (mg/dL): "))
    except (ValueError, KeyboardInterrupt):
        print("\n[LỖI] Dữ liệu không hợp lệ.")
        return

    # Tạo DataFrame và dự đoán
    df_input = pd.DataFrame([data], columns=FEATURE_NAMES)
    X_scaled = scaler.transform(df_input)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]

    # Hiển thị kết quả
    print("\n" + "=" * 50)
    print("KẾT QUẢ DỰ ĐOÁN")
    print("=" * 50)
    if prediction == 1:
        print(f"⚠️  CÓ NGUY CƠ mắc bệnh tim mạch trong 10 năm")
    else:
        print(f"✅  KHÔNG có nguy cơ mắc bệnh tim mạch trong 10 năm")
    print(f"Xác suất KHÔNG mắc bệnh: {probability[0]:.2%}")
    print(f"Xác suất CÓ mắc bệnh:   {probability[1]:.2%}")
    print("=" * 50)


def predict_from_csv(model, scaler, csv_path):
    """Dự đoán từ file CSV."""
    if not os.path.exists(csv_path):
        print(f"[LỖI] Không tìm thấy file: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"[OK] Đã tải {len(df)} mẫu từ {csv_path}")

    # Kiểm tra các cột
    missing_cols = set(FEATURE_NAMES) - set(df.columns)
    if missing_cols:
        print(f"[LỖI] Thiếu các cột: {missing_cols}")
        sys.exit(1)

    X = df[FEATURE_NAMES]
    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    df["TenYearCHD_Predicted"] = predictions
    df["CHD_Probability"] = probabilities.round(4)

    # Lưu kết quả
    output_path = csv_path.replace(".csv", "_predictions.csv")
    df.to_csv(output_path, index=False)

    print(f"\n[OK] Kết quả đã lưu tại: {output_path}")
    print(f"Tổng mẫu: {len(df)}")
    print(f"Dự đoán CÓ nguy cơ: {predictions.sum()} ({predictions.mean():.2%})")
    print(f"Dự đoán KHÔNG nguy cơ: {(1 - predictions).sum()} ({(1 - predictions).mean():.2%})")


def main():
    global MODEL_PATH, SCALER_PATH
    parser = argparse.ArgumentParser(
        description="Dự đoán nguy cơ bệnh tim mạch 10 năm - Framingham Heart Study"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Đường dẫn file CSV chứa dữ liệu mới cần dự đoán"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH,
        help="Đường dẫn đến file mô hình (.joblib)"
    )
    parser.add_argument(
        "--scaler", type=str, default=SCALER_PATH,
        help="Đường dẫn đến file scaler (.joblib)"
    )

    args = parser.parse_args()

    MODEL_PATH = args.model
    SCALER_PATH = args.scaler

    model, scaler = load_model()

    if args.csv:
        predict_from_csv(model, scaler, args.csv)
    else:
        predict_single(model, scaler)


if __name__ == "__main__":
    main()
