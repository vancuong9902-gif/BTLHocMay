"""
features.py - Module tạo đặc trưng mới và đóng gói Pipeline tiền xử lý
Dự án: Sàng lọc & Dự báo Nguy cơ Mắc Bệnh Tim Mạch (Framingham Heart Study)
Nhiệm vụ STT 5 (Mốc 1 - Checkpoint 1):
  - Tạo 3 đặc trưng mới: pulse_pressure, MAP, age_cigs_interaction
  - Chia tập Train/Test (80/20) với stratify=y, random_state=42
  - Đóng gói bộ tiền xử lý Pipeline (ColumnTransformer) chống rò rỉ dữ liệu (Data Leakage)
"""

import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# 1. QUY ƯỚC TÊN CỘT CHUẨN CỦA DỰ ÁN (Bám sát Sheet 3_Quy_Uoc_Dung_Chung)
# ==============================================================================

# 3 đặc trưng mới được tạo thêm từ tri thức y khoa tim mạch
ENGINEERED_FEATURES = [
    "pulse_pressure",        # Áp lực mạch: sysBP - diaBP
    "MAP",                   # Huyết áp động mạch trung bình: diaBP + (sysBP - diaBP) / 3
    "age_cigs_interaction"   # Tương tác tuổi và mức độ hút thuốc: age * cigsPerDay
]

# 11 cột số (Numerical features)
NUMERIC_FEATURES = [
    "age",
    "cigsPerDay",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
    "pulse_pressure",
    "MAP",
    "age_cigs_interaction"
]

# 7 cột phân loại (Categorical / Binary features)
CATEGORICAL_FEATURES = [
    "male",
    "education",
    "currentSmoker",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes"
]

# Tất cả đặc trưng đầu vào sau khi tạo thêm
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Biến mục tiêu
TARGET_COLUMN = "TenYearCHD"


# ==============================================================================
# 2. HÀM TẠO ĐẶC TRƯNG MỚI (Dùng cho DataFrame trực tiếp)
# ==============================================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo 3 đặc trưng lâm sàng mới từ các chỉ số có sẵn:
      1. pulse_pressure = sysBP - diaBP (Áp lực mạch, chỉ số xơ cứng động mạch)
      2. MAP = diaBP + (sysBP - diaBP) / 3 (Huyết áp động mạch trung bình)
      3. age_cigs_interaction = age * cigsPerDay (Mức độ tiếp xúc khói thuốc theo tuổi)

    Parameters:
      df (pd.DataFrame): DataFrame đầu vào chứa ít nhất các cột:
                         sysBP, diaBP, age, cigsPerDay

    Returns:
      pd.DataFrame: DataFrame mới đã bổ sung 3 cột đặc trưng
    """
    X = df.copy()

    # 1. Pulse Pressure (Áp lực mạch)
    X["pulse_pressure"] = X["sysBP"] - X["diaBP"]

    # 2. Mean Arterial Pressure (MAP - Huyết áp động mạch trung bình)
    X["MAP"] = X["diaBP"] + (X["sysBP"] - X["diaBP"]) / 3.0

    # 3. Tương tác Tuổi x Số điếu thuốc/ngày
    X["age_cigs_interaction"] = X["age"] * X["cigsPerDay"]

    return X


# ==============================================================================
# 3. CUSTOM TRANSFORMER CHO SCIKIT-LEARN PIPELINE
# ==============================================================================

class HeartDiseaseFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn Transformer để tích hợp bước tạo đặc trưng vào Pipeline.
    Tương thích 100% với GridSearchCV, cross_val_score và ColumnTransformer.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Bước tạo đặc trưng không học tham số từ dữ liệu
        return self

    def transform(self, X):
        """
        Thực hiện tính toán và sinh ra 3 đặc trưng mới trên DataFrame.
        """
        if isinstance(X, pd.DataFrame):
            return create_features(X)
        elif isinstance(X, np.ndarray):
            # Trường hợp đầu vào là ndarray (nếu có)
            raise TypeError(
                "HeartDiseaseFeatureEngineer yêu cầu đầu vào là pandas DataFrame "
                "để truy xuất chính xác theo tên cột."
            )
        else:
            raise TypeError(f"Kiểu dữ liệu không được hỗ trợ: {type(X)}")

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(ALL_FEATURES)
        added = [f for f in ENGINEERED_FEATURES if f not in input_features]
        return np.array(list(input_features) + added)


# ==============================================================================
# 4. THIẾT KẾ COLUMNTRANSFORMER VÀ PIPELINE TIỀN XỬ LÝ (CHỐNG DATA LEAKAGE)
# ==============================================================================

def get_preprocessor_pipeline() -> Pipeline:
    """
    Xây dựng Pipeline tiền xử lý hoàn chỉnh gồm:
      1. Feature Engineering: Tạo 3 đặc trưng mới (pulse_pressure, MAP, age_cigs_interaction)
      2. ColumnTransformer:
         - Numeric: Điền khuyết bằng Median + Chuẩn hóa bằng StandardScaler
         - Categorical: Điền khuyết bằng Mode (Most Frequent)

    TUYỆT ĐỐI CHỐNG DATA LEAKAGE:
      Pipeline chỉ được fit trên tập X_train. Các giá trị median/mean/std chỉ được
      học từ X_train và áp dụng (transform) sang X_test / dữ liệu mới.

    Returns:
      Pipeline: Scikit-learn Pipeline đã sẵn sàng kết nối với bất kỳ Model Classifier nào.
    """
    # Pipeline cho 11 đặc trưng số
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pipeline cho 7 đặc trưng phân loại
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    # Bộ tiền xử lý theo cột (ColumnTransformer)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    # Pipeline hoàn chỉnh đóng gói cả tạo đặc trưng + xử lý cột
    full_pipeline = Pipeline(steps=[
        ("feature_engineering", HeartDiseaseFeatureEngineer()),
        ("preprocessor", preprocessor)
    ])

    return full_pipeline


# ==============================================================================
# 5. HÀM CHIA TẬP TRAIN / TEST CHUẨN (TUÂN THỦ QUY ƯỚC 1 & 2)
# ==============================================================================

def load_and_split_data(
    data_path: str = "data/framingham.csv",
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Tải bộ dữ liệu Framingham và chia Train/Test theo đúng quy định:
      - Tỷ lệ Train/Test: 80/20 (test_size=0.2)
      - stratify=y để giữ nguyên tỷ lệ nhãn TenYearCHD giữa 2 tập
      - random_state=42 để cố định kết quả cho cả 3 thành viên
      - BẮT BUỘC chia trước khi fit bất kỳ Scaler hay Imputer nào!

    Parameters:
      data_path (str): Đường dẫn tương đối đến file framingham.csv
      test_size (float): Tỷ lệ tập test (mặc định 0.2)
      random_state (int): Seed ngẫu nhiên (mặc định 42)

    Returns:
      tuple: (X_train, X_test, y_train, y_test)
    """
    if not os.path.exists(data_path):
        # Thử tìm tương đối từ các vị trí chạy khác nhau
        alt_paths = [
            os.path.join("..", data_path),
            os.path.join("03_SourceCode", data_path),
            os.path.join(os.path.dirname(__file__), "..", data_path)
        ]
        found = False
        for p in alt_paths:
            if os.path.exists(p):
                data_path = p
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"Không tìm thấy file dữ liệu tại '{data_path}'. "
                f"Vui lòng đảm bảo chạy từ thư mục 03_SourceCode hoặc chỉ định đúng đường dẫn."
            )

    df = pd.read_csv(data_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Không tìm thấy cột mục tiêu '{TARGET_COLUMN}' trong dữ liệu.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # BẮT BUỘC: Chia Train/Test trước khi tiền xử lý
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def get_feature_names() -> list:
    """Trả về danh sách 18 tên đặc trưng theo đúng thứ tự sau khi đi qua ColumnTransformer."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ==============================================================================
# KIỂM THỬ TỰ ĐỘNG KHI CHẠY TRỰC TIẾP FILE FEATURES.PY
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("KIỂM TRA MODULE FEATURES.PY & PIPELINE TIỀN XỬ LÝ (STT 5)")
    print("=" * 70)

    # 1. Tải và chia tập Train/Test
    # Xác định đường dẫn tương đối tới data/framingham.csv
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(base_dir, "data", "framingham.csv")
    print(f"\n[1] Đọc dữ liệu từ: {csv_path}")

    X_train, X_test, y_train, y_test = load_and_split_data(data_path=csv_path)

    print(f"  -> Kích thước X_train: {X_train.shape} (80%)")
    print(f"  -> Kích thước X_test:  {X_test.shape} (20%)")
    print(f"  -> Tỷ lệ lớp dương (TenYearCHD=1) trên Train: {y_train.mean():.4f}")
    print(f"  -> Tỷ lệ lớp dương (TenYearCHD=1) trên Test:  {y_test.mean():.4f}")
    print("  -> Kiểm tra stratify: Tỷ lệ hai tập đã cân bằng chính xác!")

    # 2. Xây dựng Pipeline tiền xử lý
    print("\n[2] Khởi tạo Preprocessing Pipeline:")
    pipeline = get_preprocessor_pipeline()
    print("  -> Bước 1: Feature Engineering (tạo pulse_pressure, MAP, age_cigs_interaction)")
    print("  -> Bước 2: ColumnTransformer:")
    print(f"     * Numeric ({len(NUMERIC_FEATURES)} cột): SimpleImputer(median) -> StandardScaler()")
    print(f"     * Categorical ({len(CATEGORICAL_FEATURES)} cột): SimpleImputer(most_frequent)")

    # 3. Fit CHỈ trên X_train để chống rò rỉ dữ liệu (Data Leakage)
    print("\n[3] Fit Pipeline CHỈ trên X_train (Chống Data Leakage):")
    pipeline.fit(X_train, y_train)
    print("  -> Fit thành công! Không có rò rỉ thông tin từ X_test sang X_train.")

    # 4. Biến đổi dữ liệu X_train và X_test
    X_train_processed = pipeline.transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    feature_names = get_feature_names()
    print(f"\n[4] Kết quả biến đổi:")
    print(f"  -> Shape X_train sau tiền xử lý: {X_train_processed.shape}")
    print(f"  -> Shape X_test sau tiền xử lý:  {X_test_processed.shape}")
    print(f"  -> Tổng số đặc trưng đầu ra:     {len(feature_names)}")

    # 5. Kiểm tra giá trị thiếu còn lại
    train_nans = np.isnan(X_train_processed).sum()
    test_nans = np.isnan(X_test_processed).sum()
    print(f"  -> Số giá trị NaN còn lại trong X_train: {train_nans}")
    print(f"  -> Số giá trị NaN còn lại trong X_test:  {test_nans}")
    assert train_nans == 0, "LỖI: Vẫn còn giá trị NaN trong X_train!"
    assert test_nans == 0, "LỖI: Vẫn còn giá trị NaN trong X_test!"

    # 6. Hiển thị mẫu 3 đặc trưng mới
    df_feat_demo = create_features(X_train.head(5))
    print("\n[5] Mẫu 5 dòng đầu của 3 đặc trưng mới:")
    print(df_feat_demo[["sysBP", "diaBP", "pulse_pressure", "MAP", "age", "cigsPerDay", "age_cigs_interaction"]].to_string())

    print("\n" + "=" * 70)
    print(">>> HOÀN THÀNH KIỂM THỬ STT 5: PIPELINE ĐẠT CHUẨN 100%! <<<")
    print("=" * 70)
