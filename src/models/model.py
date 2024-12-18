import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load cleaned dataset
data = pd.read_csv("../../data/cleaned_data.csv")

print("Dữ liệu gốc:")
print(data.head(5))

# Load the trained model and scalers

model = joblib.load("random_forest_model.pkl")
# model = joblib.load("linear_regression_model.pkl")

PredictorScalerFit = joblib.load("predictor_scaler.pkl")
TargetVarScalerFit = joblib.load("target_var_scaler.pkl")
print("Mô hình và scaler đã được tải thành công!")


# Function for preprocessing input
def preprocess_input(ward, district, house_type, legal_paper, floors, bedrooms, area):
    valid_wards = data["Phường"].unique()
    valid_districts = data["Quận"].unique()
    valid_house_types = data["Loại hình nhà ở"].unique()
    valid_legal_papers = data["Giấy tờ pháp lý"].unique()

    # Validate user inputs
    if ward not in valid_wards:
        raise ValueError(f"Phường '{ward}' không tồn tại trong dữ liệu.")
    if district not in valid_districts:
        raise ValueError(f"Quận '{district}' không tồn tại trong dữ liệu.")
    if house_type not in valid_house_types:
        raise ValueError(f"Loại hình nhà ở '{house_type}' không hợp lệ.")
    if legal_paper not in valid_legal_papers:
        raise ValueError(f"Giấy tờ pháp lý '{legal_paper}' không hợp lệ.")

    # Create the input DataFrame
    input_data = {
        "Số tầng": [floors],
        "Số phòng ngủ": [bedrooms],
        "Diện tích": [area],
        "Quận": [district],
        "Phường": [ward],
        "Loại hình nhà ở": [house_type],
        "Giấy tờ pháp lý": [legal_paper],
    }
    input_df = pd.DataFrame(input_data)

    dummy_columns = pd.get_dummies(
        data[["Loại hình nhà ở", "Giấy tờ pháp lý", "Quận", "Phường"]],
        prefix="",  # Không có prefix
        prefix_sep="",  # Không có ký tự phân cách
    )

    input_encoded = pd.get_dummies(
        input_df[["Loại hình nhà ở", "Giấy tờ pháp lý", "Quận", "Phường"]],
        prefix="",
        prefix_sep="",
    )

    for col in dummy_columns.columns:
        if col not in input_encoded:
            input_encoded[col] = 0
        else:
            input_encoded[col] = 1

    input_encoded = input_encoded[dummy_columns.columns]

    # Add numeric features at the beginning
    input_encoded.insert(0, "Số tầng", floors)
    input_encoded.insert(1, "Số phòng ngủ", bedrooms)
    input_encoded.insert(2, "Diện tích", area)

    # Scale numeric features
    numeric_features = ["Số tầng", "Số phòng ngủ", "Diện tích"]
    input_encoded[numeric_features] = PredictorScalerFit.transform(
        input_encoded[numeric_features]
    )

    print(input_encoded.columns)

    return input_encoded


# Example input

ward = "Phường Trung Hoà"
district = "Quận Cầu Giấy"
house_type = "Nhà ngõ, hẻm"
legal_paper = "Đã có sổ"
floors = 5
bedrooms = 4
area = 65


# Preprocess input
input_processed = preprocess_input(
    ward, district, house_type, legal_paper, floors, bedrooms, area
)
print("Dữ liệu đầu vào đã tiền xử lý:")
print(input_processed)

# Predict using the trained model
prediction_scaled = model.predict(input_processed.values).reshape(-1, 1)
prediction = TargetVarScalerFit.inverse_transform(prediction_scaled)

# Display prediction result
print(f"Dự đoán giá nhà là: {prediction[0][0]:,.0f} VNĐ/m2")
