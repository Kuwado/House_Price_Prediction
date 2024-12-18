import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


@st.cache_resource
def load_data_and_models():
    # Load dữ liệu và model
    data = pd.read_csv("../data/cleaned_data.csv")
    rf_model = joblib.load("models/random_forest_model.pkl")
    lr_model = joblib.load("models/linear_regression_model.pkl")
    PredictorScalerFit = joblib.load("models/predictor_scaler.pkl")
    TargetVarScalerFit = joblib.load("models/target_var_scaler.pkl")
    print("Mô hình và scaler đã được tải thành công!")
    return data, rf_model, lr_model, PredictorScalerFit, TargetVarScalerFit


data, rf_model, lr_model, PredictorScalerFit, TargetVarScalerFit = (
    load_data_and_models()
)


# Xử lý dữ liệu đầu vào
def preprocess_input(ward, district, house_type, legal_paper, floors, bedrooms, area):
    valid_wards = data["Phường"].unique()
    valid_districts = data["Quận"].unique()
    valid_house_types = data["Loại hình nhà ở"].unique()
    valid_legal_papers = data["Giấy tờ pháp lý"].unique()

    # Kiểm tra validate
    if ward not in valid_wards:
        raise ValueError(f"Phường '{ward}' không tồn tại trong dữ liệu.")
    if district not in valid_districts:
        raise ValueError(f"Quận '{district}' không tồn tại trong dữ liệu.")
    if house_type not in valid_house_types:
        raise ValueError(f"Loại hình nhà ở '{house_type}' không hợp lệ.")
    if legal_paper not in valid_legal_papers:
        raise ValueError(f"Giấy tờ pháp lý '{legal_paper}' không hợp lệ.")

    # Tạo inputData
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
        prefix="",
        prefix_sep="",
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

    # Chèn các cột cần scale vào đầu
    input_encoded.insert(0, "Số tầng", floors)
    input_encoded.insert(1, "Số phòng ngủ", bedrooms)
    input_encoded.insert(2, "Diện tích", area)

    # Scale
    numeric_features = ["Số tầng", "Số phòng ngủ", "Diện tích"]
    input_encoded[numeric_features] = PredictorScalerFit.transform(
        input_encoded[numeric_features]
    )

    return input_encoded


# ---- Dữ liệu đầu vào ------
# ward = "Phường Trung Hoà"
# district = "Quận Cầu Giấy"
# house_type = "Nhà ngõ, hẻm"
# legal_paper = "Đã có sổ"
# floors = 5
# bedrooms = 4
# area = 65

# Tạo giao diện với Streamlit
st.markdown(
    """
    <style>
    .main {
        padding-left: 50px;
        padding-right: 50px;
        background-color: blue;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Dự đoán giá nhà tại Hà Nội")
st.write("Nhập thông tin chi tiết về ngôi nhà để dự đoán giá.")

# Input từ người dùng
ward = st.selectbox("Chọn Phường", data["Phường"].unique())
district = st.selectbox("Chọn Quận", data["Quận"].unique())
house_type = st.selectbox("Loại hình nhà ở", data["Loại hình nhà ở"].unique())
legal_paper = st.selectbox("Giấy tờ pháp lý", data["Giấy tờ pháp lý"].unique())
floors = st.number_input("Số tầng", min_value=1, max_value=15, value=1)
bedrooms = st.number_input("Số phòng ngủ", min_value=1, max_value=15, value=1)
area = st.number_input("Diện tích (m2)", min_value=10.0, max_value=500.0, value=50.0)
predict_button = st.button("Dự đoán giá nhà")

if predict_button:
    st.header("Kết quả:")
    try:
        # Tiền xử lý dữ liệu
        input_processed = preprocess_input(
            ward, district, house_type, legal_paper, floors, bedrooms, area
        )

        # Dự đoán
        lr_prediction_scaled = lr_model.predict(input_processed.values).reshape(-1, 1)
        rf_prediction_scaled = rf_model.predict(input_processed.values).reshape(-1, 1)

        lr_prediction = TargetVarScalerFit.inverse_transform(lr_prediction_scaled)
        rf_prediction = TargetVarScalerFit.inverse_transform(rf_prediction_scaled)

        # Hiển thị kết quả
        st.success(
            f"Giá nhà dự đoán (Linear Regression): {lr_prediction[0][0]:,.0f} triệu VNĐ/m2"
        )
        st.success(
            f"Giá nhà dự đoán (Random Forest): {rf_prediction[0][0]:,.0f} triệu VNĐ/m2"
        )
    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {e}")
