import warnings
import os
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor, KerasClassifier

# Ignoring future warnings and deprecation warnings so as not to make the notebook full of warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("../../data/cleaned_data.csv")
df.head()

# df["Ngày"] = pd.to_datetime(df["Ngày"])
# df["Ngày"] = df["Ngày"].astype(str)


# rename
df = df.rename(
    columns={
        # "Ngày": "date",
        "Địa chỉ": "address",
        "Quận": "district",
        "Phường": "ward",
        "Loại hình nhà ở": "type_of_housing",
        "Giấy tờ pháp lý": "legal_paper",
        "Số tầng": "num_floors",
        "Số phòng ngủ": "num_bed_rooms",
        "Diện tích": "squared_meter_area",
        "Dài": "length_meter",
        "Rộng": "width_meter",
        "Giá/m2": "price_in_million_per_square_meter",
    }
)

print(df.head())

# duplicate
df[df.duplicated(keep=False)]

# df = df.drop("Stt", axis=1)
df = df.dropna()
df = df.reset_index()

# The total records of the dataset after dropping null values
print("The total records of the dataset are: ", str(len(df)), "records.")

df = df[df["num_floors"] != "Nhiều hơn 10"]
df = df[df["num_bed_rooms"] != "nhiều hơn 10 phòng"]

df["district"] = df["district"].str.replace("Quận ", "").str.strip()
df["ward"] = df["ward"].str.replace("Phường ", "").str.strip()
df["num_floors"] = df["num_floors"].astype(float)
df["num_bed_rooms"] = df["num_bed_rooms"].astype(float)

df["squared_meter_area"] = df["squared_meter_area"].astype(float)
df["price_in_million_per_square_meter"] = df[
    "price_in_million_per_square_meter"
].astype(float)

# df["length_meter"] = df["length_meter"].str.replace(" m", "").str.strip().astype(float)
# df["width_meter"] = df["width_meter"].str.replace(" m", "").str.strip().astype(float)

print(
    df[
        [
            "num_floors",
            "num_bed_rooms",
            "squared_meter_area",
            # "length_meter",
            # "width_meter",
        ]
    ]
)

# df.groupby("price_in_million_per_square_meter").count()["date"]

# df.loc[
#     df["price_in_million_per_square_meter"].str.contains(" tỷ/m²"),
#     "price_in_million_per_square_meter",
# ] = (
#     df.loc[
#         df["price_in_million_per_square_meter"].str.contains(" tỷ/m²"),
#         "price_in_million_per_square_meter",
#     ]
#     .str.replace(" tỷ/m²", "")
#     .str.replace(".", "")
#     .str.replace(",", ".")
#     .astype(float)
#     * 1000
# )
# df.loc[
#     df["price_in_million_per_square_meter"].str.contains(" triệu/m²", na=False),
#     "price_in_million_per_square_meter",
# ] = (
#     df.loc[
#         df["price_in_million_per_square_meter"].str.contains(" triệu/m²", na=False),
#         "price_in_million_per_square_meter",
#     ]
#     .str.replace(" triệu/m²", "")
#     .str.replace(",", ".")
#     .astype(float)
# )
# df.loc[
#     df["price_in_million_per_square_meter"].str.contains(" đ/m²", na=False),
#     "price_in_million_per_square_meter",
# ] = (
#     df.loc[
#         df["price_in_million_per_square_meter"].str.contains(" đ/m²", na=False),
#         "price_in_million_per_square_meter",
#     ]
#     .str.replace(" đ/m²", "")
#     .str.replace(".", "")
#     .astype(float)
#     * 0.000001
# )

dummy_type_of_housing = pd.get_dummies(df.type_of_housing, prefix="housing_type")
dummy_legal_paper = pd.get_dummies(df.legal_paper, prefix="legal_paper")
dummy_district = pd.get_dummies(df.district, prefix="district")
dummy_ward = pd.get_dummies(df.ward, prefix="ward")

df_cleaned = pd.concat(
    [df, dummy_type_of_housing, dummy_legal_paper, dummy_district, dummy_ward], axis=1
)
df_cleaned = df_cleaned.drop(
    ["index", "address", "district", "ward", "type_of_housing", "legal_paper"],
    axis=1,
)
print(df_cleaned.head())


removed_outliers = df_cleaned


print("The final length of the dataset is", str(len(removed_outliers)), "rows.")

housing = removed_outliers

# Separate predictors and response (price) variables
X = housing.loc[
    :,
    (housing.columns != "price_in_million_per_square_meter")
    & (housing.columns != "date"),
]
y = housing[["price_in_million_per_square_meter"]]
to_be_scaled = [
    "num_floors",
    "num_bed_rooms",
    "squared_meter_area",
    # "length_meter",
    # "width_meter",
]

# Initiate scaler
PredictorScaler = StandardScaler()
TargetVarScaler = StandardScaler()

X_scaled = X
y_scaled = y

# Storing the fit object for reference and reverse the scaling later
PredictorScalerFit = PredictorScaler.fit(X_scaled[to_be_scaled])
TargetVarScalerFit = TargetVarScaler.fit(y_scaled)

# Generating the standardized values of X and y
X_scaled[to_be_scaled] = PredictorScalerFit.transform(X_scaled[to_be_scaled])
y_scaled = TargetVarScalerFit.transform(y)

X_array = np.array(X_scaled.values).astype("float32")
y_array = np.array(y_scaled).astype("float32")

X_train, X_test, y_train, y_test = train_test_split(
    X_array, y_array, test_size=0.2, random_state=2023
)

# Sanity check to see if all train and test arrays have correct dimensions
assert (
    X_train.shape[0] == y_train.shape[0]
    and X_train.shape[1] == X_test.shape[1]
    and X_test.shape[0] == y_test.shape[0]
    and y_train.shape[1] == y_test.shape[1]
), "All train and test sets should have correct dimensions."

import numpy as np
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

r_sq = model.score(X_train, y_train)
print(f"coefficient of determination: {r_sq}")

y_pred = model.intercept_ + np.sum(model.coef_ * X_test, axis=1)
print("The prediction of the statistical Linear Regression model:\n", y_pred)

# from sklearn.preprocessing import PolynomialFeatures

# X_train_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_train)
# X_train_.shape

# new_model = LinearRegression()
# new_model.fit(X_train_, y_train)

# r_sq = new_model.score(X_train_, y_train)
# print(f"coefficient of determination: {r_sq}")


# # Define a function evaluate the predictions
# def Accuracy_Score(orig, pred):
#     MAPE = np.mean(100 * (np.abs(orig - pred) / orig))
#     return 100 - MAPE


# print(
#     "Accuracy for the LR model is:",
#     str(Accuracy_Score(TestingData["Price"], TestingData["LR_predictions"])),
# )


# Define a function to evaluate the predictions
# Define a function evaluate the predictions
def Accuracy_Score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig - pred) / orig))
    return 100 - MAPE


# Dự đoán giá trị y_test từ mô hình Linear Regression
y_test_pred = model.predict(X_test)

# Đảo ngược quá trình chuẩn hóa để có giá trị gốc
y_test_actual = TargetVarScalerFit.inverse_transform(y_test)
y_test_pred_actual = TargetVarScalerFit.inverse_transform(y_test_pred)

# Tính độ chính xác
accuracy = Accuracy_Score(y_test_actual, y_test_pred_actual)
print(f"Accuracy for the LR model 1 is: {accuracy:.2f}%")
# Scaling the y_test Price data back to original price scale


y_test_orig = TargetVarScalerFit.inverse_transform(y_test)

# Scaling the test data back to original scale
Test_Data = np.concatenate(
    (PredictorScalerFit.inverse_transform(X_test[:, :5]), X_test[:, 5:]), axis=1
)
TestingData = pd.DataFrame(data=Test_Data, columns=X.columns)
TestingData["Price"] = y_test_orig

LR_predictions = model.predict(X_test)
LR_predictions = TargetVarScalerFit.inverse_transform(LR_predictions)
TestingData["LR_predictions"] = LR_predictions
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(LR_predictions, y_test))

print(
    "Accuracy for the LR model is:",
    str(Accuracy_Score(TestingData["Price"], TestingData["LR_predictions"])),
)
