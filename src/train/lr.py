import warnings
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Import tqdm for progress bar
import joblib


# Turn off warnings
warnings.filterwarnings("ignore")

# Read the dataset
data = pd.read_csv("../../data/cleaned_data.csv")

# Convert non-numeric columns to dummy variables
dummy_type_of_housing = pd.get_dummies(data["Loại hình nhà ở"])
dummy_legal_paper = pd.get_dummies(data["Giấy tờ pháp lý"])
dummy_district = pd.get_dummies(data["Quận"])
dummy_ward = pd.get_dummies(data["Phường"])

# Combine the dummy variables into the cleaned data
data_cleaned = pd.concat(
    [data, dummy_type_of_housing, dummy_legal_paper, dummy_district, dummy_ward], axis=1
)
data_cleaned = data_cleaned.drop(
    ["Địa chỉ", "Quận", "Phường", "Loại hình nhà ở", "Giấy tờ pháp lý"], axis=1
)


# Separate predictors and target (price) variables
X = data_cleaned.loc[:, data_cleaned.columns != "Giá/m2"]
y = data_cleaned[["Giá/m2"]]
print(X.columns)
# Columns to be scaled
to_be_scaled = ["Số tầng", "Số phòng ngủ", "Diện tích"]

# Initialize the scalers
PredictorScaler = StandardScaler()
TargetVarScaler = StandardScaler()

X_scaled = X.copy()
y_scaled = y.copy()

# Fit the scalers and apply transformations
PredictorScalerFit = PredictorScaler.fit(X_scaled[to_be_scaled])
TargetVarScalerFit = TargetVarScaler.fit(y_scaled)

X_scaled[to_be_scaled] = PredictorScalerFit.transform(X_scaled[to_be_scaled])
y_scaled = TargetVarScalerFit.transform(y)

print(X_scaled.head(5))

# Convert to numpy arrays for model training
X_array = np.array(X_scaled.values).astype("float32")
y_array = np.array(y_scaled).astype("float32")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_array, y_array, test_size=0.2, random_state=2032
)


# Sanity check
if (
    X_train.shape[0] == y_train.shape[0]
    and X_train.shape[1] == X_test.shape[1]
    and X_test.shape[0] == y_test.shape[0]
    and y_train.shape[1] == y_test.shape[1]
):
    print("All train and test sets have correct dimensions.")

# --------------------------
# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
r_sq = model.score(X_train, y_train)
print(f"Coefficient of determination for Linear Regression: {r_sq}")

# Predict using the Linear Regression model
y_pred = model.predict(X_test)
print(
    "Predictions from the Linear Regression model:\n", y_pred[:5]
)  # Show the first 5 predictions


# --------------------------
# Define function to evaluate predictions using Accuracy Score (MAPE)
def Accuracy_Score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig - pred) / orig))
    return 100 - MAPE


# Evaluate accuracy for the Linear Regression model
accuracy_lr = Accuracy_Score(y_test, y_pred)
print(f"Accuracy for the LR model is: {accuracy_lr}%")

# -------------------------------------------------------------------

y_test_orig = TargetVarScalerFit.inverse_transform(y_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Scaling the test data back to original scale
Test_Data = np.concatenate(
    (PredictorScalerFit.inverse_transform(X_test[:, :3]), X_test[:, 3:]), axis=1
)
TestingData = pd.DataFrame(data=Test_Data, columns=X.columns)
TestingData["Giá/m2"] = y_test_orig

LR_predictions = model.predict(X_test)
LR_predictions = TargetVarScalerFit.inverse_transform(LR_predictions)
TestingData["LR_predictions"] = LR_predictions
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(LR_predictions, y_test))

print(
    "Accuracy for the LR model is:",
    str(Accuracy_Score(TestingData["Giá/m2"], TestingData["LR_predictions"])),
)

print(X_train)

joblib.dump(model, "../models/linear_regression_model.pkl")
joblib.dump(PredictorScalerFit, "../models/predictor_scaler.pkl")
joblib.dump(TargetVarScalerFit, "../models/target_var_scaler.pkl")
print("Mô hình và scalers đã được lưu thành công!")
