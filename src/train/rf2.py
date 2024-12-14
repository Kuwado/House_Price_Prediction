import warnings
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm  # Import tqdm for progress bar
import json

# Turn off warnings
warnings.filterwarnings("ignore")

# Đọc file
data = pd.read_csv("../../data/cleaned_data.csv")

# Chuẩn hóa các cột dữ liệu không phải là số
dummy_type_of_housing = pd.get_dummies(data["Loại hình nhà ở"], prefix="housing_type")
dummy_legal_paper = pd.get_dummies(data["Giấy tờ pháp lý"], prefix="legal_paper")
dummy_district = pd.get_dummies(data["Quận"], prefix="district")
dummy_ward = pd.get_dummies(data["Phường"], prefix="ward")

# Kết hợp các cột đã chuyển đổi
data_cleaned = pd.concat(
    [data, dummy_type_of_housing, dummy_legal_paper, dummy_district, dummy_ward], axis=1
)
data_cleaned = data_cleaned.drop(
    [
        "Địa chỉ",
        "Quận",
        "Phường",
        "Loại hình nhà ở",
        "Giấy tờ pháp lý",
    ],
    axis=1,
)
print(data_cleaned.head())

# Separate predictors and response (price) variables
X = data_cleaned.loc[:, data_cleaned.columns != "Giá/m2"]
y = data_cleaned[["Giá/m2"]]

# List of columns that need to be scaled
to_be_scaled = [
    "Số tầng",
    "Số phòng ngủ",
    "Diện tích",
]

# Initialize the scalers
PredictorScaler = StandardScaler()
TargetVarScaler = StandardScaler()

X_scaled = X.copy()
y_scaled = y.copy()

# Storing the fit object for reference and reverse the scaling later
PredictorScalerFit = PredictorScaler.fit(X_scaled[to_be_scaled])
TargetVarScalerFit = TargetVarScaler.fit(y_scaled)

# Generating the standardized values of X and y
X_scaled[to_be_scaled] = PredictorScalerFit.transform(X_scaled[to_be_scaled])
y_scaled = TargetVarScalerFit.transform(y)

# Convert to numpy arrays for model training
X_array = np.array(X_scaled.values).astype("float32")
y_array = np.array(y_scaled).astype("float32")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_array, y_array, test_size=0.2, random_state=2032
)

# Sanity check to see if all train and test arrays have correct dimensions
if (
    X_train.shape[0] == y_train.shape[0]
    and X_train.shape[1] == X_test.shape[1]
    and X_test.shape[0] == y_test.shape[0]
    and y_train.shape[1] == y_test.shape[1]
):
    print("All train and test sets have correct dimensions.")

# ----------------------------------

# Load the best parameters from the saved JSON file
with open("best.json", "r") as json_file:
    RF_best_params = json.load(json_file)

# Use these parameters to initialize the RandomForestRegressor
RF = RandomForestRegressor(**RF_best_params)


# Fit the model and make predictions
RF.fit(X_train, np.ravel(y_train))
RF_predictions = RF.predict(X_test)

# Inverse scaling for predictions
RF_predictions = TargetVarScalerFit.inverse_transform(
    np.resize(RF_predictions, (1466, 1))
)

# Ensure TestingData matches the number of rows in X_test (1466)
TestingData = pd.DataFrame(X_test, columns=X.columns)

# Add the predictions to the DataFrame
TestingData["RF_predictions"] = (
    RF_predictions  # This should match the shape of X_test (1466)
)

print(TestingData.head())  # Check the first few rows to confirm the predictions


# Function to evaluate accuracy
def Accuracy_Score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig - pred) / orig))
    return 100 - MAPE


print(
    "Accuracy for the RF model is:",
    str(Accuracy_Score(TestingData["Giá/m2"], TestingData["RF_predictions"])),
)
