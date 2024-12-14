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

# Turn off warnings
warnings.filterwarnings("ignore")

# Read the dataset
data = pd.read_csv("../../data/cleaned_data.csv")

# Convert non-numeric columns to dummy variables
dummy_type_of_housing = pd.get_dummies(data["Loại hình nhà ở"], prefix="housing_type")
dummy_legal_paper = pd.get_dummies(data["Giấy tờ pháp lý"], prefix="legal_paper")
dummy_district = pd.get_dummies(data["Quận"], prefix="district")
dummy_ward = pd.get_dummies(data["Phường"], prefix="ward")

# Combine the dummy variables into the cleaned data
data_cleaned = pd.concat(
    [data, dummy_type_of_housing, dummy_legal_paper, dummy_district, dummy_ward], axis=1
)
data_cleaned = data_cleaned.drop(
    ["Địa chỉ", "Quận", "Phường", "Loại hình nhà ở", "Giấy tờ pháp lý"], axis=1
)
print(data_cleaned.head())

# Separate predictors and target (price) variables
X = data_cleaned.loc[:, data_cleaned.columns != "Giá/m2"]
y = data_cleaned[["Giá/m2"]]

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
# Polynomial Regression (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)

# Show progress while training the new model
new_model = LinearRegression()
new_model.fit(X_train_poly, y_train)

r_sq_poly = new_model.score(X_train_poly, y_train)
print(f"Coefficient of determination for Polynomial Regression: {r_sq_poly}")

# Transform the test set and show progress
X_test_poly = poly.transform(X_test)
y_pred_poly = new_model.predict(X_test_poly)
print(
    "Predictions from the Polynomial Regression model:\n", y_pred_poly[:5]
)  # Show the first 5 predictions


# --------------------------
# Define function to evaluate predictions using Accuracy Score (MAPE)
def Accuracy_Score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig - pred) / orig))
    return 100 - MAPE


# Evaluate accuracy for the Linear Regression model
accuracy_lr = Accuracy_Score(y_test, y_pred)
print(f"Accuracy for the LR model is: {accuracy_lr}%")

# Evaluate accuracy for the Polynomial Regression model
accuracy_poly = Accuracy_Score(y_test, y_pred_poly)
print(f"Accuracy for the Polynomial Regression model is: {accuracy_poly}%")
