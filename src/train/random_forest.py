import warnings
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib
from sklearn.metrics import mean_squared_error, r2_score


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

# ----------------------------------------------
# model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10)
model = RandomForestRegressor(
    bootstrap=True,
    max_depth=80,
    max_features="sqrt",
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=600,
    random_state=42,
)
model.fit(X_train, y_train)
r_sq = model.score(X_train, y_train)
print(f"Coefficient of determination for Linear Regression: {r_sq}")
y_pred = model.predict(X_test)
print(
    "Predictions from the Linear Regression model:\n", y_pred[:5]
)  # Show the first 5 predictions

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Coefficient of Determination (R^2): {r2:.2f}")


def Accuracy_Score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig - pred) / orig))
    return 100 - MAPE


y_test_orig = TargetVarScalerFit.inverse_transform(y_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Scaling the test data back to original scale
Test_Data = np.concatenate(
    (PredictorScalerFit.inverse_transform(X_test[:, :3]), X_test[:, 3:]), axis=1
)
TestingData = pd.DataFrame(data=Test_Data, columns=X.columns)
TestingData["Giá/m2"] = y_test_orig


LR_predictions = model.predict(X_test)  # Dự đoán từ mô hình
LR_predictions = LR_predictions.reshape(-1, 1)  # Chuyển sang 2D
LR_predictions = TargetVarScalerFit.inverse_transform(
    LR_predictions
)  # Chuyển về giá trị gốc
TestingData["LR_predictions"] = LR_predictions
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(LR_predictions, y_test))

print(
    "Accuracy for the LR model is:",
    str(Accuracy_Score(TestingData["Giá/m2"], TestingData["LR_predictions"])),
)

print(TestingData["Giá/m2"].head())
print(TestingData["LR_predictions"].head())

# Lưu mô hình RandomForestRegressor vào file
# joblib.dump(model, "../models/random_forest_model.pkl")

# Create a figure and axis object
plt.figure(figsize=(10, 6))

# Plot the actual and predicted house prices as line plots
plt.plot(
    TestingData.index,
    TestingData["Giá/m2"],
    label="Actual Prices",
    color="blue",
    linestyle="-",
    marker="o",
)
plt.plot(
    TestingData.index,
    TestingData["LR_predictions"],
    label="Predicted Prices",
    color="red",
    linestyle="--",
    marker="x",
)

# Add labels and a title
plt.xlabel("House Index")
plt.ylabel("Price per Square Meter (Giá/m2)")
plt.title("Comparison of Actual and Predicted House Prices")

# Show a legend
plt.legend()

# Display the plot
plt.show()
