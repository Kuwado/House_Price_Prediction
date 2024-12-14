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
# Create a dictionary of random parameters for the model with reduced ranges
RF_random_grid = {
    "n_estimators": [100, 200, 300],  # Reduce the number of trees
    "max_features": ["auto", "sqrt"],  # Reduce choices for max_features
    "max_depth": [10, 30, 50],  # Limit the depth of trees
    "min_samples_split": [2, 5],  # Limit to fewer splits
    "min_samples_leaf": [1, 2],  # Limit to fewer leaf sizes
    "bootstrap": [True, False],
}

# Create the base RF model and fit the random search
RF_regressor = RandomForestRegressor()

# Add progress bar to track the RandomizedSearchCV fitting process
print("Starting RandomizedSearchCV...")

RF_random_search = RandomizedSearchCV(
    estimator=RF_regressor,
    param_distributions=RF_random_grid,
    n_iter=10,  # Reduce number of iterations for faster search
    cv=3,  # Use 3-fold cross-validation to speed up the process
    verbose=2,  # Verbose set to 2 for more detailed information
    random_state=2022,
    n_jobs=-1,
)

# Use tqdm to show progress during fitting
for _ in tqdm(range(10), desc="Fitting RandomizedSearchCV"):  # Fewer iterations
    RF_random_search.fit(X_train, np.ravel(y_train))

RF_best_params = RF_random_search.best_params_
print(f"Best Random Search Parameters: {RF_best_params}")

# Narrowing the parameters grid based on the best parameters given by the random search, then feed the grid to a grid search
RF_param_grid = {
    "n_estimators": [
        RF_best_params["n_estimators"] - 50,
        RF_best_params["n_estimators"],
    ],
    "max_features": ["sqrt"],
    "max_depth": [RF_best_params["max_depth"] - 10, RF_best_params["max_depth"]],
    "min_samples_split": [5],
    "min_samples_leaf": [1, 2],
    "bootstrap": [True, False],
}

# Create another base RF model and fit the grid search
RF_regressor_2 = RandomForestRegressor()

# Add progress bar to track the GridSearchCV fitting process
print("Starting GridSearchCV...")

RF_grid_search = GridSearchCV(
    estimator=RF_regressor_2,
    param_grid=RF_param_grid,
    cv=3,
    n_jobs=-1,
    verbose=4,  # 3-fold CV
)

# Use tqdm to show progress during fitting
for _ in tqdm(range(10), desc="Fitting GridSearchCV"):  # Fewer iterations
    RF_grid_search.fit(X_train, np.ravel(y_train))

# Showing the best parameters
print(f"Best Grid Search Parameters: {RF_grid_search.best_params_}")

# Fitting a RF model with the best parameters
RF = RF_grid_search.best_estimator_

# Generating Predictions on testing data
RF_predictions = RF.predict(X_test)

# Scaling the predicted Price data back to original price scale
RF_predictions = TargetVarScalerFit.inverse_transform(
    np.resize(RF_predictions, (1466, 1))
)

# Assuming TestingData is the test set you're working with
# TestingData = pd.DataFrame(X_test)  # Make sure you have the right DataFrame here
# TestingData["RF_predictions"] = RF_predictions
# print(TestingData.head())

TestingData = pd.DataFrame(
    X_test, columns=X.columns
)  # Ensure it matches the X_test shape
TestingData["RF_predictions"] = RF_predictions  # Add the predictions

print(TestingData.head())


# Define a function evaluate the predictions
def Accuracy_Score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig - pred) / orig))
    return 100 - MAPE


print(
    "Accuracy for the RF model is:",
    str(Accuracy_Score(TestingData["Giá/m2"], TestingData["RF_predictions"])),
)
