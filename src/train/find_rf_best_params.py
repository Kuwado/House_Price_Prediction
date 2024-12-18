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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


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
# Create a dictionary of random parameters for the model
RF_random_grid = {
    "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [int(x) for x in np.linspace(10, 100, num=10)],
    "min_samples_split": [2, 3, 5],
    "min_samples_leaf": [1, 2, 3],
    "bootstrap": [True, False],
}

print("Starting RandomizedSearchCV...")

# Create the base RF model and fit the random search
RF_regressor = RandomForestRegressor()
RF_random_search = RandomizedSearchCV(
    estimator=RF_regressor,
    param_distributions=RF_random_grid,
    n_iter=20,
    cv=2,
    verbose=1,
    random_state=2022,
    n_jobs=-1,
).fit(X_train, np.ravel(y_train))

RF_best_params = RF_random_search.best_params_
print(f"Best Random Search Parameters: {RF_best_params}")

# ----

# Narrowing the parameters grid based on the best parameters given by the random search, then feed the grid to a grid search
RF_param_grid = {
    "n_estimators": [
        RF_best_params["n_estimators"] - 100,
        RF_best_params["n_estimators"],
    ],
    "max_features": ["sqrt", "log2"],
    "max_depth": [RF_best_params["max_depth"] - 10, RF_best_params["max_depth"]],
    "min_samples_split": [2, 3, 5],
    "min_samples_leaf": [1, 2, 3],
    "bootstrap": [True, False],
}

RF_regressor_2 = RandomForestRegressor()


RF_grid_search = GridSearchCV(
    estimator=RF_regressor_2, param_grid=RF_param_grid, cv=2, n_jobs=-1, verbose=4
).fit(X_train, np.ravel(y_train))

# Showing the best parameters
print(f"Best Grid Search Parameters: {RF_grid_search.best_params_}")
