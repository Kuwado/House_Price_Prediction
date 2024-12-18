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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor



# Load and preprocess your dataset
data = pd.read_csv("/kaggle/input/cleaned-data/cleaned_data.csv")

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

# Turn off TensorFlow messages and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_SETTINGS"] = "false"

# Create the base model
def create_regression_ANN(optimizer_trial):
    model = Sequential()
    model.add(Dense(units=10, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=optimizer_trial)
    return model

# Creathe a dictionary for trial parameters
ANN_params = {'batch_size':[10, 20, 30, 50],
             'epochs':[10, 20, 50],
             'optimizer_trial':['adam', 'rmsprop']}

ANN_trial = KerasRegressor(create_regression_ANN, verbose=0)

# Initiate the grid search and storing best parameters for later reference
ANN_grid_search = GridSearchCV(estimator=ANN_trial, param_grid=ANN_params, 
                               cv=3, n_jobs = -1).fit(X_train, y_train, verbose=0)
ANN_best_params = ANN_grid_search.best_params_

# Showing the best parameters
ANN_best_params


# Fitting the ANN to the Training set
ANN = Sequential()                
ANN.add(Dense(units=10, input_dim=X_train.shape[1], 
                kernel_initializer='normal', activation='relu'))
ANN.add(Dense(1, kernel_initializer='normal'))
ANN.compile(loss='mean_squared_error', optimizer=ANN_best_params['optimizer_trial'])
ANN.fit(X_train, y_train,batch_size = int(ANN_best_params['batch_size']),
        epochs = int(ANN_best_params['epochs']), verbose=0)

# Generating Predictions on testing data
ANN_predictions = ANN.predict(X_test)
 
# Scaling the predicted Price data back to original price scale
ANN_predictions = TargetVarScalerFit.inverse_transform(ANN_predictions)
 
# Scaling the y_test Price data back to original price scale
y_test_orig = TargetVarScalerFit.inverse_transform(y_test)
 
# Scaling the test data back to the original scale for the columns that were scaled
X_test_scaled = X_test[:, :len(to_be_scaled)]  # Extract the scaled columns
X_test_non_scaled = X_test[:, len(to_be_scaled):]  # Extract the non-scaled columns

# Apply the inverse transform only to the scaled columns
X_test_scaled_back = PredictorScalerFit.inverse_transform(X_test_scaled)

# Concatenate the scaled and non-scaled columns
Test_Data = np.concatenate((X_test_scaled_back, X_test_non_scaled), axis=1)

# Recreating the dataset, now with predicted price using the ANN model
TestingData = pd.DataFrame(data=Test_Data, columns=X.columns)
TestingData['Price'] = y_test_orig
TestingData['ANN_predictions'] = ANN_predictions

# Display the first few rows of the final dataframe
TestingData[['Price', 'ANN_predictions']].head()

# Define a function evaluate the predictions
def Accuracy_Score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig - pred) / orig))
    return(100-MAPE)


# Showing scores for both the ANN and the RF model
print("Accuracy for the ANN model is:", str(Accuracy_Score(TestingData['Price'], TestingData['ANN_predictions'])))