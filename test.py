import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
from utils import fill_missing_entries_with_nan, detect_outliers_zscore_using_train_data
from sklearn.metrics import f1_score
import time

# Load test data 
test_data = pd.read_csv("test.csv")
print("Test file loaded successfully\n")
# 1. Drop duplicates
test_data.drop_duplicates(inplace=True)
print("duplicates dropped successfully\n")

# 2. Fill missing entries using the custom function from utils.py
fill_missing_entries_with_nan(test_data, "native-country", "?")
fill_missing_entries_with_nan(test_data, "occupation", "?")
fill_missing_entries_with_nan(test_data, "workclass", "?")
print("missing values filled successfully\n")

# 3. Detect outliers using Z-score for 'age' and 'hours-per-week'
mean_age_train = joblib.load("mean_age_train.pkl")
std_age_train = joblib.load("std_age_train.pkl")

mean_hours_train = joblib.load("mean_hours_train.pkl")
std_hours_train = joblib.load("std_hours_train.pkl")


detect_outliers_zscore_using_train_data(test_data, "age", mean_age_train, std_age_train)
detect_outliers_zscore_using_train_data(
    test_data, "hours-per-week", mean_hours_train, std_hours_train
)
print("outliers detected by zscore method and removed successfully\n")

# 4. Encode categorical columns using LabelEncoder
label_encoder = joblib.load("label_encoder.pkl")
test_encoded = test_data.apply(
    lambda col: label_encoder.fit_transform(col) if col.dtype == "object" else col
)

# 5. Load the Min-Max scaler that was used on the training data
scaler = joblib.load("scaler.pkl")

# 6. Apply Min-Max scaling on the encoded test data
test_scaled = pd.DataFrame(scaler.transform(test_encoded), columns=test_encoded.columns)

# (Optional) Load the trained model and make predictions
xgb_model = joblib.load("xgb_model.pkl")
X = test_scaled.drop("income", axis=1)
y = test_scaled["income"]
print("Waiting for predictions...\n")
time.sleep(2)  # Pauses the program for 2 seconds
y_test_pred = xgb_model.predict(X)
print("predictions done successfully!\n")

# If you have the true labels for the test data:
f1 = f1_score(y, y_test_pred)
print(f"F1 Score: {f1}")
