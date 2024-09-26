# Task Overview
This project involves training and testing a machine learning model using the Adult Income dataset (adult.csv). The model has been trained using XGBoost, and the necessary preprocessing steps are stored in pickle files (.pkl). These files are essential to run the test.py script for making predictions on new data.
## Files Included:

- `adult.csv`: The dataset used for training the model.
- `Draft.ipynb`: The Jupyter notebook containing the preprocessing, training, and evaluation steps.
- `label_encoder.pkl`: Label encoder for categorical variables used during preprocessing.
- `mean_age_train.pkl`: Mean of the 'age' column used during training, needed for consistent preprocessing in test data.
- `mean_hours_train.pkl`: Mean of the 'hours-per-week' column used during training.
- `scaler.pkl`: The scaler object (likely Min-Max or Standard Scaler) used to scale the features during training.
- `std_age_train.pkl`: Standard deviation of the 'age' column used during training.
- `std_hours_train.pkl`: Standard deviation of the 'hours-per-week' column used during training.
- `xgb_model.pkl`: The trained XGBoost model used for prediction.
- `test.csv`: Test dataset for evaluating the model’s performance.
- `test.py`: Script for loading the model and preprocessing objects to run predictions on the test data.
- `utils.py`: Utility functions used for preprocessing, such as filling missing values and detecting outliers.

## Instructions:

1. **Install Dependencies**:
   To run this project, you must have Python and the necessary libraries installed. You can install the required packages by running:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure that **XGBoost** is installed, as the trained model relies on this library:

   ```bash
   pip install xgboost
   ```

2. **Run the Test Script**:
   To test the model on the test dataset (`test.csv`), simply run the `test.py` script:

   ```bash
   python test.py
   ```

This will load the necessary `.pkl` files (such as the model, scalers, and encoders) and make predictions on the test data.

## Summary of Preprocessing Steps in Machine Learning

In my machine learning project, I undertook several critical preprocessing steps to prepare our dataset for training. 

### Preprocessing Steps Taken

1. **Dropping Duplicated Entries**: 
   I first removed duplicate records from the dataset to avoid redundant information that could skew the model's learning process. This step was essential to ensure data integrity and prevent the model from learning repeated patterns that do not provide additional value.
2. **Handling Missing Values**: 
   For columns such as `native-country`, `occupation`, and `workclass`, I replaced missing values with `NaN` using a custom function from `utils.py`. Handling missing data is crucial because leaving gaps in the dataset can result in biased or inaccurate predictions.

3. **Detecting Outliers**:
   I used Z-score to detect outliers in the `age` and `hours-per-week` features. Outliers can have a significant impact on the model's performance by distorting the learning process, so they were removed.
4. **Encoding Categorical Columns**: 
   I applied `LabelEncoder` to convert categorical variables into numerical values. This step was important because machine learning algorithms typically require numerical inputs.

5. **Feature Scaling**: 
   We applied feature scaling to standardize the range of independent variables. Techniques like Min-Max Scaling were used to ensure that all features contributed equally to the model's training process, preventing any one feature from disproportionately influencing the outcome.

### Impact of Changing Preprocessing Steps (Skipping Feature Scaling)

In another model preoricessing stesp, I chose not to apply Min-Max scaling to the features. As a result, the model's F1 score dropped from 0.72 to 0.47. The significant reduction in F1 score demonstrated how unscaled features negatively impacted the model's ability to generalize.
### Conclusions on Preprocessing Impact

Preprocessing plays a vital role in the model's ability to learn and generalize from the data. Each preprocessing step, whether it’s handling missing data, encoding categorical features, or scaling, contributes to ensuring that the model can interpret the data properly. In my case, omitting the feature scaling step resulted in a notable drop in performance, which highlights how crucial it is for all features to be treated uniformly, especially when they have varying scales. Proper preprocessing ensures that the model can learn patterns effectively and perform well on unseen data.