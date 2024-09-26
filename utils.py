from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from scipy import stats


def value_counts_with_percentage(df, column_name):
    """
    This function calculates the count and percentage of each
    unique value in a specified column of a DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the column.
    column_name (str): The name of the column in the DataFrame
    for which the count and percentage need to be calculated.

    Returns:
    pandas.DataFrame: A DataFrame containing the unique values in the specified column as the index,
                      and two columns 'Count' and 'Percentage' representing the count and percentage of each unique value.
                      If the specified column does not exist in the DataFrame, a string message is returned.
    """

    if column_name in df.columns:
        value_counts = df[column_name].value_counts()
        total_count = len(df[column_name])
        percentages = (value_counts / total_count) * 100
        result = pd.DataFrame({"Count": value_counts, "Percentage": percentages})
        return result
    else:
        return f"Column '{column_name}' does not exist in the DataFrame."


def fill_missing_entries_with_nan(df, column, value):
    """
    This function replaces all occurrences of a specified value in a specified column of a DataFrame with NaN.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the column.
    column (str): The name of the column in the DataFrame where the replacement needs to be performed.
    value: The value in the specified column that needs to be replaced with NaN.

    Returns:
    pandas.DataFrame: The modified DataFrame with the specified value replaced with NaN in the specified column.
    """
    df[column] = df[column].replace(value, np.nan)
    return df


def detect_outliers_IQR(df, column):
    """
    Detect outliers in a DataFrame column using IQR based on the quantiles.

    Parameters:
    - df: DataFrame
    - column: The column in which to detect outliers
    """

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
    print(len(outliers))
    df.drop(outliers.index, inplace=True)

def detect_outliers_zscore_using_train_data(df, column, mean_train, std_train, threshold=3):
    """
    Detect outliers in a DataFrame column using Z-scores based on the mean and standard deviation 
    from the training data.

    Parameters:
    - df: DataFrame
    - column: The column in which to detect outliers
    - mean_train: The mean value of the column from the training data
    - std_train: The standard deviation of the column from the training data
    - threshold: Z-score threshold to define outliers (default is 3)
    """
    
    # Calculate Z-scores for the column using the provided mean and std from the training data
    z_scores = np.abs((df[column] - mean_train) / std_train)

    # Detect the outliers (where Z-score is greater than the threshold)
    outliers = df[z_scores > threshold]
    print(f"Number of outliers detected in {column}: {len(outliers)}")
    
    # Drop the outliers
    df.drop(outliers.index, inplace=True)
    
def detect_outliers_zscore(df, column, threshold=3):
    """
    Detect and remove outliers from a DataFrame column using Z-scores.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the column to detect outliers.
    - column (str): The name of the column in the DataFrame to detect outliers.
    - threshold (float, optional): The Z-score threshold to define outliers. Defaults to 3.

    Returns:
    pandas.DataFrame: The modified DataFrame with outliers removed.

    The function calculates the Z-scores for the specified column.
    then identifies the outliers as those rows where the Z-score is greater than the specified threshold.
    The identified outliers are then dropped from the DataFrame using the drop method.
    """
    
    z_scores = np.abs(stats.zscore(df[column]))

    # Detect the outliers (where Z-score is greater than the threshold)
    outliers = df[z_scores > threshold]
    print(len(outliers))
    df.drop(outliers.index, inplace=True)
