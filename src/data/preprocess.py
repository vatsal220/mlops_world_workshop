"""Module for preprocessing data."""
import pandas as pd

def normalize_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Normalizes the values in a DataFrame column to be between 0 and 1.

    Args:
        df (pd.DataFrame): DataFrame to normalize.
        column_name (str): Name of the column to normalize.

    Returns:
        pd.DataFrame: DataFrame with the normalized column added.
    """
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    df[column_name + '_normalized'] = (df[column_name] - min_val) / (max_val - min_val)
    return df