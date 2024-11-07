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
    # try:
    #     df[column_name + '_normalized'] = (df[column_name] - min_val) / (max_val - min_val)
    # except TypeError as e:
    #     raise TypeError(f"Column {column_name} must hold numeric values.") from e
    return df


class Embedder(object):
    def __init__(self, df: pd.DataFrame, text_column: str, model_name: str):
        """Initialize the Embedder class.

        Args:
            df (pd.DataFrame): DataFrame containing text data.
            text_column (str): Name of the column containing text data.
            model (str): Name of the pre-trained embedding model to use.
        """
        self.df = df
        self.text_column = text_column
        self.model_name = model_name

    def preprocess_text(self) -> pd.DataFrame:
        """Preprocesses the text data.

        Returns:
            pd.DataFrame: DataFrame with preprocessed text data.
        """
        return self.df

    def embed_text(self) -> pd.DataFrame:
        """Embeds the text data using the pre-trained model.

        Returns:
            pd.DataFrame: DataFrame with the embedded text data.
        """
        self.preprocess_text()
        mdl = self.load_model(self.model_name)
        # genreate embeddings
        return self.df