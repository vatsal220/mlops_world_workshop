"""Testing the preprocess module in the data directory."""
import pytest
import pandas as pd
from src.data.preprocess import normalize_column

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Provides a sample DataFrame for testing.
    """
    data = {
        'featureA': [10, 20, 30, 40, 50],
        'featureB': [5, 15, 25, 35, 45],
        'featureC': ['100', '200', '300', '400', '500']
    }
    return pd.DataFrame(data)
