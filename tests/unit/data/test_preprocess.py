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
        'featureB': [5, 15.6, 25, 0.35, 45.23],
        'featureC': ['100', '200', '300', '400', '500']
    }
    return pd.DataFrame(data)

def test_normalize_column(sample_df: pd.DataFrame):
    """Test the normalize_column function."""
    # Arrange
    df = sample_df.copy()
    
    # Calculate expected normalized values (Act)
    min_val = df['featureA'].min()
    max_val = df['featureA'].max()
    expected_normalized = (df['featureA'] - min_val) / (max_val - min_val)
    result_df = normalize_column(df, 'featureA')
    
    # Assert
    # Use pandas testing utility to compare Series
    # check_names param refers to the column name
    pd.testing.assert_series_equal(
        result_df['featureA_normalized'], expected_normalized, check_names=False 
    )
    assert 'featureA_normalized' in result_df.columns
    assert all(expected_normalized == result_df['featureA_normalized'].values)
    assert expected_normalized.equals(result_df['featureA_normalized'])


@pytest.mark.parametrize("col, err", [
    ('featureA', False),
    ('featureB', False),
    ('featureC', True)
])
def test_normalize_column_with_error_handling(sample_df: pd.DataFrame, col: str, err: bool):
    """Test the normalize_column function with parameterized columns, including error handling."""
    df = sample_df.copy()
    if err:
        with pytest.raises(TypeError):
            normalize_column(df, col)
    else:
        result_df = normalize_column(df, col)
        
        # Calculate expected normalized values
        min_val = df[col].min()
        max_val = df[col].max()
        expected_normalized = (df[col] - min_val) / (max_val - min_val)

        assert f'{col}_normalized' in result_df.columns
        assert all(expected_normalized == result_df[f'{col}_normalized'].values)
        assert expected_normalized.equals(result_df[f'{col}_normalized'])
