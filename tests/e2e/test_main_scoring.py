"""Testing the main scoring module."""
import os
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.main_scoring import (
    get_data_from_db,
    get_model_from_registry,
    generate_predictions,
    upload_results_to_s3,
    insert_results_to_db,
    main
)

class TestScoring(unittest.TestCase):
    @patch('src.main_scoring.trino.dbapi.connect')
    def test_get_data_from_db(self, mock_connect):
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock cursor execution and fetchall
        sample_data = [('Alice', 25), ('Bob', 30)]
        mock_cursor.description = [('name',), ('age',)]
        mock_cursor.fetchall.return_value = sample_data

        query = "SELECT * FROM input_table"
        result = get_data_from_db(query)

        expected_df = pd.DataFrame(sample_data, columns=['name', 'age'])
        pd.testing.assert_frame_equal(result, expected_df)

        # Check that cursor methods are called
        mock_cursor.execute.assert_called_once_with(query)
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch('builtins.open')
    @patch('src.main_scoring.pickle.load')
    def test_get_model_from_registry(self, mock_pickle_load, mock_open):
        # Mock model loading
        mock_model = MagicMock()
        mock_pickle_load.return_value = mock_model

        model_name = 'test_model'
        environment = 'test_env'
        version = '1.0'
        os.environ['MODEL_REGISTRY_PATH'] = '/models'

        result = get_model_from_registry(model_name, environment, version)
        self.assertEqual(result, mock_model)
        mock_open.assert_called_once()
        mock_pickle_load.assert_called_once()

    def test_generate_predictions(self):
        # Mock model and data
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0])
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        result = generate_predictions(mock_model, data)
        np.testing.assert_array_equal(result, np.array([0, 1, 0]))
        mock_model.predict.assert_called_once_with(data)

    @patch('src.main_scoring.boto3.client')
    def test_upload_results_to_s3(self, mock_boto_client):
        # Mock S3 upload
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3

        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'prediction': [0, 1, 0]
        })
        bucket_name = 'test-bucket'
        s3_key = 'test/key.csv'

        upload_results_to_s3(data, bucket_name, s3_key)
        mock_s3.put_object.assert_called_once()
        args, kwargs = mock_s3.put_object.call_args
        self.assertEqual(kwargs['Bucket'], bucket_name)
        self.assertEqual(kwargs['Key'], s3_key)
        self.assertIn('Body', kwargs)

    @patch('src.main_scoring.trino.dbapi.connect')
    def test_insert_results_to_db(self, mock_connect):
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        data = pd.DataFrame({
            'feature1': [1, 2],
            'prediction': [0, 1]
        })
        table_name = 'predictions_table'

        insert_results_to_db(data, table_name)

        # Check that execute is called correctly
        expected_calls = []
        for row in data.to_records(index=False):
            expected_calls.append(
                ((f"INSERT INTO {table_name} (feature1, prediction) VALUES (?, ?)", tuple(row)),)
            )
        self.assertEqual(mock_cursor.execute.call_count, len(data))
        mock_cursor.execute.assert_has_calls(expected_calls, any_order=False)

        # Check that cursor and connection are closed
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch('src.main_scoring.insert_results_to_db')
    @patch('src.main_scoring.upload_results_to_s3')
    @patch('src.main_scoring.generate_predictions')
    @patch('src.main_scoring.get_model_from_registry')
    @patch('src.main_scoring.get_data_from_db')
    def test_main(
        self,
        mock_get_data_from_db,
        mock_get_model_from_registry,
        mock_generate_predictions,
        mock_upload_results_to_s3,
        mock_insert_results_to_db
    ):
        # Mock the entire pipeline
        mock_data = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [3, 4]
        })
        mock_get_data_from_db.return_value = mock_data

        mock_model = MagicMock()
        mock_get_model_from_registry.return_value = mock_model

        mock_preds = np.array([0, 1])
        mock_generate_predictions.return_value = mock_preds

        # Set environment variables
        os.environ['MODEL_NAME'] = 'test_model'
        os.environ['ENVIRONMENT'] = 'test_env'
        os.environ['MODEL_VERSION'] = '1.0'
        os.environ['S3_BUCKET'] = 'test-bucket'
        os.environ['OUTPUT_TABLE'] = 'predictions_table'

        main()

        mock_get_data_from_db.assert_called_once()
        mock_get_model_from_registry.assert_called_once_with('test_model', 'test_env', '1.0')
        mock_generate_predictions.assert_called_once_with(mock_model, mock_data)
        self.assertIn('prediction', mock_data.columns)
        mock_upload_results_to_s3.assert_called_once_with(mock_data, 'test-bucket', 'predictions/output.csv')
        mock_insert_results_to_db.assert_called_once_with(mock_data, 'predictions_table')

if __name__ == '__main__':
    unittest.main()