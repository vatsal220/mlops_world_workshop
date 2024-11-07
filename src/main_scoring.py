# -*- coding: utf-8 -*-
"""Main - Scoring
Driver script for scoring a classification model.
"""
import os
import boto3
import pickle
import trino
import pandas as pd

def get_database_connection():
    """
    Establishes a database connection using credentials from environment variables.
    """
    trino_host = os.getenv('TRINO_HOST', 'localhost')
    trino_port = int(os.getenv('TRINO_PORT', 8080))
    trino_user = os.getenv('TRINO_USER', 'user')
    trino_catalog = os.getenv('TRINO_CATALOG', 'hive')
    trino_schema = os.getenv('TRINO_SCHEMA', 'default')

    conn = trino.dbapi.connect(
        host=trino_host,
        port=trino_port,
        user=trino_user,
        catalog=trino_catalog,
        schema=trino_schema,
    )
    return conn

def get_data_from_db(query: str) -> pd.DataFrame:
    """
    Retrieves data from the database based on the provided query.
    """
    conn = get_database_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    # Fetch column names
    columns = [desc[0] for desc in cursor.description]
    # Fetch all data
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)
    cursor.close()
    conn.close()
    return df

def get_model_from_registry(model_name: str, environment: str, version: str) -> object:
    """
    Retrieves the model from a model registry.
    For simplicity, we simulate the registry with a local file system.
    """
    # Simulate model registry path
    model_registry_path = os.getenv('MODEL_REGISTRY_PATH', './models')
    model_path = f"{model_registry_path}/{environment}/{model_name}/{version}/model.pkl"
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def generate_predictions(model: object, data: pd.DataFrame) -> pd.Series:
    """
    Generates predictions using the provided model and data.
    """
    predictions = model.predict(data)
    return predictions

def upload_results_to_s3(df: pd.DataFrame, bucket_name: str, s3_key: str) -> None:
    """
    Uploads the DataFrame as a CSV to an S3 bucket.
    """
    csv_buffer = df.to_csv(index=False)
    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_buffer)

def insert_results_to_db(df: pd.DataFrame, table_name: str) -> None:
    """
    Inserts the results into a database table.
    """
    conn = get_database_connection()
    cursor = conn.cursor()

    # Generate insert query
    columns = ', '.join(df.columns)
    placeholders = ', '.join(['?'] * len(df.columns))
    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

    # Prepare data for insertion
    data_tuples = [tuple(x) for x in df.to_numpy()]
    
    # Execute insert statements
    for row in data_tuples:
        cursor.execute(insert_query, row)
    cursor.close()
    conn.close()

def main():
    # Step 1: Get data from database
    query = "SELECT * FROM input_table"
    data = get_data_from_db(query)
    
    # Step 2: Get model from registry
    model_name = os.getenv('MODEL_NAME')
    environment = os.getenv('ENVIRONMENT')
    version = os.getenv('MODEL_VERSION')
    model = get_model_from_registry(model_name, environment, version)
    
    # Step 3: Generate predictions
    predictions = generate_predictions(model, data)
    data['prediction'] = predictions
    
    # Step 4: Upload results
    # Upload to S3
    s3_bucket = os.getenv('S3_BUCKET')
    s3_key = os.getenv('S3_KEY', 'predictions/output.csv')
    upload_results_to_s3(data, s3_bucket, s3_key)
    
    # Insert into database
    output_table = os.getenv('OUTPUT_TABLE', 'predictions_table')
    insert_results_to_db(data, output_table)

if __name__ == "__main__":
    main()