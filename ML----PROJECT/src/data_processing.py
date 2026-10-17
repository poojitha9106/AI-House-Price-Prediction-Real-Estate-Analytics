"""
=============================================================================
HOUSE PRICE PREDICTION - DATA PROCESSING & PREPROCESSING MODULE
=============================================================================
Purpose:
  Handles data ingestion, data cleaning, duplicate removal, train/test 
  splitting, and constructs a Scikit-Learn `ColumnTransformer` feature 
  preprocessing pipeline for Indian real estate data.
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define global column lists for Indian real estate dataset
NUMERICAL_FEATURES = ['area_sqyards', 'area_sqft', 'bedrooms', 'bathrooms', 'floors', 'year_built']
CATEGORICAL_FEATURES = ['city_village', 'parking', 'condition']
TARGET_COLUMN = 'price'

def load_data(csv_path):
    """
    Loads dataset from CSV file, checks for duplicates, and validates columns.
    """
    df = pd.read_csv(csv_path)
    
    # 1. Remove duplicate rows if any exist
    initial_len = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_len:
        print(f"[DATA CLEANING]: Removed {initial_len - len(df)} duplicate rows.")
        
    # 2. Validate target column existence
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset CSV.")
        
    return df

def build_preprocessor():
    """
    Constructs a Scikit-learn ColumnTransformer pipeline for feature preprocessing.
    """
    # 1. Numerical Pipeline: Median Imputation -> Standard Scaling
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical Pipeline: Most-Frequent Imputation -> One-Hot Encoding
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Combine both pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUMERICAL_FEATURES),
            ('cat', cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    
    return preprocessor

def get_train_test_data(csv_path, test_size=0.2, random_state=42):
    """
    Loads data and splits into X_train, X_test, y_train, y_test.
    """
    df = load_data(csv_path)
    
    # Separate input features (X) from target price (y)
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]
    
    # Perform 80% train / 20% test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    import os
    sample_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'house_data.csv')
    X_tr, X_te, y_tr, y_te = get_train_test_data(sample_path)
    print(f"Data Processing Check: X_train shape = {X_tr.shape}, X_test shape = {X_te.shape}")
