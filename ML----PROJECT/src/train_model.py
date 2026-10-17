"""
=============================================================================
HOUSE PRICE PREDICTION - MODEL TRAINING & COMPARISON SCRIPT
=============================================================================
Purpose:
  Trains multiple regression algorithms (Linear Regression, Ridge, Random 
  Forest, Gradient Boosting), compares their evaluation metrics (MAE, MSE, 
  RMSE, R² score), automatically selects the top-performing model, and 
  saves the complete Scikit-Learn pipeline to a .pkl file.

Key Machine Learning Concepts for Beginners:
  1. Pipeline: Combines preprocessor and model into a single executable object.
  2. Model Evaluation:
     - MAE (Mean Absolute Error): Average dollar/rupee prediction error.
     - MSE (Mean Squared Error): Average squared prediction error.
     - RMSE (Root Mean Squared Error): Square root of MSE (penalizes large errors).
     - R² Score: Percentage of price variance explained (1.0 = perfect model).
  3. Model Serialization: `joblib.dump()` saves trained models to disk.
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_processing import get_train_test_data, build_preprocessor

def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluates model predictions on the independent test set.
    
    Parameters:
      pipeline (Pipeline): Fitted Scikit-learn Pipeline object.
      X_test (DataFrame): Test features.
      y_test (Series): Ground truth test prices.
      
    Returns:
      dict: Dictionary containing MAE, MSE, RMSE, and R2_Score.
    """
    # Generate predictions on test features
    y_pred = pipeline.predict(X_test)
    
    # Calculate statistical metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'MAE': round(float(mae), 2),
        'MSE': round(float(mse), 2),
        'RMSE': round(float(rmse), 2),
        'R2_Score': round(float(r2), 4)
    }

def train_and_evaluate_models(data_path):
    """
    Trains multiple regression algorithms and benchmarks their performance.
    
    Parameters:
      data_path (str): Path to dataset CSV.
      
    Returns:
      tuple: (best_fitted_pipeline, metrics_dictionary, best_model_name)
    """
    print("Loading data and splitting into 80% train / 20% test sets...")
    X_train, X_test, y_train, y_test = get_train_test_data(data_path)
    print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")
    
    # Define regression algorithms to compare
    algorithms = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=120, learning_rate=0.1, max_depth=5, random_state=42)
    }
    
    preprocessor = build_preprocessor()
    
    results = {}
    fitted_pipelines = {}
    best_model_name = None
    best_r2 = -float('inf')
    
    print("\nTraining and evaluating regression models:")
    print("=" * 70)
    print(f"{'Algorithm':<20} | {'MAE (INR)':<14} | {'RMSE (INR)':<14} | {'R2 Score':<10}")
    print("-" * 70)
    
    for name, model in algorithms.items():
        # Build unified pipeline: Preprocessor -> Algorithm
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Fit pipeline ONLY on training data (prevents data leakage!)
        pipeline.fit(X_train, y_train)
        
        # Evaluate pipeline on test data
        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics
        fitted_pipelines[name] = pipeline
        
        print(f"{name:<20} | {metrics['MAE']:<12,.2f} | {metrics['RMSE']:<12,.2f} | {metrics['R2_Score']:<10.4f}")
        
        # Compare R2 score to select the best performing model
        if metrics['R2_Score'] > best_r2:
            best_r2 = metrics['R2_Score']
            best_model_name = name

    print("=" * 70)
    print(f"\n[SELECTED BEST MODEL]: {best_model_name} (R2 Accuracy = {best_r2:.4f})")
    
    return fitted_pipelines[best_model_name], results, best_model_name

def save_artifacts(best_pipeline, results, best_model_name, output_dir):
    """
    Saves the best trained pipeline object (.pkl) and evaluation metrics (.json).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "house_price_pipeline.pkl")
    metrics_path = os.path.join(output_dir, "metrics.json")
    
    # 1. Save Scikit-Learn pipeline to disk via Joblib
    joblib.dump(best_pipeline, model_path)
    print(f"Best model pipeline saved to: {model_path}")
    
    # 2. Save evaluation metrics JSON report
    export_metrics = {
        'best_model': best_model_name,
        'model_comparison': results
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(export_metrics, f, indent=4)
        
    print(f"Model metrics report saved to: {metrics_path}")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "house_data.csv")
    output_dir = os.path.join(project_root, "models")
    
    best_pipeline, results, best_name = train_and_evaluate_models(data_path)
    save_artifacts(best_pipeline, results, best_name, output_dir)
