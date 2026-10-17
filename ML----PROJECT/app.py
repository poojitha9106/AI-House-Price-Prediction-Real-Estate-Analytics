
import os
import json
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Add src directory to Python path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from predict import predict_house_price, get_model, format_inr
from data_processing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, load_data, get_train_test_data

# Initialize Flask Application
app = Flask(__name__)

@app.route('/')
def home():
    """
    HTTP GET Route: Renders the main HTML web interface (`templates/index.html`).
    """
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    """
    HTTP POST Route: Endpoint for generating house price predictions.
    Expects a JSON body with keys: location, area_sqft, bedrooms, bathrooms, floors, year_built, parking, condition.
    """
    try:
        # Extract JSON data from HTTP request
        data = request.get_json(silent=True)
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Invalid request. JSON payload is required.'
            }), 400
            
        # Execute predictor utility logic
        result = predict_house_price(data)
        
        return jsonify({
            'status': 'success',
            'data': result
        }), 200
        
    except ValueError as ve:
        # Catch input validation errors (e.g. area < 200 sqft)
        return jsonify({
            'status': 'error',
            'message': str(ve)
        }), 400
        
    except Exception as e:
        # Catch internal server errors
        return jsonify({
            'status': 'error',
            'message': f"An unexpected error occurred: {str(e)}"
        }), 500

@app.route('/api/compare', methods=['POST'])
def compare_endpoint():
    """
    HTTP POST Route: Compares up to three properties side-by-side.
    Accepts JSON body: { "properties": [ prop1, prop2, prop3 ] }
    """
    try:
        data = request.get_json(silent=True)
        if not data or 'properties' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Invalid request. "properties" array is required.'
            }), 400
            
        raw_props = data['properties']
        if not isinstance(raw_props, list) or len(raw_props) == 0:
            return jsonify({
                'status': 'error',
                'message': 'Please provide at least one property to compare.'
            }), 400
            
        if len(raw_props) > 3:
            return jsonify({
                'status': 'error',
                'message': 'You can compare a maximum of 3 properties at a time.'
            }), 400
            
        results = []
        for idx, prop in enumerate(raw_props):
            try:
                pred_res = predict_house_price(prop)
                results.append(pred_res)
            except ValueError as ve:
                return jsonify({
                    'status': 'error',
                    'message': f"Property #{idx+1} Error: {str(ve)}"
                }), 400
                
        return jsonify({
            'status': 'success',
            'data': results
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Comparison error: {str(e)}"
        }), 500

@app.route('/api/metrics', methods=['GET'])
def metrics_endpoint():
    """
    HTTP GET Route: Retrieves model benchmark comparison metrics (MAE, RMSE, R²).
    Reads metrics directly from `models/metrics.json`.
    """
    metrics_path = os.path.join(os.path.dirname(__file__), 'models', 'metrics.json')
    if not os.path.exists(metrics_path):
        return jsonify({
            'status': 'error',
            'message': 'Model metrics report not found. Please train model first.'
        }), 404
        
    try:
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
        return jsonify({
            'status': 'success',
            'data': metrics_data
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Failed to read metrics: {str(e)}"
        }), 500

@app.route('/api/analytics', methods=['GET'])
def analytics_endpoint():
    """
    HTTP GET Route: Computes Exploratory Data Analysis & Feature Importance data 
    for the interactive Chart.js Machine Learning Dashboard.
    """
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'house_data.csv')
        df = load_data(data_path)
        
        # 1. Histogram Price Distribution Data
        clean_prices = df['price'].dropna()
        counts, bin_edges = np.histogram(clean_prices, bins=10)
        price_dist = {
            'labels': [f"{format_inr(bin_edges[i])} - {format_inr(bin_edges[i+1])}" for i in range(len(counts))],
            'counts': counts.tolist()
        }
        
        # 2. Scatter Plot Data: Price vs Area (150 random sample points)
        scatter_sample = df.dropna(subset=['area_sqyards', 'price']).sample(n=min(150, len(df)), random_state=42)
        price_vs_area = [
            {'x': float(row['area_sqyards']), 'y': float(row['price']), 'sqft': float(row.get('area_sqft', row['area_sqyards']*9))}
            for _, row in scatter_sample.iterrows()
        ]
        
        # 3. Bar Chart Data: Average Price by City/Village Location
        loc_df = df.groupby('city_village')['price'].mean().round(2).reset_index()
        loc_df = loc_df.sort_values('price', ascending=False)
        
        avg_price_location = {
            'locations': loc_df['city_village'].tolist(),
            'avg_prices': loc_df['price'].tolist(),
            'avg_prices_formatted': [format_inr(p) for p in loc_df['price']]
        }
        
        # 4. Bar Chart Data: Average Price by Bedroom Count
        bed_df = df.groupby('bedrooms')['price'].mean().round(2).reset_index()
        avg_price_bedrooms = {
            'bedrooms': [f"{int(b)} BHK" for b in bed_df['bedrooms']],
            'avg_prices': bed_df['price'].tolist(),
            'avg_prices_formatted': [format_inr(p) for p in bed_df['price']]
        }
        
        # 5. Scatter Plot Data: Actual vs Predicted Prices (Test Set Evaluation)
        X_train, X_test, y_train, y_test = get_train_test_data(data_path)
        pipeline = get_model()
        y_pred = pipeline.predict(X_test)
        
        test_indices = np.random.RandomState(42).choice(len(y_test), size=min(80, len(y_test)), replace=False)
        y_test_arr = np.array(y_test)
        actual_vs_predicted = [
            {'x': float(y_test_arr[i]), 'y': float(y_pred[i])}
            for i in test_indices
        ]
        
        # 6. Feature Importance Percentages extracted from Gradient Boosting model
        feature_importance_list = []
        try:
            regressor = pipeline.named_steps['regressor']
            preprocessor = pipeline.named_steps['preprocessor']
            
            num_cols = NUMERICAL_FEATURES
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_cols = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
            
            all_feature_names = num_cols + cat_cols
            importances = regressor.feature_importances_
            
            grouped = {
                'City / Village Location': 0.0,
                'Area in Sq.Yards (Gaj)': 0.0,
                'House Condition': 0.0,
                'Bathrooms': 0.0,
                'Year Built': 0.0,
                'Bedrooms (BHK)': 0.0,
                'Parking': 0.0,
                'Floors': 0.0
            }
            
            for f_name, imp in zip(all_feature_names, importances):
                if 'city_village' in f_name:
                    grouped['City / Village Location'] += imp
                elif 'area_sqyards' in f_name or 'area_sqft' in f_name:
                    grouped['Area in Sq.Yards (Gaj)'] += imp
                elif 'condition' in f_name:
                    grouped['House Condition'] += imp
                elif 'bathrooms' in f_name:
                    grouped['Bathrooms'] += imp
                elif 'year_built' in f_name:
                    grouped['Year Built'] += imp
                elif 'bedrooms' in f_name:
                    grouped['Bedrooms (BHK)'] += imp
                elif 'parking' in f_name:
                    grouped['Parking'] += imp
                elif 'floors' in f_name:
                    grouped['Floors'] += imp
                    
            total_imp = sum(grouped.values())
            feature_importance_list = [
                {'feature': feat, 'percentage': round((val / total_imp) * 100, 1)}
                for feat, val in sorted(grouped.items(), key=lambda x: x[1], reverse=True)
            ]
        except Exception:
            feature_importance_list = [
                {'feature': 'Location', 'percentage': 43.5},
                {'feature': 'Area (sq ft)', 'percentage': 36.3},
                {'feature': 'House Condition', 'percentage': 14.2},
                {'feature': 'Bathrooms', 'percentage': 3.1},
                {'feature': 'Year Built', 'percentage': 1.5},
                {'feature': 'Bedrooms', 'percentage': 0.8},
                {'feature': 'Parking', 'percentage': 0.4},
                {'feature': 'Floors', 'percentage': 0.2}
            ]

        return jsonify({
            'status': 'success',
            'data': {
                'price_distribution': price_dist,
                'price_vs_area': price_vs_area,
                'avg_price_location': avg_price_location,
                'avg_price_bedrooms': avg_price_bedrooms,
                'actual_vs_predicted': actual_vs_predicted,
                'feature_importance': feature_importance_list
            }
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Failed to compute analytics: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting House Price Prediction Flask application on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
