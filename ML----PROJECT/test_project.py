"""
=============================================================================
HOUSE PRICE PREDICTION - AUTOMATED TEST & VERIFICATION SUITE
=============================================================================
Purpose:
  Systematically verifies all files, imports, preprocessing, model training,
  serialization, Flask backend endpoints, input validation, dynamic inference,
  and dashboard analytics API endpoints.
=============================================================================
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import urllib.request

def run_tests():
    print("=" * 70)
    print("HOUSE PRICE PREDICTION PROJECT - SYSTEMATIC TEST SUITE")
    print("=" * 70)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(project_root, 'src'))
    
    # TEST 1: Check File Existence
    print("\n[TEST 1/12] Verifying Project File Structure...")
    required_files = [
        "requirements.txt",
        "README.md",
        "PROJECT_REPORT.md",
        "app.py",
        "data/generate_dataset.py",
        "data/house_data.csv",
        "models/house_price_pipeline.pkl",
        "models/metrics.json",
        "src/__init__.py",
        "src/data_processing.py",
        "src/train_model.py",
        "src/predict.py",
        "templates/index.html",
        "static/css/style.css",
        "static/js/main.js"
    ]
    
    missing = []
    for rel_path in required_files:
        full_path = os.path.join(project_root, rel_path)
        if not os.path.exists(full_path):
            missing.append(rel_path)
            
    if missing:
        print(f"FAILED: Missing files: {missing}")
        sys.exit(1)
    print("PASSED: All 15 required project files exist.")
    
    # TEST 2: Check Core Package Imports
    print("\n[TEST 2/12] Verifying Package Imports...")
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import joblib
        import flask
        print(f"PASSED: Imports functional (pandas {pd.__version__}, scikit-learn {sklearn.__version__}).")
    except Exception as e:
        print(f"FAILED: Import error: {e}")
        sys.exit(1)
        
    # TEST 3: Check Requirements.txt
    print("\n[TEST 3/12] Verifying requirements.txt...")
    req_path = os.path.join(project_root, "requirements.txt")
    with open(req_path, 'r') as f:
        reqs = f.read()
    assert "pandas" in reqs and "numpy" in reqs and "scikit-learn" in reqs and "flask" in reqs
    print("PASSED: requirements.txt includes all required core dependencies.")

    # TEST 4: Check Dataset Loading
    print("\n[TEST 4/12] Verifying Dataset Loading...")
    from data_processing import load_data
    df = load_data(os.path.join(project_root, "data", "house_data.csv"))
    assert len(df) >= 1000, "Dataset should have at least 1000 rows"
    assert "price" in df.columns, "Dataset missing price target"
    print(f"PASSED: Dataset loaded successfully ({len(df)} rows, {len(df.columns)} columns).")

    # TEST 5: Check Data Preprocessing
    print("\n[TEST 5/12] Verifying Data Preprocessing Pipeline...")
    from data_processing import get_train_test_data, build_preprocessor
    X_train, X_test, y_train, y_test = get_train_test_data(os.path.join(project_root, "data", "house_data.csv"))
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    assert X_train_trans.shape[0] == len(X_train)
    print(f"PASSED: Preprocessing pipeline built and transformed data without data leakage (Transformed shape: {X_train_trans.shape}).")

    # TEST 6 & 7: Check Model Serialization & Inference
    print("\n[TEST 6 & 7/12] Verifying Model Pipeline Serialization & Joblib Loader...")
    model_path = os.path.join(project_root, "models", "house_price_pipeline.pkl")
    pipeline = joblib.load(model_path)
    test_sample = pd.DataFrame([{
        'city_village': 'Mumbai - Bandra West',
        'area_sqyards': 250,
        'area_sqft': 2250,
        'bedrooms': 3,
        'bathrooms': 3.0,
        'floors': 2,
        'year_built': 2022,
        'parking': 'Covered Car Park',
        'condition': 'Fully Furnished / Renovated'
    }])
    pred_val = pipeline.predict(test_sample)[0]
    assert pred_val > 100000, "Predicted value should be realistic"
    print(f"PASSED: Model pipeline loaded cleanly from joblib. Test prediction output = INR {pred_val:,.2f}")

    # TEST 8: Check Frontend Homepage Route
    print("\n[TEST 8/12] Verifying Frontend Homepage Endpoint (GET /)...")
    resp = urllib.request.urlopen("http://127.0.0.1:5000/")
    assert resp.getcode() == 200
    html_content = resp.read().decode('utf-8')
    assert "House Price Prediction" in html_content
    print("PASSED: Frontend homepage returned HTTP 200 OK with title 'House Price Prediction'.")

    # TEST 9: Check Prediction API Route & Non-Hardcoded Dynamic Prediction
    print("\n[TEST 9/12] Verifying Prediction REST API Endpoint (POST /api/predict)...")
    payload1 = {'city_village':'Palampur Village (HP)','area_sqyards':80,'bedrooms':1,'bathrooms':1.0,'floors':1,'year_built':1980,'parking':'None','condition':'Fixer-Upper / Old'}
    payload2 = {'city_village':'Mumbai - Bandra West','area_sqyards':400,'bedrooms':4,'bathrooms':4.0,'floors':3,'year_built':2024,'parking':'Private Multi-Car Garage','condition':'Luxury Villa / Ultra-Modern'}
    
    req1 = urllib.request.Request("http://127.0.0.1:5000/api/predict", data=json.dumps(payload1).encode('utf-8'), headers={'Content-Type':'application/json'})
    req2 = urllib.request.Request("http://127.0.0.1:5000/api/predict", data=json.dumps(payload2).encode('utf-8'), headers={'Content-Type':'application/json'})
    
    res1 = json.loads(urllib.request.urlopen(req1).read().decode('utf-8'))['data']['predicted_price']
    res2 = json.loads(urllib.request.urlopen(req2).read().decode('utf-8'))['data']['predicted_price']
    
    assert res1 != res2, "Predictions must be dynamic based on ML model, not hardcoded!"
    assert res2 > res1 * 2, "Luxury Mumbai villa should predict significantly higher price than Rural village house!"
    print(f"PASSED: Dynamic prediction confirmed! (Village House: INR {res1:,.2f} vs Mumbai Villa: INR {res2:,.2f})")

    # TEST 10: Check User Input Validation Error Handling
    print("\n[TEST 10/12] Verifying User Input Validation & Error Handling...")
    invalid_payload = {'location':'InvalidLocation','area_sqft':-50,'bedrooms':15}
    req_err = urllib.request.Request("http://127.0.0.1:5000/api/predict", data=json.dumps(invalid_payload).encode('utf-8'), headers={'Content-Type':'application/json'})
    
    try:
        urllib.request.urlopen(req_err)
        print("FAILED: Server should return HTTP 400 for invalid inputs!")
        sys.exit(1)
    except urllib.error.HTTPError as e:
        assert e.code == 400
        err_res = json.loads(e.read().decode('utf-8'))
        assert err_res['status'] == 'error'
        print(f"PASSED: Input validation caught invalid input gracefully! Error returned: '{err_res['message']}'")

    # TEST 11: Check Model Benchmark Metrics Route
    print("\n[TEST 11/12] Verifying Model Benchmark Metrics Endpoint (GET /api/metrics)...")
    res_metrics = json.loads(urllib.request.urlopen("http://127.0.0.1:5000/api/metrics").read().decode('utf-8'))
    assert res_metrics['status'] == 'success'
    best_model = res_metrics['data']['best_model']
    print(f"PASSED: Metrics endpoint active. Selected Best Model: '{best_model}'")

    # TEST 12: Check ML Dashboard Analytics Route
    print("\n[TEST 12/12] Verifying ML Dashboard Analytics Endpoint (GET /api/analytics)...")
    res_analytics = json.loads(urllib.request.urlopen("http://127.0.0.1:5000/api/analytics").read().decode('utf-8'))
    assert res_analytics['status'] == 'success'
    an_data = res_analytics['data']
    assert 'price_distribution' in an_data and 'price_vs_area' in an_data and 'feature_importance' in an_data
    print(f"PASSED: ML Dashboard analytics API functional with {len(an_data['feature_importance'])} feature importance metrics.")

    print("\n" + "=" * 70)
    print("ALL 12 VERIFICATION TESTS COMPLETED SUCCESSFULLY! PROJECT IS 100% OPERATIONAL.")
    print("=" * 70)

if __name__ == '__main__':
    run_tests()
