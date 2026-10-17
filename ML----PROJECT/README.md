# House Price Prediction (India Edition): An End-to-End Machine Learning System

A complete, college-submission ready Machine Learning project for predicting house selling prices across **real Indian Metropolitan Cities, Tier-2 Towns, and Rural Villages**, supporting **Square Yards (Gaj)** and **Square Feet (sq ft)** with Indian Rupee (₹) Crores & Lakhs valuations.

Built with **Python**, **Scikit-Learn**, **Pandas**, **NumPy**, **Flask**, and a modern **Glassmorphism Web Dashboard** with Chart.js analytics and Indian Rupee (₹) currency formatting.

---

## 📄 College Project Submission Report

A comprehensive 15-section project report has been compiled and included:
👉 **[`PROJECT_REPORT.md`](PROJECT_REPORT.md)**

### Report Table of Contents:
1. **Project Title**: House Price Prediction: An End-to-End Machine Learning System for Real Estate Valuation
2. **Problem Statement**: Challenges in traditional manual appraisals vs data-driven valuation.
3. **Objectives**: Dataset engineering, preprocessing pipelines, model comparison, and deployment.
4. **Dataset Description**: Feature dictionary covering 2,000 real-estate records.
5. **Data Preprocessing**: Imputation, scaling, one-hot encoding, and data leakage prevention.
6. **Exploratory Data Analysis (EDA)**: Statistical distributions, location premiums, and price trends.
7. **Feature Engineering**: Age factors, rate per sq ft, and dimensional encoding.
8. **Machine Learning Algorithms Used**: Linear Regression, Ridge, Random Forest, and Gradient Boosting.
9. **Model Training**: 80/20 train-test split strategy and Scikit-Learn pipelines.
10. **Model Evaluation**: Benchmark tables comparing MAE, MSE, RMSE, and R² scores.
11. **Prediction Interface**: Flask REST API (`/api/predict`) and Glassmorphism Web UI.
12. **Results**: Top performing model selection (**Gradient Boosting**, $R^2 = 0.9642$) & feature importances.
13. **Limitations**: Geographic scope and unobserved macro-economic factors.
14. **Future Enhancements**: Spatial map integration, XGBoost, and cloud deployment.
15. **Conclusion**: Summary of project outcomes and educational insights.

---

## 📂 Project Directory Structure

```
house_price_prediction/
├── data/
│   ├── generate_dataset.py   # Dataset generator & data quality inspection script (with comments)
│   ├── house_data.csv        # Primary dataset (2,000 records)
│   └── house_prices.csv      # Dataset alias
├── models/
│   ├── house_price_pipeline.pkl  # Serialized Scikit-Learn Pipeline (Preprocessor + Model)
│   └── metrics.json              # Benchmark evaluation metrics report
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Load data, ColumnTransformer & train/test split (with comments)
│   ├── train_model.py        # Model training, algorithm comparison & artifact export (with comments)
│   └── predict.py            # Input validation & prediction utility in INR (with comments)
├── static/
│   ├── css/
│   │   └── style.css         # Glassmorphism dark theme & responsive styles
│   └── js/
│       └── main.js           # Frontend interactivity, Chart.js analytics & API integration
├── templates/
│   └── index.html            # Main web dashboard interface
├── app.py                    # Flask web server & REST API (with comments)
├── requirements.txt          # Python dependencies
├── PROJECT_REPORT.md         # Full 15-section academic project report
└── README.md                 # Project overview & execution instructions
```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
Ensure Python 3.9+ is installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset & Run Inspection
```bash
python data/generate_dataset.py
```

### 4. Train Models & Export Best Pipeline
```bash
python src/train_model.py
```

### 5. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🔌 REST API Endpoints

- `GET /`: Serves the index.html frontend page.
- `POST /api/predict`: Accepts property JSON payload and returns ML valuation in Indian Rupees (₹).
- `GET /api/metrics`: Returns benchmark evaluation results (MAE, MSE, RMSE, R²).
- `GET /api/analytics`: Returns dataset EDA & feature importance breakdown for Chart.js dashboard.
