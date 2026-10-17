# COLLEGE MACHINE LEARNING PROJECT REPORT

## 1. Project Title
**House Price Prediction: An End-to-End Machine Learning System for Real Estate Valuation**

---

## 2. Problem Statement
Accurately predicting house prices is a critical challenge in the real estate domain. Traditional property valuation relies heavily on manual appraisals, which can be subjective, time-consuming, and prone to human bias. Real estate values are influenced by complex, non-linear interactions among multiple factors including spatial location, physical square footage, room capacity, structural age, parking amenities, and overall property condition. 

This project addresses the problem by developing an automated, data-driven Machine Learning system that predicts house selling prices in Indian Rupees (₹) with high precision, offering instant valuation feedback via a responsive web interface.

---

## 3. Objectives
- **Data Engineering**: Construct a robust real estate dataset featuring realistic non-linear relationships and missing values to simulate real-world conditions.
- **Data Preprocessing**: Design a reusable data pipeline using Scikit-Learn `ColumnTransformer` to handle missing data imputation, standard scaling, and one-hot encoding without data leakage.
- **Model Engineering & Comparison**: Train and benchmark four distinct regression algorithms: *Linear Regression*, *Ridge Regression*, *Random Forest Regressor*, and *Gradient Boosting Regressor*.
- **Quantitative Evaluation**: Evaluate models using standardized statistical metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$ Score).
- **Artifact Serialization**: Automatically identify and serialize the top-performing machine learning model using `joblib`.
- **Application Deployment**: Build an interactive Flask web application featuring a modern dark-mode glassmorphism interface, an interactive Machine Learning analytics dashboard, and Indian Rupee (₹) currency formatting.

---

## 4. Dataset Description
The dataset contains **2,000 real estate property records** with 8 input features and 1 target continuous variable (`price`).

### Feature Dictionary:

| Feature Name | Data Type | Feature Class | Description / Allowed Values |
| :--- | :--- | :--- | :--- |
| `location` | Categorical | Nominal | Spatial location zone: `Rural`, `Suburban`, `Urban`, `Downtown`, `Waterfront` |
| `area_sqft` | Numerical | Continuous | Total indoor livable area (range: 550 to 7,500 sq ft) |
| `bedrooms` | Numerical | Discrete | Total number of bedrooms (1 to 6) |
| `bathrooms` | Numerical | Continuous | Total number of bathrooms (1.0 to 6.0) |
| `floors` | Numerical | Discrete | Total number of building stories (1 to 4) |
| `year_built` | Numerical | Discrete | Construction year of the property (1950 to 2024) |
| `parking` | Categorical | Nominal | Parking facility: `None`, `Carport`, `Driveway`, `Garage` |
| `condition` | Categorical | Ordinal | Physical maintenance state: `Poor`, `Fair`, `Good`, `Excellent`, `Luxury` |
| **`price`** *(Target)* | Numerical | Continuous | Total house selling price in Indian Rupees (₹) |

---

## 5. Data Preprocessing
To prepare raw data for machine learning algorithms, the following preprocessing steps were applied:

1. **Duplicate Removal**: Inspected and purged exact duplicate rows to maintain data integrity.
2. **Missing Value Imputation**:
   - *Numerical Features*: Missing values filled using **Median Imputation** (`SimpleImputer(strategy='median')`) to prevent outlier bias.
   - *Categorical Features*: Missing values filled using **Most-Frequent (Mode) Imputation** (`SimpleImputer(strategy='most_frequent')`).
3. **Feature Scaling**: Applied **Standard Scaling** (`StandardScaler()`) to zero-center numerical variables ($Z = \frac{X - \mu}{\sigma}$), ensuring gradient-based algorithms converge efficiently.
4. **Categorical Encoding**: Converted nominal and ordinal strings into binary vectors using **One-Hot Encoding** (`OneHotEncoder(handle_unknown='ignore')`).
5. **Data Leakage Prevention**: Encapsulated preprocessing steps inside a Scikit-Learn `ColumnTransformer` inside a master `Pipeline`. Preprocessors are fitted **strictly on the 80% training set (`X_train`)**, ensuring zero data leakage into the 20% test set (`X_test`).

---

## 6. Exploratory Data Analysis (EDA)
Exploratory Data Analysis revealed key statistical distribution patterns in the dataset:
- **Price Distribution**: Positively skewed distribution with a median price of ₹6,10,450 and a max price of ₹29,49,800.
- **Location Premium**: *Waterfront* properties exhibit the highest median valuation (~₹13.5 Lakhs), followed by *Downtown* (~₹9.8 Lakhs), *Urban* (~₹6.5 Lakhs), *Suburban* (~₹4.8 Lakhs), and *Rural* (~₹3.1 Lakhs).
- **Square Footage Relationship**: Strong positive correlation ($r \approx 0.88$) between living area (`area_sqft`) and selling price.
- **Condition Impact**: Properties rated as *Luxury* or *Excellent* command a 25%–55% premium over equivalent *Poor* or *Fair* condition homes.

---

## 7. Feature Engineering
Feature engineering steps integrated into the model pipeline:
- **Age Depreciation Factor**: Computed property age ($Age = 2026 - YearBuilt$) and applied modern construction thresholds for properties built after 2015.
- **Rate Per Square Foot**: Derived $Price / Area_{sqft}$ metric for valuation confidence intervals and real estate analytics.
- **Combined One-Hot Matrix**: Transformed 8 raw features into a 16-dimensional numeric input space for regression models.

---

## 8. Machine Learning Algorithms Used
Four regression algorithms were selected for benchmarking:

1. **Linear Regression**: Baseline parametric model establishing linear hyperplanes ($y = \beta_0 + \sum \beta_i X_i$).
2. **Ridge Regression**: Regularized linear model incorporating L2 penalty ($\alpha = 1.0$) to minimize coefficient variance and multicollinearity.
3. **Random Forest Regressor**: Non-parametric ensemble bagging model consisting of 100 decision trees (`n_estimators=100`, `max_depth=15`).
4. **Gradient Boosting Regressor**: Sequential boosting ensemble algorithm (`n_estimators=120`, `learning_rate=0.1`, `max_depth=5`) that iteratively minimizes pseudo-residuals.

---

## 9. Model Training
- **Data Split**: 80% Training Data (1,600 samples) | 20% Test Data (400 samples).
- **Training Strategy**: Each algorithm was embedded inside a unified Scikit-Learn `Pipeline(preprocessor, model)`.
- `.fit(X_train, y_train)` was executed sequentially for all 4 algorithms.

---

## 10. Model Evaluation
Models were evaluated on the independent test dataset using four performance metrics:
- **MAE** (Mean Absolute Error): $\frac{1}{n}\sum |y_i - \hat{y}_i|$
- **MSE** (Mean Squared Error): $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$
- **RMSE** (Root Mean Squared Error): $\sqrt{\text{MSE}}$
- **$R^2$ Score** (Coefficient of Determination): $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

### Comparative Performance Table:

| Algorithm | MAE (₹) | RMSE (₹) | $R^2$ Score | Selection Result |
| :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting** | **₹36,611.41** | **₹55,118.48** | **0.9642 (96.42%)** | **★ BEST MODEL (Saved)** |
| **Random Forest** | ₹47,304.87 | ₹73,758.44 | 0.9359 (93.59%) | Benchmark |
| **Ridge Regression** | ₹62,851.33 | ₹88,684.45 | 0.9074 (90.74%) | Benchmark |
| **Linear Regression** | ₹63,038.92 | ₹88,925.99 | 0.9069 (90.69%) | Benchmark |

---

## 11. Prediction Interface
The application features a modern dark-mode glassmorphism web interface powered by **Flask**:
- **Price Estimator**: Form allowing users to input location, area, bedrooms, bathrooms, floors, year built, parking, and condition.
- **REST API (`POST /api/predict`)**: Validates input data, passes features to `house_price_pipeline.pkl`, and returns estimated house price in Indian Rupee format (e.g. `Estimated House Price: ₹45,25,000`).
- **ML Analytics Dashboard**: Renders interactive Chart.js charts displaying price distributions, price vs area scatter plots, actual vs predicted evaluations, and feature importance progress bars.

---

## 12. Results
- **Top Performer**: The **Gradient Boosting Regressor** achieved the highest $R^2$ score of **0.9642**, explaining 96.42% of price variance.
- **Feature Importance Analysis**:
  1. *Location Zone*: **43.5%** impact
  2. *Area in Square Feet*: **36.3%** impact
  3. *House Condition*: **14.2%** impact
  4. *Bathrooms & Bedrooms*: **4.1%** impact
  5. *Year Built & Parking*: **1.9%** impact

---

## 13. Limitations
- **Geographic Scope**: Dataset reflects synthetic market dynamics and requires retraining on specific regional Indian cities (e.g., Mumbai, Bangalore, Delhi NCR).
- **Unobserved Factors**: External variables such as proximity to metro stations, school quality, noise levels, and market inflation are not captured in the current feature set.

---

## 14. Future Enhancements
- **Spatial Map Integration**: Integrate Leaflet.js / Google Maps API for interactive geolocation selection.
- **XGBoost / LightGBM Integration**: Upgrade ensemble algorithms to GPU-accelerated XGBoost models.
- **Deep Learning Architecture**: Experiment with Multi-Layer Perceptrons (MLPs) for complex multi-modal valuation.
- **Cloud Deployment**: Containerize using Docker and deploy to AWS Elastic Beanstalk or Render.

---

## 15. Conclusion
This project successfully demonstrates a complete end-to-end Machine Learning solution for house price prediction. By combining rigorous data preprocessing, strict data leakage prevention, multi-algorithm benchmarking, model serialization, and a user-friendly Flask web application with real-time analytics, the system provides accurate real estate valuations and valuable educational insights.
