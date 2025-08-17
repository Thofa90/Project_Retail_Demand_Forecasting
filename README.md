# 📌 Project Title

**Grocery/Retail Sales Forecasting – Corporación Favorita Case Study**

⸻

# 🎯 Project Goal

The objective of this project is to develop a predictive model that accurately forecasts daily product sales across multiple grocery stores. By leveraging historical sales data, store information, and external factors (e.g., holidays, promotions), we aim to create a robust forecasting system that can generalize well to future periods.The project compare ARIMA-based methods, tree-based models (XGBoost), and neural networks (LSTM) to evaluate forecasting accuracy and business applicability.

⸻

# 🏢 Business Context

Corporación Favorita is one of the largest grocery store chains in Ecuador. With thousands of products and multiple store locations, predicting future sales is a critical task for managing operations efficiently. Accurate forecasts enable the company to:

	•	📦 Optimize inventory — ensure products are available when customers need them while minimizing overstock.
	•	🛒 Reduce waste — especially for perishable goods by aligning supply with demand.
	•	💰 Improve profitability — through better pricing, promotions, and stock allocation.
	•	🚚 Enhance logistics — streamline supply chain management and replenishment scheduling.

⸻

# 🌍 Real-World Application

Sales forecasting plays a crucial role in retail and supply chain optimization. The techniques developed here are directly applicable to:

	•	🏪 Retail chains managing multiple stores with varied customer demand.
	•	🥦 Supermarkets & groceries aiming to cut food waste and stockouts.
	•	📊 E-commerce platforms predicting order volume for better fulfillment planning.
	•	🚢 Logistics companies optimizing warehouse and delivery schedules.
	•	🎯 Promotional planning — aligning discounts and marketing campaigns with predicted demand peaks.

# 📂 Project Structure

1. data_store_sales/.    # Raw CSV files (items.csv, stores.csv, transactions. csv, oil.csv, holidays_events.csv, train.csv, etc.)

2. DataLoad_EDA_Merging.ipynb.    # File 1 - Data loading, cleaning, preprocessing & EDA

3. FeatureEng_BasicModel(ARIMA with exo feature).ipynb   # File 2 - Feature engineering & ARIMA/SARIMAX models

4. XGBOOST_Model.ipynb.    # File 3 - XGBoost baseline + tuned model

5. LSTM_Model.ipynb.    # File 4 - LSTM + tuned LSTM model

6. saved_models/.    # Trained model artifacts (XGBoost, ARIMA, LSTM)

7. README.md.    # Project documentation

# 📊 Dataset

The datasets are stored in the data_store_sales/ folder.

	•	Items.csv → Item details
	•	Stores.csv → Store details
	•	Transactions.csv → Daily transaction counts
	•	Oil.csv → Daily oil prices (external factor)
	•	Holidays_events.csv → Special events and holidays
	•	Train.csv → Main sales data (store_nbr, item_nbr, date, unit_sales, onpromotion)

# 📂 Workflow Overview

**1. Importing Data**

	•	Items: Product information including perishability and family classification.
	•	Stores: Store details including location and type.
	•	Transactions: Daily transaction counts per store.
	•	Oil: Historical oil prices (as a macroeconomic factor).
	•	Holidays Event: National & local holidays with event types.
	•	Train: Historical sales data.
	•	Filtering: Data limited to Guayas state and top 3 product families.

⸻

**2. Preprocessing**

	1.	Dealing with Missing Values – Filling or removing null entries.
	2.	Converting Negative Sales to Zero – Fixing incorrect data entries.
	3.	Outlier Analysis – Identifying extreme values in sales & transactions.
	4.	Filling Missing Dates with Zero Sales – Ensuring continuous daily data.
	5.	Merging Full Range Dataset – Combining:
	•	Outlier-adjusted sales
	•	Oil prices
	•	Transactions
	•	Holidays

⸻

**3. Exploratory Data Analysis (EDA)**

	1.	Time-Based Features – Trends in sales over years, months, and days.
	2.	Impact of Holidays – Analyzing how holidays influence sales volume.
	3.	Oil Price Impact – Correlation between oil price fluctuations and sales.
	4.	Perishable vs Non-Perishable Sales – Sales patterns by product type.
	5.	Store vs Item Analysis – Item-level sales performance per store.
	6.	Transaction vs Sales Analysis – Relationship between transaction count and unit sales.
	7.	Weekly Transaction Volume – Identifying peak shopping days.
	8.	Holiday Sales Impact – Comparing transactions on holidays vs non-holidays.
	9.	Basket Size Distribution – Average items per transaction, identifying bulk-buying behavior.

**4. 📊 Key Insights from EDA**

	•	Certain stores dominate sales volume, while others serve fewer but larger transactions.
	•	Perishable items have lower total sales compared to non-perishables.
	•	Holidays tend to increase both transactions and sales, but the effect varies by store.
	•	Oil price changes show potential indirect influence on consumer spending.
	•	Bulk buying is more common in specific stores, impacting stock planning.

**5. Feature Engineering & ARIMA Models with features**

	•	Created time-based features (day, week, month, year, etc.)
	•	Built lag features (1, 7, 14, 30, 60 days)
	•	Rolling & exponentially weighted averages
	•	Promotion effects and external drivers (oil, holidays)
	•	Implemented ARIMA, ARIMAX, SARIMAX, Holt-Winters
	•	Reported metrics (MAE, RMSE, MAPE)

**6. XGBoost Model**

	•	Baseline XGBoost with default params
	•	Hyperparameter tuning with RandomizedSearchCV
	•	Saved tuned XGBoost model to saved_models/
	•	Evaluation: MAE, RMSE, R², MAPE, sMAPE, MASE

**7.  LSTM & Tuned LSTM Model**

	•	Scaled data with MinMaxScaler / StandardScaler
	•	Sequence building (lookback windows of 30 & 60 days)
	•	Baseline LSTM and tuned LSTM (layers, units, dropout, learning rate)
	•	Hyperparameter tuning with grid search (window size × units × depth)
	•	Compared baseline vs tuned model performance
	•	Plotted actual vs predicted sales for last 30 & 90 days

**8. 📈 Evaluation Metrics**

Report includes:

	•	MAE (Mean Absolute Error)
	•	RMSE (Root Mean Squared Error)
	•	R² (Coefficient of Determination)
	•	MAPE (Mean Absolute Percentage Error)
	•	sMAPE (Symmetric MAPE)
	•	MASE (Mean Absolute Scaled Error)

**9. 🚀 Key Findings & Recommendations**

	•	ARIMA/SARIMAX captured seasonality but struggled with external regressors.
	•	XGBoost handled multiple features well and gave strong baseline results.
	•	LSTM (tuned) showed improvement in short-term predictions but still had difficulty capturing high variance in sales.

**Business Recommendation:**

	•	Use XGBoost for operational daily forecasts (robust, handles features well).
	•	Use LSTM for items with strong temporal dependencies and retrain frequently.

**10. ✅ Conclusion**

This project demonstrates how classical models (ARIMA), tree-based models (XGBoost), and deep learning (LSTM) can be applied to real-world retail forecasting.
The experiments highlight the trade-offs between interpretability, accuracy, and computational complexity.

# Model Comparison

## 📊 Model Comparison

| Model              | MAE   | RMSE  | R²      | Strengths | Weaknesses | Business Implication |
|---------------------|-------|-------|---------|-----------|------------|----------------------|
| **ARIMA**           | 1.000 | 1.092 | -2.188  | Simple, fast, captures autocorrelation | No external drivers, misses events | Too inaccurate → over/understocking risk |
| **ARIMAX**          | 0.613 | 0.681 | -0.242  | Uses promotions, holidays, lags | Still weak for irregular spikes | Best traditional model, actionable forecasts |
| **SARIMAX**         | 0.616 | 0.724 | -0.403  | Adds seasonality | Seasonal cycles not strong here | No gain over ARIMAX, unnecessary complexity |
| **Holt-Winters**    | 2.547 | 2.640 | -17.641 | Smooths trend/seasonality | No exogenous features, misses spikes | Unusable for sporadic sales, flat forecasts |
| **XGBoost (Baseline)** | 1.437 | 4.792 | 0.696 | Captures non-linear effects | Higher RMSE baseline | Useful but needs tuning |
| **XGBoost (Tuned)** | 1.441 | 4.738 | 0.703 | Better RMSE, higher R² | MAE unchanged | Stronger than baseline, learns exogenous impact |
| **LSTM (Previous)** | 0.642 | 0.823 | -0.812 | Captures sequence memory | Struggles with variance | Moderate accuracy, unstable |
| **LSTM (Tuned)**    | 0.365 | 0.464 | -1.389 | Lowest MAE/RMSE | Negative R², poor generalization | Good for short-term, unreliable long-term |


## ✅ Recommendation

Based on the comparison:

- **Best Overall Traditional Model → ARIMAX**
  - Lowest MAE & RMSE among classical models.
  - Incorporates external drivers (promotions, holidays, lags).
  - Reliable for operational business forecasting (inventory planning, promotions, staffing).

- **Best ML Model → Tuned XGBoost**
  - Higher R² (better fit) and lower RMSE than baseline.
  - Learns non-linear effects from external features.
  - Scales better than ARIMA-family models for large datasets.

- **Deep Learning Model (LSTM)**
  - Tuned LSTM achieved lowest MAE/RMSE, but negative R² indicates poor generalization.
  - Useful for **short-term forecasts**, but not stable for production.

- **✅ Takeaway:**  
  - Use **ARIMAX** if you want interpretability and external features matter.  
  - Use **XGBoost** if dataset is larger and has non-linear relationships.  
  - Use **LSTM** when sequential dependencies are critical (but validate carefully).  

  ## 🔍 Model Selection Flow

When deciding which forecasting model to use:
            ┌────────────────────┐
            │   Size of Data?    │
            └─────────┬──────────┘
                      │
      ┌───────────────┴───────────────┐
Small / Medium Data             Large / Complex Data
│                               │
┌──────▼──────┐                 ┌──────▼───────┐
│   ARIMAX    │                 │   XGBoost    │
│ (best for   │                 │ (handles     │
│ explainable │                 │ non-linear   │
│ forecasts)  │                 │ features)    │
└─────────────┘                 └──────────────┘
│                               │
│                               │
│                               │
┌──────▼───────┐                ┌──────▼──────┐
│  Stable,     │                │ Tuned for   │
│ interpretable│                │ better fit  │
│ forecasts    │                │ on big data │
└──────────────┘                └─────────────┘
│
│
┌─────────▼─────────┐
│   Sequential /    │
│  Short-term Data? │
└─────────┬─────────┘
│.        |
┌──────▼──────┐
│    LSTM     │
│ (captures   │
│ sequence &  │
│ memory)     │
└─────────────┘
    

### 🚀 Practical Usage

- Use **ARIMAX** as the baseline model for explainability and operational use.  
- For improved performance on larger datasets, prefer **Tuned XGBoost**.  
- Keep **LSTM** for experimental or short-term forecasting — but not as the main business driver.  
- Consider **hybrid models**:  
  - ARIMAX for stable baseline forecasts.  
  - ML models (XGBoost/LightGBM) for residual corrections and anomaly handling.

# 💾 Model Saving & Loading


All trained models are saved in the project root (e.g., saved_models/xgboost_tuned.pkl, lstm_tuned.h5).

# 🛠️ Requirements

✅ With this file in your project root, anyone can install everything with:

pip install -r requirements.txt

**Core Python packages**

numpy
pandas
scipy

**Visualization**

matplotlib
seaborn

**Time Series Models**

statsmodels

**Machine Learning**

scikit-learn
xgboost

**Deep Learning**

tensorflow

**Jupyter Notebook**

notebook
ipykernel