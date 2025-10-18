# Stock Price Prediction

## A Production-Ready Machine Learning Pipeline for Next Day Closing Prices Forecasting

This repository contains a complete machine learning pipeline for predicting next-day stock closing prices using `ARIMA`, `XGBoost`, and `LSTM models`. The project includes **data preprocessing**, **end-to-end modeling**, **hyperparameter optimization**, and **deployment-ready artifacts**.

**Primary Goal:** Build a robust, interpretable, and high-performance regression model suitable for real-world deployment in stock price analyzing and prediction.

**Status:** Trained, evaluated, and validated  
**Output:** model file ready for integration 

---

## 📋 Table of Contents

- [Overview](#overview)
    - [Key Engineering Decisions](#key-engineering-decisions)
- [Dataset & Features](#dataset-features)
- [Modeling Pipeline](#modeling-pipeline)
- [Model Selection](#model-selection)
    - [Modeling Process](#modeling-process)
    - [Overall Model Review](#overall-model-review)
    - [Best Performing Model](#best-performing-model)
    - [Final Recommendation](#final-recommendation)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
    - [Clone the Repository](#clone-the-repository)
    - [Create a virtual environment](#create-a-virtual-environment)
    - [Install Dependencies](#install-dependencies)
- [Model Inference](#model-inference)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [Contact](#contact)
- [License](#license)

---

## Overview

This repository contains a complete end-to-end project for forecasting the next-day closing price of Apple Inc. (AAPL) stock using historical daily data from 2012 to 2024. Three advanced models — **ARIMA**, **XGBoost**, and **LSTM** — were rigorously evaluated. The **LSTM model** was selected as the final production-ready solution due to its superior accuracy and generalization capability.

The code is organized into modular directories for data, source code, visualizations, and artifacts, ensuring maintainability and reproducibility.

### Key Engineering Decisions

- **Time-Based Splits**: Strict chronological splitting (Train: 2012–2021, Val: 2022, Test: 2023–2024) to prevent data leakage and simulate real-world forecasting.
- **Model-Specific Optimization**: Used `auto_arima` for ARIMA, `Optuna` for XGBoost, and manual tuning for LSTM to ensure fair, state-of-the-art hyperparameter optimization.
- **Native Serialization**: Saved models in their optimal formats (`.keras` for LSTM, `.pkl` for others) for reliability and performance.
- **Comprehensive Evaluation**: Employed multiple metrics (RMSE, MAE, MAPE, R²) and time-series cross-validation to assess both performance and robustness.

---

## Dataset & Features

- **Source**: Historical stock data for **AAPL** (Apple Inc.) downloaded via `yfinance`.
- **Time Range**: January 1, 2012 – December 31, 2024.
- **Target Variable**: Daily closing price (`Close`).
- **Features**: The primary model (LSTM) uses a **univariate time series** of the last 10 closing prices to predict the next day's price. This simple, interpretable feature set avoids look-ahead bias and ensures real-time applicability.

---

## Modeling Pipeline

The end-to-end pipeline consists of the following stages:

1. **Data Acquisition**: Fetch raw OHLCV data from `yfinance`.
2. **Preprocessing**: Clean data, handle missing values (forward-fill weekends), and create a univariate `Close` price series.
3. **Time-Based Splitting**: Partition data into training, validation, and test sets without shuffling.
4. **Model Training & Optimization**:
   - **ARIMA**: Automatic order selection using `pmdarima`.
   - **XGBoost**: Hyperparameter tuning with `Optuna` using lagged features.
   - **LSTM**: Architecture tuning (units, dropout, learning rate) on raw price sequences.
5. **Evaluation**: Compute metrics on both validation and test sets. Perform time-series cross-validation on the training set.
6. **Model Selection**: Choose the best model based on test set performance and overfitting analysis.
7. **Persistence**: Save the final model and its artifacts (e.g., scaler) to the `artifacts/` directory.

---

## Model Selection

### Modeling Process

This project implemented and evaluated three state-of-the-art forecasting models—**ARIMA (auto_arima)**, **XGBoost (Optuna)**, and **LSTM (Manual Tuning)**—to predict the next-day closing price of stock using historical daily price data from 2012 to 2024. Each model underwent a rigorous, multi-stage evaluation process. Performance was meticulously assessed on both training and test sets using key regression metrics: **RMSE, MAE, MAPE, MSE, and R²**. Time-series cross-validation was used to gauge model robustness, and a comprehensive overfitting analysis compared cross-validated and test performance to ensure generalization.

---

### Overall Model Review

Among all evaluated models, **LSTM** emerged as the superior performer, demonstrating an exceptional combination of **high accuracy, strong generalization, and robustness** on unseen future data.

- **LSTM delivers outstanding test performance** with the lowest Test RMSE (8.71), Test MAE (7.595), and Test MAPE (1.117%), indicating highly precise price predictions. Its Test R² score of **0.919** confirms it explains 91% of the variance in the test set, a remarkable achievement for stock forecasting.
- **LSTM exhibits exceptional generalization**. The overfitting ratio, significantly lower than other models, indicates that the model's performance on unseen data is *better* than its cross-validated performance on historical data, suggesting it has learned robust temporal patterns rather than overfitting to noise.
- **ARIMA**, while showing a low test RMSE (31.828), suffers from a catastrophically negative R² score (-0.071) and a very high CV MAPE (12.54%), indicating poor explanatory power and instability.
- **XGBoost**, despite strong cross-validation results (CV RMSE: 10.055), shows clear signs of overfitting with an Overfitting Ratio of 5.913. Its test performance (RMSE: 59.453) is significantly worse than its training performance, making it unreliable for future predictions.

Therefore, **LSTM (Manual Tuning)** is selected as the final, production-ready model for the Advanced Stock Price Forecasting System.

---

### Best Performing Model

Based on a holistic evaluation prioritizing **test set accuracy, generalization ability (low overfitting ratio), and explanatory power (R²)**, LSTM (Manual Tuning) is the definitive best model:

- **Cross-Validation RMSE:** `0.086`
- **Test RMSE:** `8.71`
- **Test MAE:** `7.595`
- **Test MAPE:** `4.117%`
- **Test R²:** `0.919`
- **Model Status:** `Good`

---

### Final Recommendation

Based on its unparalleled accuracy, robust generalization, and practical utility, the **LSTM** model is strongly recommended for production deployment in the Advanced Stock Price Forecasting System.

The model, along with its associated scaler and configuration, has been successfully persisted for immediate use in a prediction pipeline. Future enhancements could include:
- Incorporating additional features (e.g., trading volume, technical indicators, macroeconomic data) to potentially improve accuracy further.
- Implementing a rolling retraining strategy to keep the model updated with the latest market data.
- Developing a confidence interval or prediction interval around the point forecast to quantify uncertainty.
- Exploring more advanced architectures like Transformer models or hybrid CNN-LSTM models for even more complex pattern recognition.

---

## File Structure

The project is organized into logical directories for clarity and scalability.
```bash
Stock-Price-Forecasting/
├── artifacts/
│   ├── model-performance/
│   │   ├── a_ModelPerformance.csv
│   │   ├── a_overfittingAnalysis.csv
│   │   ├── arimaPerformance.csv
│   │   ├── lstmPerformance.csv
│   │   └── xgboostPerformance.csv
│   ├── models/
│   │   ├── model_Arima.pkl
│   │   ├── model_LSTM.keras
│   │   └── model_XGBoost.keras
│   └── report.md
├── data/
│   └── AAPL_preprocessed.csv
├── notebooks/
│   ├── DataPreprocessing.ipynb
│   ├── model_Arima.ipynb
│   ├── model_LSTM.ipynb
│   ├── model_XGBoost.ipynb
│   └── ModelSelection.ipynb
├── src/
│    ├── utilities.py
├── visualizations/
│    ├── ActualvsPredicted.png
│    ├── ClosingPriceAnalysis.png
│    ├── DailyReturnAnalysis.png
│    ├── ModelPerformance.png
│    ├── MovingAverages.png
│    ├── OverfittingAnalysis.png
│    ├── OverfittingIndicator.png
│    └── VolumeofSalesAnalysis.png
├── .gitignore
├── app.py
├── LICENSE
├── README.md
├── requirements.txt
└── runtime.txt

```

---

## Dependencies

The project requires the following key Python libraries:

```python
# Core Python libraries
numpy==1.26.4
Cython==3.1.4
pandas==2.3.2
yfinance==0.2.66
matplotlib==3.10.6
seaborn==0.13.2
joblib==1.5.2

# Scikit-learn utilities
scikit-learn==1.7.2
scikeras==0.13.0

# Machine Learning & Optimization
xgboost==3.0.5
optuna==4.5.0

# ARIMA models for time series forecasting
statsmodels==0.14.5
pmdarima==2.0.3

# Deep Learning with TensorFlow
tensorflow==2.16.1

# Building web apps with Streamlit
streamlit==1.50.0
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/tshihab07/Stock-Price-Forecasting.git
```

```bash
cd Stock-Price-Forecasting
```

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows
```

### Install Dependencies

```python
pip install -r requirements.txt
```

---

## Model Inference
```python
streamlit run app.py
```

---

## Future Improvements

- Incorporate additional features (e.g., trading volume, technical indicators like RSI or MACD).
- Implement a rolling retraining strategy to adapt to evolving market conditions.
- Develop prediction intervals to quantify forecast uncertainty.
- Experiment with more advanced architectures (e.g., Transformers, Temporal Fusion Transformers).
- Add real-time data ingestion for live forecasting.

---

## Contributing

Contributions are welcome! Please feel free to submit a pull request.
- Fork the project.
- Create your feature branch
- Commit changes
- Push
- Open a Pull Request

---

## Contact

E-mail: tushar.shihab13@gmail.com <br>
More Projects: 👉🏿 [Projects](https://github.com/tshihab07?tab=repositories)<br>
LinkedIn: [Tushar Shihab](https://www.linkedin.com/in/tshihab07/)

---


## License

This project is licensed under the [MIT License](LICENSE).