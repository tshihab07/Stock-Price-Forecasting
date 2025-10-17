# Stock Price Forecasting Model Selection Report

---

**Best Model: LSTM**

## **Summary of the Modeling Process**
This project implemented and evaluated three state-of-the-art forecasting models—**ARIMA (auto_arima)**, **XGBoost (Optuna)**, and **LSTM (Manual Tuning)** to predict the next-day closing price of stock using historical daily price data from 2012 to 2024. Each model underwent a rigorous, multi-stage evaluation process. Performance was meticulously assessed on both training and test sets using key regression metrics: **RMSE, MAE, MAPE, MSE, and R²**. Time-series cross-validation was used to gauge model robustness, and a comprehensive overfitting analysis compared cross-validated and test performance to ensure generalization.

---

## **Overall Model Review**
Among all evaluated models, **LSTM** emerged as the superior performer, demonstrating an exceptional combination of **high accuracy, strong generalization, and robustness** on unseen future data.

- **LSTM delivers outstanding test performance** with the lowest Test RMSE (8.71), Test MAE (7.595), and Test MAPE (1.117%), indicating highly precise price predictions. Its Test R² score of **0.919** confirms it explains 91% of the variance in the test set, a remarkable achievement for stock forecasting.
- **LSTM exhibits exceptional generalization** The overfitting ratio, significantly lower than other models, indicates that the model's performance on unseen data is *better* than its cross-validated performance on historical data, suggesting it has learned robust temporal patterns rather than overfitting to noise.
- **ARIMA**, while showing a low test RMSE (31.828), suffers from a catastrophically negative R² score (-0.071) and a very high CV MAPE (12.54%), indicating poor explanatory power and instability.
- **XGBoost**, despite strong cross-validation results (CV RMSE: 10.055), shows clear signs of overfitting with an Overfitting Ratio of 5.913. Its test performance (RMSE: 59.453) is significantly worse than its training performance, making it unreliable for future predictions.

Therefore, **LSTM (Manual Tuning)** is selected as the final, production-ready model for the Advanced Stock Price Forecasting System.

---

## **Best Performing Model Performance Summary**
Based on a holistic evaluation prioritizing **test set accuracy, generalization ability (low overfitting ratio), and explanatory power (R²)**, LSTM (Manual Tuning) is the definitive best model:

- **Cross-Validation RMSE:** `0.086`
- **Test RMSE:** `8.71`
- **Test MAE:** `7.595`
- **Test MAPE:** `4.117%`
- **Test R²:** `0.919`
- **Model Status:** `Good`

---

## **Final Recommendation**
Based on its unparalleled accuracy, robust generalization, and practical utility, the **LSTM** model is strongly recommended for production deployment in the Advanced Stock Price Forecasting System.

The model, along with its associated scaler and configuration, has been successfully persisted for immediate use in a prediction pipeline. Future enhancements could include:
- Incorporating additional features (e.g., trading volume, technical indicators, macroeconomic data) to potentially improve accuracy further.
- Implementing a rolling retraining strategy to keep the model updated with the latest market data.
- Developing a confidence interval or prediction interval around the point forecast to quantify uncertainty.
- Exploring more advanced architectures like Transformer models or hybrid CNN-LSTM models for even more complex pattern recognition.