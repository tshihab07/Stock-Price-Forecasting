# Import Required Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")


# Handles data splitting, feature engineering, and sequence creation for stock price prediction
class StockDataProcessor:
    
    # Split dataset based on time ratio
    @staticmethod
    def time_based_split(df, train_ratio=0.8, val_ratio=0.1):
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
    

    # Create lagged features for time-series modeling
    @staticmethod
    def create_lagged_features(series, lags=10):
        data = pd.DataFrame(series, columns=['Close'])
        
        for lag in range(1, lags + 1):
            data[f'lag_{lag}'] = data['Close'].shift(lag)
        
        data.dropna(inplace=True)
        return data
    

    # Generate sequential input-output pairs
    @staticmethod
    def create_sequences(data, seq_length):
        X, y = [], []
        
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)


# Provides unified metric calculation and reporting for regression model performance
class Evaluator:
    
    # Compute regression evaluation metrics includes MSE, MAE, RMSE, R², and MAPE
    @staticmethod
    def calculate_metrics(actual, pred, prefix=""):
        if len(actual) == 0 or len(pred) == 0:
            raise ValueError("Empty input for metrics calculation.")
        
        mse = mean_squared_error(actual, pred)
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, pred)
        
        mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100  # avoid div by zero
        
        return {
            f"{prefix}MSE": mse,
            f"{prefix}MAE": mae,
            f"{prefix}RMSE": rmse,
            f"{prefix}R2": r2,
            f"{prefix}MAPE": mape
        }
    

    # Print and return a comparison table showing Train vs Test performance metrics
    @staticmethod
    def print_evaluation_tables(model_name, train_metrics, test_metrics):
        perf = pd.DataFrame({
            'Train': train_metrics,
            'Test': test_metrics
        })
        
        print(f"--- Performance Comparison: Train vs Test ({model_name}) ---")
        
        return perf.round(3)


# Handles saving trained models and performance results to organized directories
class ModelPersister:
    
    # Save model artifacts to disk that supports both sklearn models and LSTM (Keras)
    @staticmethod
    def save_model(model, model_name, model_dir="../artifacts/models"):
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        if model_name == "LSTM":
            model.save(path / "model_LSTM.keras")
            print(f"✅ Model saved: {path}/{model_name}.keras")
        
        else:
            joblib.dump(model, path / f"{model_name}.pkl")
            print(f"✅ Model saved: {path}/{model_name}.pkl or .keras")
        
    
    # Save model performance results as CSV
    @staticmethod
    def save_results(data, filename, results_dir="../artifacts/model-performance"):
        path = Path(results_dir)
        path.mkdir(parents=True, exist_ok=True)
        if isinstance(data, pd.DataFrame):
            data.to_csv(path / filename, index=False)
        
        else:
            pd.DataFrame([data]).to_csv(path / filename, index=False)
        print(f"✅ Results saved: {path}/{filename}")