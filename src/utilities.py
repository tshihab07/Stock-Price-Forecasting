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
    def time_based_split(df, train_ratio=0.8):
        n = len(df)
        train_end = int(n * train_ratio)
        
        return df.iloc[:train_end], df.iloc[train_end:]
    

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
    def calculate_metrics(actual, pred):
        if len(actual) == 0 or len(pred) == 0:
            raise ValueError("Empty input for metrics calculation.")
        
        mse = mean_squared_error(actual, pred)
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, pred)
        
        mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
        
        return [mse, mae, rmse, r2, mape]
    

    # Print and return a comparison table showing Train vs Test performance metrics
    @staticmethod
    def print_evaluation_tables(model_name, train_metrics, test_metrics):
        perf = pd.DataFrame({
            'Metric' : ['MSE', 'MAE', 'RMSE', 'R2 Score', 'MAPE'],
            'Training': train_metrics,
            'Test': test_metrics
        })
        
        print(f"--- Performance Comparison: Train vs Test ({model_name}) ---")
        
        return perf.round(3)


# Handles saving trained models and performance results to organized directories
class ModelPersister:
    
    def __init__(self, model_name, artifacts_root="../artifacts"):
        self.model_name = model_name
        self.artifacts_root = Path(artifacts_root)
        self.model_dir = self.artifacts_root / "models"
        self.performance_dir = self.artifacts_root / "model-performance"
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.performance_dir.mkdir(parents=True, exist_ok=True)
    

    # Save the trained model in appropriate format
    def save_model(self, model):
        if self.model_name.lower() == "lstm":
            model.save(self.model_dir / "model_LSTM.keras")
        
        else:
            joblib.dump(model, self.model_dir / f"model_{self.model_name.title()}.pkl")
        
        print(f"Model saved: {self.model_dir}/{self.model_name.lower()}.pkl")
    

    # Save full train/test/CV metrics for this model only
    def save_performance(self, performance_df):
        filename = f"{self.model_name.lower()}Performance.csv"
        path = self.performance_dir / filename
        performance_df.to_csv(path, index=False)
        print(f"{self.model_name} performance saved: {path}")
    

    # Append this model's summary metrics to the shared performance file
    def aggregated_performance(self, df):
        path = self.performance_dir / "a_ModelPerformance.csv"
        
        
        # Append or create
        if path.exists():
            model_perf = pd.read_csv(path)                          # open previous loaded data
            df = pd.concat([model_perf, df], ignore_index=True)     # append new data
            df.to_csv(path, index=False)
        
        else:
            df.to_csv(path, index=False)
        
        print(f"Appended to aggregated performance: {path}")
    

    # Append this model's overfitting metrics to the shared overfitting file
    def append_overfitting(self, df):
        path = self.performance_dir / "a_overfittingAnalysis.csv"
        
        if path.exists():
            overfit_df = pd.read_csv(path)                          # open previous loaded data
            df = pd.concat([overfit_df, df], ignore_index=True)     # append new data
            df.to_csv(path, index=False)
        
        else:
            df.to_csv(path, index=False)
        
        print(f"Appended to overfitting analysis: {path}")