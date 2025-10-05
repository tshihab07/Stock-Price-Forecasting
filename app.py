import streamlit as st
import joblib
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model


MODEL_DIR = Path("artifacts/models")  

# --- Load the Model and Scaler ---
@st.cache_resource
def load_model_and_scaler():
    """
    Loads the LSTM model (.h5) and its associated scaler (.pkl).
    """
    
    model_path = MODEL_DIR / "LSTM.h5"
    model = load_model(model_path)
    
    scaler_path = MODEL_DIR / "LSTM_scaler.pkl"
    scaler = joblib.load(scaler_path)
    
    return model, scaler

# Load the model and scaler
try:
    model, scaler = load_model_and_scaler()
    st.success("Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model or scaler: {e}")
    st.stop()

# --- App UI ---
st.title("Stock Price Forecaster")
st.write("This app uses a trained LSTM model to predict the next day's closing price.")

# --- User Input / Data Source ---
st.subheader("Provide the last 10 closing prices (in USD)")

# Create 10 input boxes for the user
prices = []
for i in range(10, 0, -1):
    days_ago = i
    default_price = 190.0 + (10 - i) * 0.5
    price = st.number_input(
        f"Price from {days_ago} day(s) ago", 
        min_value=0.01, 
        value=float(default_price), 
        key=f"price_{i}"
    )
    prices.append(price)

if st.button("Predict Next Day's Closing Price"):
    try:
        input_array = np.array(prices).reshape(-1, 1)
        
        scaled_input = scaler.transform(input_array)
        
        scaled_input = scaled_input.reshape(1, -1, 1)
        
        # --- Make Prediction ---
        scaled_prediction = model.predict(scaled_input, verbose=0)
        
        # --- Inverse transform to get the actual price ---
        prediction = scaler.inverse_transform(scaled_prediction).flatten()[0]
        
        # --- Display Result ---
        st.success(f"**Predicted Closing Price:** ${prediction:.2f}")
        st.balloons()
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.exception(e)