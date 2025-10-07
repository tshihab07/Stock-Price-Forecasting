import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import yfinance as yf

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

import io
import os
import traceback
import streamlit as st
import tempfile


# Streamlit Configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <h1 style='text-align: center; color: #FF4B4B; margin-top: -45px;'>
        📈 Stock Price Predictor
    </h1>
    """
    "Enter a stock ticker symbol (e.g., <code>AAPL</code>, <code>GOOG</code>, <code>TSLA</code>)",
    unsafe_allow_html=True
)

# Load Pre-trained Model
MODEL_PATH = os.path.join("artifacts", "models", "model_LSTM.keras")
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}.")
    st.stop()

try:
    model = load_model(MODEL_PATH)

except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Helper Functions
def fig_to_image(fig):
    """ Convert Matplotlib figure to RGB image array. """
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    plt.close(fig)
    img = Image.open(buf).convert("RGB")
    return img


def predict_stock(stock_symbol: str):
    """ Run the LSTM stock prediction. """
    
    try:
        stock_symbol = stock_symbol.strip().upper()
        if not stock_symbol:
            return None, None, None, None, None, "Please enter a valid stock symbol (e.g., AAPL, GOOG)."

        start, end = "2012-01-01", "2024-12-31"
        data = yf.download(stock_symbol, start=start, end=end)

        if data.empty or "Close" not in data.columns:
            return None, None, None, None, None, f"No valid data found for symbol: {stock_symbol}. Try AAPL, MSFT, or GOOG."

        # Split data
        split_index = int(len(data) * 0.8)
        data_train = pd.DataFrame(data["Close"][:split_index])
        data_test = pd.DataFrame(data["Close"][split_index:])

        # Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        past_100_days = data_train.tail(100)
        data_test = pd.concat([past_100_days, data_test], ignore_index=True)
        data_test_scaled = scaler.fit_transform(data_test)

        x, y = [], []
        for i in range(100, len(data_test_scaled)):
            x.append(data_test_scaled[i - 100:i])
            y.append(data_test_scaled[i, 0])
        x, y = np.array(x), np.array(y)

        if x.size == 0:
            return None, None, None, None, None, "Not enough data to make predictions."

        predictions = model.predict(x, verbose=0)
        scale_factor = 1 / scaler.scale_[0]
        predictions = predictions * scale_factor
        y = y * scale_factor

        # ======== PLOTS ========
        # Price vs MA50
        ma_50 = data["Close"].rolling(window=50).mean()
        fig1 = plt.figure(figsize=(10, 6))
        plt.plot(ma_50, "r", label="MA50")
        plt.plot(data["Close"], "g", label="Close Price")
        plt.title(f"{stock_symbol} - Price vs MA50")
        plt.legend()
        img1 = fig_to_image(fig1)

        # MA50 vs MA100
        ma_100 = data["Close"].rolling(window=100).mean()
        fig2 = plt.figure(figsize=(10, 6))
        plt.plot(ma_50, "r", label="MA50")
        plt.plot(ma_100, "b", label="MA100")
        plt.plot(data["Close"], "g", label="Close Price")
        plt.title(f"{stock_symbol} - Price vs MA50 vs MA100")
        plt.legend()
        img2 = fig_to_image(fig2)

        # MA100 vs MA200
        ma_200 = data["Close"].rolling(window=200).mean()
        fig3 = plt.figure(figsize=(10, 6))
        plt.plot(ma_100, "r", label="MA100")
        plt.plot(ma_200, "b", label="MA200")
        plt.plot(data["Close"], "g", label="Close Price")
        plt.title(f"{stock_symbol} - Price vs MA100 vs MA200")
        plt.legend()
        img3 = fig_to_image(fig3)

        # Original vs Predicted
        fig4 = plt.figure(figsize=(10, 6))
        plt.plot(predictions, "r", label="Predicted Price")
        plt.plot(y, "g", label="Original Price")
        plt.title(f"{stock_symbol} - Original vs Predicted Price")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        img4 = fig_to_image(fig4)

        # ======== CSV ========
        pred_df = pd.DataFrame(
            {"Original_Price": y.flatten(), "Predicted_Price": predictions.flatten()}
        )

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        pred_df.to_csv(temp_file.name, index=False)

        return img1, img2, img3, img4, temp_file.name, "✅ Prediction completed successfully!"

    except Exception as e:
        error_details = f"Error: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
        print(error_details)
        return None, None, None, None, None, error_details


# Streamlit UI
with st.form("predict_form"):
    stock_symbol = st.text_input("Stock Symbol", value="GOOG", placeholder="e.g., AAPL, TSLA")
    submitted = st.form_submit_button("Predict")

if submitted:
    with st.spinner("Fetching data and predicting..."):
        img1, img2, img3, img4, csv_path, status_msg = predict_stock(stock_symbol)

    st.text_area("Status", status_msg, height=100)

    if "✅" in status_msg:
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            st.image(img1, caption="Price vs MA50")
        with col2:
            st.image(img2, caption="Price vs MA50 vs MA100")
        with col3:
            st.image(img3, caption="Price vs MA100 vs MA200")
        with col4:
            st.image(img4, caption="Original vs Predicted Price")

        with open(csv_path, "rb") as f:
            st.download_button(
                label="📥 Download Original vs Predicted Price in CSV Format",
                data=f,
                file_name="stock_predictions.csv",
                mime="text/csv"
            )