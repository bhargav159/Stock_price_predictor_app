#run command streamlit run Streamlit_App.py
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Stock Price Predictor App")

# List of well-known stock tickers and names
well_known_stocks = {
    "AAPL": "Apple Inc.",
    "GOOG": "Alphabet Inc.",
    "MSFT": "Microsoft Corp.",
    "TSLA": "Tesla Inc.",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "NVDA": "Nvidia Corp.",
    "JPM":  "JPMorgan Chase & Co.",
    "V": "Visa Inc.",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart Inc.",
    "MA" : "Mastercard Inc.",
    "PG" : "Procter & Gamble Co."
}

# Using a session state for efficient handling of multiple calls
if 'stock_data' not in st.session_state:
    st.session_state['stock_data'] = None

if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None

if 'selected_stock_symbol' not in st.session_state:
    st.session_state['selected_stock_symbol'] = "GOOG"

# Stock selection using a dropdown or direct input
selected_stock_name = st.selectbox("Select a well-known company (Optional)", [None] + list(well_known_stocks.values()))
if selected_stock_name:
    stock = next(key for key, value in well_known_stocks.items() if value == selected_stock_name)
else:
    stock = st.text_input("Or enter the Stock ID", "GOOG").upper()



if st.session_state['selected_stock_symbol'] != stock:
    st.session_state['stock_data'] = None
    st.session_state['predictions'] = None
    st.session_state['selected_stock_symbol'] = stock

if st.session_state['stock_data'] is None:
    from datetime import datetime
    end = datetime.now()
    start = datetime(end.year-20,end.month,end.day)

    try:
        google_data = yf.download(stock, start, end)
    except Exception as e:
        st.error(f"Error downloading data for {stock}. Please ensure the ticker is valid and try again. Details: {e}")
        st.stop()

    if google_data.empty:
        st.error(f"No data available for the ticker {stock}. Please check the ticker and try again")
        st.stop()
    st.session_state['stock_data'] = google_data
    model = load_model("Latest_stock_price_model.keras")
    splitting_len = int(len(google_data)*0.7)
    close_prices = google_data['Close'].values.reshape(-1,1)
    x_test = close_prices[splitting_len:]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(x_test)

    x_data = []
    y_data = []

    for i in range(100,len(scaled_data)):
        x_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    predictions = model.predict(x_data)

    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    ploting_data = pd.DataFrame(
        {
        'original_test_data': inv_y_test.reshape(-1),
            'predictions': inv_pre.reshape(-1)
        } ,
        index = google_data.index[splitting_len+100:]
    )
    st.session_state['predictions'] = ploting_data
    st.session_state['scaler'] = scaler
else:
    google_data = st.session_state['stock_data']
    ploting_data = st.session_state['predictions']


def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data['Close'], 'b')
    if extra_data:
        plt.plot(extra_dataset,'g')
    return fig

st.subheader(f"Stock Data for {stock}")
st.write(google_data)

st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data['Close'].rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'],google_data,0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data['Close'].rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data['Close'].rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))


st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(google_data.Close[:int(len(google_data)*0.7)+100],'b', label = "Original data")
plt.plot(ploting_data, 'orange', label= "Predicted data")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"Original Price and Prediction Price for {stock}")
plt.legend()
st.pyplot(fig)