# train_model.py

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Fetch historical data
def fetch_historical_data(ticker):
    return yf.download(ticker, start="2010-01-01", end="2024-01-01")

# Preprocess the data
def preprocess_data(data):
    data = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Create sequences for LSTM model
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Build and train the LSTM model
def train_model(ticker):
    historical_data = fetch_historical_data(ticker)
    scaled_data, scaler = preprocess_data(historical_data)
    seq_length = 60
    X_train, y_train = create_sequences(scaled_data, seq_length)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    return model, scaler

if __name__ == "__main__":
    model, scaler = train_model('AAPL')
    model.save('stock_model.h5')
    with open('scaler.pkl', 'wb') as f:
        import pickle
        pickle.dump(scaler, f)
