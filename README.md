# Nm-project
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load stock data
def load_data(ticker='AAPL', period='5y'):
    data = yf.download(ticker, period=period)
    return data['Close'].values.reshape(-1, 1), data.index

# Preprocess data
def preprocess(data, time_step=60):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i - time_step:i, 0])
        y.append(data_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Build model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main
if __name__ == '__main__':
    raw_data, dates = load_data('AAPL')
    X, y, scaler = preprocess(raw_data)
    
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = build_model((X.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot
    plt.plot(actual, color='blue', label='Actual Stock Price')
    plt.plot(predictions, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
