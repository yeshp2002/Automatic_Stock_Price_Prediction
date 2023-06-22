import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import date, timedelta

# Define the ticker symbol for the stock you want to predict
ticker_symbol = 'AMZN'

# Retrieve historical stock data using yfinance
end_date = date.today() - timedelta(days=1)  # Exclude current day
start_date = end_date - timedelta(days=365)  # Retrieve data for the past year
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Preprocess the data
data = data[['Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare the training data
X_train = []
y_train = []

# Use the previous one year's data to predict a day's stock for each LSTM layer
lookback = 365

for i in range(lookback, len(scaled_data)):
    X_train.append(scaled_data[i - lookback: i, 0])
    y_train.append(scaled_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the training data for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Get today's stock price
today = date.today()
today_data = yf.download(ticker_symbol, start=today, end=today)
today_stock_price = today_data['Close'].iloc[0]

# Prepare the input data for prediction
last_year_data = data.tail(lookback)
inputs = scaler.transform(last_year_data)

X_test = []
X_test.append(inputs)
X_test = np.array(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Predict today's stock price
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Compare predicted price with today's stock price
print("--------------------Using LSTM Layers ------------------------")
print(f"Predicted stock price for today: {predicted_stock_price[0][0]}")
print(f"Actual stock price for today: {today_stock_price}")

# Calculate the difference between predicted and actual prices
price_difference = predicted_stock_price[0][0] - today_stock_price
if price_difference > 0:
    print(f"The predicted price is higher than the actual price by {price_difference}")
elif price_difference < 0:
    print(f"The predicted price is lower than the actual price by {-price_difference}")
else:
    print("The predicted price is equal to the actual price")
