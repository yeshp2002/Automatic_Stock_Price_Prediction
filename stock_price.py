import yfinance as yf
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression

# Define the ticker symbol for the stock you want to predict
ticker_symbol = 'AAPL'

# Retrieve historical stock data using yfinance
end_date = date.today() - timedelta(days=1)  # Exclude current day
start_date = end_date - timedelta(days=365)  # Retrieve data for the past year
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Prepare the data for model training and prediction
data['NextDayClose'] = data['Close'].shift(-1)  # Create target variable shifted by 1 day
data.dropna(inplace=True)  # Remove rows with missing values

# Split the data into features (X) and target (y)
X = data[['Close']]
y = data['NextDayClose']

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get today's stock price
today = date.today()
today_data = yf.download(ticker_symbol, start=today, end=today)
today_stock_price = today_data['Close'].iloc[0]

# Predict today's stock price
predicted_stock_price = model.predict([[today_stock_price]])

# Compare predicted price with today's stock price
print(f"Predicted stock price for today: {predicted_stock_price[0]}")
print(f"Actual stock price for today: {today_stock_price}")

# Calculate the difference between predicted and actual prices
price_difference = predicted_stock_price[0] - today_stock_price
if price_difference > 0:
    print(f"The predicted price is higher than the actual price by {price_difference}")
elif price_difference < 0:
    print(f"The predicted price is lower than the actual price by {-price_difference}")
else:
    print("The predicted price is equal to the actual price")
