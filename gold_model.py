# === Step 1: Import Libraries ===
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# === Set ticker symbols ===
gold = yf.download("GC=F", start="2023-01-01", end=None)  # Gold Futures
usd = yf.download("DX-Y.NYB", start="2023-01-01", end=None)  # US Dollar Index
oil = yf.download("CL=F", start="2023-01-01", end=None)  # Crude Oil Futures

# === Align dates and clean data ===
df = pd.DataFrame({
    'Date': gold.index,
    'Gold_Price': gold['Adj Close'],
    'USD_Index': usd['Adj Close'],
    'Oil_Price': oil['Adj Close']
})

# Drop rows with any missing data
df.dropna(inplace=True)

# Save to CSV
df.to_csv('gold_data.csv', index=False)
print("✅ Live gold_data.csv has been created using real market data.")

# === Step 2: Load Your Data ===
# Replace 'gold_data.csv' with your actual file name
df = pd.read_csv('gold_data.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
df = df.sort_index()

# === Step 3: Compute Log Returns ===
df['Gold_Return'] = np.log(df['Gold_Price']).diff()
df['USD_Return'] = np.log(df['USD_Index']).diff()
df['Oil_Return'] = np.log(df['Oil_Price']).diff()
df.dropna(inplace=True)

# === Step 4: Stationarity Check ===
result = adfuller(df['Gold_Return'])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# === Step 5: Train-Test Split ===
train = df.iloc[:-100]
test = df.iloc[-100:]

# === Step 6: Build ARIMAX Model ===
exog_train = train[['USD_Return', 'Oil_Return']]
exog_test = test[['USD_Return', 'Oil_Return']]

model = SARIMAX(train['Gold_Return'], exog=exog_train, order=(1, 0, 1))
results = model.fit(disp=False)
print(results.summary())

# === Step 7: Forecast ===
forecast = results.get_forecast(steps=100, exog=exog_test)
predicted = forecast.predicted_mean
true = test['Gold_Return']

# === Step 8: Evaluate Forecast ===
rmse = np.sqrt(mean_squared_error(true, predicted))
print(f'RMSE: {rmse:.6f}')

# === Step 9: Plot Forecast ===
plt.figure(figsize=(12, 6))
plt.plot(true.index, true, label='Actual Returns')
plt.plot(predicted.index, predicted, label='Forecasted Returns', linestyle='--')
plt.title('Gold Price Returns Forecast (ARIMAX)')
plt.legend()
plt.show()
