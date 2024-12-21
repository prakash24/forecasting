import json
from datetime import datetime
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt    
import yfinance as yf
#from alpha_vantage.timeseries import TimeSeries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

def adjust_for_splits(prices, splits):
    adjusted_prices = prices.copy()  # Make a copy of the prices
    for date, ratio in splits.iterrows():
        split_date = ratio['splitDate']
        split_ratio = ratio['splitCoefficient']

        # If the date of the split is in our historical data, we adjust the prices
        if split_date in adjusted_prices.index:
            # Adjust prices by the split ratio
            adjusted_prices.loc[split_date:] = adjusted_prices.loc[split_date:] / split_ratio
    return adjusted_prices

def save_stock_data_to_file(data, symbol, meta_data=''):
    #make directory under stock_data directory
    dir_path = os.path.join('stock_data', symbol)
    os.makedirs(dir_path, exist_ok=True)
    #write data to file
    data.to_csv(os.path.join(dir_path, symbol +'.csv'), index=False)

    if meta_data is not None:
        #write metadata to metadata file
        with open(os.path.join(dir_path, symbol + "_metadata.json"), 'w') as file:
            json.dump(meta_data, file, indent=4)

def get_current_date():
    current_datetime = datetime.now()
    current_date = current_datetime.date()
    formatted_date = current_date.strftime('%Y-%m-%d')
    print("Formatted date:", formatted_date)
    return formatted_date

def adf_test(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print(f"Critical Values: {result[4]}")
    
    if result[1] <= 0.05:
        print("The series is stationary.")
        return True
    else:
        print("The series is not stationary. We need to difference the data.")
        return False

# Differencing until the series is stationary
def make_stationary(data):
    data_diff = data.diff().dropna()
    while not adf_test(data_diff):
        data_diff = data_diff.diff().dropna()
    return data_diff


def use_yfinance(symbol):
    print("Fetching data from Yahoo Finance")
    
    # Fetch historical stock data (AAPL for Apple)
    data = yf.download(symbol, start='2004-01-01', end=get_current_date()) #'2024-01-01')

    # Display the first few rows of the data
    print(data.tail())
    
    save_stock_data_to_file(data, symbol)

    # Plot the adjusted closing prices to visualize the data
    data['Adj Close'].plot(figsize=(10, 6))
    plt.title(f'{symbol} Adjusted Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price ($)')
    plt.show()


    series = data['Adj Close']
    diff_series = series
    while True:
        # Perform the Augmented Dickey-Fuller test
        result = adfuller(diff_series.dropna())  # Remove missing data
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print(f"Critical Values: {result[4]}")

        if result[1] <= 0.05:
            print("The series is stationary.")
            break
        else:
            print("The series is not stationary. We need to difference the data.")

        # Difference the series to make it stationary
        diff_series = diff_series.diff().dropna()
        # Check if the differenced data is stationary

    # Plot the differenced series
    plt.figure(figsize=(10, 6))
    diff_series.plot(title="Differenced {{symbol}} Adjusted Close Price")
    plt.xlabel('Date')
    plt.ylabel('Differenced Price')
    plt.show()

    # Use pmdarima to automatically find the best ARIMA model
    model_auto = auto_arima(diff_series, seasonal=False, stepwise=True, trace=True)

    print("***************** ARIMA MODEL *****************")
    # Print the model summary
    print(model_auto.summary())


    forecast_steps = 100
    print("Forecasting for the next " + str(forecast_steps) + " days****************")
    forecast_diff = model_auto.predict(n_periods=forecast_steps)
    
    # To get the forecast on the original scale, we add the last known value of the original series
    last_value = series.iloc[-1]

    # Convert the differenced forecast back to the original scale
    #forecast_original = last_value + np.cumsum(forecast_diff)  # Cumulative sum of the differenced forecast
    forecast_original = [last_value]
    for value in forecast_diff:
        forecast_original.append(forecast_original[-1] + value)

    forecast_original = forecast_original[1:]  # remove the first element (last known value)

    # Step 7: Create a corresponding time index for the forecasted values
    forecast_index = pd.date_range(start=series.index[-1], periods=forecast_steps+1, freq='B')[1:]

    # Step 8: Create a forecast series
    forecast_series = pd.Series(forecast_original, index=forecast_index)

    # Step 9: Plot the original series and forecasted values
    plt.figure(figsize=(10, 6))
    plt.plot(series, label='Historical Data')
    plt.plot(forecast_series, label='Forecasted Data', color='red')
    plt.title(f'{symbol} Stock Price Forecast using ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Price in $')
    plt.legend()
    plt.show()

    # Optionally, print the forecasted values
    print("Forecasted Values:")
    print(forecast_series)
    


# Take input from stdin (user input)
symbol = input("Enter Stock Symbol: ")
print("You entered:", symbol)
use_yfinance(symbol)



"""
def use_alpha_vintage(symbol):
    # Alpha Vantage API key
    api_key = "<<API_KEY>>"  # Replace with your Alpha Vantage API key

    # Initialize TimeSeries object
    ts = TimeSeries(key=api_key, output_format='pandas')

    # Fetch historical data for a stock (e.g., 'AAPL' for Apple)
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')  # 'full' gives 20 years of data0
    print(data.head())
    # get splits
    save_stock_data_to_file(data, meta_data, symbol)

    print("**************************************")

    adjusted_data, adjusted_meta_data = ts.get_daily_adjusted(symbol=symbol, outputsize='full')
    # Display the first few rows of the data
    print(adjusted_data.head())

    #plotting data without considering split
    
    # Select the 'close' price and ensure it is in datetime format
    df = data['4. close']
    # Optionally, we can visualize the closing prices
    df.plot(figsize=(10,6))
    plt.title(f"Stock Price ({symbol}) Over Time")
    plt.show()
    
    # Use the adjusted close price for the stock
    adjusted_close_prices = adjusted_data['5. adjusted close']

    # Plot the adjusted stock prices
    adjusted_close_prices.plot(figsize=(10,6))
    plt.title("Adjusted Apple Stock Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price ($)")
    plt.show()
"""
