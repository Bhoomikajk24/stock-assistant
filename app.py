import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Define the top companies and their ticker symbols
top_companies = {
    'apple': 'AAPL',
    'microsoft': 'MSFT',
    'google': 'GOOGL',
    'amazon': 'AMZN',
    'tesla': 'TSLA',
    'facebook': 'FB',
    'nvidia': 'NVDA',
    'berkshire hathaway': 'BRK.B',
    'visa': 'V',
    'jpmorgan': 'JPM',
    'unitedhealth': 'UNH',
    'procter & gamble': 'PG',
    'mastercard': 'MA',
    'coca-cola': 'KO',
    'pepsico': 'PEP',
    'walmart': 'WMT',
    'exxon mobil': 'XOM',
    'home depot': 'HD',
    'pfizer': 'PFE',
    'intel': 'INTC',
    'salesforce': 'CRM',
    'abbvie': 'ABBV',
    'broadcom': 'AVGO',
    'adobe': 'ADBE',
    'nike': 'NKE',
    'caterpillar': 'CAT',
    'starbucks': 'SBUX',
    'boeing': 'BA',
    'costco': 'COST',
    'merck': 'MRK',
    'thermo fisher': 'TMO'
}

# Function to get ticker from input
def get_company_name_from_input(input):
    for key in top_companies:
        if key in input.lower():
            return top_companies[key]
    return input  # Assume the input is a ticker symbol if not found

# Alpha Vantage API key
API_KEY = "K0AWNF84KI0CAX1L"

# Function to get stock price data
def get_stock_price(ticker):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval=5min&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    
    try:
        last_price = float(list(data["Time Series (5min)"].values())[0]["4. close"])
        return {
            'ticker': ticker,
            'price': last_price
        }
    except KeyError:
        return None

# Fetch historical data for the past year
def get_historical_data(ticker):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    
    try:
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index", dtype=float)
        df.columns = ["open", "high", "low", "close", "volume"]
        df = df.sort_index()
        return df
    except KeyError:
        return None

# Simple AI models for stock prediction
def predict_with_ai(df):
    # Prepare data
    df["Date"] = pd.to_datetime(df.index)
    df["Date_ordinal"] = df["Date"].map(pd.Timestamp.toordinal)
    X = df[["Date_ordinal"]].values
    y = df["close"].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    
 # Moving Average model (simple model to predict next day based on average of past n days)
    df["Moving_Avg"] = df["close"].rolling(window=5).mean()
    y_pred_ma = df["Moving_Avg"].iloc[-1]  # Next day's price prediction based on the moving average

    return y_pred_lr, y_pred_ma, y_test, y

# Streamlit UI
st.title("Stock Price App with AI Predictions")
user_input = st.text_input("Ask about a stock...")

if st.button("Get Stock Price"):
    company_name = get_company_name_from_input(user_input)
    stock_data = get_stock_price(company_name)

    if stock_data:
        st.write(f"The current price of **{company_name}** ({stock_data['ticker']}) is **${stock_data['price']:.2f}**")
    else:
        st.error("Could not retrieve stock price data. Please try again.")
        
if st.button("Get Technical Indicators and Historical Data"):
    company_name = get_company_name_from_input(user_input)
    df = get_historical_data(company_name)

    if df is not None:
        st.write(f"Displaying historical data for **{company_name}**")
        st.line_chart(df["close"])  # Display close prices over time

        # Display AI model predictions
        y_pred_lr, y_pred_ma, y_test, y = predict_with_ai(df)
        st.write(f"**Linear Regression Prediction for test data:** {y_pred_lr[-5:]}")
        st.write(f"**Moving Average Prediction for next day:** {y_pred_ma}")

        # Display Technical Indicators (RSI, EMA, SMA, MACD)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        ema = df["close"].ewm(span=20, adjust=False).mean()
        sma = df["close"].rolling(window=50).mean()
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        
        st.write(f"The current RSI is **{rsi.iloc[-1]:.2f}**")
        st.write(f"The current EMA is **{ema.iloc[-1]:.2f}**")
        st.write(f"The current SMA is **{sma.iloc[-1]:.2f}**")
        st.write(f"The current MACD is **{macd.iloc[-1]:.2f}**")
    else:
        st.error("Could not retrieve historical data. Please try again.")
