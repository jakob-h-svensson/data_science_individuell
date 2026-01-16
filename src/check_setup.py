import yfinance as yf
import pandas as pd

# Define the dictionary
omx30 = {
    "Volvo B": "VOLV-B.ST",
    "Atlas Copco A": "ATCO-A.ST"
}

print(f"Dictionary defined: {omx30}")

try:
    print("Fetching data for Volvo B...")
    # Use the dictionary
    ticker_symbol = omx30["Volvo B"]
    stock = yf.Ticker(ticker_symbol)
    
    # Fetch history
    data = stock.history(period="1mo")
    
    if data.empty:
        print("Data is empty (might be a network/symbol issue), but code executed.")
    else:
        print("Data fetched successfully!")
        print(data.head())

except NameError as e:
    print(f"Caught expected NameError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
