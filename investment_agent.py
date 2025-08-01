import streamlit as st
import yfinance as yf
import requests
import pandas as pd

st.title("AI Investment Agent 📈🤖")
st.caption("Live stock comparison using Mistral-7B via OpenRouter")

openrouter_key = st.text_input("OpenRouter API Key", type="password")

@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="6mo")
    return {
        "symbol": ticker,
        "longName": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "price": info.get("currentPrice", "N/A"),
        "marketCap": info.get("marketCap", "N/A"),
        "peRatio": info.get("trailingPE", "N/A"),
        "dividendYield": info.get("dividendYield", "N/A"),
        "history": hist.tail(5)[["Close"]]
    }

def generate_prompt(data1, data2):
    return f"""
Compare the following two stocks and give an investment analysis:

### Stock 1: {data1['longName']} ({data1['symbol']})
- Sector: {data1['sector']}
- Current Price: ${data1['price']}
- Market Cap: {data1['marketCap']}
- PE Ratio: {data1['peRatio']}
- Dividend Yield: {data1['dividendYield']}

Recent 5-Day Closing Prices:
{data1['history'].to_markdown()}

---

### Stock 2: {data2['longName']} ({data2['symbol']})
- Sector: {data2['sector']}
- Current Price: ${data2['price']}
- Market Cap: {data2['marketCap']}
- PE Ratio: {data2['peRatio']}
- Dividend Yield: {data2['dividendYield']}

Recent 5-Day Closing Prices:
{data2['history'].to_markdown()}

---

Which stock is a better buy and why? Give markdown-formatted investment advice.
"""

def call_openrouter(prompt, api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful financial analyst."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Error: {response.status_code} - {response.text}"

if openrouter_key:
    col1, col2 = st.columns(2)
    with col1:
        stock1 = st.text_input("Enter first stock symbol (e.g. AAPL)")
    with col2:
        stock2 = st.text_input("Enter second stock symbol (e.g. MSFT)")

    if stock1 and stock2:
        with st.spinner("Fetching stock data and generating report..."):
            try:
                data1 = fetch_stock_data(stock1.upper())
                data2 = fetch_stock_data(stock2.upper())
                prompt = generate_prompt(data1, data2)
                result = call_openrouter(prompt, openrouter_key)
                st.markdown(result)
            except Exception as e:
                st.error(f"❌ Failed to generate report: {e}")
print("hello")
print("Harsh mkc")