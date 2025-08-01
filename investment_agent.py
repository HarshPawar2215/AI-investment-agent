import streamlit as st
import yfinance as yf
import requests
import pandas as pd
from datetime import date, timedelta

# --- NEW: Load API Key from secrets.toml ---
try:
    openrouter_key = st.secrets["openrouter_key"]
except KeyError:
    st.error("❌ OpenRouter API Key not found. Please add your key to `.streamlit/secrets.toml`.")
    st.stop()

st.title("AI Investment Agent 📈🤖")
st.caption("Live stock comparison using Mistral-7B via OpenRouter")

def format_large_number(number, currency_symbol):
    if number is None or number == "N/A":
        return "N/A"
    
    if currency_symbol == "₹":
        if number >= 10**7:
            return f"{number / 10**7:.2f} Cr"
        elif number >= 10**5:
            return f"{number / 10**5:.2f} L"
        else:
            return f"{number:.2f}"
    else:
        if number >= 10**12:
            return f"{number / 10**12:.2f}T"
        elif number >= 10**9:
            return f"{number / 10**9:.2f}B"
        elif number >= 10**6:
            return f"{number / 10**6:.2f}M"
        else:
            return f"{number:.2f}"

@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info.get("longName"):
            return None

        currency = info.get("currency", "N/A")
        currency_map = {
            "INR": "₹",
            "USD": "$"
        }
        currency_symbol = currency_map.get(currency, currency)

        hist = stock.history(period="1y") 
        financials = stock.financials.loc[['Total Revenue', 'Gross Profit']].transpose().iloc[0]
        summary = info.get("longBusinessSummary", "No summary available.")
        
        today = date.today()
        one_year_ago = today - timedelta(days=365)
        five_years_ago = today - timedelta(days=365 * 5)
        
        hist_1y = stock.history(start=one_year_ago, end=today)
        hist_5y = stock.history(start=five_years_ago, end=today)

        if not hist_1y.empty and not hist_5y.empty:
            price_1y_change = ((hist_1y['Close'].iloc[-1] - hist_1y['Close'].iloc[0]) / hist_1y['Close'].iloc[0]) * 100
            price_5y_change = ((hist_5y['Close'].iloc[-1] - hist_5y['Close'].iloc[0]) / hist_5y['Close'].iloc[0]) * 100
        else:
            price_1y_change, price_5y_change = "N/A", "N/A"

        return {
            "symbol": ticker,
            "longName": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "price": info.get("currentPrice", "N/A"),
            "marketCap": info.get("marketCap", "N/A"),
            "peRatio": info.get("trailingPE", "N/A"),
            "forwardPERatio": info.get("forwardPE", "N/A"),
            "pbRatio": info.get("priceToBook", "N/A"),
            "dividendYield": info.get("dividendYield", "N/A"),
            "beta": info.get("beta", "N/A"),
            "annualRevenue": financials.get('Total Revenue', "N/A"),
            "grossProfit": financials.get('Gross Profit', "N/A"),
            "longBusinessSummary": summary,
            "price1yChange": price_1y_change,
            "price5yChange": price_5y_change,
            "history": hist.tail(30)[["Close"]],
            "currency_symbol": currency_symbol
        }
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def generate_prompt(data1, data2):
    mc1 = format_large_number(data1['marketCap'], data1['currency_symbol'])
    rev1 = format_large_number(data1['annualRevenue'], data1['currency_symbol'])
    gp1 = format_large_number(data1['grossProfit'], data1['currency_symbol'])
    
    mc2 = format_large_number(data2['marketCap'], data2['currency_symbol'])
    rev2 = format_large_number(data2['annualRevenue'], data2['currency_symbol'])
    gp2 = format_large_number(data2['grossProfit'], data2['currency_symbol'])
    
    return f"""
You are a highly experienced financial analyst. Compare the following two stocks and give a detailed investment analysis based on the provided data.

### Stock 1: {data1['longName']} ({data1['symbol']})
- Sector: {data1['sector']}
- Current Price: {data1['currency_symbol']}{data1['price']:.2f}
- Market Cap: {data1['currency_symbol']}{mc1}
- P/E Ratio (Trailing): {data1['peRatio']}
- P/E Ratio (Forward): {data1['forwardPERatio']}
- P/B Ratio: {data1['pbRatio']}
- Dividend Yield: {data1['dividendYield']}
- Beta: {data1['beta']}
- Annual Revenue: {data1['currency_symbol']}{rev1}
- Gross Profit: {data1['currency_symbol']}{gp1}
- 1-Year Price Change: {data1['price1yChange']:.2f}%
- 5-Year Price Change: {data1['price5yChange']:.2f}%

Company Summary:
{data1['longBusinessSummary']}

Recent 30-Day Closing Prices:
{data1['history'].to_markdown()}

---

### Stock 2: {data2['longName']} ({data2['symbol']})
- Sector: {data2['sector']}
- Current Price: {data2['currency_symbol']}{data2['price']:.2f}
- Market Cap: {data2['currency_symbol']}{mc2}
- P/E Ratio (Trailing): {data2['peRatio']}
- P/E Ratio (Forward): {data2['forwardPERatio']}
- P/B Ratio: {data2['pbRatio']}
- Dividend Yield: {data2['dividendYield']}
- Beta: {data2['beta']}
- Annual Revenue: {data2['currency_symbol']}{rev2}
- Gross Profit: {data2['currency_symbol']}{gp2}
- 1-Year Price Change: {data2['price1yChange']:.2f}%
- 5-Year Price Change: {data2['price5yChange']:.2f}%

Company Summary:
{data2['longBusinessSummary']}

Recent 30-Day Closing Prices:
{data2['history'].to_markdown()}

---

Based on this data, provide a detailed comparison and a clear, actionable investment recommendation. Use Markdown formatting to organize your response with sections like "Valuation Analysis", "Growth & Profitability", "Risk & Outlook", and a final "Conclusion & Recommendation".
"""

def call_openrouter(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful financial analyst."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Error: {response.status_code} - {response.text}"

col1, col2 = st.columns(2)
with col1:
    stock1 = st.text_input("Enter first stock symbol (e.g. TSLA or RELIANCE.NS)")
with col2:
    stock2 = st.text_input("Enter second stock symbol (e.g. MSFT or HDFCBANK.NS)")

if stock1 and stock2:
    with st.spinner("Fetching stock data and generating report..."):
        try:
            data1 = fetch_stock_data(stock1.upper())
            data2 = fetch_stock_data(stock2.upper())

            if data1 is None or data2 is None:
                st.error("❌ Failed to fetch data for one or both stocks. Please ensure the ticker symbols are correct, including exchange suffixes for non-US stocks (e.g., '.NS' for India).")
            else:
                st.subheader("📊 Stock Data Summary")
                col_data1, col_data2 = st.columns(2)

                with col_data1:
                    st.subheader(f"{data1['longName']} ({data1['symbol']})")
                    st.markdown(f"**Sector:** {data1['sector']}")
                    st.metric("Current Price", f"{data1['currency_symbol']}{data1['price']:.2f}")
                    st.metric("Market Cap", f"{data1['currency_symbol']}{format_large_number(data1['marketCap'], data1['currency_symbol'])}")
                    st.metric("P/E Ratio", f"{data1['peRatio']:.2f}")
                    st.metric("1-Year Price Change", f"{data1['price1yChange']:.2f}%")
                    st.line_chart(data1['history'])

                with col_data2:
                    st.subheader(f"{data2['longName']} ({data2['symbol']})")
                    st.markdown(f"**Sector:** {data2['sector']}")
                    st.metric("Current Price", f"{data2['currency_symbol']}{data2['price']:.2f}")
                    st.metric("Market Cap", f"{data2['currency_symbol']}{format_large_number(data2['marketCap'], data2['currency_symbol'])}")
                    st.metric("P/E Ratio", f"{data2['peRatio']:.2f}")
                    st.metric("1-Year Price Change", f"{data2['price1yChange']:.2f}%")
                    st.line_chart(data2['history'])

                st.subheader("🤖 AI Investment Analysis")
                prompt = generate_prompt(data1, data2)
                # Note: call_openrouter no longer takes the api_key as an argument
                result = call_openrouter(prompt) 
                st.markdown(result)
        except Exception as e:
            st.error(f"❌ Failed to generate report: {e}")