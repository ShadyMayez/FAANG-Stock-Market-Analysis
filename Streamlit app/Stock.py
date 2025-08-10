import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import concurrent.futures
from sklearn.linear_model import LinearRegression


# Define the dataset paths for each ticker
datasets = {
    'AMZN - Amazon': r'https://raw.githubusercontent.com/madhan96p/FAANG/refs/heads/main/FAANG_CSV/amazon_data.csv',
    'AAPL - Apple': r'https://raw.githubusercontent.com/madhan96p/FAANG/refs/heads/main/FAANG_CSV/apple_data.csv',
    'GOOGL - Google': r'https://raw.githubusercontent.com/madhan96p/FAANG/refs/heads/main/FAANG_CSV/google_data.csv',
    'NFLX - Netflix': r'https://raw.githubusercontent.com/madhan96p/FAANG/refs/heads/main/FAANG_CSV/netflix_data.csv',
    'META - Facebook': r'https://raw.githubusercontent.com/madhan96p/FAANG/refs/heads/main/FAANG_CSV/facebook_data.csv',
}

# Title Section
st.markdown("<center><h1>FAANG Stock Market</h1></center>",True)

def convert_large_numbers(value):
    if value >= 1e12:
        return f"{value / 1e12:.2f}T"
    elif value >= 1e9:
        return f"{value / 1e9:.2f}B"
    elif value >= 1e6:
        return f"{value / 1e6:.2f}M"
    elif value >= 1e3:
        return f"{value / 1e3:.2f}K"
    else:
        return str(value)
marquee_data = ""


marquee_data = ""
for stock, url in datasets.items():
    try:
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df['Date'] = df['Date'].dt.strftime('%d-%m-%Y')
        latest_row = df.iloc[-1]

        company = str(latest_row.get('Company', stock)).strip()
        close_price = float(latest_row.get('Close', 0))
        last_date = latest_row.get('Date', 'N/A')

        marquee_data += f"{company} | {close_price:.2f} USD | {last_date} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
    except Exception as e:
        print(f"Error processing {stock}: {e}")

# Display moving marquee
st.markdown(
    f"""
    <style>
    .marquee {{
        white-space: nowrap;
        overflow: hidden;
        box-sizing: border-box;
    }}
    .marquee span {{
        display: inline-block;
        padding-left: 100%;
        animation: marquee 20s linear infinite;
        color: white;
        font-size: 18px;
        font-family: Arial;
    }}
    @keyframes marquee {{
        0%   {{ transform: translateX(0%); }}
        100% {{ transform: translateX(-100%); }}
    }}
    </style>
    <div class="marquee"><span>{marquee_data}</span></div>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose Analysis Stage:", ["📊 Stage 1: Individual Company", "📈 Stage 2: Company Comparison"])

# TECHNICAL INDICATOR FUNCTIONS
def compute_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    """Calculate MACD and Signal Line"""
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def find_support_resistance(price_data, window=20):
    """Identify dynamic support/resistance levels"""
    support = price_data['Close'].rolling(window).min()
    resistance = price_data['Close'].rolling(window).max()
    return support, resistance

def compute_volatility(data, window=20):
    """Calculate rolling volatility"""
    return data['Close'].pct_change().rolling(window=window).std()

def compute_bollinger_bands(data, window=20):
    """Calculate Bollinger Bands"""
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    return sma, upper_band, lower_band

# MAIN APP CODE
if page == "📊 Stage 1: Individual Company":
    st.sidebar.header("Select a Stock to Analyze")
    selected_stock = st.sidebar.selectbox("Choose a stock:", list(datasets.keys()))

    # Data Loading and Preparation
    df = pd.read_csv(datasets[selected_stock])
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    
    # Timeframe Selection
    timeframe = st.radio(
        "Select Analysis Period", 
        ['5D', '1M', '1Y', '5Y', 'Max'], 
        horizontal=True,
        help="Choose the time horizon for technical analysis"
    )

    # Apply timeframe filter
    if timeframe == '5D':
        df_filtered = df.tail(5)
    elif timeframe == '1M':
        df_filtered = df.tail(30)
    elif timeframe == '1Y':
        df_filtered = df.tail(252)
    elif timeframe == '5Y':
        df_filtered = df.tail(1260)
    else:
        df_filtered = df

    # Calculate Technical Indicators
    df_filtered['RSI'] = compute_rsi(df_filtered)
    df_filtered['MACD'], df_filtered['MACD_Signal'] = compute_macd(df_filtered)
    df_filtered['Support'], df_filtered['Resistance'] = find_support_resistance(df_filtered)
    df_filtered['Volatility'] = compute_volatility(df_filtered)
    df_filtered['BB_MA'], df_filtered['BB_Upper'], df_filtered['BB_Lower'] = compute_bollinger_bands(df_filtered)
    
    # Price Performance Header
    latest_data = df_filtered.iloc[-1]
    price = latest_data['Close']
    change = price - df_filtered.iloc[0]['Close'] if len(df_filtered) > 1 else 0
    percent_change = (change / df_filtered.iloc[0]['Close']) * 100 if len(df_filtered) > 1 else 0
    
    st.markdown(f"""
    ### 📈 {selected_stock} Performance Summary
    <div style="background-color:#1e1e1e;padding:20px;border-radius:10px;margin-bottom:20px;">
        <span style="color:{'green' if change > 0 else 'red'};font-size:28px;">
            ${price:.2f} 
        </span>
        <span style="color:{'green' if change > 0 else 'red'};font-size:24px;">
            {'+' if change > 0 else ''}{change:.2f} ({'+' if change > 0 else ''}{percent_change:.2f}%)
        </span>
        <p style="color:#aaa;margin-top:8px;">
            {df_filtered.iloc[0]['Date'].strftime('%b %d, %Y')} → {latest_data['Date'].strftime('%b %d, %Y')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # MAIN PRICE CHART
    st.markdown("### 📊 Price Trend Analysis")
    fig_price = px.line(
        df_filtered, 
        x="Date", 
        y="Close", 
        title=f"{selected_stock} Closing Prices",
        labels={"Close": "Price (USD)", "Date": "Date"},
        template="plotly_dark"
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # CANDLESTICK ANALYSIS (without tabs)
    st.markdown("## 🕯️ Candlestick Analysis")
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df_filtered['Date'],
        open=df_filtered['Open'],
        high=df_filtered['High'],
        low=df_filtered['Low'],
        close=df_filtered['Close'],
        increasing_line_color='#2ECC71',
        decreasing_line_color='#E74C3C'
    )])
    fig_candle.update_layout(
        title=f"{selected_stock} Price Action",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    st.markdown("""
    **Candlestick Patterns - What They Show:**
    - **Green Candles:** Price closed higher than it opened (bullish)
    - **Red Candles:** Price closed lower than it opened (bearish)
    - **Wicks/Shadows:** Show price rejection (highs/lows that didn't hold)
    - **Common Patterns:**
    - **Hammer/Hanging Man:** Potential reversals
    - **Engulfing Patterns:** Strong momentum shifts
    - **Doji:** Indecision in the market
    """)
        # TECHNICAL ANALYSIS SECTION
    st.markdown("## 📉 Technical Indicators")

    # Create tabs for technical indicators
    tab_rsi, tab_sr, tab_macd = st.tabs([
        "🔄 RSI", 
        "⚖️ Support/Resistance", 
        "📶 MACD"
    ])

    with tab_rsi:
        st.markdown("### Relative Strength Index (RSI)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df_filtered['Date'], 
            y=df_filtered['RSI'], 
            name='RSI', 
            line=dict(color='#F39C12')
        ))
        fig_rsi.add_hline(y=70, line=dict(color='red', dash='dash'), annotation_text='Overbought (70)')
        fig_rsi.add_hline(y=30, line=dict(color='green', dash='dash'), annotation_text='Oversold (30)')
        fig_rsi.update_layout(yaxis_range=[0,100], template="plotly_dark")
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        st.markdown("""
        **RSI (Relative Strength Index) - What It Shows:**
        - Measures speed and change of price movements (0-100 scale)
        - **Above 70:** Overbought (potential pullback)
        - **Below 30:** Oversold (potential bounce)
        - **Divergences:** When price makes new highs but RSI doesn't, may signal reversal
        """)

    with tab_sr:
        st.markdown("### Support & Resistance Levels")
        fig_sr = go.Figure()
        fig_sr.add_trace(go.Scatter(
            x=df_filtered['Date'], 
            y=df_filtered['Close'], 
            name='Price', 
            line=dict(color='white'))
        )
        fig_sr.add_trace(go.Scatter(
            x=df_filtered['Date'], 
            y=df_filtered['Support'], 
            name='Support', 
            line=dict(color='#2ECC71', dash='dash'))
        )
        fig_sr.add_trace(go.Scatter(
            x=df_filtered['Date'], 
            y=df_filtered['Resistance'], 
            name='Resistance', 
            line=dict(color='#E74C3C', dash='dash'))
        )
        fig_sr.update_layout(template="plotly_dark")
        st.plotly_chart(fig_sr, use_container_width=True)
        
        st.markdown("""
        **Support & Resistance - What It Shows:**
        - **Support (Green):** Price level where buying interest emerges
        - **Resistance (Red):** Price level where selling pressure increases
        - **Breakouts:** When price moves through these levels with conviction
        - **Bounces:** Price reactions at these levels can signal trading opportunities
        """)

    with tab_macd:
        st.markdown("### MACD (Moving Average Convergence Divergence)")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=df_filtered['Date'], 
            y=df_filtered['MACD'], 
            name='MACD', 
            line=dict(color='#3498DB')
        ))
        fig_macd.add_trace(go.Scatter(
            x=df_filtered['Date'], 
            y=df_filtered['MACD_Signal'], 
            name='Signal Line', 
            line=dict(color='#E74C3C')
        ))
        fig_macd.update_layout(template="plotly_dark")
        st.plotly_chart(fig_macd, use_container_width=True)
        
        st.markdown("""
        **MACD - What It Shows:**
        - **Blue Line (MACD):** Difference between 12-day and 26-day EMAs
        - **Red Line (Signal):** 9-day EMA of MACD line
        - **Crossovers:** When MACD crosses above/below Signal line
        - **Zero Line Crosses:** When MACD crosses above/below zero
        - **Divergences:** Can signal weakening trends
        """)



    # --- VOLUME ANALYSIS ---
    st.markdown("### 📊 Volume & Liquidity Analysis")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=df_filtered['Date'],
        y=df_filtered['Volume'],
        name='Volume',
        marker_color='#9B59B6'
    ))
    fig_vol.update_layout(
        title='Trading Volume',
        template="plotly_dark"
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    st.info("""
    **Volume Signals:**  
    - High volume on up days = Strong buying interest
    - High volume on down days = Strong selling pressure
    - Volume spikes often precede big price moves
    """)

    # --- FUNDAMENTAL METRICS ---
    st.markdown("### 🧾 Fundamental Analysis Metrics")

    fundamental_data = {
        "Metric": [
            "P/E Ratio",
            "EPS (Earnings Per Share)",
            "Revenue Growth",
            "Net Profit Margin",
            "ROE (Return on Equity)",
            "Debt to Equity", 
            "Dividend Yield",
            "Free Cash Flow"
        ],
        "Value": [
            f"{latest_data['PE Ratio']:.2f}" if 'PE Ratio' in latest_data else "N/A",
            f"{latest_data['EPS']:.2f} USD" if 'EPS' in latest_data else "N/A",
            f"{latest_data['Revenue Growth'] * 100:.2f}%" if 'Revenue Growth' in latest_data else "N/A",
            f"{latest_data['Net Profit Margin'] * 100:.2f}%" if 'Net Profit Margin' in latest_data else "N/A",
            f"{latest_data['Return on Equity (ROE)'] * 100:.2f}%" if 'Return on Equity (ROE)' in latest_data else "N/A",
            f"{latest_data['Debt to Equity']:.2f}" if 'Debt to Equity' in latest_data else "N/A",
            f"{latest_data['Dividend Yield'] * 100:.2f}%" if 'Dividend Yield' in latest_data else "N/A",
            f"${latest_data['Free Cash Flow'] / 1e9:.2f}B" if 'Free Cash Flow' in latest_data else "N/A"
        ],
        "Description": [
            "Price-to-Earnings ratio shows valuation relative to earnings (lower is typically better)",
            "Earnings per share indicates profitability per outstanding share",
            "Year-over-year revenue growth percentage (higher = faster growth)",
            "Percentage of revenue that becomes net profit (measures efficiency)",
            "Measures profitability relative to shareholder equity",
            "Compares company's debt to its equity (lower = less leveraged)",
            "Annual dividend payment as percentage of stock price", 
            "Cash generated after operating expenses and capital expenditures"
        ]
    }

    fundamental_df = pd.DataFrame(fundamental_data)
    st.dataframe(fundamental_df)

    st.markdown("## 📊 Quantitative Analysis")

    # Initialize variables
    merged = None
    beta = None
    alpha = None

    # Load market data
    try:
        spy = yf.download('^GSPC', 
                        start=df_filtered['Date'].min().strftime('%Y-%m-%d'),
                        end=pd.Timestamp.today(),
                        progress=False)
        
        if not spy.empty:
            spy_price = spy[['Adj Close']].rename(columns={'Adj Close': 'SP500'}) if 'Adj Close' in spy.columns else spy[['Close']].rename(columns={'Close': 'SP500'})
            
            # Calculate returns
            spy_price['Return'] = spy_price['SP500'].pct_change()
            stock_returns = df_filtered.set_index('Date')['Close'].pct_change().dropna()
            
            # Merge datasets
            merged = pd.merge(
                stock_returns.rename("Stock Return"),
                spy_price['Return'].rename("Market Return"), 
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            # Calculate alpha and beta if enough data
            if len(merged) > 1:
                X = merged[['Market Return']].values
                y = merged['Stock Return'].values
                model = LinearRegression().fit(X, y)
                beta = model.coef_[0]
                alpha = model.intercept_

    except Exception as e:
        st.warning(f"⚠️ Error loading market data: {str(e)}")

    # Display CAPM metrics if available
    if beta is not None and alpha is not None:
        st.markdown("### 📈 CAPM Metrics (Risk Analysis)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Beta (Market Sensitivity)", f"{beta:.2f}")
            st.info("""
            **Beta Interpretation:**
            - β = 1: Moves with the market  
            - β > 1: More volatile than market
            - β < 1: Less volatile than market
            - Negative β: Moves opposite to market
            """)
        
        with col2:
            st.metric("Alpha (Excess Return)", f"{alpha:.4f}")
            st.info("""
            **Alpha Interpretation:**
            - α > 0: Outperforming expectations
            - α = 0: Meeting expectations
            - α < 0: Underperforming
            - Higher = better risk-adjusted returns
            """)
    elif merged is not None and len(merged) <= 1:
        st.warning("⚠️ Not enough overlapping trading days to calculate Alpha/Beta")
    else:
        st.warning("⚠️ Market benchmark data unavailable for comparison")

        # --- Compute Additional KPIs ---
    from sklearn.linear_model import LinearRegression
    import numpy as np

    st.markdown("## 🧠 Key Performance Analysis (KPA)")

    # Example calculations (ensure these columns exist and are not NaN)
    close_price = df_filtered['Close'].iloc[-1]
    pe_ratio = latest_data['PE Ratio']
    eps = latest_data['EPS']
    market_cap = latest_data['Market Cap']
    volume = latest_data['Volume']
    target_price = latest_data.get('Target Price', None)

    # KPA 1: Is the stock currently overvalued or undervalued?
    st.markdown("###  Is the stock currently overvalued or undervalued?")
    if pe_ratio < 15:
        valuation = "undervalued"
    elif pe_ratio > 30:
        valuation = "overvalued"
    else:
        valuation = "fairly valued"
    st.markdown(f"**Answer:** The stock's P/E ratio is **{pe_ratio:.2f}**, which suggests it is **{valuation}**.")
    st.info(" **Explanation:** A lower P/E ratio (under 15) may indicate the stock is undervalued, while a higher P/E (over 30) might mean it's overvalued relative to earnings.")

    # KPA 2: Is the company profitable?
    st.markdown("###  Is the company profitable?")
    if eps > 0:
        st.markdown(f"**Answer:** Yes, the company has a positive EPS of **{eps:.2f}**.")
    else:
        st.markdown(f"**Answer:** No, the company has a negative EPS of **{eps:.2f}**.")
    st.info(" **Explanation:** Earnings per Share (EPS) indicates how much profit is allocated per share. Positive EPS means the company is generating profit.")

    # KPA 3: Is the stock price likely to grow?
    st.markdown("###  Is the stock price likely to grow?")
    if target_price:
        price_diff = target_price - close_price
        direction = "increase" if price_diff > 0 else "decrease"
        st.markdown(f"**Answer:** Analyst target price is **{target_price:.2f}**, suggesting a potential **{direction}** of **{abs(price_diff):.2f} USD**.")
    else:
        st.markdown("**Answer:** Target price data is not available.")
    st.info(" **Explanation:** Target prices are analyst predictions for future stock value. A higher target than current price suggests expected growth.")

    # KPA 4: Is the stock actively traded?
    st.markdown("### 📊 Is the stock actively traded?")
    st.markdown(f"**Answer:** The average daily trading volume is **{convert_large_numbers(volume)}**.")
    st.info(" **Explanation:** High volume indicates strong investor interest and better liquidity, making it easier to buy or sell the stock.")

    import numpy as np

    # --- ATR ---
    df_filtered['TR'] = np.maximum(
        df_filtered['High'] - df_filtered['Low'],
        np.maximum(
            abs(df_filtered['High'] - df_filtered['Close'].shift(1)),
            abs(df_filtered['Low'] - df_filtered['Close'].shift(1))
        )
    )
    df_filtered['ATR_14'] = df_filtered['TR'].rolling(window=14).mean()
    fig_atr = go.Figure()
    fig_atr.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['ATR_14'],
                                mode='lines', name='ATR (14)', line=dict(color='orange')))
    fig_atr.update_layout(title="ATR (Average True Range - 14 days)",
                        xaxis_title="Date", yaxis_title="ATR Value",
                        template="plotly_dark")
    st.plotly_chart(fig_atr)
    st.info(
        "**What does the Average True Range (ATR) tell us?**\n\n"
        "ATR measures how much a stock typically moves (up or down) in a given day.\n\n"
        "**Why is it useful?**\n"
        "- It quantifies **volatility** — not direction.\n"
        "- A high ATR means big daily swings (more risk/opportunity), while a low ATR means the stock is stable.\n"
        "**Use it to:** Set stop-loss levels or gauge if a stock fits your risk tolerance."
    )


    # --- Golden/Death Cross ---
    df_filtered['MA_50'] = df_filtered['Close'].rolling(window=50).mean()
    df_filtered['MA_200'] = df_filtered['Close'].rolling(window=200).mean()

    fig_cross = go.Figure()
    fig_cross.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Close'], name='Close', line=dict(color='white')))
    fig_cross.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['MA_50'], name='MA 50', line=dict(color='blue')))
    fig_cross.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['MA_200'], name='MA 200', line=dict(color='red')))
    fig_cross.update_layout(title="Golden Cross / Death Cross Detection",
                            xaxis_title="Date", yaxis_title="Price",
                            template="plotly_dark")
    st.plotly_chart(fig_cross)
    st.info(
        "**What are the Golden Cross and Death Cross signals?**\n\n"
        "These are momentum-based signals based on moving averages:\n"
        "- **Golden Cross**: When the 50-day MA crosses **above** the 200-day MA → **Bullish signal**.\n"
        "- **Death Cross**: When the 50-day MA crosses **below** the 200-day MA → **Bearish signal**.\n\n"
        "**Why is it useful?**\n"
        "- It helps investors spot long-term trend changes.\n"
        "**Use it to:** Confirm entry/exit points in longer-term strategies."
    )
    # --- OBV ---
    df_filtered['OBV'] = (np.sign(df_filtered['Close'].diff()) * df_filtered['Volume']).fillna(0).cumsum()
    fig_obv = go.Figure()
    fig_obv.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['OBV'],
                                mode='lines', name='OBV', line=dict(color='green')))
    fig_obv.update_layout(title="OBV (On-Balance Volume)",
                        xaxis_title="Date", yaxis_title="OBV Value",
                        template="plotly_dark")
    st.plotly_chart(fig_obv)
    st.info(
        "**What does the On-Balance Volume (OBV) indicator show?**\n\n"
        "OBV tracks the flow of volume in or out of a stock to detect hidden buying/selling pressure.\n\n"
        "**Why is it useful?**\n"
        "- Rising OBV with rising price = **strong bullish confirmation**.\n"
        "- Falling OBV while price rises = **bearish divergence** (price may reverse).\n"
        "**Use it to:** Confirm trends or detect when volume disagrees with price."
    )
        
    # --- Sharpe & Sortino Ratio ---
    daily_returns = df_filtered['Close'].pct_change().dropna()
    risk_free = 0.01 / 252
    sharpe = (daily_returns.mean() - risk_free) / daily_returns.std()
    sortino = (daily_returns.mean() - risk_free) / daily_returns[daily_returns < 0].std()

    st.markdown("## 📊 Advanced KPIs")
    st.write(f"**Sharpe Ratio:** {sharpe:.2f} — Measures excess return per unit of total risk.")
    st.info(
        "**What does the Sharpe Ratio tell us?**\n\n"
        "It measures how much excess return you're getting for each unit of total risk (volatility).\n\n"
        "**Why is it useful?**\n"
        "- A higher Sharpe Ratio (> 1.0) means your investment is performing well relative to its risk.\n"
        "- It considers both good and bad volatility equally.\n"
        "**Use it to:** Compare risk-adjusted returns across stocks or portfolios."
    )


    st.write(f"**Sortino Ratio:** {sortino:.2f} — Measures excess return per unit of *downside* risk.")
    st.info(
        "**How is Sortino Ratio different from Sharpe?**\n\n"
        "Sortino focuses only on **downside risk**, ignoring positive (upward) volatility.\n\n"
        "**Why is it useful?**\n"
        "- It gives a clearer picture when returns are volatile but mostly positive.\n"
        "- A higher Sortino means better returns with **less downside risk**.\n"
        "**Use it to:** Evaluate investments that may have high returns but only occasional dips."
    )
    # --- Monte Carlo Simulation ---
    simulations = 1000
    T = 252
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    last_price = df_filtered['Close'].iloc[-1]

    simulated_paths = np.zeros((T, simulations))
    for i in range(simulations):
        rand_returns = np.random.normal(mu, sigma, T)
        simulated_paths[:, i] = last_price * np.exp(np.cumsum(rand_returns))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(simulated_paths[:, :50], alpha=0.3)
    ax.set_title("Monte Carlo Simulation (50 out of 1000 paths)")
    st.pyplot(fig)
    st.info(
        "**What is Monte Carlo Simulation used for?**\n\n"
        "It runs thousands of simulations of future stock prices based on historical volatility and returns.\n\n"
        "**Why is it useful?**\n"
        "- Helps estimate potential future price ranges and probabilities of gains/losses.\n"
        "- It doesn't predict the future, but helps assess risk.\n"
        "**Use it to:** Understand uncertainty and the probability of hitting specific investment goals."
    )

elif page == "📈 Stage 2: Company Comparison":
    st.sidebar.header("Compare Multiple Stocks")

    comparison_data = {
    "Company": ["Apple", "Facebook", "Google", "Amazon", "Netflix"],
    "Close":        [232.15, 576.93, 162.93, 187.53, 687.65],
    "Target Price": [240.78, 601.58, 200.20, 218.90, 718.88],
    "Price_difference": [8.63, 24.65, 37.27, 31.37, 31.23],
    "PE Ratio":         [35.79, 29.61, 23.49, 45.50, 42.82],
    "PEG Ratio": [1.12, 0.95, 1.30, 2.10, 1.88],
    "P/S Ratio": [6.5, 10.2, 5.9, 8.7, 7.6],
    "EV/EBITDA": [21.3, 18.5, 19.2, 23.1, 20.6],
    "EPS":       [6.57, 19.56, 6.97, 4.18, 17.67],
    "Volume": [32978900, 8687000, 21339400, 24993600, 8820000],
    "Market Cap": [3575090000000, 1465350000000, 2024580000000, 1996000000000, 324753000000], # Newly Added Metrics
    "ROE":        [28.3, 25.1, 21.5, 16.9, 30.2],  # %
    "Net Margin": [22.5, 26.1, 20.2, 10.7, 17.4],  # %
    "ROA":        [18.1, 14.5, 13.7, 7.9, 12.3],  # %
    "Revenue Growth": [7.2, 6.9, 9.1, 8.3, 6.0],  # %
    "EPS Growth":     [5.8, 7.4, 8.2, 4.1, 9.0],  # %
    "5Y CAGR":        [12.3, 10.8, 11.5, 9.9, 13.2],
    "Alpha" :         [0.12,0.25,-0.08, 0.03,-0.15], 
    "Beta":           [1.18, 1.05, 1.12, 1.25, 0.95],
    "Sharpe Ratio":   [1.10, 0.95, 1.20, 1.00, 1.30],
    "Sortino Ratio":  [1.45, 1.30, 1.55, 1.40, 1.60],
    "Drawdown":       [-3.2, -2.6, -1.8, -4.3, -0.9]  # %
}


    comparison_df = pd.DataFrame(comparison_data)

    selected_stocks = st.sidebar.multiselect(
        "Select up to 5 stocks to analyze:",
        options=comparison_df['Company'].tolist(),
        default=comparison_df['Company'].tolist()[:3]
    )

    df_selected = comparison_df[comparison_df['Company'].isin(selected_stocks)]

    if df_selected.empty:
        st.warning("Please select at least one stock to view the comparison.")
    else:
        df_selected['Volume'] = df_selected['Volume'].apply(convert_large_numbers)
        df_selected['Market Cap'] = df_selected['Market Cap'].apply(convert_large_numbers)

        st.markdown("### 📊 Stock Comparison Table")
        st.dataframe(df_selected)

        st.markdown("### 📘 KPI Descriptions")
        st.markdown("""
        - **P/E Ratio (Price to Earnings)**: A ratio of 15–20 is typically considered fair value. A high ratio may indicate overvaluation, while a low ratio might suggest undervaluation.
        - **EPS (Earnings Per Share)**: Shows company profitability. The higher the EPS, the better.
        - **Target Price**: Analysts’ projection. A target price higher than the current price suggests growth potential.
        - **Price Difference**: How much room the stock has to grow based on analyst estimates.
        - **Volume**: Reflects trading activity and investor interest.
        - **Market Cap**: Represents the total value of the company; helps compare company sizes.

        - **PEG Ratio (Price/Earnings to Growth)**: Adjusts the P/E ratio based on earnings growth. A value < 1 is considered undervalued relative to growth.
        - **P/S Ratio (Price to Sales)**: Compares stock price to revenue. Lower ratios can indicate undervaluation.
        - **EV/EBITDA**: A valuation metric; lower values generally mean a stock is more attractively priced.
        - **ROE (Return on Equity)**: Measures how efficiently a company generates profits from shareholders’ equity. Higher is better.
        - **ROA (Return on Assets)**: Indicates how efficiently a company uses its assets to generate profit.
        - **Net Margin**: The percentage of revenue that turns into profit. Higher margins suggest better cost control.
        - **Revenue Growth**: Measures how fast a company's sales are increasing year-over-year.
        - **EPS Growth**: Shows how fast earnings per share are increasing. Sustained growth is a positive indicator.
        - **5Y CAGR (Compound Annual Growth Rate)**: Represents the company's average annual growth over five years. Higher values suggest strong long-term growth.
        - **Alpha**: Measures excess return compared to a benchmark. Positive alpha = outperformance.
        - **Beta**: Indicates volatility relative to the market. Beta > 1 = more volatile; Beta < 1 = more stable.
        - **Sharpe Ratio**: Measures return per unit of total risk. Higher values (>1) indicate good risk-adjusted performance.
        - **Sortino Ratio**: Similar to Sharpe but considers only downside risk. Better for evaluating asymmetric volatility.
        - **Drawdown**: The maximum observed loss from a peak. Smaller drawdowns indicate better downside protection.
        """)


        

        if len(df_selected) > 1:
            # Identify best stocks by key metrics
            best_pe = df_selected.loc[df_selected['PE Ratio'].astype(float).idxmin()]
            best_eps = df_selected.loc[df_selected['EPS'].astype(float).idxmax()]
            best_target_price = df_selected.loc[df_selected['Price_difference'].astype(float).idxmax()]

            st.markdown("---")
            st.markdown("### 📌 **Top Picks Based on Financial KPIs**")

            col1, col2, col3 = st.columns(3)
            col1.metric(label=" Value Stock (Lowest P/E)", value=best_pe['Company'], delta=f"P/E: {best_pe['PE Ratio']}")
            col2.metric(label=" Most Profitable (Highest EPS)", value=best_eps['Company'], delta=f"EPS: {best_eps['EPS']}")
            col3.metric(label=" Highest Upside Potential", value=best_target_price['Company'], delta=f"+{best_target_price['Price_difference']} USD")

            # Summary Table
            recommendations_data = {
                " Category": ["Value", "Profitability", "Growth Potential"],
                " Company": [best_pe['Company'], best_eps['Company'], best_target_price['Company']],
                " Metric": [
                    f"P/E Ratio: {best_pe['PE Ratio']}",
                    f"EPS: {best_eps['EPS']}",
                    f"Price Diff: +{best_target_price['Price_difference']} USD"
                ]
            }

            recommendations_df = pd.DataFrame(recommendations_data)
            st.table(recommendations_df)

            # Interactive Donut Chart
            st.markdown("###  Potential Upside by Company (Analyst Target - Current Price)")

            fig = go.Figure(data=[go.Pie(
                labels=df_selected['Company'],
                values=df_selected['Price_difference'].astype(float),
                hole=0.5,
                marker=dict(colors=px.colors.qualitative.Set3, line=dict(color='black', width=5)),
                hoverinfo='label+percent+value',
                textinfo='label+percent',
            )])

            fig.update_layout(
                height=400,
                showlegend=True,
                margin=dict(t=50, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                title=dict(text="Estimated Growth (%) Based on Analyst Target", x=0.18, font=dict(size=18))
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
        else:
            st.info("Select two or more companies to see recommendations and pie chart.")
    # -- Sidebar stock selector
    st.sidebar.header("Select Stocks to Compare")
    selected_stocks = st.sidebar.multiselect(
        "Choose up to 5 companies:",
        options=list(datasets.keys()),
        default=list(datasets.keys())[:3]
    )

    # -- Load all data into one combined DataFrame
    combined_df = pd.DataFrame()
    for name, url in datasets.items():
        try:
            df = pd.read_csv(url)
            df["Date"] = pd.to_datetime(df["Date"])
            df["Company"] = name
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            st.warning(f"❌ Failed to load data for {name}: {e}")

    # -- Closing Price Plot
    if not combined_df.empty:
        df_filtered = combined_df[combined_df["Company"].isin(selected_stocks)][["Date", "Close", "Company"]].dropna()
        fig = px.line(
            df_filtered,
            x="Date",
            y="Close",
            color="Company",
            title="📈 Closing Prices Over Time - Selected FAANG Stocks",
            labels={"Close": "Closing Price (USD)", "Date": "Date"},
            template="plotly_dark"
        )
        fig.update_traces(mode="lines", line=dict(width=2))
        fig.update_layout(height=500, width=1000)
        st.plotly_chart(fig, use_container_width=True)

    # -- Daily Volume Plot
    st.markdown("## 📊 Daily Trading Volume of Selected Stocks")
    fig = go.Figure()
    for label in selected_stocks:
        try:
            df = pd.read_csv(datasets[label])
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            if 'Volume' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Volume'],
                    mode='lines',
                    name=label
                ))
        except Exception as e:
            st.warning(f"❌ Could not load data for {label}: {e}")

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Volume",
        title="Daily Trading Volume",
        template="plotly_dark",
        height=500,
        margin=dict(t=50, b=40, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -- Moving Averages
    st.markdown("## 🧮 Smoothed Moving Averages (6M, 12M, 24M)")
    colors = {
        'Close': '#1f77b4',
        'MA_6M': '#2ca02c',
        'MA_12M': '#ff7f0e',
        'MA_24M': '#d62728',
    }
    fig = make_subplots(
        rows=len(selected_stocks), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=selected_stocks
    )
    for i, label in enumerate(selected_stocks, start=1):
        try:
            df = pd.read_csv(datasets[label])
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df['MA_6M'] = df['Close'].rolling(window=126).mean()
            df['MA_12M'] = df['Close'].rolling(window=252).mean()
            df['MA_24M'] = df['Close'].rolling(window=504).mean()

            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close',
                                    line=dict(color=colors['Close'], width=1.8), showlegend=(i==1)), row=i, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_6M'], mode='lines', name='6M MA',
                                    line=dict(color=colors['MA_6M'], width=1.2), showlegend=(i==1)), row=i, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_12M'], mode='lines', name='12M MA',
                                    line=dict(color=colors['MA_12M'], width=1.2), showlegend=(i==1)), row=i, col=1)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_24M'], mode='lines', name='24M MA',
                                    line=dict(color=colors['MA_24M'], width=1.2), showlegend=(i==1)), row=i, col=1)
        except Exception as e:
            st.warning(f"⚠️ Failed to load data for {label}: {e}")

    fig.update_layout(
        height=400 * len(selected_stocks),
        width=1000,
        title_text="📉 Moving Averages for Selected Stocks",
        showlegend=True,
        template="plotly_dark",
        margin=dict(t=60, b=40, l=40, r=20),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
 
    # Add this section after your other visualizations in the "Company Comparison" page

    st.markdown("## 📉 Daily Return Distributions")

    # Calculate daily returns for each selected stock
    return_dfs = []
    for company in selected_stocks:
            try:
                df = pd.read_csv(datasets[company])
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                df['Daily Return'] = df['Close'].pct_change()
                df['Company'] = company
                return_dfs.append(df[['Date', 'Daily Return', 'Company']].dropna())
            except Exception as e:
                st.warning(f"Could not calculate returns for {company}: {e}")

    if return_dfs:
        returns_df = pd.concat(return_dfs)
        
        # Create subplots - one histogram per company
        fig = make_subplots(
            rows=3, 
            cols=2,
            subplot_titles=selected_stocks,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        for i, company in enumerate(selected_stocks, 1):
            row = (i-1)//2 + 1
            col = (i-1)%2 + 1
            
            company_data = returns_df[returns_df['Company'] == company]
            
            fig.add_trace(
                go.Histogram(
                    x=company_data['Daily Return'],
                    nbinsx=50,
                    name=company,
                    marker_color=px.colors.qualitative.Plotly[i-1],
                    opacity=0.75
                ),
                row=row, col=col
            )
            
            # Update subplot axes
            fig.update_xaxes(
                title_text="Daily Return",
                row=row, col=col
            )
            fig.update_yaxes(
                title_text="Count",
                row=row, col=col
            )
        
        # Update overall layout
        fig.update_layout(
            height=800,
            width=1000,
            title_text="Distribution of Daily Returns",
            showlegend=False,
            template="plotly_dark",
            margin=dict(t=100, b=50))
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No daily return data available for the selected stocks.")
    # Get closing prices for all selected stocks (shared for all visualizations)
    closing_data = []
    for company in selected_stocks:
        try:
            df = pd.read_csv(datasets[company])
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            closes = df.set_index('Date')['Close'].rename(company)
            closing_data.append(closes)
        except Exception as e:
            st.warning(f"Could not load data for {company}: {e}")

    if len(closing_data) > 1:
        closing_df = pd.concat(closing_data, axis=1).dropna()
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Pairwise Relationships", 
            "⚖️ Relative Strength", 
            "🔄 Rolling Correlation",
            "🔥 Correlation Matrix"
        ])
        
        with tab1:
            st.markdown("### Pairwise Closing Price Relationships")
            fig1 = px.scatter_matrix(
                closing_df,
                dimensions=closing_df.columns,
                title="Price Relationships",
                height=800,
                template="plotly_dark"
            )
            fig1.update_traces(
                diagonal_visible=False,
                showupperhalf=True,
                marker=dict(size=3, opacity=0.6)
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with tab2:
            st.markdown("### Relative Strength Comparison")
            base_stock = st.selectbox(
                "Select base stock for comparison:",
                selected_stocks,
                key="base_stock_selector"
            )
            
            ratios = closing_df.div(closing_df[base_stock], axis=0)
            fig2 = px.line(
                ratios,
                title=f"Price Ratios Relative to {base_stock}",
                template="plotly_dark"
            )
            fig2.update_layout(
                yaxis_title=f"Price Ratio (X/{base_stock})",
                hovermode="x unified",
                height=600
            )
            fig2.add_hline(y=1.0, line_dash="dot", 
                        line_color="gray", 
                        annotation_text=f"{base_stock} Baseline")
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.markdown("### 90-Day Rolling Correlation")
            rolling_corr = closing_df.rolling(window=90).corr().dropna()
            pairs = [(a,b) for i,a in enumerate(selected_stocks) 
                    for j,b in enumerate(selected_stocks) if i < j]
            
            selected_pairs = st.multiselect(
                "Select pairs to display:",
                [f"{a} vs {b}" for a,b in pairs],
                default=[f"{a} vs {b}" for a,b in pairs[:3]],
                key="pair_selector"
            )
            
            fig3 = go.Figure()
            for pair in selected_pairs:
                a, b = pair.split(" vs ")
                corr_series = rolling_corr.xs(a, level=1)[b]
                fig3.add_trace(go.Scatter(
                    x=corr_series.index,
                    y=corr_series,
                    name=pair,
                    mode="lines"
                ))
            
            fig3.update_layout(
                yaxis_range=[-1,1],
                hovermode="x unified",
                template="plotly_dark",
                height=600
            )
            fig3.add_hline(y=0, line_color="gray")
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab4:
            st.markdown("### Correlation Heatmap")
            corr_matrix = closing_df.corr()
            
            fig4 = go.Figure(
                go.Heatmap(
                    z=corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate="%{text}",
                    hoverongaps=False
                )
            )
            
            fig4.update_layout(
                title="Correlation of Stock Closing Prices",
                xaxis_title="Companies",
                yaxis_title="Companies",
                height=600,
                width=800,
                template="plotly_dark",
                margin=dict(l=100, r=100, t=100, b=100)
            )
            
            # Add annotation about correlation interpretation
            st.info("""
            **Interpreting Correlation Values:**
            - 1.0: Perfect positive correlation
            - 0.0: No correlation
            - -1.0: Perfect negative correlation
            """)
            
            st.plotly_chart(fig4, use_container_width=True)

    else:
        st.warning("Need at least 2 stocks to generate comparisons")
     
     
        # Fundamental Metrics Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Valuation", 
        "💼 Profitability", 
        "📈 Growth", 
        "⚠️ Risk"
    ])

    with tab1:  # Valuation Metrics
        st.subheader("📊 Valuation Ratio Analysis")
        valuation_cols = ["PE Ratio", "PEG Ratio", "P/S Ratio", "EV/EBITDA"]
        
        # Data Display
        st.dataframe(df_selected[["Company"] + valuation_cols])
        
        # Original Definitions
        st.info("""
        **P/E Ratio (Price to Earnings):**  
        This ratio shows how much investors are willing to pay per dollar of earnings. 
        Lower P/E may indicate undervaluation, while a higher P/E could imply growth expectations.
        """)
        
        st.info("""
        **PEG Ratio (P/E to Growth):**  
        PEG accounts for earnings growth. A PEG < 1 suggests undervalued growth, 
        which is more meaningful than P/E alone.
        """)
        
        st.info("""
        **P/S Ratio (Price to Sales):**  
        Used when companies have low or no earnings. A lower P/S ratio may suggest undervaluation.
        """)
        
        st.info("""
        **EV/EBITDA (Enterprise Value to EBITDA):**  
        This ratio helps compare firms with different capital structures. 
        It's better than P/E for debt-heavy companies.
        """)
        
        # Visualization
        st.plotly_chart(
            go.Figure([
                go.Bar(
                    name=metric, 
                    x=df_selected["Company"], 
                    y=df_selected[metric], 
                    text=df_selected[metric],
                    textposition='auto'
                ) for metric in valuation_cols
            ]).update_layout(
                barmode="group", 
                title="Valuation Ratios",
                template="plotly_dark"
            ),
            use_container_width=True
        )

    with tab2:  # Profitability Metrics
        st.subheader("💼 Profitability Metrics")
        profit_cols = ["EPS", "ROE", "Net Margin", "ROA"]
        
        # Data Display
        st.dataframe(df_selected[["Company"] + profit_cols])
        
        # Original Definitions
        st.info("""
        **EPS (Earnings per Share):**  
        Shows how much profit is assigned to each share. Higher is better.
        """)
        
        st.info("""
        **ROE (Return on Equity):**  
        Evaluates how efficiently equity is used. Over 15% is strong.
        """)
        
        st.info("""
        **Net Margin:**  
        The percentage of revenue turned into profit. Higher = more efficient.
        """)
        
        st.info("""
        **ROA (Return on Assets):**  
        Reflects how well assets are used to generate earnings.
        """)
        
        # Visualization
        st.plotly_chart(
            go.Figure([
                go.Bar(
                    name=metric, 
                    x=df_selected["Company"], 
                    y=df_selected[metric], 
                    text=df_selected[metric],
                    textposition='auto'
                ) for metric in profit_cols
            ]).update_layout(
                barmode="group", 
                title="Profitability Metrics",
                template="plotly_dark"
            ),
            use_container_width=True
        )

    with tab3:  # Growth Metrics
        st.subheader("📈 Growth Metrics")
        growth_cols = ["Revenue Growth", "EPS Growth", "5Y CAGR"]
        
        # Data Display
        st.dataframe(df_selected[["Company"] + growth_cols])
        
        # Original Definitions
        st.info("""
        **Revenue Growth (%):**  
        Reveals how fast company sales are growing year over year.
        """)
        
        st.info("""
        **EPS Growth (%):**  
        Shows how profits are growing over time, a sign of solid performance.
        """)
        
        st.info("""
        **5Y CAGR:**  
        The Compound Annual Growth Rate. It smooths growth over time.
        """)
        
        # Visualization
        st.plotly_chart(
            go.Figure([
                go.Bar(
                    name=metric, 
                    x=df_selected["Company"], 
                    y=df_selected[metric], 
                    text=df_selected[metric],
                    textposition='auto'
                ) for metric in growth_cols
            ]).update_layout(
                barmode="group", 
                title="Growth Metrics",
                template="plotly_dark"
            ),
            use_container_width=True
        )

    with tab4:  # Risk Metrics
        st.subheader("⚠️ Risk & Volatility Metrics")
        risk_cols = ["Drawdown", "Beta", "Alpha", "Sharpe Ratio", "Sortino Ratio"]
        
        # Data Display
        st.dataframe(df_selected[["Company"] + risk_cols])
        
        # Original Definitions
        st.info("""
        **Beta:**  
        Shows volatility relative to the market (1.0 = same as market).
        """)
        
        st.info("""
        **Sharpe Ratio:**  
        Balances return vs. total risk. Higher than 1.0 is good.
        """)
        
        st.info("""
        **Sortino Ratio:**  
        Is like Sharpe but focuses only on downside risk (losses).
        """)
        
        st.info("""
        **Drawdown:**  
        Measures the largest drop from a peak. Lower is better.
        """)
        
        # Visualization
        st.plotly_chart(
            go.Figure([
                go.Bar(
                    name=metric, 
                    x=df_selected["Company"], 
                    y=df_selected[metric], 
                    text=df_selected[metric],
                    textposition='auto'
                ) for metric in risk_cols
            ]).update_layout(
                barmode="group", 
                title="Risk & Volatility Metrics",
                template="plotly_dark"
            ),
            use_container_width=True
        )
   # Add this as a new tab in your existing app
    risk_tab, return_tab = st.tabs(["⚠️ Risk Analysis", "📈 Return Analysis"])

    with risk_tab:
        st.markdown("## 📉 Risk-Return Profile of Selected Stocks")
        
        # Calculate daily returns
        returns_df = closing_df.pct_change().dropna()
        
        if len(returns_df.columns) >= 1:  # Need at least 1 stock
            # Calculate risk (std) and return (mean)
            risk_return = pd.DataFrame({
                'Stock': returns_df.columns,
                'Expected Return': returns_df.mean(),
                'Risk': returns_df.std()
            })
            
            # Create the scatter plot
            fig = px.scatter(
                risk_return,
                x='Expected Return',
                y='Risk',
                text='Stock',
                size=[30]*len(risk_return),  # Constant bubble size
                title='Risk vs Expected Return',
                labels={
                    'Expected Return': 'Expected Daily Return (%)',
                    'Risk': 'Risk (Std Dev of Daily Returns)'
                },
                template="plotly_dark"
            )
            
            # Add arrows and format axes
            fig.update_traces(
                textposition='top center',
                marker=dict(size=20),
                textfont=dict(size=12)
            )
            
            fig.update_layout(
                height=600,
                xaxis_tickformat=".2%",
                yaxis_tickformat=".2%",
                hovermode="closest"
            )
            
            # Add explanation
            st.info("""
            **How to interpret this chart:**
            - **Right**: Higher expected returns
            - **Top**: Higher risk (volatility)
            - Ideal investments are bottom-right (high return, low risk)
            """)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show numerical data
            st.markdown("### 📊 Risk-Return Metrics")
            st.dataframe(risk_return.set_index('Stock').style.format("{:.2%}"))
            
        else:
            st.warning("Need at least 1 stock to calculate risk/return profile")

    with return_tab:
        st.markdown("## 📈 Cumulative Returns Over Time")
        
        if len(returns_df.columns) >= 1:
            cum_returns = (1 + returns_df).cumprod() - 1
            
            fig = px.line(
                (cum_returns * 100).reset_index(),  # Convert to percentage
                x='Date',
                y=cum_returns.columns,
                title="Growth of $1 Investment",
                labels={"value": "Return (%)", "Date": "Date"},
                template="plotly_dark"
            )
            
            fig.update_layout(
                hovermode="x unified",
                yaxis_title="Cumulative Return (%)",
                height=500
            )
            fig.update_yaxes(ticksuffix="%")
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Need at least 1 stock to show cumulative returns")