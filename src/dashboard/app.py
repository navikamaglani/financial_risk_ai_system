# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import feedparser

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load(
    "models/risk_classifier.pkl"
)

# -----------------------------
# SHAP Explainer
# -----------------------------
explainer = shap.TreeExplainer(model)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Portfolio Risk Intelligence Platform",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title(
    "AI Portfolio Risk Intelligence Platform"
)

st.markdown(
    """
    Real-time portfolio risk prediction and
    financial analytics using machine learning.
    """
)

# -----------------------------
# Available Stocks
# -----------------------------
available_tickers = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "SPY",
    "QQQ"
]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Investment Configuration")

primary_ticker = st.sidebar.selectbox(
    "Primary Stock",
    available_tickers,
    index=2
)

comparison_tickers = st.sidebar.multiselect(
    "Comparison Stocks",
    available_tickers,
    default=["AAPL", "MSFT", "SPY"]
)

time_period = st.sidebar.selectbox(
    "Time Horizon",
    ["6mo", "1y", "3y", "5y"],
    index=1
)

analyze = st.sidebar.button(
    "Run AI Analysis"
)

# -----------------------------
# Validation
# -----------------------------
if len(comparison_tickers) == 0:
    st.warning(
        "Please select comparison stocks."
    )
    st.stop()

# -----------------------------
# Main Analysis
# -----------------------------
if analyze:

    # -----------------------------
    # Download Primary Stock Data
    # -----------------------------
    df = yf.download(
        primary_ticker,
        period=time_period,
        auto_adjust=True,
        progress=False
    )

    # Flatten MultiIndex Columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = (
            df.columns.get_level_values(0)
        )

    # -----------------------------
    # Technical Indicators
    # -----------------------------
    df["Daily_Return"] = (
        df["Close"].pct_change(
            fill_method=None
        )
    )

    df["Rolling_Volatility"] = (
        df["Daily_Return"]
        .rolling(window=21)
        .std()
    )

    df["SMA_20"] = (
        df["Close"]
        .rolling(window=20)
        .mean()
    )

    df["SMA_50"] = (
        df["Close"]
        .rolling(window=50)
        .mean()
    )

    rsi = RSIIndicator(
        close=df["Close"],
        window=14
    )

    df["RSI"] = rsi.rsi()

    macd = MACD(
        close=df["Close"]
    )

    df["MACD"] = macd.macd()

    df["MACD_Signal"] = (
        macd.macd_signal()
    )

    df["Momentum_10"] = (
        df["Close"]
        - df["Close"].shift(10)
    )

    bb = BollingerBands(
        close=df["Close"],
        window=20
    )

    df["BB_High"] = (
        bb.bollinger_hband()
    )

    df["BB_Low"] = (
        bb.bollinger_lband()
    )

    # -----------------------------
    # Remove Missing Rows
    # -----------------------------
    df = df.dropna()

    # -----------------------------
    # Feature Columns
    # -----------------------------
    feature_columns = [
        "Daily_Return",
        "Rolling_Volatility",
        "SMA_20",
        "SMA_50",
        "RSI",
        "MACD",
        "MACD_Signal",
        "BB_High",
        "BB_Low",
        "Momentum_10"
    ]

    # -----------------------------
    # Model Features
    # -----------------------------
    latest_row = df.iloc[-1]

    X = pd.DataFrame([
        latest_row[feature_columns]
    ])

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(X)[0]

    probabilities = (
        model.predict_proba(X)[0]
    )

    risk_mapping = {
        0: "LOW RISK",
        1: "MEDIUM RISK",
        2: "HIGH RISK"
    }

    predicted_risk = (
        risk_mapping[prediction]
    )

    # -----------------------------
    # Risk Display
    # -----------------------------
    st.subheader(
        f"Risk Prediction for {primary_ticker}"
    )

    if predicted_risk == "LOW RISK":
        st.success(predicted_risk)

    elif predicted_risk == "MEDIUM RISK":
        st.warning(predicted_risk)

    else:
        st.error(predicted_risk)

    # -----------------------------
    # AI Summary
    # -----------------------------
    risk_summary = f"""
    {primary_ticker} is currently classified
    as {predicted_risk}.

    RSI is {latest_row['RSI']:.2f} indicating
    {'overbought' if latest_row['RSI'] > 70 else 'oversold' if latest_row['RSI'] < 30 else 'neutral'}
    market conditions.

    Rolling volatility is
    {latest_row['Rolling_Volatility']:.4f}.

    MACD currently indicates
    {'bullish' if latest_row['MACD'] > latest_row['MACD_Signal'] else 'bearish'}
    momentum behavior.
    """

    st.subheader("AI Risk Summary")

    st.info(risk_summary)

    # -----------------------------
    # Prediction Confidence
    # -----------------------------
    st.subheader(
        "Prediction Confidence"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Low Risk Probability",
        f"{probabilities[0] * 100:.2f}%"
    )

    col2.metric(
        "Medium Risk Probability",
        f"{probabilities[1] * 100:.2f}%"
    )

    col3.metric(
        "High Risk Probability",
        f"{probabilities[2] * 100:.2f}%"
    )

    # -----------------------------
    # Price Trend
    # -----------------------------
    st.subheader(
        "Stock Price Trend"
    )

    fig_price, ax_price = plt.subplots(
        figsize=(12, 5)
    )

    ax_price.plot(
        df.index,
        df["Close"]
    )

    ax_price.set_title(
        f"{primary_ticker} Closing Price"
    )

    st.pyplot(fig_price)

    # -----------------------------
    # RSI Chart
    # -----------------------------
    st.subheader("RSI Indicator")

    fig_rsi, ax_rsi = plt.subplots(
        figsize=(12, 4)
    )

    ax_rsi.plot(
        df.index,
        df["RSI"]
    )

    ax_rsi.axhline(
        70,
        linestyle="--"
    )

    ax_rsi.axhline(
        30,
        linestyle="--"
    )

    st.pyplot(fig_rsi)

    # -----------------------------
    # MACD Chart
    # -----------------------------
    st.subheader("MACD Indicator")

    fig_macd, ax_macd = plt.subplots(
        figsize=(12, 4)
    )

    ax_macd.plot(
        df.index,
        df["MACD"],
        label="MACD"
    )

    ax_macd.plot(
        df.index,
        df["MACD_Signal"],
        label="Signal"
    )

    ax_macd.legend()

    st.pyplot(fig_macd)

    # -----------------------------
    # Volatility Chart
    # -----------------------------
    st.subheader(
        "Rolling Volatility"
    )

    fig_vol, ax_vol = plt.subplots(
        figsize=(12, 4)
    )

    ax_vol.plot(
        df.index,
        df["Rolling_Volatility"]
    )

    st.pyplot(fig_vol)

    # -----------------------------
    # SHAP Explainability
    # -----------------------------
    st.subheader(
        "Model Explainability (SHAP)"
    )

    shap_values = explainer.shap_values(
        X,
        check_additivity=False
    )

    shap_df = pd.DataFrame({
        "Feature": feature_columns,
        "Impact": shap_values[
            0, :, prediction
        ]
    })

    shap_df["AbsImpact"] = (
        shap_df["Impact"].abs()
    )

    shap_df = shap_df.sort_values(
        by="AbsImpact",
        ascending=False
    )

    fig_shap, ax_shap = plt.subplots(
        figsize=(10, 5)
    )

    ax_shap.barh(
        shap_df["Feature"],
        shap_df["Impact"]
    )

    ax_shap.set_title(
        "Feature Impact on Prediction"
    )

    st.pyplot(fig_shap)

    # -----------------------------
    # News & Sentiment
    # -----------------------------
    st.header(
        "Market News & Sentiment"
    )

    news_url = (
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={primary_ticker}&region=US&lang=en-US"
    )

    feed = feedparser.parse(
        news_url
    )

    analyzer = (
        SentimentIntensityAnalyzer()
    )

    news_titles = []
    sentiment_scores = []

    for entry in feed.entries[:5]:

        title = entry.title

        sentiment = (
            analyzer.polarity_scores(
                title
            )["compound"]
        )

        news_titles.append(title)

        sentiment_scores.append(
            sentiment
        )

    news_df = pd.DataFrame({
        "Headline": news_titles,
        "Sentiment": sentiment_scores
    })

    for _, row in news_df.iterrows():

        if row["Sentiment"] > 0.2:
            st.success(
                f"🟢 {row['Headline']}"
            )

        elif row["Sentiment"] < -0.2:
            st.error(
                f"🔴 {row['Headline']}"
            )

        else:
            st.warning(
                f"🟡 {row['Headline']}"
            )

    avg_sentiment = (
        news_df["Sentiment"].mean()
    )

    if avg_sentiment > 0.2:
        sentiment_label = "Bullish"

    elif avg_sentiment < -0.2:
        sentiment_label = "Bearish"

    else:
        sentiment_label = "Neutral"

    st.metric(
        "Overall Market Sentiment",
        sentiment_label
    )

    # -----------------------------
    # Portfolio Analysis
    # -----------------------------
    st.header(
        "Portfolio Risk Analysis"
    )

    portfolio_tickers = (
        [primary_ticker]
        + comparison_tickers
    )

    portfolio_tickers = list(
        set(portfolio_tickers)
    )

    weights = np.array(
        [1 / len(portfolio_tickers)]
        * len(portfolio_tickers)
    )

    portfolio_data = yf.download(
        portfolio_tickers,
        period=time_period,
        auto_adjust=True,
        progress=False
    )["Close"]

    returns = (
        portfolio_data
        .pct_change(fill_method=None)
        .dropna()
    )

    portfolio_returns = (
        returns * weights
    ).sum(axis=1)

    expected_return = (
        portfolio_returns.mean() * 252
    )

    portfolio_volatility = (
        portfolio_returns.std()
        * np.sqrt(252)
    )

    # -----------------------------
    # Portfolio Risk Label
    # -----------------------------
    if portfolio_volatility < 0.20:
        portfolio_risk = "LOW RISK"

    elif portfolio_volatility < 0.35:
        portfolio_risk = "MEDIUM RISK"

    else:
        portfolio_risk = "HIGH RISK"

    # -----------------------------
    # Portfolio Metrics
    # -----------------------------
    st.subheader(
        "Portfolio Metrics"
    )

    pcol1, pcol2, pcol3 = st.columns(3)

    pcol1.metric(
        "Expected Annual Return",
        f"{expected_return:.2%}"
    )

    pcol2.metric(
        "Annual Volatility",
        f"{portfolio_volatility:.2%}"
    )

    pcol3.metric(
        "Portfolio Risk",
        portfolio_risk
    )

    # -----------------------------
    # Sector Exposure
    # -----------------------------
    st.subheader(
        "Sector Exposure"
    )

    sector_map = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "NVDA": "Technology",
        "GOOGL": "Communication Services",
        "META": "Communication Services",
        "AMZN": "Consumer Discretionary",
        "TSLA": "Automotive",
        "SPY": "ETF",
        "QQQ": "ETF"
    }

    sector_counts = {}

    for stock in portfolio_tickers:

        sector = sector_map.get(
            stock,
            "Other"
        )

        sector_counts[sector] = (
            sector_counts.get(
                sector,
                0
            ) + 1
        )

    sector_df = pd.DataFrame({
        "Sector": sector_counts.keys(),
        "Count": sector_counts.values()
    })

    fig_sector, ax_sector = plt.subplots(
        figsize=(7, 5)
    )

    ax_sector.pie(
        sector_df["Count"],
        labels=sector_df["Sector"],
        autopct="%1.1f%%"
    )

    ax_sector.set_title(
        "Portfolio Sector Exposure"
    )

    st.pyplot(fig_sector)

    # -----------------------------
    # Benchmark Comparison
    # -----------------------------
    st.subheader(
        "Portfolio vs SPY Benchmark"
    )

    spy_data = yf.download(
        "SPY",
        period=time_period,
        auto_adjust=True,
        progress=False
    )["Close"]

    spy_returns = (
        spy_data
        .pct_change(fill_method=None)
        .dropna()
    )

    spy_cumulative = (
        1 + spy_returns
    ).cumprod()

    portfolio_cumulative = (
        1 + portfolio_returns
    ).cumprod()

    fig_compare, ax_compare = plt.subplots(
        figsize=(12, 5)
    )

    ax_compare.plot(
        portfolio_cumulative.index,
        portfolio_cumulative,
        label="Portfolio"
    )

    ax_compare.plot(
        spy_cumulative.index,
        spy_cumulative,
        label="SPY Benchmark"
    )

    ax_compare.legend()

    ax_compare.set_title(
        "Portfolio Performance vs SPY"
    )

    st.pyplot(fig_compare)

    # -----------------------------
    # Relative Performance
    # -----------------------------
    st.subheader(
        "Relative Performance Comparison"
    )

    comparison_list = (
        [primary_ticker]
        + comparison_tickers
    )

    comparison_data = yf.download(
        comparison_list,
        period=time_period,
        auto_adjust=True,
        progress=False
    )["Close"]

    comparison_returns = (
        comparison_data.pct_change(
            fill_method=None
        ).dropna()
    )

    comparison_cumulative = (
        1 + comparison_returns
    ).cumprod()

    fig_relative, ax_relative = plt.subplots(
        figsize=(12, 5)
    )

    for stock in comparison_cumulative.columns:

        ax_relative.plot(
            comparison_cumulative.index,
            comparison_cumulative[stock],
            label=stock
        )

    ax_relative.set_title(
        "Relative Performance Comparison"
    )

    ax_relative.legend()

    st.pyplot(fig_relative)

    # -----------------------------
    # Quant Metrics
    # -----------------------------
    volatility = (
        comparison_returns.std()
        * np.sqrt(252)
    )

    sharpe_ratio = (
        comparison_returns.mean() * 252
    ) / volatility

    # -----------------------------
    # Beta vs SPY
    # -----------------------------
    beta_values = {}

    if "SPY" in comparison_returns.columns:

        spy_returns_beta = (
            comparison_returns["SPY"]
        )

        for stock in comparison_returns.columns:

            if stock != "SPY":

                covariance = np.cov(
                    comparison_returns[stock],
                    spy_returns_beta
                )[0][1]

                variance = np.var(
                    spy_returns_beta
                )

                beta = covariance / variance

                beta_values[stock] = beta

    # -----------------------------
    # Relative Strength Ranking
    # -----------------------------
    ranking_data = []

    for stock in comparison_returns.columns:

        total_return = (
            comparison_cumulative[stock]
            .iloc[-1] - 1
        )

        vol = volatility[stock]

        sharpe = sharpe_ratio[stock]

        beta = (
            beta_values.get(stock, np.nan)
        )

        if vol < 0.20:
            risk = "Low"

        elif vol < 0.35:
            risk = "Medium"

        else:
            risk = "High"

        if stock == primary_ticker:
            sentiment = sentiment_label
        else:
            sentiment = "Neutral"

        ranking_data.append({
            "Stock": stock,
             "Return": f"{total_return:.2%}",
             "Volatility": f"{vol:.2%}",
             "Sharpe Ratio": round(sharpe, 2),

            "Beta vs SPY": (
                round(float(beta), 2)
                if not np.isnan(beta)
                else np.nan
             ),

            "Risk": risk,
            "Sentiment": sentiment
            })

    ranking_df = pd.DataFrame(
        ranking_data
    )
    ranking_df = ranking_df.astype(object)
 

    st.subheader(
        "Relative Strength Ranking"
    )

    st.dataframe(
        ranking_df.sort_values(
            by="Sharpe Ratio",
            ascending=False
        ),
        use_container_width=True
    )

    # -----------------------------
    # Correlation Matrix
    # -----------------------------
    st.subheader(
        "Portfolio Correlation Matrix"
    )

    corr_matrix = returns.corr()

    fig_corr, ax_corr = plt.subplots(
        figsize=(8, 6)
    )

    cax = ax_corr.matshow(
        corr_matrix,
        cmap="coolwarm"
    )

    fig_corr.colorbar(cax)

    ax_corr.set_xticks(
        range(len(portfolio_tickers))
    )

    ax_corr.set_yticks(
        range(len(portfolio_tickers))
    )

    ax_corr.set_xticklabels(
        portfolio_tickers,
        rotation=45
    )

    ax_corr.set_yticklabels(
        portfolio_tickers
    )

    st.pyplot(fig_corr)

    # -----------------------------
    # Latest Metrics
    # -----------------------------
    st.subheader(
        "Latest Market Metrics"
    )

    latest_close = float(
        df["Close"].iloc[-1]
    )

    latest_volume = int(
        df["Volume"].iloc[-1]
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Latest Closing Price",
        f"${latest_close:.2f}"
    )

    col2.metric(
        "Latest Volume",
        f"{latest_volume:,}"
    )

    # -----------------------------
    # AI Investment Insight
    # -----------------------------
    st.header(
        "AI Investment Insight"
    )

    ai_summary = f"""
    {primary_ticker} currently shows
    {predicted_risk.lower()} characteristics.

    Key drivers include:
    • volatility profile
    • momentum behavior
    • technical indicator positioning
    • benchmark relative performance
    • recent market sentiment

    Compared to peers,
    {primary_ticker} demonstrates
    higher return potential but elevated
    risk concentration.
    """

    st.info(ai_summary)