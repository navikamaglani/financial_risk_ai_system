# app_pro.py

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

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Portfolio Intelligence Platform",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================
st.title(
    "AI Portfolio Intelligence Platform"
)

st.markdown(
    """
    Institutional-grade portfolio analytics,
    explainable AI risk modeling,
    benchmark comparison,
    Monte Carlo forecasting,
    and financial sentiment intelligence.
    """
)

# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load(
    "models/risk_classifier.pkl"
)

explainer = shap.TreeExplainer(model)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header(
    "Investment Configuration"
)

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

st.sidebar.markdown("---")

st.sidebar.success(
    "AI-Driven Quantitative Portfolio Analytics"
)

# =====================================================
# MAIN ANALYSIS
# =====================================================
if analyze:

    # =====================================================
    # TABS
    # =====================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Technical Analysis",
        "Portfolio Intelligence",
        "AI Explainability",
        "News & Sentiment"
    ])

    # =====================================================
    # DOWNLOAD DATA
    # =====================================================
    df = yf.download(
        primary_ticker,
        period=time_period,
        auto_adjust=True,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = (
            df.columns.get_level_values(0)
        )

    # =====================================================
    # TECHNICAL INDICATORS
    # =====================================================
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

    df = df.dropna()

    # =====================================================
    # FEATURES
    # =====================================================
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

    latest_row = df.iloc[-1]

    X = pd.DataFrame([
        latest_row[feature_columns]
    ])

    # =====================================================
    # MODEL PREDICTION
    # =====================================================
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

    # =====================================================
    # NEWS + SENTIMENT
    # =====================================================
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

    for entry in feed.entries[:15]:

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

    avg_sentiment = (
        news_df["Sentiment"].mean()
    )

    if avg_sentiment > 0.15:
        sentiment_label = "Bullish"

    elif avg_sentiment < -0.15:
        sentiment_label = "Bearish"

    else:
        sentiment_label = "Neutral"

    # =====================================================
    # PORTFOLIO
    # =====================================================
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

    # =====================================================
    # PORTFOLIO RISK
    # =====================================================
    if portfolio_volatility < 0.20:
        portfolio_risk = "LOW RISK"

    elif portfolio_volatility < 0.35:
        portfolio_risk = "MEDIUM RISK"

    else:
        portfolio_risk = "HIGH RISK"

    # =====================================================
    # SHARPE RATIO
    # =====================================================
    risk_free_rate = 0.02

    sharpe_ratio = (
        expected_return - risk_free_rate
    ) / portfolio_volatility

    # =====================================================
    # MAX DRAWDOWN
    # =====================================================
    portfolio_cumulative = (
        1 + portfolio_returns
    ).cumprod()

    running_max = (
        portfolio_cumulative.cummax()
    )

    drawdown = (
        portfolio_cumulative
        / running_max
    ) - 1

    max_drawdown = drawdown.min()

    # =====================================================
    # MONTE CARLO
    # =====================================================
    simulation_days = 252
    num_simulations = 100

    mean_return = (
        portfolio_returns.mean()
    )

    std_return = (
        portfolio_returns.std()
    )

    simulation_results = np.zeros(
        (simulation_days, num_simulations)
    )

    initial_portfolio = 10000

    for sim in range(num_simulations):

        prices = [initial_portfolio]

        for day in range(simulation_days):

            simulated_return = np.random.normal(
                mean_return,
                std_return
            )

            next_price = (
                prices[-1]
                * (1 + simulated_return)
            )

            prices.append(next_price)

        simulation_results[:, sim] = (
            prices[1:]
        )

    ending_values = (
        simulation_results[-1, :]
    )

    mc_mean = np.mean(
        ending_values
    )

    mc_median = np.median(
        ending_values
    )

    mc_min = np.min(
        ending_values
    )

    mc_max = np.max(
        ending_values
    )

    var_95 = np.percentile(
        ending_values,
        5
    )

    # =====================================================
    # TAB 1 — OVERVIEW
    # =====================================================
    with tab1:

        st.subheader(
            f"Risk Prediction for {primary_ticker}"
        )

        if predicted_risk == "LOW RISK":
            st.success(predicted_risk)

        elif predicted_risk == "MEDIUM RISK":
            st.warning(predicted_risk)

        else:
            st.error(predicted_risk)

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Risk Level",
            predicted_risk
        )

        k2.metric(
            "Annual Return",
            f"{expected_return:.2%}"
        )

        k3.metric(
            "Volatility",
            f"{portfolio_volatility:.2%}"
        )

        k4.metric(
            "Sentiment",
            sentiment_label
        )

        st.subheader(
            "Prediction Confidence"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Low Risk",
            f"{probabilities[0]*100:.2f}%"
        )

        c2.metric(
            "Medium Risk",
            f"{probabilities[1]*100:.2f}%"
        )

        c3.metric(
            "High Risk",
            f"{probabilities[2]*100:.2f}%"
        )

        st.subheader(
            "Stock Price Trend"
        )

        fig_price, ax_price = plt.subplots(
            figsize=(14, 6)
        )

        ax_price.plot(
            df.index,
            df["Close"],
            linewidth=2
        )

        ax_price.set_title(
            f"{primary_ticker} Closing Price Trend",
            fontsize=18,
            fontweight="bold"
        )

        ax_price.set_xlabel(
            "Date"
        )

        ax_price.set_ylabel(
            "Price ($)"
        )

        ax_price.grid(alpha=0.3)

        plt.xticks(rotation=15)

        st.pyplot(fig_price)

    # =====================================================
    # TAB 2 — TECHNICAL ANALYSIS
    # =====================================================
    with tab2:

        # RSI
        st.subheader(
            "RSI Indicator"
        )

        fig_rsi, ax_rsi = plt.subplots(
            figsize=(14, 5)
        )

        ax_rsi.plot(
            df.index,
            df["RSI"],
            linewidth=2,
            label="RSI"
        )

        ax_rsi.axhline(
            70,
            linestyle="--",
            color="red",
            label="Overbought"
        )

        ax_rsi.axhline(
            30,
            linestyle="--",
            color="green",
            label="Oversold"
        )

        ax_rsi.legend()

        ax_rsi.grid(alpha=0.3)

        st.pyplot(fig_rsi)

        # MACD
        st.subheader(
            "MACD Indicator"
        )

        fig_macd, ax_macd = plt.subplots(
            figsize=(14, 5)
        )

        ax_macd.plot(
            df.index,
            df["MACD"],
            linewidth=2,
            label="MACD"
        )

        ax_macd.plot(
            df.index,
            df["MACD_Signal"],
            linewidth=2,
            label="Signal"
        )

        ax_macd.legend()

        ax_macd.grid(alpha=0.3)

        st.pyplot(fig_macd)

        # VOLATILITY
        st.subheader(
            "Rolling Volatility"
        )

        fig_vol, ax_vol = plt.subplots(
            figsize=(14, 5)
        )

        ax_vol.plot(
            df.index,
            df["Rolling_Volatility"],
            linewidth=2
        )

        ax_vol.grid(alpha=0.3)

        st.pyplot(fig_vol)

    # =====================================================
    # TAB 3 — PORTFOLIO
    # =====================================================
    with tab3:

        st.subheader(
            "Advanced Portfolio Metrics"
        )

        m1, m2, m3, m4, m5 = st.columns(5)

        m1.metric(
            "Expected Return",
            f"{expected_return:.2%}"
        )

        m2.metric(
            "Volatility",
            f"{portfolio_volatility:.2%}"
        )

        m3.metric(
            "Sharpe Ratio",
            round(sharpe_ratio, 2)
        )

        m4.metric(
            "Max Drawdown",
            f"{max_drawdown:.2%}"
        )

        m5.metric(
            "Portfolio Risk",
            portfolio_risk
        )

        # =====================================================
        # PORTFOLIO VS SPY
        # =====================================================
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

        fig_compare, ax_compare = plt.subplots(
            figsize=(14, 6)
        )

        ax_compare.plot(
            portfolio_cumulative.index,
            portfolio_cumulative,
            linewidth=3,
            label="Portfolio"
        )

        ax_compare.plot(
            spy_cumulative.index,
            spy_cumulative,
            linewidth=3,
            label="SPY"
        )

        ax_compare.legend()

        ax_compare.grid(alpha=0.3)

        st.pyplot(fig_compare)

        # =====================================================
        # PORTFOLIO ALLOCATION
        # =====================================================
        st.subheader(
            "Portfolio Allocation"
        )

        allocation_df = pd.DataFrame({
            "Stock": portfolio_tickers,
            "Weight": weights
        })

        fig_alloc, ax_alloc = plt.subplots(
            figsize=(7, 7)
        )

        ax_alloc.pie(
            allocation_df["Weight"],
            labels=allocation_df["Stock"],
            autopct="%1.1f%%"
        )

        st.pyplot(fig_alloc)

        # =====================================================
        # RISK VS RETURN
        # =====================================================
        st.subheader(
            "Risk vs Return Profile"
        )

        asset_returns = (
            returns.mean() * 252
        )

        asset_volatility = (
            returns.std() * np.sqrt(252)
        )

        fig_scatter, ax_scatter = plt.subplots(
            figsize=(10, 7)
        )

        ax_scatter.scatter(
            asset_volatility,
            asset_returns,
            s=150
        )

        for stock in returns.columns:

            ax_scatter.annotate(
                stock,
                (
                    asset_volatility[stock],
                    asset_returns[stock]
                )
            )

        ax_scatter.grid(alpha=0.3)

        st.pyplot(fig_scatter)

        # =====================================================
        # ROLLING VOLATILITY
        # =====================================================
        st.subheader(
            "Rolling Volatility Comparison"
        )

        rolling_volatility = (
            returns.rolling(21).std()
            * np.sqrt(252)
        )

        fig_roll, ax_roll = plt.subplots(
            figsize=(14, 6)
        )

        for stock in rolling_volatility.columns:

            ax_roll.plot(
                rolling_volatility.index,
                rolling_volatility[stock],
                linewidth=2,
                label=stock
            )

        ax_roll.legend()

        ax_roll.grid(alpha=0.3)

        st.pyplot(fig_roll)

        # =====================================================
        # MONTE CARLO
        # =====================================================
        st.subheader(
            "Monte Carlo Portfolio Simulation"
        )

        fig_mc, ax_mc = plt.subplots(
            figsize=(14, 7)
        )

        ax_mc.plot(
            simulation_results,
            alpha=0.12
        )

        ax_mc.grid(alpha=0.3)

        st.pyplot(fig_mc)

        # =====================================================
        # MONTE CARLO METRICS
        # =====================================================
        st.subheader(
            "Simulation Metrics"
        )

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)

        mc1.metric(
            "Expected Value",
            f"${mc_mean:,.0f}"
        )

        mc2.metric(
            "Median Value",
            f"${mc_median:,.0f}"
        )

        mc3.metric(
            "Worst Case",
            f"${mc_min:,.0f}"
        )

        mc4.metric(
            "Best Case",
            f"${mc_max:,.0f}"
        )

        mc5.metric(
            "95% VaR",
            f"${var_95:,.0f}"
        )

        # =====================================================
        # DISTRIBUTION
        # =====================================================
        st.subheader(
            "Distribution of Final Portfolio Values"
        )

        fig_dist, ax_dist = plt.subplots(
            figsize=(12, 6)
        )

        ax_dist.hist(
            ending_values,
            bins=30,
            edgecolor="black"
        )

        ax_dist.axvline(
            var_95,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"95% VaR = ${var_95:,.0f}"
        )

        ax_dist.axvline(
            mc_mean,
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Mean = ${mc_mean:,.0f}"
        )

        ax_dist.legend()

        ax_dist.grid(alpha=0.3)

        st.pyplot(fig_dist)

    # =====================================================
    # TAB 4 — AI EXPLAINABILITY
    # =====================================================
    with tab4:

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
            ascending=True
        )

        # =====================================================
        # SHAP PLOT
        # =====================================================
        fig_shap, ax_shap = plt.subplots(
            figsize=(13, 6)
        )

        colors = [
            "green" if x > 0 else "red"
            for x in shap_df["Impact"]
        ]

        bars = ax_shap.barh(
            shap_df["Feature"],
            shap_df["Impact"],
            color=colors
        )

        for bar in bars:

            width = bar.get_width()

            ax_shap.text(
                width,
                bar.get_y() + bar.get_height()/2,
                f"{width:.3f}",
                va="center",
                fontsize=10
            )

        ax_shap.axvline(
            0,
            color="black"
        )

        ax_shap.grid(alpha=0.3)

        st.pyplot(fig_shap)

        # =====================================================
        # AI SUMMARY
        # =====================================================
        top_positive = (
            shap_df.sort_values(
                by="Impact",
                ascending=False
            )
            .head(3)
        )

        top_negative = (
            shap_df.sort_values(
                by="Impact",
                ascending=True
            )
            .head(3)
        )

        dominant_feature = (
            shap_df.iloc[-1]["Feature"]
        )

        dominant_value = (
            shap_df.iloc[-1]["Impact"]
        )

        ai_summary = f"""
# Executive Portfolio Intelligence Summary

The AI risk engine classifies **{primary_ticker}**
as **{predicted_risk}**.

### Dominant Risk Driver
- {dominant_feature}
- SHAP Impact Score: {dominant_value:.3f}

---

# Technical Indicators

- RSI: **{latest_row['RSI']:.2f}**
- MACD Signal:
**{'Bullish' if latest_row['MACD'] > latest_row['MACD_Signal'] else 'Bearish'}**
- Rolling Volatility:
**{portfolio_volatility:.2%}**
- Sharpe Ratio:
**{sharpe_ratio:.2f}**
- Maximum Drawdown:
**{max_drawdown:.2%}**

---

# Sentiment Intelligence

Market sentiment is currently:
## **{sentiment_label}**

Average Sentiment Score:
**{avg_sentiment:.3f}**

---

# Positive Contributors to Risk

{chr(10).join([f"- {row['Feature']} increased portfolio risk confidence." for _, row in top_positive.iterrows()])}

---

# Negative Contributors to Risk

{chr(10).join([f"- {row['Feature']} reduced portfolio risk exposure." for _, row in top_negative.iterrows()])}

---

# Monte Carlo Forecast

- Worst Case:
${mc_min:,.0f}

- Expected Case:
${mc_mean:,.0f}

- Best Case:
${mc_max:,.0f}

- 95% VaR:
${var_95:,.0f}

---

# Final AI Assessment

The portfolio currently demonstrates
**{portfolio_risk.lower()} characteristics**
with volatility and momentum remaining
the strongest drivers of future performance.
"""

        st.markdown(ai_summary)
    # =====================================================
# FEATURE IMPORTANCE TABLE
# =====================================================
st.subheader(
    "Detailed Feature Importance"
)

# Create normal Python dataframe
shap_display = pd.DataFrame({
    "Feature": [
        str(x)
        for x in shap_df["Feature"].tolist()
    ],
    "Impact": [
        round(float(x), 4)
        for x in shap_df["Impact"].tolist()
    ],
    "Absolute Impact": [
        round(float(x), 4)
        for x in shap_df["AbsImpact"].tolist()
    ]
})

# Sort values
shap_display = shap_display.sort_values(
    by="Absolute Impact",
    ascending=False
)

# Convert dataframe to plain dictionary
safe_data = shap_display.to_dict(
    orient="records"
)

# Render manually
for row in safe_data:

    c1, c2, c3 = st.columns([3, 2, 2])

    c1.write(row["Feature"])

    c2.write(row["Impact"])

    c3.write(row["Absolute Impact"])


    # =====================================================
    # TAB 5 — NEWS
    # =====================================================
    with tab5:

        st.subheader(
            "Market News & Sentiment"
        )

        for _, row in news_df.iterrows():

            if row["Sentiment"] > 0.15:

                st.success(
                    f"🟢 {row['Headline']}"
                )

            elif row["Sentiment"] < -0.15:

                st.error(
                    f"🔴 {row['Headline']}"
                )

            else:

                st.warning(
                    f"🟡 {row['Headline']}"
                )

        st.metric(
            "Overall Market Sentiment",
            sentiment_label
        )

        # =====================================================
        # SENTIMENT DISTRIBUTION
        # =====================================================
        st.subheader(
            "Sentiment Score Distribution"
        )

        fig_sent, ax_sent = plt.subplots(
            figsize=(12, 5)
        )

        ax_sent.bar(
            range(len(news_df)),
            news_df["Sentiment"]
        )

        ax_sent.set_title(
            "Headline Sentiment Scores",
            fontsize=18,
            fontweight="bold"
        )

        ax_sent.set_xlabel(
            "News Headlines"
        )

        ax_sent.set_ylabel(
            "Sentiment Score"
        )

        ax_sent.grid(alpha=0.3)

        st.pyplot(fig_sent)