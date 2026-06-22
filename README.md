# AI Portfolio Intelligence Platform

An AI-powered portfolio analytics platform that combines quantitative finance, machine learning, explainable AI, Monte Carlo simulation, and financial sentiment analysis into a single interactive dashboard.

The platform is designed to help investors understand not only portfolio risk, but also the underlying factors driving investment decisions through explainable machine learning.

---

## Overview

Traditional portfolio dashboards focus on historical performance and basic metrics. This project extends portfolio analytics by integrating:

- Machine Learning-based Risk Classification
- Explainable AI using SHAP
- Portfolio Forecasting via Monte Carlo Simulation
- Technical Analysis Indicators
- Financial News Sentiment Analysis
- Risk-Adjusted Performance Metrics
- Interactive Visual Analytics

The objective is to provide a more transparent and intelligent approach to portfolio analysis.

---

## Key Features

### Portfolio Risk Prediction

A trained machine learning model evaluates market and technical indicators to classify investments into:

- Low Risk
- Medium Risk
- High Risk

The model generates prediction probabilities and confidence scores for each classification.

---

### Explainable AI (SHAP)

Instead of treating predictions as a black box, SHAP values are used to explain:

- Which features increased portfolio risk
- Which features reduced portfolio risk
- Relative importance of each factor

This provides transparency into the model's decision-making process.

---

### Technical Analysis

The platform automatically calculates:

- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- 20-Day Moving Average
- 50-Day Moving Average
- Rolling Volatility
- Momentum Indicators

These indicators are used both for visualization and risk prediction.

---

### Portfolio Analytics

Institutional-style portfolio metrics include:

- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Portfolio Risk Classification
- Benchmark Comparison Against SPY

---

### Monte Carlo Simulation

Future portfolio outcomes are simulated using Monte Carlo methods.

Outputs include:

- Expected Portfolio Value
- Best Case Scenario
- Worst Case Scenario
- Median Portfolio Value
- 95% Value at Risk (VaR)

This helps estimate future uncertainty and downside risk.

---

### Financial News Sentiment Analysis

The platform retrieves real-time financial headlines and applies NLP-based sentiment scoring using VADER.

Sentiment categories:

- Bullish
- Neutral
- Bearish

This provides additional market context alongside quantitative metrics.

---

## Dashboard Components

### Overview

- Risk Classification
- Prediction Confidence
- Portfolio KPIs
- Stock Trend Analysis

### Technical Analysis

- RSI Visualization
- MACD Signals
- Rolling Volatility

### Portfolio Intelligence

- Portfolio vs SPY Benchmark
- Allocation Analysis
- Risk vs Return Visualization
- Monte Carlo Forecasting
- Value at Risk Analysis

### AI Explainability

- SHAP Feature Importance
- Risk Driver Analysis
- Executive Portfolio Intelligence Summary

### News & Sentiment

- Financial Headlines
- Sentiment Scoring
- Market Sentiment Distribution

---

## Technology Stack

### Data Sources

- Yahoo Finance API (yFinance)
- Yahoo Finance RSS Feeds

### Machine Learning

- Scikit-Learn
- SHAP

### Quantitative Finance

- NumPy
- Pandas
- TA-Lib / Technical Analysis Library

### Visualization

- Streamlit
- Matplotlib

### NLP

- VADER Sentiment Analysis

---

## Project Architecture

```
Yahoo Finance
       |
       v
Data Collection
       |
       v
Feature Engineering
(RSI, MACD, Volatility,
Momentum, Bollinger Bands)
       |
       v
Risk Classification Model
       |
       +------> SHAP Explainability
       |
       +------> Portfolio Analytics
       |
       +------> Monte Carlo Simulation
       |
       +------> Sentiment Analysis
       |
       v
Streamlit Dashboard
```

---

## Example Insights

Some of the most interesting findings from the project include:

- Portfolio risk is often driven more by volatility and momentum than by market sentiment alone.
- Explainable AI reveals how different indicators influence risk predictions under changing market conditions.
- Monte Carlo simulations provide a more realistic view of uncertainty than point estimates alone.
- Combining quantitative analytics with sentiment intelligence creates a richer investment decision framework.

---

## Future Enhancements

- Portfolio Optimization using Efficient Frontier
- Multi-Factor Risk Models
- Sector Exposure Analysis
- Real-Time Market Streaming
- LLM-Based Investment Insights
- Alternative Data Integration
- Scenario Stress Testing

---

## Author

**Navika Maglani**

Data Scientist | AI Engineer | Machine Learning Enthusiast

Focused on building intelligent systems that combine machine learning, explainable AI, and data-driven decision making.
