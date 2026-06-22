# AI Portfolio Intelligence Platform

An AI-powered portfolio analytics platform that combines **Machine Learning**, **Explainable AI (XAI)**, **Quantitative Finance**, **Monte Carlo Simulation**, and **Financial News Sentiment Analysis** into a single interactive dashboard.

The platform helps investors move beyond traditional performance metrics by providing transparent, data-driven insights into portfolio risk, market conditions, and future uncertainty through explainable machine learning.

---

##  Overview

Traditional portfolio dashboards primarily focus on historical performance and basic financial metrics. This project extends portfolio analytics by integrating artificial intelligence, quantitative finance, and natural language processing into a unified decision-support system.

### Key Capabilities

- Machine Learning-Based Risk Classification
- Explainable AI using SHAP
- Monte Carlo Portfolio Forecasting
- Technical Analysis Indicators
- Financial News Sentiment Analysis
- Risk-Adjusted Performance Metrics
- Interactive Portfolio Visualization

The objective is to provide a more transparent and intelligent framework for evaluating portfolio performance and investment risk.

---

## Key Features

###  Portfolio Risk Prediction

A machine learning model evaluates market behavior and technical indicators to classify investments into:

- Low Risk
- Medium Risk
- High Risk

The model also generates probability scores and confidence levels for each prediction.

---

###  Explainable AI (SHAP)

Rather than treating predictions as a black box, SHAP (SHapley Additive exPlanations) is used to explain:

- Which factors increased portfolio risk
- Which factors reduced portfolio risk
- Relative contribution of each feature
- Dominant risk drivers behind predictions

This provides transparency into the model's decision-making process and improves trust in AI-generated insights.

---

###  Technical Analysis

The platform automatically calculates:

- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- 20-Day Moving Average (SMA-20)
- 50-Day Moving Average (SMA-50)
- Rolling Volatility
- Momentum Indicators

These indicators are used both for visualization and machine learning feature engineering.

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

###  Monte Carlo Simulation

Future portfolio outcomes are simulated using Monte Carlo forecasting techniques.

Simulation outputs include:

- Expected Portfolio Value
- Best Case Scenario
- Worst Case Scenario
- Median Portfolio Value
- 95% Value at Risk (VaR)

These forecasts help quantify uncertainty and estimate downside risk.

---

###  Financial News Sentiment Analysis

The platform retrieves real-time financial headlines and performs sentiment scoring using VADER NLP.

Sentiment categories include:

- Bullish
- Neutral
- Bearish

This provides additional market context alongside quantitative portfolio metrics.

---

## Dashboard Components

###  Overview

Provides a high-level summary of portfolio performance and AI predictions.

**Features:**

- Risk Classification
- Prediction Confidence
- Portfolio KPIs
- Stock Trend Analysis
- Market Sentiment Summary

---

###  Technical Analysis

Visualizes key technical indicators.

**Includes:**

- RSI Analysis
- MACD Signals
- Rolling Volatility Trends
- Price Momentum Indicators

---

### Portfolio Intelligence

Provides portfolio-level risk and performance analytics.

**Features:**

- Portfolio vs SPY Benchmark
- Portfolio Allocation Analysis
- Risk vs Return Visualization
- Rolling Volatility Comparison
- Sharpe Ratio Analysis
- Maximum Drawdown Analysis
- Monte Carlo Forecasting
- Value at Risk (VaR)

---

###  AI Explainability

Explains model predictions using SHAP.

**Features:**

- SHAP Feature Importance Visualization
- Dominant Risk Driver Analysis
- Positive and Negative Risk Contributors
- Executive AI Portfolio Intelligence Summary

---

###  News & Sentiment Intelligence

Analyzes financial news headlines and market sentiment.

**Includes:**

- Financial Headlines Feed
- Sentiment Classification
- Sentiment Distribution Analysis
- Market Sentiment Overview

---

## Technology Stack

### Data Sources

- Yahoo Finance API (yFinance)
- Yahoo Finance RSS Feeds

### Machine Learning

- Scikit-Learn
- Joblib

### Explainable AI

- SHAP

### Quantitative Finance

- NumPy
- Pandas
- Technical Analysis (TA)

### Natural Language Processing

- VADER Sentiment Analysis

### Visualization

- Streamlit
- Matplotlib

---

##  Project Architecture

```text
Yahoo Finance
       │
       ▼
Data Collection
       │
       ▼
Feature Engineering
(RSI, MACD, Volatility,
Momentum, Bollinger Bands)
       │
       ▼
Risk Classification Model
       │
       ├── SHAP Explainability
       │
       ├── Portfolio Analytics
       │
       ├── Monte Carlo Simulation
       │
       ├── Sentiment Analysis
       │
       ▼
Streamlit Dashboard
```

---

##  Project Structure

```text
AI-Portfolio-Intelligence/
│
├── app_pro.py
│
├── models/
│   └── risk_classifier.pkl
│
├── requirements.txt
│
├── README.md
│
└── assets/
    ├── dashboard.png
    ├── portfolio_analysis.png
    ├── shap_explainability.png
    └── sentiment_analysis.png
```

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/AI-Portfolio-Intelligence.git

cd AI-Portfolio-Intelligence
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

##  Running the Application

Launch the Streamlit dashboard:

```bash
streamlit run app_pro.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 📈 Example Insights

Some key observations generated by the platform include:

- Portfolio risk is often driven more by volatility and momentum than by sentiment alone.
- Explainable AI reveals how technical indicators influence risk predictions under changing market conditions.
- Monte Carlo simulations provide a more realistic representation of uncertainty than single-point forecasts.
- Combining quantitative analytics with sentiment intelligence creates a richer investment decision framework.
- Risk-adjusted performance metrics provide deeper insight than returns alone.

---

##  Future Enhancements

- Efficient Frontier Optimization
- Portfolio Rebalancing Engine
- Multi-Factor Risk Models
- Sector Exposure Analysis
- Real-Time Market Streaming
- LLM-Based Investment Insights
- Alternative Data Integration
- Scenario Stress Testing
- Power BI Executive Dashboard
- AWS Cloud Deployment
- Portfolio Recommendation Engine

---

##  Dashboard Preview

### Overview Dashboard

- Risk Classification
- Performance Metrics
- Market Sentiment Overview

### Portfolio Intelligence

- Allocation Analysis
- Benchmark Comparison
- Monte Carlo Forecasting
- Risk Metrics

### Explainable AI

- SHAP Visualizations
- Feature Importance Ranking
- AI Investment Intelligence Summary

### Sentiment Analytics

- Financial Headlines
- Sentiment Distribution
- Market Mood Analysis

---

##  Author

**Navika Maglani**

Data Scientist | AI Engineer | Quantitative Analytics | Explainable AI

Focused on building intelligent systems that combine machine learning, explainable AI, quantitative finance, and data-driven decision making.

- LinkedIn: www.linkedin.com/in/navikamaglani
- GitHub: github.com/NavikaMaglani

---

##  Support

If you found this project useful:

- Star the repository
- Share feedback
- Connect on LinkedIn
- Explore other AI and Data Science projects

---
**Built with Machine Learning, Quantitative Finance, Explainable AI, and Streamlit**
