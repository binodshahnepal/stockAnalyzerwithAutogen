import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from serpapi import GoogleSearch

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Stock Analyst Pro",
    page_icon="📈",
    layout="wide",
)

# =========================================================
# SECRET HELPER
# =========================================================

def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def compute_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series):
    short = series.ewm(span=12).mean()
    long = series.ewm(span=26).mean()

    macd = short - long
    signal = macd.ewm(span=9).mean()

    hist = macd - signal

    return macd, signal, hist


# =========================================================
# STOCK DATA
# =========================================================

@st.cache_data(ttl=1800)
def get_stock_data(ticker: str, period="6mo"):

    stock = yf.Ticker(ticker)

    hist = stock.history(period=period)
    info = stock.info

    hist["MA20"] = hist["Close"].rolling(20).mean()
    hist["MA50"] = hist["Close"].rolling(50).mean()

    hist["RSI"] = compute_rsi(hist["Close"])

    hist["MACD"], hist["MACD_SIGNAL"], hist["MACD_HIST"] = compute_macd(hist["Close"])

    return hist, info


# =========================================================
# NEWS FETCH
# =========================================================

@st.cache_data(ttl=900)
def get_news(company, ticker, serpapi_key):

    if not serpapi_key:
        return []

    params = {
        "engine": "google",
        "q": f"{company} {ticker} stock news",
        "api_key": serpapi_key,
        "tbm": "nws",
        "num": 5,
    }

    results = GoogleSearch(params).get_dict()

    news = results.get("news_results", [])

    cleaned = []

    for n in news:

        cleaned.append(
            {
                "title": n.get("title"),
                "source": n.get("source"),
                "snippet": n.get("snippet"),
                "link": n.get("link"),
            }
        )

    return cleaned


# =========================================================
# MARKET SNAPSHOT
# =========================================================

def build_snapshot(hist, info, ticker):

    latest = hist.iloc[-1]

    return f"""
Ticker: {ticker}
Company: {info.get("longName",ticker)}

Current Price: {latest["Close"]}

MA20: {latest["MA20"]}
MA50: {latest["MA50"]}

RSI: {latest["RSI"]}

MACD: {latest["MACD"]}

Sector: {info.get("sector")}
Industry: {info.get("industry")}
Market Cap: {info.get("marketCap")}
"""


# =========================================================
# AUTOGEN ANALYSIS (Gemini)
# =========================================================

async def run_agents(gemini_key, company, ticker, snapshot, news_text):

    model_client = OpenAIChatCompletionClient(

        model="gemini-1.5-flash",

        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",

        api_key=gemini_key,

        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "google",
            "structured_output": False,
        },
    )

    news_agent = AssistantAgent(
        name="news_analyst",
        model_client=model_client,
        system_message="Analyze news sentiment for stock investors.",
    )

    tech_agent = AssistantAgent(
        name="technical_analyst",
        model_client=model_client,
        system_message="Analyze technical indicators.",
    )

    invest_agent = AssistantAgent(
        name="investment_committee",
        model_client=model_client,
        system_message="Produce final BUY HOLD SELL recommendation.",
    )

    news_result = await news_agent.run(
        task=f"""
Analyze news for {company} ({ticker})

News:
{news_text}

Provide bullish and bearish insights.
"""
    )

    news_summary = news_result.messages[-1].content

    tech_result = await tech_agent.run(
        task=f"""
Analyze this technical data

{snapshot}

Explain RSI, MACD and moving averages.
"""
    )

    tech_summary = tech_result.messages[-1].content

    final_result = await invest_agent.run(
        task=f"""
News analysis:
{news_summary}

Technical analysis:
{tech_summary}

Give final recommendation:

BUY HOLD or SELL

Include reasoning.
"""
    )

    report = final_result.messages[-1].content

    return report, news_summary, tech_summary


def generate_report(key, company, ticker, snapshot, news_text):

    return asyncio.run(run_agents(key, company, ticker, snapshot, news_text))


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("Configuration")

    gemini_key = get_secret("GEMINI_API_KEY")
    serpapi_key = get_secret("SERPAPI_API_KEY")

    ticker = st.text_input("Ticker", "AAPL")
    company = st.text_input("Company", "Apple")

    run = st.button("Analyze Stock")


# =========================================================
# UI
# =========================================================

st.title("AI Stock Analyst Pro")

if not run:
    st.stop()


hist, info = get_stock_data(ticker)

news = get_news(company, ticker, serpapi_key)

snapshot = build_snapshot(hist, info, ticker)

news_text = "\n".join([n["title"] + " " + n["snippet"] for n in news])


report, news_summary, tech_summary = generate_report(
    gemini_key,
    company,
    ticker,
    snapshot,
    news_text,
)

st.subheader("AI Recommendation")

st.write(report)


tabs = st.tabs(["Final Report", "News Analysis", "Technical Analysis"])

with tabs[0]:
    st.write(report)

with tabs[1]:
    st.write(news_summary)

with tabs[2]:
    st.write(tech_summary)