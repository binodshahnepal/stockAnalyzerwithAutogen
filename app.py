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
    initial_sidebar_state="expanded",
)


# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #0b1020 0%, #0f172a 100%);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 1.35rem 1.5rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(37,99,235,0.22), rgba(16,185,129,0.18));
            border: 1px solid rgba(148,163,184,0.18);
            box-shadow: 0 10px 30px rgba(0,0,0,0.18);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.15rem;
        }
        .hero p {
            margin: 0.4rem 0 0 0;
        }
        .muted {
            color: #94a3b8;
        }
        .section-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(15,23,42,0.72);
            border: 1px solid rgba(148,163,184,0.18);
            margin-bottom: 1rem;
        }
        .recommend-pill {
            display: inline-block;
            padding: 0.5rem 0.85rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 0.75rem;
        }
        .buy {
            background: rgba(16,185,129,0.18);
            color: #34d399;
            border: 1px solid rgba(16,185,129,0.35);
        }
        .hold {
            background: rgba(245,158,11,0.18);
            color: #fbbf24;
            border: 1px solid rgba(245,158,11,0.35);
        }
        .sell {
            background: rgba(239,68,68,0.18);
            color: #f87171;
            border: 1px solid rgba(239,68,68,0.35);
        }
        .small-note {
            color: #94a3b8;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SECRET / ENV HELPERS
# =========================================================
def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


# =========================================================
# TECHNICAL INDICATOR HELPERS
# =========================================================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    series: pd.Series,
    short_span: int = 12,
    long_span: int = 26,
    signal_span: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_short = series.ewm(span=short_span, adjust=False).mean()
    ema_long = series.ewm(span=long_span, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_span, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def classify_recommendation(text: str) -> str:
    upper_text = text.upper()
    if "SELL" in upper_text:
        return "SELL"
    if "BUY" in upper_text:
        return "BUY"
    return "HOLD"


def recommendation_class(rec: str) -> str:
    return {"BUY": "buy", "HOLD": "hold", "SELL": "sell"}.get(rec, "hold")


def format_large_number(value: Any) -> str:
    if value in (None, "N/A"):
        return "N/A"
    try:
        num = float(value)
    except Exception:
        return str(value)

    if abs(num) >= 1_000_000_000_000:
        return f"{num / 1_000_000_000_000:.2f}T"
    if abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    return f"{num:,.0f}"


# =========================================================
# DATA FETCHING
# =========================================================
@st.cache_data(show_spinner=False, ttl=1800)
def get_stock_data(ticker: str, period: str = "6mo") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    info = stock.info

    if hist.empty:
        raise ValueError(f"No market data found for {ticker}.")

    hist = hist.copy()
    hist["MA_20"] = hist["Close"].rolling(20).mean()
    hist["MA_50"] = hist["Close"].rolling(50).mean()
    hist["RSI"] = compute_rsi(hist["Close"])
    hist["MACD"], hist["MACD_SIGNAL"], hist["MACD_HIST"] = compute_macd(hist["Close"])

    return hist, info


@st.cache_data(show_spinner=False, ttl=900)
def get_stock_news(
    company_name: str,
    ticker: str,
    serpapi_key: str,
    num_results: int = 5,
) -> List[Dict[str, str]]:
    if not serpapi_key:
        return []

    try:
        params = {
            "engine": "google",
            "q": f"{company_name} {ticker} stock latest news",
            "api_key": serpapi_key,
            "tbm": "nws",
            "num": num_results,
        }

        results = GoogleSearch(params).get_dict()
        news = results.get("news_results", results.get("organic_results", []))

        cleaned: List[Dict[str, str]] = []
        for item in news[:num_results]:
            cleaned.append(
                {
                    "title": item.get("title", "No title"),
                    "source": item.get("source", "Unknown source"),
                    "snippet": item.get("snippet", item.get("description", "")),
                    "link": item.get("link", ""),
                    "date": item.get("date", ""),
                }
            )
        return cleaned
    except Exception:
        return []


# =========================================================
# SNAPSHOT BUILDERS
# =========================================================
def build_market_snapshot(hist: pd.DataFrame, info: Dict[str, Any], ticker: str) -> str:
    latest = hist.iloc[-1]
    last_5 = hist.tail(5)

    current_price = latest["Close"]
    start_5d = last_5["Close"].iloc[0]
    change_5d = ((current_price - start_5d) / start_5d) * 100
    high_5d = last_5["High"].max()
    low_5d = last_5["Low"].min()
    avg_volume_5d = int(last_5["Volume"].mean())

    trend = "Bullish" if latest["MA_20"] > latest["MA_50"] else "Bearish"

    rsi_signal = "Neutral"
    if pd.notna(latest["RSI"]):
        if latest["RSI"] > 70:
            rsi_signal = "Overbought"
        elif latest["RSI"] < 30:
            rsi_signal = "Oversold"

    macd_signal = "Bullish crossover" if latest["MACD"] > latest["MACD_SIGNAL"] else "Bearish crossover"

    snapshot = f"""
Ticker: {ticker}
Company: {info.get('longName', ticker)}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}
Market Cap: {format_large_number(info.get('marketCap', 'N/A'))}
Current Price: ${current_price:.2f}
5-Day Change: {change_5d:+.2f}%
5-Day High: ${high_5d:.2f}
5-Day Low: ${low_5d:.2f}
Average Volume (5D): {avg_volume_5d:,}
MA20: ${latest['MA_20']:.2f}
MA50: ${latest['MA_50']:.2f}
Trend Signal: {trend}
RSI(14): {latest['RSI']:.2f}
RSI Interpretation: {rsi_signal}
MACD: {latest['MACD']:.4f}
MACD Signal: {latest['MACD_SIGNAL']:.4f}
MACD Histogram: {latest['MACD_HIST']:.4f}
MACD Interpretation: {macd_signal}
52 Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}
52 Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}
Trailing PE: {info.get('trailingPE', 'N/A')}
Forward PE: {info.get('forwardPE', 'N/A')}
Dividend Yield: {info.get('dividendYield', 'N/A')}
""".strip()

    return snapshot


def build_news_text(news_items: List[Dict[str, str]]) -> str:
    if not news_items:
        return "No external news available. Analyze using stock data and technical indicators only."

    parts: List[str] = []
    for idx, item in enumerate(news_items, start=1):
        parts.append(
            f"{idx}. Title: {item['title']}\n"
            f"Source: {item['source']}\n"
            f"Date: {item['date']}\n"
            f"Summary: {item['snippet']}"
        )
    return "\n\n".join(parts)


# =========================================================
# AUTOGEN HELPERS
# =========================================================
def _extract_result_text(task_result: Any) -> str:
    messages = getattr(task_result, "messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    return "No report generated."


async def _run_autogen_analysis(
    groq_api_key: str,
    company_name: str,
    ticker: str,
    market_snapshot: str,
    news_text: str,
    risk_profile: str,
    horizon: str,
) -> Tuple[str, str, str]:
    model_client = OpenAIChatCompletionClient(
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
        temperature=0.3,
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
    )

    news_agent = AssistantAgent(
        name="news_analyst",
        model_client=model_client,
        system_message=(
            "You are a market news analyst. Analyze stock-related news for sentiment, catalysts, risks, "
            "and what investors should watch. Be balanced, concise, and practical."
        ),
    )

    technical_agent = AssistantAgent(
        name="technical_analyst",
        model_client=model_client,
        system_message=(
            "You are a technical stock analyst. Interpret moving averages, RSI, MACD, price trend, "
            "momentum, and near-term setup using the supplied market snapshot."
        ),
    )

    investment_agent = AssistantAgent(
        name="investment_committee",
        model_client=model_client,
        system_message=(
            "You are an investment committee chair. Combine the news analysis and technical analysis "
            "into a final investor report. You must end with exactly one recommendation: BUY, HOLD, or SELL. "
            "Include a confidence score from 0 to 100 and a risk level of Low, Medium, or High. "
            "This is educational analysis, not guaranteed financial advice."
        ),
    )

    try:
        news_result = await news_agent.run(
            task=f"""
Analyze these latest news items for {company_name} ({ticker}).

Investor profile:
- Risk appetite: {risk_profile}
- Time horizon: {horizon}

News items:
{news_text}

Return markdown with these sections:
1. Overall Sentiment
2. Bullish Catalysts
3. Bearish Risks
4. What Investors Should Watch
"""
        )
        news_summary = _extract_result_text(news_result)

        technical_result = await technical_agent.run(
            task=f"""
Analyze this market snapshot for {company_name} ({ticker}).

Investor profile:
- Risk appetite: {risk_profile}
- Time horizon: {horizon}

Market snapshot:
{market_snapshot}

Return markdown with these sections:
1. Trend
2. Momentum
3. Technical Strengths
4. Technical Weaknesses
5. Near-Term Setup
"""
        )
        technical_summary = _extract_result_text(technical_result)

        final_result = await investment_agent.run(
            task=f"""
Create a final investor report for {company_name} ({ticker}).

Investor profile:
- Risk appetite: {risk_profile}
- Time horizon: {horizon}

News analysis:
{news_summary}

Technical analysis:
{technical_summary}

Return markdown with these exact sections:
1. Executive Summary
2. News Highlights
3. Technical Analysis
4. Fundamental/Business Context
5. Final Recommendation
6. Risk Level and Confidence
7. Short-Term Outlook
8. Important Disclaimer

Requirements:
- Mention RSI, MACD, and moving averages in the technical section.
- Clearly state one recommendation only: BUY, HOLD, or SELL.
- Include a confidence score from 0 to 100.
- Include a risk level: Low, Medium, or High.
- Keep it practical and under about 500 words.
"""
        )
        final_report = _extract_result_text(final_result)
        return final_report, news_summary, technical_summary
    finally:
        await model_client.close()


def generate_ai_report(
    groq_api_key: str,
    company_name: str,
    ticker: str,
    market_snapshot: str,
    news_text: str,
    risk_profile: str,
    horizon: str,
) -> Tuple[str, str, str]:
    return asyncio.run(
        _run_autogen_analysis(
            groq_api_key=groq_api_key,
            company_name=company_name,
            ticker=ticker,
            market_snapshot=market_snapshot,
            news_text=news_text,
            risk_profile=risk_profile,
            horizon=horizon,
        )
    )


# =========================================================
# CHARTS
# =========================================================
def build_chart(hist: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.56, 0.22, 0.22],
        subplot_titles=(f"{ticker} Price", "RSI", "MACD"),
    )

    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA_20"], name="MA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA_50"], name="MA 50"), row=1, col=1)

    fig.add_trace(go.Scatter(x=hist.index, y=hist["RSI"], name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", row=2, col=1)

    fig.add_trace(go.Scatter(x=hist.index, y=hist["MACD"], name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MACD_SIGNAL"], name="Signal"), row=3, col=1)
    fig.add_trace(go.Bar(x=hist.index, y=hist["MACD_HIST"], name="Histogram"), row=3, col=1)

    fig.update_layout(
        height=850,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.caption("API keys are loaded securely from Streamlit secrets or environment variables.")

    groq_api_key = get_secret("GROQ_API_KEY")
    serpapi_key = get_secret("SERPAPI_API_KEY")

    if groq_api_key:
        st.success("Groq key loaded")
    else:
        st.warning("Groq key not found")

    if serpapi_key:
        st.success("SerpAPI key loaded")
    else:
        st.info("SerpAPI key not found. News analysis will be limited.")

    st.markdown("---")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    company_name = st.text_input("Company name", value="Apple Inc.")
    period = st.selectbox("Chart period", ["3mo", "6mo", "1y"], index=1)
    horizon = st.selectbox("Investment horizon", ["Short-term", "Medium-term", "Long-term"], index=0)
    risk_profile = st.selectbox("Risk appetite", ["Conservative", "Moderate", "Aggressive"], index=1)
    news_count = st.slider("Number of news articles", 3, 10, 5)
    run_analysis = st.button("🚀 Analyze Stock", use_container_width=True)


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class='hero'>
        <h1>📈 AI Stock Analyst Pro</h1>
        <p class='muted'>Multi-agent stock analysis using AutoGen + Groq, with live charts, indicators, news, and AI-powered investment summaries.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Educational use only. This app does not provide guaranteed financial advice.")

if not run_analysis:
    st.info("Choose a ticker and click **Analyze Stock**.")
    st.stop()

if not groq_api_key:
    st.error("Groq API key is missing. Add it in Streamlit secrets or environment variables.")
    st.stop()


# =========================================================
# MAIN APP
# =========================================================
try:
    with st.spinner("Fetching stock data..."):
        hist, info = get_stock_data(ticker, period)
        news_items = get_stock_news(company_name, ticker, serpapi_key, news_count)
        snapshot = build_market_snapshot(hist, info, ticker)
        news_text = build_news_text(news_items)

    latest = hist.iloc[-1]
    last_5 = hist.tail(5)
    current_price = latest["Close"]
    start_5d = last_5["Close"].iloc[0]
    change_5d = ((current_price - start_5d) / start_5d) * 100
    rsi_value = latest["RSI"]
    trend_label = "Bullish" if latest["MA_20"] > latest["MA_50"] else "Bearish"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${current_price:.2f}")
    m2.metric("5-Day Change", f"{change_5d:+.2f}%")
    m3.metric("RSI(14)", f"{rsi_value:.2f}")
    m4.metric("Trend", trend_label)

    left, right = st.columns([1.35, 0.9])

    with left:
        st.markdown("### 📊 Technical Dashboard")
        st.plotly_chart(build_chart(hist, ticker), use_container_width=True)

    with right:
        st.markdown("### 🧾 Market Snapshot")
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.write(f"**Company:** {info.get('longName', company_name)}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")
        st.write(f"**Market Cap:** {format_large_number(info.get('marketCap', 'N/A'))}")
        st.write(f"**52W High:** {info.get('fiftyTwoWeekHigh', 'N/A')}")
        st.write(f"**52W Low:** {info.get('fiftyTwoWeekLow', 'N/A')}")
        st.write(f"**Trailing PE:** {info.get('trailingPE', 'N/A')}")
        st.write(f"**Forward PE:** {info.get('forwardPE', 'N/A')}")
        st.write(f"**Dividend Yield:** {info.get('dividendYield', 'N/A')}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 📰 Latest News")
        if news_items:
            for item in news_items:
                with st.container(border=True):
                    st.markdown(f"**{item['title']}**")
                    st.caption(f"{item['source']} • {item['date']}")
                    st.write(item["snippet"])
                    if item["link"]:
                        st.link_button("Open article", item["link"], use_container_width=True)
        else:
            st.warning("No news fetched. Add a SerpAPI key to enable live news analysis.")

    st.markdown("### 🤖 AI Investment Report")
    with st.spinner("Generating AutoGen report..."):
        report, news_summary, technical_summary = generate_ai_report(
            groq_api_key=groq_api_key,
            company_name=company_name,
            ticker=ticker,
            market_snapshot=snapshot,
            news_text=news_text,
            risk_profile=risk_profile,
            horizon=horizon,
        )

    recommendation = classify_recommendation(report)
    rec_class = recommendation_class(recommendation)

    st.markdown(
        f"<span class='recommend-pill {rec_class}'>Recommendation: {recommendation}</span>",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Final Report", "News Agent", "Technical Agent"])

    with tab1:
        st.markdown(report)

    with tab2:
        st.markdown(news_summary)

    with tab3:
        st.markdown(technical_summary)

    st.download_button(
        "⬇️ Download report as Markdown",
        data=report,
        file_name=f"{ticker.lower()}_investment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.stop()