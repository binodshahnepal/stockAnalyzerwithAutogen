import asyncio
import os
import re
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
            background:
                radial-gradient(circle at top right, rgba(37,99,235,0.12), transparent 30%),
                radial-gradient(circle at top left, rgba(16,185,129,0.10), transparent 26%),
                linear-gradient(180deg, #070b16 0%, #0b1120 100%);
        }
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            max-width: 1450px;
        }
        .hero {
            padding: 1.45rem 1.6rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(37,99,235,0.26), rgba(16,185,129,0.16));
            border: 1px solid rgba(148,163,184,0.14);
            box-shadow: 0 18px 40px rgba(0,0,0,0.20);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.3rem;
            line-height: 1.1;
        }
        .muted {
            color: #94a3b8;
        }
        .mini-card {
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(15,23,42,0.70);
            border: 1px solid rgba(148,163,184,0.14);
            backdrop-filter: blur(8px);
        }
        .section-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(15,23,42,0.72);
            border: 1px solid rgba(148,163,184,0.16);
            margin-bottom: 1rem;
        }
        .recommend-pill {
            display: inline-block;
            padding: 0.5rem 0.9rem;
            border-radius: 999px;
            font-weight: 800;
            font-size: 1rem;
            margin-bottom: 0.8rem;
        }
        .buy {
            background: rgba(16,185,129,0.18);
            color: #34d399;
            border: 1px solid rgba(16,185,129,0.32);
        }
        .hold {
            background: rgba(245,158,11,0.18);
            color: #fbbf24;
            border: 1px solid rgba(245,158,11,0.32);
        }
        .sell {
            background: rgba(239,68,68,0.18);
            color: #f87171;
            border: 1px solid rgba(239,68,68,0.32);
        }
        .subtle {
            color: #cbd5e1;
            font-size: 0.94rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


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


def format_large_number(value: Any) -> str:
    if value in (None, "", "N/A"):
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


def classify_recommendation(text: str) -> str:
    upper = text.upper()
    if "SELL" in upper:
        return "SELL"
    if "BUY" in upper:
        return "BUY"
    return "HOLD"


def recommendation_css_class(rec: str) -> str:
    return {"BUY": "buy", "HOLD": "hold", "SELL": "sell"}.get(rec, "hold")


def extract_confidence(text: str) -> str:
    match = re.search(r"confidence[^0-9]{0,20}(\d{1,3})", text, flags=re.IGNORECASE)
    return f"{match.group(1)}%" if match else "N/A"


def extract_risk(text: str) -> str:
    match = re.search(r"risk level[^A-Za-z]{0,20}(low|medium|high)", text, flags=re.IGNORECASE)
    return match.group(1).capitalize() if match else "N/A"


def summarize_signal(latest: pd.Series) -> Tuple[str, str]:
    trend = "Bullish" if latest["MA_20"] > latest["MA_50"] else "Bearish"

    rsi_signal = "Neutral"
    if pd.notna(latest["RSI"]):
        if latest["RSI"] > 70:
            rsi_signal = "Overbought"
        elif latest["RSI"] < 30:
            rsi_signal = "Oversold"

    macd_signal = "Bullish Crossover" if latest["MACD"] > latest["MACD_SIGNAL"] else "Bearish Crossover"
    return trend, f"{rsi_signal} • {macd_signal}"


# =========================================================
# DATA
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


def build_news_text(news_items: List[Dict[str, str]]) -> str:
    if not news_items:
        return "No external news was retrieved. Use market data and technical indicators only."

    parts: List[str] = []
    for idx, item in enumerate(news_items, start=1):
        parts.append(
            f"{idx}. Title: {item['title']}\n"
            f"Source: {item['source']}\n"
            f"Date: {item['date']}\n"
            f"Summary: {item['snippet']}"
        )
    return "\n\n".join(parts)


def build_market_snapshot(hist: pd.DataFrame, info: Dict[str, Any], ticker: str) -> str:
    latest = hist.iloc[-1]
    last_5 = hist.tail(5)

    current_price = latest["Close"]
    start_5d = last_5["Close"].iloc[0]
    change_5d = ((current_price - start_5d) / start_5d) * 100
    high_5d = last_5["High"].max()
    low_5d = last_5["Low"].min()
    avg_volume_5d = int(last_5["Volume"].mean())

    trend, combined_signal = summarize_signal(latest)

    return f"""
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
RSI(14): {latest['RSI']:.2f}
MACD: {latest['MACD']:.4f}
MACD Signal: {latest['MACD_SIGNAL']:.4f}
Trend Signal: {trend}
Combined Indicator Signal: {combined_signal}
52 Week High: {info.get('fiftyTwoWeekHigh', 'N/A')}
52 Week Low: {info.get('fiftyTwoWeekLow', 'N/A')}
Trailing PE: {info.get('trailingPE', 'N/A')}
Forward PE: {info.get('forwardPE', 'N/A')}
Dividend Yield: {info.get('dividendYield', 'N/A')}
""".strip()


# =========================================================
# AUTOGEN + GEMINI (2 AGENTS)
# =========================================================
def _extract_result_text(task_result: Any) -> str:
    messages = getattr(task_result, "messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content
    return "No analysis returned."


async def _safe_run_agent(agent: AssistantAgent, task: str, retries: int = 3, delay: int = 4):
    last_error = None
    for attempt in range(retries):
        try:
            return await agent.run(task=task)
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            if any(x in msg for x in ["rate limit", "too many requests", "429", "resource_exhausted"]) and attempt < retries - 1:
                await asyncio.sleep(delay)
                continue
            raise e
    raise last_error


async def _run_autogen_analysis(
    gemini_key: str,
    company_name: str,
    ticker: str,
    market_snapshot: str,
    news_text: str,
    risk_profile: str,
    horizon: str,
) -> Tuple[str, str]:
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.5-flash",
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

    analysis_agent = AssistantAgent(
        name="market_research_analyst",
        model_client=model_client,
        system_message=(
            "You are a senior stock research analyst. Analyze both stock news and technical indicators. "
            "Be concise, balanced, practical, and investor-focused."
        ),
    )

    decision_agent = AssistantAgent(
        name="investment_committee",
        model_client=model_client,
        system_message=(
            "You are an investment committee chair. Convert research analysis into a final investor report. "
            "State exactly one recommendation: BUY, HOLD, or SELL. Include confidence score 0-100 and risk level "
            "Low, Medium, or High. This is educational analysis, not guaranteed advice."
        ),
    )

    try:
        research_result = await _safe_run_agent(
            analysis_agent,
            task=f"""
Analyze {company_name} ({ticker}) using both the news and technical snapshot below.

Investor profile:
- Risk appetite: {risk_profile}
- Time horizon: {horizon}

News:
{news_text}

Technical snapshot:
{market_snapshot}

Return markdown with these sections:
1. Overall Sentiment
2. Bullish Drivers
3. Bearish Risks
4. Technical Interpretation
5. What Investors Should Watch
""",
        )
        research_summary = _extract_result_text(research_result)

        final_result = await _safe_run_agent(
            decision_agent,
            task=f"""
Using the following research analysis, create a final investor report for {company_name} ({ticker}).

Research analysis:
{research_summary}

Return markdown with these exact sections:
1. Executive Summary
2. News and Market Drivers
3. Technical Analysis
4. Final Recommendation
5. Risk Level and Confidence
6. Short-Term Outlook
7. Important Disclaimer

Requirements:
- Mention RSI, MACD, and moving averages in the technical section.
- Clearly state exactly one recommendation: BUY, HOLD, or SELL.
- Include a confidence score from 0 to 100.
- Include a risk level: Low, Medium, or High.
- Keep it practical and under about 450 words.
""",
        )
        final_report = _extract_result_text(final_result)
        return final_report, research_summary
    finally:
        await model_client.close()


def generate_ai_report(
    gemini_key: str,
    company_name: str,
    ticker: str,
    market_snapshot: str,
    news_text: str,
    risk_profile: str,
    horizon: str,
) -> Tuple[str, str]:
    return asyncio.run(
        _run_autogen_analysis(
            gemini_key=gemini_key,
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
        vertical_spacing=0.055,
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
        height=860,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=16, r=16, t=50, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.caption("Keys are loaded from Streamlit Secrets or environment variables.")

    gemini_key = get_secret("GEMINI_API_KEY")
    serpapi_key = get_secret("SERPAPI_API_KEY")

    if gemini_key:
        st.success("Gemini key loaded")
    else:
        st.warning("Gemini key not found")

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
    news_count = st.slider("News articles", 3, 8, 4)
    run_analysis = st.button("🚀 Analyze Stock", use_container_width=True)


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class='hero'>
        <h1>📈 AI Stock Analyst Pro</h1>
        <p class='muted'>Slick multi-agent stock analysis with AutoGen + Gemini, combining live market news, technical indicators, and investor-focused AI summaries.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Educational use only. This app does not provide guaranteed financial advice.")

if not run_analysis:
    st.info("Choose a ticker and click **Analyze Stock**.")
    st.stop()

if not gemini_key:
    st.error("Gemini API key is missing. Add it in Streamlit Secrets or environment variables.")
    st.stop()


# =========================================================
# MAIN
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
    trend_label, combined_signal = summarize_signal(latest)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Current Price", f"${current_price:.2f}")
    mc2.metric("5-Day Change", f"{change_5d:+.2f}%")
    mc3.metric("RSI(14)", f"{latest['RSI']:.2f}")
    mc4.metric("Signal", trend_label)

    top_left, top_right = st.columns([1.35, 0.9])

    with top_left:
        st.markdown("### 📊 Technical Dashboard")
        st.plotly_chart(build_chart(hist, ticker), use_container_width=True)

    with top_right:
        st.markdown("### 🧾 Snapshot")
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
        st.write(f"**Combined Signal:** {combined_signal}")
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
    with st.spinner("Running AutoGen agents on Gemini..."):
        report, research_summary = generate_ai_report(
            gemini_key=gemini_key,
            company_name=company_name,
            ticker=ticker,
            market_snapshot=snapshot,
            news_text=news_text,
            risk_profile=risk_profile,
            horizon=horizon,
        )

    recommendation = classify_recommendation(report)
    confidence = extract_confidence(report)
    risk_level = extract_risk(report)

    pill_class = recommendation_css_class(recommendation)
    st.markdown(
        f"<span class='recommend-pill {pill_class}'>Recommendation: {recommendation}</span>",
        unsafe_allow_html=True,
    )

    info1, info2 = st.columns(2)
    info1.markdown(
        f"<div class='mini-card'><strong>Confidence</strong><br><span class='subtle'>{confidence}</span></div>",
        unsafe_allow_html=True,
    )
    info2.markdown(
        f"<div class='mini-card'><strong>Risk Level</strong><br><span class='subtle'>{risk_level}</span></div>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Final Report", "Research Agent"])

    with tabs[0]:
        st.markdown(report)

    with tabs[1]:
        st.markdown(research_summary)

    st.download_button(
        "⬇️ Download report as Markdown",
        data=report,
        file_name=f"{ticker.lower()}_investment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

except Exception as e:
    err = str(e).lower()
    if "rate limit" in err or "too many requests" in err or "429" in err or "resource_exhausted" in err:
        st.warning("Rate limit reached. Please wait a bit and try again.")
    elif "notfound" in err or "404" in err:
        st.error("Gemini model or endpoint was not accepted. Please verify the updated code and API key.")
    else:
        st.error(f"Something went wrong: {e}")
    st.stop()