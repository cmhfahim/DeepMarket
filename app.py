import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import json
import pickle
import os
from PIL import Image
import joblib
import io
import base64

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="DeepMarket",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Global CSS — modern dark theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Hide default Streamlit chrome ── */
header {visibility: hidden !important; display: none !important;}
footer {visibility: hidden !important; height: 0 !important; display: none !important;}
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
.stAppDeployButton {display: none !important;}

/* ── Root background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    color: #e0e0e0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}
section[data-testid="stSidebar"] .stRadio > label {
    color: #a5b4fc !important;
    font-weight: 600;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: rgba(99, 102, 241, 0.06);
    border-radius: 8px;
    padding: 8px 14px;
    margin-bottom: 4px;
    transition: all 0.2s ease;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(99, 102, 241, 0.15);
}

/* ── Cards / containers ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 14px;
    padding: 22px 26px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(99,102,241,0.15);
}
.metric-card h3 {
    margin: 0 0 6px 0;
    font-size: 14px;
    color: #a5b4fc;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-card p {
    margin: 0;
    font-size: 28px;
    font-weight: 700;
    color: #e0e0e0;
}

/* ── Headings ── */
.page-title {
    text-align: center;
    font-size: 56px;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.page-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 20px;
    font-weight: 400;
    margin-bottom: 32px;
}
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #c4b5fd;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(99,102,241,0.25);
    display: inline-block;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
    margin: 36px 0;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 32px;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.45);
}

/* ── Inputs / Select boxes ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
    color: #e0e0e0 !important;
}

/* ── Data frame ── */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Plotly chart wrapper spacing ── */
.stPlotlyChart {
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 12px;
}

/* ── Feedback form ── */
.modern-form input,
.modern-form textarea {
    width: 100%;
    margin: 10px 0;
    padding: 14px 16px;
    border-radius: 10px;
    border: 1px solid rgba(99,102,241,0.25);
    background: rgba(255,255,255,0.05);
    color: #e0e0e0;
    font-size: 15px;
    font-family: inherit;
    transition: border-color 0.2s ease;
}
.modern-form input:focus,
.modern-form textarea:focus {
    outline: none;
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15);
}
.modern-form input::placeholder,
.modern-form textarea::placeholder {
    color: #64748b;
}
.modern-form button {
    width: 100%;
    margin-top: 12px;
    padding: 14px;
    border-radius: 10px;
    border: none;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3);
}
.modern-form button:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.45);
}

/* ── Member card ── */
.member-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(139,92,246,0.06));
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 16px;
    transition: transform 0.2s ease;
}
.member-card:hover {
    transform: translateY(-2px);
}
.member-card strong {
    color: #c4b5fd;
    font-size: 17px;
}
.member-card a {
    color: #94a3b8;
    text-decoration: none;
}
.member-card a:hover {
    color: #a5b4fc;
}

/* ── Prediction result badge ── */
.prediction-result {
    text-align: center;
    margin-top: 28px;
    padding: 28px;
    border-radius: 16px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,102,241,0.15);
}
.prediction-result .label {
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 8px;
}
.prediction-result .detail {
    font-size: 18px;
    color: #94a3b8;
}
.pred-up   { color: #34d399; }
.pred-down { color: #f87171; }
.pred-flat { color: #fbbf24; }

/* ── Disclaimer ── */
.disclaimer {
    text-align: center;
    margin-top: 48px;
    color: #64748b;
    font-size: 14px;
    line-height: 1.7;
}
.disclaimer hr {
    width: 30%;
    margin: 12px auto;
    border-color: rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Plotly theme helper
# ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c4b5fd", family="Inter, sans-serif"),
    title_font=dict(size=18, color="#e0e0e0"),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a5b4fc"),
    ),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
    margin=dict(t=50, b=30, l=40, r=20),
)

ACCENT_COLORS = ["#818cf8", "#a78bfa", "#c084fc", "#f472b6", "#fb923c", "#34d399", "#38bdf8"]


# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
@st.cache_data
def load_vis_data():
    df = pd.read_csv("display_data_set.csv", parse_dates=["DATE"])
    df["MONTH"] = df["DATE"].dt.month
    df["YEAR_MONTH"] = df["DATE"].dt.to_period("M").astype(str)
    return df

df_vis = load_vis_data()

@st.cache_data
def load_vis_data2():
    df = pd.read_csv("Final_data_for_ML.csv", parse_dates=["DATE"])
    df["MONTH"] = df["DATE"].dt.month
    df["YEAR_MONTH"] = df["DATE"].dt.to_period("M").astype(str)
    return df

df_vis2 = load_vis_data2()

with open("company_encoding.json", "r") as f:
    enc_dict = json.load(f)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="text-align:center; margin-bottom:24px;">
            <span style="font-size:32px;">📈</span>
            <p style="font-size:22px; font-weight:700; color:#a5b4fc; margin:4px 0 0 0;">DeepMarket</p>
            <p style="font-size:12px; color:#64748b; margin:0;">Stock Analytics Platform</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Home", "Market Analysis", "Visualization", "Prediction", "Feedback"], label_visibility="collapsed")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#475569; font-size:12px;">© 2026 Team QuantumTalk</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════
if page == "Home":
    st.markdown('<div style="height:40px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">DeepMarket</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Dhaka Stock Market Analysis &amp; Price Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Hero metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>Companies Tracked</h3>
                <p>{}</p>
            </div>
        """.format(df_vis["TRADING CODE"].nunique()), unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h3>Data Points</h3>
                <p>{:,}</p>
            </div>
        """.format(len(df_vis2)), unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h3>ML Models</h3>
                <p>3</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:36px;"></div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="glass-card" style="max-width:860px; margin:0 auto;">
            <h3 style="color:#c4b5fd; text-align:center; margin-top:0;">About This Platform</h3>
            <p style="color:#94a3b8; font-size:16px; line-height:1.8; text-align:center;">
                Explore trends, visualize insights, and predict future movement of stocks from the
                <strong style="color:#a5b4fc;">Dhaka Stock Exchange</strong> using interactive tools.
                This platform leverages historical data to understand stock behavior and uses machine learning
                models — <strong style="color:#a5b4fc;">LightGBM</strong>, <strong style="color:#a5b4fc;">XGBoost</strong>,
                and <strong style="color:#a5b4fc;">Random Forest</strong> — to forecast whether a company's
                stock is likely to go <span style="color:#34d399;">up</span>,
                stay <span style="color:#fbbf24;">unchanged</span>, or go
                <span style="color:#f87171;">down</span>.
            </p>
            <p style="color:#64748b; font-size:14px; text-align:center; margin-top:16px;">
                Built with Python · Streamlit · Plotly · Pandas · Seaborn · Scikit-learn
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Feature highlights
    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("📊", "Market Analysis", "Macro-level trends, volume, and monthly direction insights."),
        ("📈", "Visualization", "Company-level charts: rolling averages, distributions, correlations."),
        ("🤖", "Prediction", "Predict stock direction using LightGBM, XGBoost, or Random Forest."),
        ("💬", "Feedback", "Share your thoughts and help us improve."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
        with col:
            st.markdown(f"""
                <div class="metric-card" style="text-align:left; min-height:160px;">
                    <div style="font-size:28px; margin-bottom:8px;">{icon}</div>
                    <h3 style="text-align:left; font-size:16px;">{title}</h3>
                    <p style="font-size:13px; color:#94a3b8; font-weight:400;">{desc}</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="height:40px;"></div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#475569; font-size:14px;">Built by <strong style="color:#a5b4fc;">Team QuantumTalk</strong></p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MARKET ANALYSIS PAGE
# ══════════════════════════════════════════════
elif page == "Market Analysis":
    st.markdown('<div class="page-title" style="font-size:40px;">Market Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Macro-level trends across the Dhaka Stock Exchange</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Monthly Average Trend (Donut) ──
    st.markdown('<div class="section-title">📊 Monthly Average Direction</div>', unsafe_allow_html=True)
    df_vis2["MONTH_PERIOD"] = pd.to_datetime(df_vis2["DATE"]).dt.to_period("M")
    monthly_avg = df_vis2.groupby("MONTH_PERIOD")["TARGET"].mean().reset_index()

    def categorize_trend(val):
        if val > 0.05:
            return "Up"
        elif val < -0.05:
            return "Down"
        else:
            return "No Change"

    monthly_avg["Trend"] = monthly_avg["TARGET"].apply(categorize_trend)
    trend_counts = monthly_avg["Trend"].value_counts().reset_index()
    trend_counts.columns = ["Trend", "Count"]

    fig_pie = px.pie(
        trend_counts,
        names="Trend",
        values="Count",
        hole=0.55,
        title="Market Monthly Average Trend Distribution",
        color_discrete_map={"Up": "#34d399", "Down": "#f87171", "No Change": "#64748b"},
    )
    fig_pie.update_traces(textfont_size=13, textinfo="percent+label")
    fig_pie.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_pie, use_container_width=True)

    # ── Overall Target Distribution ──
    st.markdown('<div class="section-title">🎯 Overall Target Distribution</div>', unsafe_allow_html=True)
    target_counts = df_vis2["TARGET"].value_counts().reindex([1, 0, -1], fill_value=0)
    target_labels = ["Up", "No Change", "Down"]

    fig_market_target = px.pie(
        values=target_counts.values,
        names=target_labels,
        title="Overall Market Target Distribution",
        color=target_labels,
        color_discrete_map={"Up": "#34d399", "Down": "#f87171", "No Change": "#64748b"},
    )
    fig_market_target.update_traces(textfont_size=13, textinfo="percent+label")
    fig_market_target.update_layout(**PLOTLY_LAYOUT, showlegend=True)
    st.plotly_chart(fig_market_target, use_container_width=True)

    # ── Total Market Volume ──
    st.markdown('<div class="section-title">📦 Total Market Volume Over Time</div>', unsafe_allow_html=True)
    market_volume = df_vis2.groupby("DATE")["VOLUME"].sum().reset_index()
    fig_market_vol = px.area(
        market_volume, x="DATE", y="VOLUME",
        title="Total Trading Volume Across All Companies",
        color_discrete_sequence=["#818cf8"],
    )
    fig_market_vol.update_traces(line=dict(width=1.5), fillcolor="rgba(129,140,248,0.15)")
    fig_market_vol.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_market_vol, use_container_width=True)

    # ── Circular Monthly View ──
    st.markdown('<div class="section-title">🔄 Monthly Market Behavior (Circular View)</div>', unsafe_allow_html=True)
    monthly_trend = (
        df_vis2.groupby(df_vis2["DATE"].dt.to_period("M"))["TARGET"]
        .mean()
        .reset_index()
    )
    monthly_trend["MONTH"] = monthly_trend["DATE"].dt.strftime("%b")
    monthly_trend["Trend_Value"] = monthly_trend["TARGET"].apply(
        lambda x: 1 if x > 0.05 else (-1 if x < -0.05 else 0)
    )
    monthly_trend["Trend_Label"] = monthly_trend["Trend_Value"].map(
        {1: "Up", 0: "No Change", -1: "Down"}
    )

    fig_polar = px.bar_polar(
        monthly_trend,
        r="Trend_Value",
        theta="MONTH",
        color="Trend_Label",
        color_discrete_map={"Up": "#34d399", "Down": "#f87171", "No Change": "#64748b"},
        title="Average Market Trend by Month (Circular Axis)",
    )
    fig_polar.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(showticklabels=False, ticks="", gridcolor="rgba(255,255,255,0.05)"),
            angularaxis=dict(direction="clockwise", gridcolor="rgba(255,255,255,0.05)", color="#a5b4fc"),
        ),
    )
    st.plotly_chart(fig_polar, use_container_width=True)


# ══════════════════════════════════════════════
# VISUALIZATION PAGE
# ══════════════════════════════════════════════
elif page == "Visualization":
    st.markdown('<div class="page-title" style="font-size:40px;">Data Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Explore individual company performance in depth</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    selected_company = st.selectbox("Select a company", sorted(df_vis["TRADING CODE"].unique()))
    company_df2 = df_vis[df_vis["TRADING CODE"] == selected_company].copy()

    st.markdown('<div class="section-title">📋 Raw Data</div>', unsafe_allow_html=True)
    st.dataframe(company_df2, use_container_width=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    company_df = df_vis2[df_vis2["TRADING CODE"] == selected_company].copy()

    # ── Close Price Trend ──
    st.markdown('<div class="section-title">📈 Close Price Over Time</div>', unsafe_allow_html=True)
    fig1 = px.area(company_df, x="DATE", y="CLOSEP*", title=f"{selected_company} – Close Price Trend",
                   color_discrete_sequence=["#818cf8"])
    fig1.update_traces(line=dict(width=1.5), fillcolor="rgba(129,140,248,0.12)")
    fig1.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig1, use_container_width=True)

    # ── Rolling Avg ──
    st.markdown('<div class="section-title">📉 30-Day Rolling Avg &amp; Median</div>', unsafe_allow_html=True)
    company_df["MA30"] = company_df["CLOSEP*"].rolling(30, min_periods=1).mean()
    company_df["MED30"] = company_df["CLOSEP*"].rolling(30, min_periods=1).median()
    fig_rolling = px.line(
        company_df, x="DATE", y=["CLOSEP*", "MA30", "MED30"],
        labels={"value": "Price", "variable": "Legend"},
        title=f"{selected_company} – Close Price with 30-Day MA & Median",
        color_discrete_map={"CLOSEP*": "#818cf8", "MA30": "#fb923c", "MED30": "#34d399"},
    )
    fig_rolling.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_rolling, use_container_width=True)

    # ── Volume ──
    st.markdown('<div class="section-title">📦 Volume by Date</div>', unsafe_allow_html=True)
    fig2 = px.bar(company_df, x="DATE", y="VOLUME", title=f"{selected_company} – Trading Volume",
                  color_discrete_sequence=["#a78bfa"])
    fig2.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Daily % Change ──
    st.markdown('<div class="section-title">📊 Daily % Change Histogram</div>', unsafe_allow_html=True)
    company_df["PCT_CHANGE"] = company_df["CLOSEP*"].pct_change() * 100
    fig_hist = px.histogram(
        company_df, x="PCT_CHANGE", nbins=30,
        title=f"{selected_company} – Daily % Change",
        color_discrete_sequence=["#38bdf8"],
    )
    fig_hist.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Box Plot ──
    st.markdown('<div class="section-title">📦 Close Price Distribution</div>', unsafe_allow_html=True)
    fig_box = px.box(
        company_df, x="CLOSEP*", points="all",
        color_discrete_sequence=["#818cf8"],
        title=f"{selected_company} – Close Price Distribution",
    )
    fig_box.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Monthly Avg Close ──
    st.markdown('<div class="section-title">📅 Monthly Average Close Price</div>', unsafe_allow_html=True)
    monthly_avg = company_df.groupby("YEAR_MONTH")["CLOSEP*"].mean()
    fig_monthly = px.line(
        x=monthly_avg.index, y=monthly_avg.values,
        title=f"{selected_company} – Monthly Avg Close",
        labels={"x": "Year-Month", "y": "Avg Close"},
        markers=True, color_discrete_sequence=["#c084fc"],
    )
    fig_monthly.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_monthly, use_container_width=True)

    # ── Circular Monthly ──
    st.markdown('<div class="section-title">🔄 Circular Monthly Avg Close Price</div>', unsafe_allow_html=True)
    company_df["MONTH"] = company_df["MONTH"].astype(int)
    monthly_data = company_df.groupby("MONTH")["CLOSEP*"].mean().reindex(range(1, 13), fill_value=0)
    fig_polar = px.bar_polar(
        r=monthly_data.values,
        theta=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        color=monthly_data.values,
        color_continuous_scale="Viridis",
        title=f"{selected_company} – Circular Monthly Avg Close",
    )
    fig_polar.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        polar=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_polar, use_container_width=True)

    # Consistent target colors
    target_color_map = {1: "#34d399", 0: "#fbbf24", -1: "#f87171"}
    target_labels_list = ["1 = Price Up", "0 = No Change", "-1 = Price Down"]

    # ── Monthly Target Histogram ──
    st.markdown('<div class="section-title">📊 Monthly Target Histogram</div>', unsafe_allow_html=True)
    fig4 = px.histogram(
        company_df, x="MONTH", color="TARGET",
        category_orders={"MONTH": list(range(1, 13))},
        color_discrete_map=target_color_map,
        title="Target by Month",
    )
    fig4.update_layout(**PLOTLY_LAYOUT, bargap=0.15, bargroupgap=0.05)
    st.plotly_chart(fig4, use_container_width=True)

    # ── Target Donut ──
    st.markdown('<div class="section-title">🎯 Target Distribution</div>', unsafe_allow_html=True)
    pie_data = company_df["TARGET"].value_counts().reindex([1, 0, -1], fill_value=0)

    fig3 = px.pie(
        values=pie_data.values,
        names=target_labels_list,
        color=pie_data.index.astype(str),
        color_discrete_map={str(k): v for k, v in target_color_map.items()},
        hole=0.55,
        title="Target Distribution",
    )
    fig3.update_traces(textfont_size=13, textinfo="percent+label")
    fig3.update_layout(**PLOTLY_LAYOUT, showlegend=True)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Scatter ──
    st.markdown('<div class="section-title">🔗 Volume vs Close Price</div>', unsafe_allow_html=True)
    fig_scatter = px.scatter(
        company_df, x="VOLUME", y="CLOSEP*", color="TARGET",
        color_discrete_map=target_color_map,
        title=f"{selected_company} – Volume vs Close Price",
        opacity=0.7,
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Correlation Heatmap ──
    st.markdown('<div class="section-title">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols = ["OPENP*", "HIGH", "LOW", "CLOSEP*", "TRADE", "VOLUME"]
    fig_corr = px.imshow(
        company_df[num_cols].corr(),
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title=f"{selected_company} – Correlation Heatmap",
    )
    fig_corr.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Lag Plot ──
    st.markdown('<div class="section-title">⏪ Lag Plot of Close Price</div>', unsafe_allow_html=True)
    company_df["CLOSE_LAG1"] = company_df["CLOSEP*"].shift(1)
    lag_df = company_df.dropna(subset=["CLOSE_LAG1", "CLOSEP*"])
    fig_lag = px.scatter(
        lag_df, x="CLOSE_LAG1", y="CLOSEP*",
        title=f"{selected_company} – Lag Plot (t vs t-1)",
        labels={"CLOSE_LAG1": "Previous Day Close", "CLOSEP*": "Today Close"},
        color_discrete_sequence=["#c084fc"],
    )
    fig_lag.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_lag, use_container_width=True)

    # ── Volatility ──
    st.markdown('<div class="section-title">🌊 30-Day Rolling Volatility</div>', unsafe_allow_html=True)
    company_df["RET"] = company_df["CLOSEP*"].pct_change()
    company_df["VOLATILITY"] = company_df["RET"].rolling(30, min_periods=1).std()
    fig_vol = px.line(
        company_df, x="DATE", y="VOLATILITY",
        title=f"{selected_company} – 30-Day Rolling Volatility",
        color_discrete_sequence=["#f87171"],
    )
    fig_vol.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_vol, use_container_width=True)


# ══════════════════════════════════════════════
# PREDICTION PAGE
# ══════════════════════════════════════════════
elif page == "Prediction":
    st.markdown('<div class="page-title" style="font-size:40px;">Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Forecast stock direction with machine learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Company & Model selection in two columns
    sel1, sel2 = st.columns(2)
    with sel1:
        company_name = st.selectbox("Select Company", sorted(enc_dict.keys()))
    with sel2:
        model_choice = st.selectbox("Select Model", ["LightGBM", "XGBoost", "Random Forest"])

    company_id = enc_dict[company_name]

    # Load model
    try:
        if model_choice == "LightGBM":
            model = joblib.load("lgbm_model.pkl")
        elif model_choice == "Random Forest":
            model = joblib.load("rf_model.pkl")
        elif model_choice == "XGBoost":
            with open("xgboost_model.pkl", "rb") as f:
                model = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load {model_choice} model: {e}")
        st.stop()

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 Enter Feature Values</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        month = st.selectbox("Month", list(range(1, 13)), key="month")
    with col2:
        openp = st.number_input("OPENP*", min_value=0.0, value=100.0, key="openp")

    with col1:
        high = st.number_input("HIGH", min_value=0.0, value=105.0, key="high")
    with col2:
        low = st.number_input("LOW", min_value=0.0, value=95.0, key="low")

    with col1:
        closep = st.number_input("CLOSEP*", min_value=0.0, value=102.0, key="closep")
    with col2:
        trade = st.number_input("TRADE", min_value=0, value=500, key="trade")

    vol_col1, vol_col2, vol_col3 = st.columns([1, 2, 1])
    with vol_col2:
        volume = st.number_input("VOLUME", min_value=0, value=10000, key="volume")

    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

    btn_col1, btn_col2, btn_col3 = st.columns([3, 1, 3])
    with btn_col2:
        predict_clicked = st.button("🔮 Predict")

    if predict_clicked:
        input_df = pd.DataFrame([{
            "COMPANY_ID": company_id,
            "MONTH": month,
            "OPENP*": openp,
            "HIGH": high,
            "LOW": low,
            "CLOSEP*": closep,
            "TRADE": trade,
            "VOLUME": volume,
        }])

        prediction = model.predict(input_df)[0]
        label_map = {1: "Price Up ↑", 0: "No Change ●", -1: "Price Down ↓"}
        css_class = {1: "pred-up", 0: "pred-flat", -1: "pred-down"}

        st.markdown(f"""
            <div class="prediction-result">
                <div class="label {css_class[prediction]}">{label_map[prediction]}</div>
                <div class="detail">
                    Model <strong style="color:#a5b4fc;">{model_choice}</strong> predicts
                    <strong>{label_map[prediction]}</strong> for
                    <strong style="color:#c4b5fd;">{company_name}</strong>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="disclaimer">
            <hr>
            ⚠️ <strong>Disclaimer</strong><br>
            This prediction is for <strong>research purposes only</strong>.<br>
            Investment decisions should be made independently.<br>
            The development team is <strong>not responsible</strong> for any outcomes.
        </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# FEEDBACK PAGE
# ══════════════════════════════════════════════
elif page == "Feedback":
    st.markdown('<div class="page-title" style="font-size:40px;">Feedback</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">We value your thoughts — help us improve</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="glass-card" style="max-width:620px; margin:0 auto;">
            <form action="https://formsubmit.co/choowdhuryfahim03@gmail.com" method="POST" class="modern-form">
                <input type="hidden" name="_captcha" value="false">
                <input type="text" name="name" placeholder="Your Name" required>
                <input type="email" name="email" placeholder="Your Email" required>
                <textarea name="message" placeholder="Share your feedback…" rows="5" required></textarea>
                <button type="submit">🚀 Send Feedback</button>
            </form>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align:center; margin-top:36px; color:#64748b; font-size:14px;">
            📩 Your feedback helps us improve this platform!
        </div>
    """, unsafe_allow_html=True)
