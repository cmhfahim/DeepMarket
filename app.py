import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import json
import pickle
import os
from PIL import Image
import joblib
import io
import base64

st.set_page_config(page_title="📈 DeepMarket", layout="wide")

# ---- Custom Sidebar Font Size ----
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            font-size: 60px !important;
        }
        /* Style form inputs and button */
        form input, form textarea, form button {
            width: 100%;
            margin: 8px 0;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #ccc;
            font-size: 16px;
        }
        form button {
            background-color: #4B8BBE;
            color: white;
            border: none;
            cursor: pointer;
            font-weight: bold;
        }
        form button:hover {
            background-color: #3a6d9c;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Load data ----
@st.cache_data
def load_vis_data():
    df = pd.read_csv("cleaned_sorted_dse_data_3.csv", parse_dates=["DATE"])
    df["MONTH"] = df["DATE"].dt.month
    df["YEAR_MONTH"] = df["DATE"].dt.to_period("M").astype(str)
    return df

df_vis = load_vis_data()

# Load encoding and model
with open("company_encoding.json", "r") as f:
    enc_dict = json.load(f)

model = joblib.load("lgbm_model.pkl")

# Sidebar
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Visualization", "📌 Prediction", "📘 Real Life Experience","🚀 Project Journey", "📝 Feedback"])

# ---- Pages ----

if page == "🏠 Home":
    st.markdown("""
        <div style="text-align: center;">
            <h1 style='color:black; font-size: 70px;'>DeepMarket</h1>
            <h3 style='color:#1b1f3a; font-size: 28px;'>Dhaka Stock Market Analysis and Price Prediction</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Description with spacing
    st.markdown("""
        <div style='height:40px;'></div>

        <div style="text-align: center; max-width: 900px; margin: 0 auto; color:#241717; font-size: 18px; line-height: 1.6;">
            <h2>🌐 Description</h2>
            <p>
                Explore trends, visualize insights, and predict future movement of stocks from Dhaka Stock Exchange using interactive tools. This platform leverages historical data to understand stock behavior and uses machine learning models (LightGBM) to forecast whether a company's stock is likely to go up, stay unchanged, or go down. With rich visualizations, stock-wise filtering, and an interactive prediction interface, users can gain deeper insights into the market's rhythm. Whether you're a curious learner, a data enthusiast, or a researcher, DeepMarket offers a compact yet powerful window into financial analytics. Built using <strong>Python, Streamlit, Plotly, LightGBM, Pandas,</strong> and <strong>Seaborn</strong>, this project aims to bridge the gap between data science and financial decision-making.
            </p>
        </div>

        <div style='height:60px;'></div>
    """, unsafe_allow_html=True)

    # Team section title
    st.markdown("<h2 style='text-align:center;'>👥 Team Members</h2>", unsafe_allow_html=True)
    st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)

    # Member card HTML template
    def member_card(name, email):
        return f"""
            <div style="
                background-color: #14252b;
                color: white;
                border-radius: 10px;
                padding: 15px 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.5);
            ">
                <strong style='font-size:18px;'>{name}</strong><br>
                📧 <a href='mailto:{email}' style='color:#dddddd;'>{email}</a>
            </div>
        """

    # First 4 members in 2 columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(member_card("Atkia Mona Rahi", "atkiamona.rahi2003@gmail.com"), unsafe_allow_html=True)
        st.markdown(member_card("Shafayat Hossain Ornob", "ornobhossain121@gmail.com"), unsafe_allow_html=True)

    with col2:
        st.markdown(member_card("Abu Zafor Mohammad Saleh", "abuzaforsaleh11@gmail.com"), unsafe_allow_html=True)
        st.markdown(member_card("Chowdhury Manjurul Hasan", "cmhfahim@gmail.com"), unsafe_allow_html=True)

    # Spacer before the last member
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # Center the 5th member using 3-column layout
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown(member_card("Pijush Das", "pijushdas123@gmail.com"), unsafe_allow_html=True)

    # Footer
    st.markdown("<p style='text-align:center; margin-top:50px; color:black;'>💡 Built by <strong>Team QuantumTalk</strong></p>", unsafe_allow_html=True)


elif page == "📊 Visualization":
    st.markdown("<h2 style='text-align:center; font-size:36px; color:white;'>📊 Data Visualization</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    selected_company = st.selectbox("Select a company", sorted(df_vis["TRADING CODE"].unique()))
    company_df = df_vis[df_vis["TRADING CODE"] == selected_company]

    st.subheader("📄 Raw Data")
    st.dataframe(company_df, use_container_width=True)

    st.markdown("---")

    # Close Price Trend
    st.subheader("📈 Close Price Over Time")
    fig1 = px.area(company_df, x="DATE", y="CLOSEP*", title=f"{selected_company} – Close Price Trend", color_discrete_sequence=["#4B8BBE"])
    st.plotly_chart(fig1, use_container_width=True)

    # Rolling Mean & Median
    st.subheader("🔄 30-Day Rolling Mean & Median of Close Price")
    company_df['MA30'] = company_df['CLOSEP*'].rolling(30).mean()
    company_df['MED30'] = company_df['CLOSEP*'].rolling(30).median()
    fig_rolling = px.line(
        company_df,
        x="DATE",
        y=["CLOSEP*", "MA30", "MED30"],
        labels={"value":"Price", "variable":"Legend"},
        title=f"{selected_company} – Close Price with 30-Day MA & Median"
    )
    st.plotly_chart(fig_rolling, use_container_width=True)

    # Volume
    st.subheader("📦 Volume by Date")
    fig2 = px.bar(company_df, x="DATE", y="VOLUME", title=f"{selected_company} – Trading Volume", color_discrete_sequence=["#ff7f0e"])
    st.plotly_chart(fig2, use_container_width=True)

    # Daily % Change Histogram
    st.subheader("📊 Daily % Change Histogram")
    company_df['PCT_CHANGE'] = company_df['CLOSEP*'].pct_change() * 100
    fig_hist = px.histogram(company_df, x='PCT_CHANGE', nbins=30, title=f"{selected_company} – Daily % Change", color_discrete_sequence=["#17becf"])
    st.plotly_chart(fig_hist, use_container_width=True)

    # Target Pie
    st.subheader("🥧 Target Distribution")
    pie_data = company_df["TARGET"].value_counts().reindex([1, 0, -1], fill_value=0)
    pie_labels = ["1 = Up", "0 = No Change", "-1 = Down"]
    fig3 = px.pie(values=pie_data.values, names=pie_labels, color_discrete_sequence=["#2ecc71", "#f1c40f", "#e74c3c"])
    st.plotly_chart(fig3, use_container_width=True)

    # Monthly Target Histogram
    st.subheader("📅 Monthly Target Histogram")
    fig4 = px.histogram(
        company_df,
        x="MONTH",
        color="TARGET",
        category_orders={"MONTH": list(range(1, 13))},
        color_discrete_map={1: "#2ecc71", 0: "#f1c40f", -1: "#e74c3c"},
        title="Target by Month",
        width=900,
        height=400
    )
    fig4.update_layout(bargap=0.15, bargroupgap=0.05)
    st.plotly_chart(fig4, use_container_width=True)

    # Correlation Heatmap
    st.subheader("📊 Correlation Heatmap")
    num_cols = ['OPENP*', 'HIGH', 'LOW', 'CLOSEP*', 'TRADE', 'VOLUME']
    corr = company_df[num_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title=f"{selected_company} – Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Lag Plot
    st.subheader("🔁 Lag Plot of Close Price")
    import matplotlib.pyplot as plt
    from pandas.plotting import lag_plot
    fig_lag, ax = plt.subplots()
    lag_plot(company_df['CLOSEP*'], ax=ax)
    ax.set_title(f"{selected_company} – Lag Plot")
    st.pyplot(fig_lag)

elif page == "📌 Prediction":
    st.markdown("<h2 style='text-align:center; font-size:36px; color:white;'>🔮 Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<p style='text-align:center;'>Enter the feature values below:</p>", unsafe_allow_html=True)

    company_name = st.selectbox("Select company", sorted(enc_dict.keys()))
    company_id = enc_dict[company_name]

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

    # Center the last field (VOLUME)
    volume_col1, volume_col2, volume_col3 = st.columns([1, 2, 1])
    with volume_col2:
        volume = st.number_input("VOLUME", min_value=0, value=10000, key="volume")

    # Center the Predict button
    btn_col1, btn_col2, btn_col3 = st.columns([3,1,3])
    with btn_col2:
        predict_clicked = st.button("📊 Predict")

    if predict_clicked:
        input_df = pd.DataFrame([{
            "COMPANY_ID": company_id,
            "MONTH": month,
            "OPENP*": openp,
            "HIGH": high,
            "LOW": low,
            "CLOSEP*": closep,
            "TRADE": trade,
            "VOLUME": volume
        }])

        prediction = model.predict(input_df)[0]
        label_map = {1: "📈 Price Up", 0: "➖ No Change", -1: "📉 Price Down"}

        # Larger font size for result, centered, green color
        st.markdown(f"""
            <div style='text-align:center; margin-top: 20px;'>
                <h2 style='color:green; font-size: 36px;'>{label_map[prediction]}</h2>
                <p style='font-weight:bold; font-size:28px;'>📊 Model predicts: <strong>{label_map[prediction]}</strong> for {company_name}</p>
            </div>
        """, unsafe_allow_html=True)

    # Disclaimer at the bottom, always visible, centered, black text
    st.markdown("""
        <div style='text-align:center; margin-top: 60px; color: black; font-size: 16px;'>
            <hr style='width:40%; margin: 15px auto; border-color:#ccc;'>
            ⚠️ <strong>Disclaimer</strong>:<br>
            This prediction is for <strong>research purposes only</strong>.<br>
            Investment decisions should be made independently.<br>
            The development team is <strong>not responsible</strong> for any outcomes.
        </div>
    """, unsafe_allow_html=True)

elif page == "🚀 Project Journey":
    st.markdown("<h2 style='text-align:center; font-size:36px; color:white;'>🛤️ Project Journey</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    #st.markdown("## 🛤️ Project Journey")

    def image_to_base64(img):
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    image_dir = r"project_pic"
    valid_exts = (".jpg", ".jpeg", ".png")

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    if not image_files:
        st.warning("⚠️ No JPG/PNG images found in the folder.")
    else:
        if "img_index" not in st.session_state:
            st.session_state.img_index = 0
        if "full_size" not in st.session_state:
            st.session_state.full_size = False

        cols = st.columns(5)
        with cols[1]:
            if st.button("⬅️ Previous"):
                st.session_state.img_index = max(0, st.session_state.img_index - 1)
        with cols[2]:
            toggle_label = "Exit Full Size" if st.session_state.full_size else "Full Size"
            if st.button(toggle_label):
                st.session_state.full_size = not st.session_state.full_size
        with cols[3]:
            if st.button("Next ➡️"):
                st.session_state.img_index = min(len(image_files) - 1, st.session_state.img_index + 1)

        img_path = os.path.join(image_dir, image_files[st.session_state.img_index])
        img = Image.open(img_path)

        max_size = (1200, 900) if st.session_state.full_size else (800, 600)
        img.thumbnail(max_size)

        st.markdown(
            f'<div style="display:flex; justify-content:center;">'
            f'<img src="data:image/png;base64,{image_to_base64(img)}" style="max-width:100%; height:auto;">'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"<p style='text-align:center; margin-top:10px;'>Step {st.session_state.img_index + 1} of {len(image_files)}</p>", unsafe_allow_html=True)

elif page == "📘 Real Life Experience":
    # White colored main heading
    st.markdown("<h2 style='text-align:center; font-size:36px; color:white;'>📘 Real Life Experience</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Black text content
    st.markdown("""
        <div style='color:black; font-size:18px;'>
            <h3>🎯 Model Performance</h3>
            <ul>
                <li><strong>Accuracy</strong>: <span style='color:white;'>0.8809</span></li>
                <li><strong>F1 Score</strong>: <span style='color:white;'>0.8804</span></li>
                <li><strong>Precision</strong>: <span style='color:white;'>0.8937</span></li>
                <li><strong>Recall</strong>: <span style='color:white;'>0.8911</span></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-top: 1px solid #ccc;'>", unsafe_allow_html=True)

    st.markdown("<h3 style='color:black;'>📊 Classification Report</h3>", unsafe_allow_html=True)

    html_table = """
    <table style='width:100%; border-collapse: collapse; color:black; font-size:16px;'>
      <thead>
        <tr>
          <th style='border: 1px solid #ddd; padding: 8px;'>Label</th>
          <th style='border: 1px solid #ddd; padding: 8px;'>Precision</th>
          <th style='border: 1px solid #ddd; padding: 8px;'>Recall</th>
          <th style='border: 1px solid #ddd; padding: 8px;'>F1-Score</th>
        </tr>
      </thead>
      <tbody>
        <tr><td style='border: 1px solid #ddd; padding: 8px;'>-1 (Down)</td><td style='border: 1px solid #ddd; padding: 8px; color:white;'>0.86</td><td style='border: 1px solid #ddd; padding: 8px; color:white;'>0.89</td><td style='border: 1px solid #ddd; padding: 8px; color:white;'>0.87</td></tr>
        <tr><td style='border: 1px solid #ddd; padding: 8px;'>0 (No Change)</td><td style='border: 1px solid #ddd; padding: 8px; color:white;'>0.95</td><td style='border: 1px solid #ddd; padding: 8px; color:white;'>0.92</td><td style='border: 1px solid #ddd; padding: 8px; color:white;'>0.92</td></tr>
        <tr><td style='border: 1px solid #ddd; padding: 8px;'>1 (Up)</td><td style='border: 1px solid #ddd; padding: 8px; color:white;'>0.89</td><td style='border: 1px solid #ddd; padding: 8px; color:white;'>0.86</td><td style='border: 1px solid #ddd; padding: 8px; color:white;'>0.87</td></tr>
      </tbody>
    </table>
    """
    st.markdown(html_table, unsafe_allow_html=True)

    st.markdown("<hr style='border-top: 1px solid #ccc;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:black;'>🔁 Confusion Matrix</h3>", unsafe_allow_html=True)

    confusion_matrix = pd.DataFrame(
        [[2267, 71, 213], [57, 3189, 227], [177, 248, 2192]],
        columns=["Pred: -1", "Pred: 0", "Pred: 1"],
        index=["Actual: -1", "Actual: 0", "Actual: 1"]
    )

    fig_cm = px.imshow(
        confusion_matrix,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
        title="Confusion Matrix"
    )
    st.plotly_chart(fig_cm, use_container_width=True)



elif page == "📝 Feedback":
    st.markdown("<h2 style='text-align:center; font-size:36px; color:white;'>📝 Feedback</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align: center; font-size: 18px; color:black;">
            We value your thoughts and suggestions! Please fill out the form below to share your feedback.
        </div>
        <br>
    """, unsafe_allow_html=True)

    contact_form = """
    <form action="https://formsubmit.co/choowdhuryfahim03@gmail.com" method="POST" style="max-width: 600px; margin: 0 auto;">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your Name" required>
        <input type="email" name="email" placeholder="Your Email" required>
        <textarea name="message" placeholder="Give your Feedback" rows="5" required></textarea>
        <button type="submit">Send Feedback</button>
    </form>
    """

    st.markdown(contact_form, unsafe_allow_html=True)

    st.markdown("""
        <div style='text-align:center; margin-top: 40px; color: black; font-size: 16px;'>
            📩 Your feedback helps us improve this platform!
        </div>
    """, unsafe_allow_html=True)

