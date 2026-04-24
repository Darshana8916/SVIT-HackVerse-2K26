import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Healthcare AI Premium", layout="wide")

# -----------------------------
# CUSTOM CSS 🔥 PREMIUM UI
# -----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
}
.card {
    background: rgba(255,255,255,0.15);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 20px;
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.03);
}
.title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.markdown('<div class="title">🏥 Healthcare AI Premium Dashboard</div>', unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    return pd.read_csv(url)

data = load_data()
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestClassifier()
model.fit(X, y)

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("🧾 Patient Details")

preg = st.sidebar.slider("Pregnancies", 0, 10, 1)
glucose = st.sidebar.slider("Glucose", 50, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 50, 140, 80)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
age = st.sidebar.slider("Age", 18, 80, 30)

# -----------------------------
# PREDICTION
# -----------------------------
if st.sidebar.button("🚀 Analyze Patient"):

    input_data = pd.DataFrame([{
        "Pregnancies": preg,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": bmi,
        "DiabetesPedigreeFunction": 0.5,
        "Age": age
    }])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    confidence = round(max(prob)*100,2)

    risk = "HIGH 🔴" if pred == 1 else "LOW 🟢"

    # -----------------------------
    # ROW LAYOUT
    # -----------------------------
    col1, col2 = st.columns(2)

    # -----------------------------
    # CARD 1: RESULT
    # -----------------------------
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧠 Prediction Result")
        st.write(f"Risk: {risk}")
        st.write(f"Confidence: {confidence}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # CARD 2: GAUGE CHART 🔥
    # -----------------------------
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Risk Meter")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text': "Confidence"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if pred==1 else "green"},
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # PIE CHART
    # -----------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Risk Distribution")

    pie = go.Figure(data=[go.Pie(
        labels=["Low Risk", "High Risk"],
        values=[prob[0]*100, prob[1]*100],
        hole=.4
    )])

    st.plotly_chart(pie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # EXPLANATION
    # -----------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Key Factors")

    if glucose > 140:
        st.write("• High Glucose")
    if bmi > 30:
        st.write("• High BMI")
    if age > 50:
        st.write("• Age Risk")
    if bp > 90:
        st.write("• High Blood Pressure")

    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------
    # DOWNLOAD REPORT
    # -----------------------------
    report = f"""
    Patient Risk Report
    -------------------
    Risk: {risk}
    Confidence: {confidence}%
    """

    st.download_button("📄 Download Report", report, file_name="report.txt")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("⚡ Premium Healthcare AI | Built with Streamlit")