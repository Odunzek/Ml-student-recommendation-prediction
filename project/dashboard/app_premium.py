"""
Premium Student Success Predictor Dashboard
Professional color scheme with premium UI components
"""

import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
import numpy as np

# ====================================================
# PAGE CONFIGURATION
# ====================================================
st.set_page_config(
    page_title="Student Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================================================
# PREMIUM CSS WITH BETTER COLORS & SLIDERS
# ====================================================
st.markdown("""
<style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }

    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Main Background - Clean White/Light Gray */
    .main {
        background: #f8fafc;
    }

    /* Headers */
    .main-header {
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.2);
        text-align: center;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .main-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.95);
        font-weight: 400;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #0f172a;
        margin: 2.5rem 0 1.5rem 0;
        padding: 1rem 1.5rem;
        background: white;
        border-radius: 12px;
        border-left: 5px solid #0ea5e9;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .section-icon {
        font-size: 1.5rem;
    }

    /* Cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        height: 100%;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }

    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.95rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    /* Risk Cards - Better Colors */
    .risk-card {
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
        border: 2px solid;
    }

    .risk-high {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border-color: #f43f5e;
    }

    .risk-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-color: #22c55e;
    }

    .risk-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
    }

    .risk-high .risk-label {
        color: #dc2626;
    }

    .risk-low .risk-label {
        color: #16a34a;
    }

    .risk-score {
        font-size: 5rem;
        font-weight: 800;
        line-height: 1;
        margin: 1rem 0;
    }

    .risk-high .risk-score {
        background: linear-gradient(135deg, #f43f5e 0%, #dc2626 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .risk-low .risk-score {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .risk-badge {
        background: rgba(255, 255, 255, 0.9);
        padding: 0.6rem 1.2rem;
        border-radius: 999px;
        font-size: 0.95rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0.5rem 0 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* PREMIUM SLIDERS */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #0ea5e9, #06b6d4) !important;
    }

    .stSlider > div > div > div {
        background-color: #e2e8f0 !important;
    }

    .stSlider [role="slider"] {
        background: white !important;
        border: 3px solid #0ea5e9 !important;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3) !important;
        width: 24px !important;
        height: 24px !important;
    }

    .stSlider [role="slider"]:hover {
        box-shadow: 0 6px 16px rgba(14, 165, 233, 0.5) !important;
    }

    /* PREMIUM NUMBER INPUTS */
    .stNumberInput input {
        border: 2px solid #e2e8f0 !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }

    .stNumberInput input:focus {
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1) !important;
    }

    /* PREMIUM SELECT BOXES */
    .stSelectbox > div > div {
        border: 2px solid #e2e8f0 !important;
        border-radius: 10px !important;
        background: white !important;
        transition: all 0.3s ease !important;
    }

    .stSelectbox > div > div:hover {
        border-color: #0ea5e9 !important;
    }

    .stSelectbox [data-baseweb="select"] {
        border-radius: 10px !important;
    }

    /* PREMIUM BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 2.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.4) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(14, 165, 233, 0.5) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* PREMIUM FILE UPLOADER */
    [data-testid="stFileUploader"] {
        border: 2px dashed #0ea5e9 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        background: #f8fafc !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        background: #f1f5f9 !important;
        border-color: #06b6d4 !important;
    }

    /* Sidebar - Clean Professional */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    [data-testid="stSidebar"] .stRadio > label {
        background: rgba(255, 255, 255, 0.1) !important;
        padding: 0.75rem 1rem !important;
        border-radius: 8px !important;
        margin: 0.25rem 0 !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stSidebar"] .stRadio > label:hover {
        background: rgba(255, 255, 255, 0.2) !important;
    }

    [data-testid="stSidebar"] .stRadio [data-checked="true"] {
        background: rgba(14, 165, 233, 0.3) !important;
    }

    /* Insights Boxes */
    .insights-box {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 4px solid;
    }

    .insights-success {
        border-left-color: #22c55e;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }

    .insights-warning {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    }

    .insights-info {
        border-left-color: #3b82f6;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    }

    .insights-box h4 {
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }

    .insights-box ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .insights-box li {
        padding: 0.5rem 0;
        display: flex;
        align-items: start;
        gap: 0.75rem;
        line-height: 1.6;
    }

    .insights-box li::before {
        content: "✓";
        font-weight: 700;
        font-size: 1.2rem;
        flex-shrink: 0;
    }

    .insights-success li::before {
        color: #22c55e;
    }

    .insights-warning li::before {
        content: "→";
        color: #f59e0b;
    }

    .insights-info li::before {
        color: #3b82f6;
    }

    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #0ea5e9, #06b6d4) !important;
        border-radius: 999px !important;
    }

    .stProgress > div {
        background-color: #e2e8f0 !important;
        border-radius: 999px !important;
    }

    /* Data Tables */
    .dataframe {
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* Download Button */
    .stDownloadButton > button {
        background: white !important;
        color: #0ea5e9 !important;
        border: 2px solid #0ea5e9 !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stDownloadButton > button:hover {
        background: #0ea5e9 !important;
        color: white !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: white !important;
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        font-weight: 600 !important;
    }

    /* Info/Success/Warning/Error Messages */
    .stAlert {
        border-radius: 10px !important;
        border: none !important;
        padding: 1rem 1.5rem !important;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #0ea5e9 !important;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================
# CONFIGURATION
# ====================================================
PIPELINE_PATH = "student_regression_pipeline.pkl"
DB_PATH = "predictions.db"

# ====================================================
# DATABASE FUNCTIONS
# ====================================================
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            predicted_score REAL,
            risk_status TEXT,
            hours_studied REAL,
            attendance REAL,
            sleep_hours REAL,
            previous_scores REAL,
            threshold REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_prediction(pred_score, is_risk, input_data, threshold):
    """Save prediction to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions
        (timestamp, predicted_score, risk_status, hours_studied, attendance,
         sleep_hours, previous_scores, threshold)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        pred_score,
        "At Risk" if is_risk else "Not At Risk",
        input_data["Hours_Studied"],
        input_data["Attendance"],
        input_data["Sleep_Hours"],
        input_data["Previous_Scores"],
        threshold
    ))
    conn.commit()
    conn.close()

# ====================================================
# LOAD PIPELINE
# ====================================================
@st.cache_resource
def load_pipeline():
    """Load the trained pipeline"""
    try:
        with open(PIPELINE_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"⚠️ Pipeline file not found at {PIPELINE_PATH}")
        st.info("📝 Please run: `python create_pipeline_mlflow.py`")
        st.stop()

pipeline = load_pipeline()

# ====================================================
# PREPROCESSING FUNCTION
# ====================================================
def preprocess_input(input_dict):
    """Encode categorical variables"""
    df = pd.DataFrame([input_dict])

    # Ordinal mappings
    lmh = {"Low": 1, "Medium": 2, "High": 3}
    df["Parental_Involvement"] = df["Parental_Involvement"].map(lmh)
    df["Access_to_Resources"] = df["Access_to_Resources"].map(lmh)
    df["Teacher_Quality"] = df["Teacher_Quality"].map(lmh)
    df["Family_Income"] = df["Family_Income"].map(lmh)
    df["Motivation_Level"] = df["Motivation_Level"].map(lmh)

    df["Distance_from_Home"] = df["Distance_from_Home"].map(
        {"Near": 1, "Moderate": 2, "Far": 3}
    )

    df["Peer_Influence"] = df["Peer_Influence"].map(
        {"Negative": 1, "Neutral": 2, "Positive": 3}
    )

    # Binary mappings
    bin_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    df["Extracurricular_Activities"] = df["Extracurricular_Activities"].map(bin_map)
    df["Internet_Access"] = df["Internet_Access"].map(bin_map)
    df["Learning_Disabilities"] = df["Learning_Disabilities"].map(bin_map)
    df["Gender"] = df["Gender"].map(bin_map)

    df["School_Type"] = df["School_Type"].map({"Public": 0, "Private": 1})

    # One-hot encoding
    df["Parental_Education_Level_High School"] = (
        1 if input_dict["Parental_Education_Level"] == "High School" else 0
    )
    df["Parental_Education_Level_Postgraduate"] = (
        1 if input_dict["Parental_Education_Level"] == "Postgraduate" else 0
    )
    df["Parental_Education_Level_Unknown"] = (
        1 if input_dict["Parental_Education_Level"] == "Unknown" else 0
    )

    # Column order
    expected_cols = [
        "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
        "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores", "Motivation_Level",
        "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
        "School_Type", "Peer_Influence", "Physical_Activity", "Learning_Disabilities",
        "Distance_from_Home", "Gender", "Parental_Education_Level_High School",
        "Parental_Education_Level_Postgraduate", "Parental_Education_Level_Unknown",
    ]

    df = df.reindex(columns=expected_cols, fill_value=0)
    return df

# ====================================================
# VISUALIZATION FUNCTIONS
# ====================================================
def create_gauge_chart(score, threshold):
    """Create premium gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={
            'suffix': "",
            'font': {'size': 64, 'color': '#0f172a', 'family': 'Poppins', 'weight': 700},
            'valueformat': '.1f'
        },
        title={
            'text': "Predicted Score",
            'font': {'size': 20, 'color': '#64748b', 'family': 'Poppins', 'weight': 500}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': "#cbd5e1",
                'tickfont': {'size': 12}
            },
            'bar': {
                'color': "#0ea5e9",
                'thickness': 0.35,
            },
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 40], 'color': '#fee2e2'},
                {'range': [40, 50], 'color': '#fed7aa'},
                {'range': [50, 60], 'color': '#fef3c7'},
                {'range': [60, 70], 'color': '#d9f99d'},
                {'range': [70, 85], 'color': '#bbf7d0'},
                {'range': [85, 100], 'color': '#a7f3d0'}
            ],
            'threshold': {
                'line': {'color': "#f43f5e", 'width': 5},
                'thickness': 0.8,
                'value': threshold
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'family': 'Poppins'},
        height=350,
        margin=dict(l=30, r=30, t=80, b=30)
    )

    return fig

def create_feature_chart(input_data):
    """Create feature comparison radar chart"""
    categories = ['Study Hours', 'Attendance', 'Sleep', 'Previous Scores', 'Activity']

    values = [
        min(input_data['Hours_Studied'] / 30 * 100, 100),
        input_data['Attendance'],
        input_data['Sleep_Hours'] / 12 * 100,
        input_data['Previous_Scores'],
        min(input_data['Physical_Activity'] / 10 * 100, 100)
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(14, 165, 233, 0.15)',
        line=dict(color='#0ea5e9', width=3),
        name='Student Profile'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=11),
                gridcolor='#e2e8f0'
            ),
            angularaxis=dict(
                gridcolor='#e2e8f0'
            ),
            bgcolor="white"
        ),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'family': 'Poppins', 'size': 13, 'color': '#0f172a'},
        height=380,
        margin=dict(l=90, r=90, t=50, b=50)
    )

    return fig

# ====================================================
# SIDEBAR
# ====================================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem; font-size: 1.8rem;'>⚙️ Navigation</h2>", unsafe_allow_html=True)

    page = st.radio(
        "",
        ["🎯 Single Prediction", "📊 Batch Upload", "📈 History"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 1.3rem; margin-bottom: 1rem;'>⚡ Settings</h3>", unsafe_allow_html=True)

    risk_threshold = st.slider(
        "Risk Threshold",
        min_value=40,
        max_value=90,
        value=65,
        step=1,
        help="Students scoring below this threshold are flagged as at-risk"
    )

    st.markdown(f"""
        <div style='background: rgba(255,255,255,0.1); padding: 1.2rem; border-radius: 10px; margin-top: 1.5rem;'>
            <p style='margin: 0; font-size: 0.95rem; line-height: 1.6;'>
                📊 <strong>Current Threshold: {risk_threshold}</strong><br>
                <span style='opacity: 0.9; font-size: 0.85rem;'>Students scoring below {risk_threshold} will be flagged for intervention</span>
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div style='font-size: 0.9rem; padding: 1rem; line-height: 1.8;'>
            <p style='font-weight: 600; margin-bottom: 0.75rem;'>💡 Quick Tips:</p>
            <ul style='padding-left: 1.5rem; margin: 0;'>
                <li style='margin-bottom: 0.5rem;'>Adjust threshold dynamically</li>
                <li style='margin-bottom: 0.5rem;'>Upload CSV for batch processing</li>
                <li style='margin-bottom: 0.5rem;'>Track predictions over time</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# ====================================================
# PAGE: SINGLE STUDENT PREDICTION
# ====================================================
if page == "🎯 Single Prediction":
    # Header
    st.markdown("""
        <div class='main-header fade-in'>
            <div class='main-title'>🎓 Student Success Predictor</div>
            <div class='main-subtitle'>AI-Powered Early Warning System for Academic Excellence</div>
        </div>
    """, unsafe_allow_html=True)

    # Input Form
    st.markdown("<div class='section-header'><span class='section-icon'>📚</span> Academic Performance Metrics</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        hours_studied = st.number_input("📖 Study Hours per Week", 0, 168, 15, help="Total weekly study hours")
        attendance = st.slider("🎯 Attendance Percentage", 0, 100, 80, help="Class attendance rate")

    with col2:
        previous_scores = st.slider("📊 Previous Exam Scores", 0, 100, 70, help="Average of past exam scores")
        tutoring_sessions = st.number_input("👨‍🏫 Tutoring Sessions/Month", 0, 30, 2, help="Monthly tutoring frequency")

    with col3:
        sleep_hours = st.slider("😴 Sleep Hours per Night", 0, 12, 7, help="Average nightly sleep duration")
        physical_activity = st.number_input("🏃 Physical Activity hrs/week", 0, 20, 3, help="Weekly exercise hours")

    st.markdown("<div class='section-header'><span class='section-icon'>👨‍👩‍👧</span> Family & Support System</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        parental_involvement = st.selectbox("👪 Parental Involvement", ["Low", "Medium", "High"])
        parental_education = st.selectbox("🎓 Parental Education", ["High School", "Postgraduate", "Unknown"])

    with col2:
        family_income = st.selectbox("💰 Family Income Level", ["Low", "Medium", "High"])
        access_resources = st.selectbox("📚 Resource Access", ["Low", "Medium", "High"])

    with col3:
        internet_access = st.selectbox("🌐 Internet Access", ["Yes", "No"])
        peer_influence = st.selectbox("👥 Peer Influence", ["Positive", "Neutral", "Negative"])

    with col4:
        motivation_level = st.selectbox("🔥 Motivation Level", ["Low", "Medium", "High"])
        extra_curricular = st.selectbox("⚽ Extracurricular Activities", ["Yes", "No"])

    st.markdown("<div class='section-header'><span class='section-icon'>🏫</span> School Environment</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        school_type = st.selectbox("🏛️ School Type", ["Public", "Private"])

    with col2:
        teacher_quality = st.selectbox("👨‍🏫 Teacher Quality", ["Low", "Medium", "High"])

    with col3:
        distance_home = st.selectbox("📍 Distance from Home", ["Near", "Moderate", "Far"])

    with col4:
        learning_disabilities = st.selectbox("🧩 Learning Disabilities", ["No", "Yes"])
        gender = st.selectbox("👤 Gender", ["Male", "Female"])

    # Build input dict
    input_data = {
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_resources,
        "Extracurricular_Activities": extra_curricular,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Motivation_Level": motivation_level,
        "Internet_Access": internet_access,
        "Tutoring_Sessions": tutoring_sessions,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "School_Type": school_type,
        "Peer_Influence": peer_influence,
        "Physical_Activity": physical_activity,
        "Learning_Disabilities": learning_disabilities,
        "Distance_from_Home": distance_home,
        "Gender": gender,
        "Parental_Education_Level": parental_education,
    }

    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🚀 Analyze Student Performance", use_container_width=True, type="primary")

    if predict_btn:
        with st.spinner("🔮 Analyzing student data..."):
            try:
                # Preprocess and predict
                X_input = preprocess_input(input_data)
                pred_score = float(pipeline.predict(X_input)[0])

                is_risk = pred_score < risk_threshold

                # Save to database
                save_prediction(pred_score, is_risk, input_data, risk_threshold)

                st.markdown("<br>", unsafe_allow_html=True)

                # Results
                col1, col2 = st.columns([1.3, 1])

                with col1:
                    # Gauge Chart
                    fig_gauge = create_gauge_chart(pred_score, risk_threshold)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col2:
                    # Risk Card
                    risk_class = "risk-high" if is_risk else "risk-low"
                    risk_emoji = "⚠️" if is_risk else "✅"
                    risk_text = "AT RISK" if is_risk else "ON TRACK"

                    gap = pred_score - risk_threshold
                    gap_text = f"{gap:+.1f}"
                    gap_color = "#dc2626" if gap < 0 else "#16a34a"

                    st.markdown(f"""
                        <div class='risk-card {risk_class} fade-in'>
                            <div class='risk-label'>{risk_emoji} {risk_text}</div>
                            <div class='risk-score'>{pred_score:.1f}</div>
                            <div style='font-size: 1rem; color: #64748b; margin-bottom: 1.5rem;'>Predicted Exam Score</div>
                            <div>
                                <span class='risk-badge'>📊 Threshold: {risk_threshold}</span>
                                <span class='risk-badge' style='color: {gap_color};'>📈 Gap: {gap_text} pts</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                # Feature Radar Chart
                st.markdown("<div class='section-header'><span class='section-icon'>📊</span> Student Profile Overview</div>", unsafe_allow_html=True)
                fig_radar = create_feature_chart(input_data)
                st.plotly_chart(fig_radar, use_container_width=True)

                # SHAP Explainability
                st.markdown("<div class='section-header'><span class='section-icon'>🔍</span> Feature Importance Analysis</div>", unsafe_allow_html=True)

                with st.spinner("Generating explainability insights..."):
                    regressor = pipeline.named_steps['regressor']
                    X_transformed = pipeline.named_steps['preprocessor'].transform(X_input)

                    explainer = shap.TreeExplainer(regressor)
                    shap_values = explainer.shap_values(X_transformed)

                    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
                    shap.summary_plot(shap_values, X_transformed, feature_names=X_input.columns.tolist(),
                                    plot_type="bar", show=False, color='#0ea5e9')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # Insights & Recommendations
                st.markdown("<div class='section-header'><span class='section-icon'>💡</span> Insights & Actionable Recommendations</div>", unsafe_allow_html=True)

                recommendations = []
                strengths = []

                if attendance < 75:
                    recommendations.append("**Boost attendance** - Currently below 75%. Target 85%+ for better outcomes")
                elif attendance >= 90:
                    strengths.append("**Outstanding attendance** - Consistent class participation is excellent!")

                if hours_studied < 10:
                    recommendations.append("**Increase study time** - Aim for 10-15 focused hours weekly")
                elif hours_studied >= 20:
                    strengths.append("**Dedicated study routine** - Strong academic commitment!")

                if sleep_hours < 6:
                    recommendations.append("**Improve sleep hygiene** - 7-8 hours nightly enhances learning")
                elif 7 <= sleep_hours <= 9:
                    strengths.append("**Optimal sleep pattern** - Well-rested for peak performance!")

                if motivation_level == "Low":
                    recommendations.append("**Build motivation** - Set achievable goals and celebrate progress")
                elif motivation_level == "High":
                    strengths.append("**High intrinsic motivation** - Keep channeling this energy!")

                if tutoring_sessions < 1 and is_risk:
                    recommendations.append("**Consider tutoring** - 1-2 monthly sessions can make a difference")

                if extra_curricular == "No" and not is_risk:
                    recommendations.append("**Explore activities** - Well-rounded development supports growth")

                col1, col2 = st.columns(2)

                with col1:
                    if strengths:
                        st.markdown(f"""
                            <div class='insights-box insights-success'>
                                <h4 style='color: #16a34a;'>✨ Key Strengths</h4>
                                <ul>
                                    {''.join(f'<li>{s}</li>' for s in strengths)}
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)

                with col2:
                    if recommendations:
                        st.markdown(f"""
                            <div class='insights-box insights-warning'>
                                <h4 style='color: #d97706;'>🎯 Action Plan</h4>
                                <ul>
                                    {''.join(f'<li>{r}</li>' for r in recommendations)}
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div class='insights-box insights-info'>
                                <h4 style='color: #2563eb;'>🎉 Excellent Profile!</h4>
                                <p style='margin: 0; line-height: 1.6;'>The student demonstrates strong performance across key indicators. Continue monitoring progress and maintaining current strategies.</p>
                            </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ Prediction error: {e}")
                with st.expander("🔍 Error Details"):
                    import traceback
                    st.code(traceback.format_exc())

# ====================================================
# PAGE: BATCH UPLOAD
# ====================================================
elif page == "📊 Batch Upload":
    st.markdown("""
        <div class='main-header fade-in'>
            <div class='main-title'>📊 Batch Student Analysis</div>
            <div class='main-subtitle'>Process multiple students simultaneously via CSV upload</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='metric-card'>
            <h3 style='margin-bottom: 1rem; color: #0f172a;'>📋 CSV Requirements</h3>
            <p style='color: #64748b; line-height: 1.6; margin-bottom: 0.75rem;'>
                Your CSV file should contain the following columns (in any order):
            </p>
            <code style='background: #f1f5f9; padding: 0.75rem; border-radius: 6px; display: block; font-size: 0.85rem;'>
                Hours_Studied, Attendance, Parental_Involvement, Access_to_Resources, ...
            </code>
            <p style='margin-top: 1rem; color: #64748b;'>
                📄 See <strong>sample_students_batch.csv</strong> for a complete example
            </p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload Your CSV File",
        type="csv",
        help="Select a CSV file containing student data"
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)

            st.success(f"✅ Successfully loaded **{len(df_upload)}** students!")

            st.markdown("<div class='section-header'><span class='section-icon'>👀</span> Data Preview</div>", unsafe_allow_html=True)
            st.dataframe(df_upload.head(10), use_container_width=True, height=400)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Run Batch Analysis", use_container_width=True, type="primary"):
                    with st.spinner("🔮 Processing all students..."):
                        predictions = []
                        progress_bar = st.progress(0)

                        for idx, row in df_upload.iterrows():
                            input_dict = row.to_dict()
                            X_input = preprocess_input(input_dict)
                            pred_score = float(pipeline.predict(X_input)[0])
                            is_risk = pred_score < risk_threshold

                            predictions.append({
                                "Predicted_Score": round(pred_score, 1),
                                "Risk_Status": "At Risk" if is_risk else "Not At Risk"
                            })

                            progress_bar.progress((idx + 1) / len(df_upload))

                        progress_bar.empty()

                        # Results
                        results_df = df_upload.copy()
                        results_df["Predicted_Score"] = [p["Predicted_Score"] for p in predictions]
                        results_df["Risk_Status"] = [p["Risk_Status"] for p in predictions]

                        st.success("✅ Batch analysis complete!")

                        # Summary Stats
                        at_risk_count = sum(1 for p in predictions if p["Risk_Status"] == "At Risk")
                        not_at_risk_count = len(predictions) - at_risk_count
                        avg_score = np.mean([p["Predicted_Score"] for p in predictions])

                        st.markdown("<div class='section-header'><span class='section-icon'>📊</span> Summary Statistics</div>", unsafe_allow_html=True)

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.markdown(f"""
                                <div class='metric-card'>
                                    <div class='metric-label'>Total Students</div>
                                    <div class='metric-value'>{len(predictions)}</div>
                                </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            st.markdown(f"""
                                <div class='metric-card'>
                                    <div class='metric-label'>At Risk</div>
                                    <div class='metric-value' style='background: linear-gradient(135deg, #f43f5e 0%, #dc2626 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{at_risk_count}</div>
                                    <div style='color: #64748b; font-size: 0.95rem; margin-top: 0.5rem;'>{at_risk_count/len(predictions)*100:.1f}% of total</div>
                                </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            st.markdown(f"""
                                <div class='metric-card'>
                                    <div class='metric-label'>Not At Risk</div>
                                    <div class='metric-value' style='background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{not_at_risk_count}</div>
                                    <div style='color: #64748b; font-size: 0.95rem; margin-top: 0.5rem;'>{not_at_risk_count/len(predictions)*100:.1f}% of total</div>
                                </div>
                            """, unsafe_allow_html=True)

                        with col4:
                            st.markdown(f"""
                                <div class='metric-card'>
                                    <div class='metric-label'>Average Score</div>
                                    <div class='metric-value'>{avg_score:.1f}</div>
                                    <div style='color: #64748b; font-size: 0.95rem; margin-top: 0.5rem;'>Mean prediction</div>
                                </div>
                            """, unsafe_allow_html=True)

                        # Results Table
                        st.markdown("<div class='section-header'><span class='section-icon'>📋</span> Detailed Results</div>", unsafe_allow_html=True)
                        st.dataframe(results_df, use_container_width=True, height=500)

                        # Download
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
            with st.expander("🔍 Error Details"):
                import traceback
                st.code(traceback.format_exc())

# ====================================================
# PAGE: HISTORY
# ====================================================
elif page == "📈 History":
    st.markdown("""
        <div class='main-header fade-in'>
            <div class='main-title'>📈 Prediction History</div>
            <div class='main-subtitle'>Track and analyze all predictions over time</div>
        </div>
    """, unsafe_allow_html=True)

    conn = sqlite3.connect(DB_PATH)
    history_df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()

    if len(history_df) == 0:
        st.markdown("""
            <div class='metric-card' style='text-align: center; padding: 4rem 2rem;'>
                <div style='font-size: 4rem; margin-bottom: 1rem;'>📭</div>
                <h2 style='color: #0f172a; margin-bottom: 1rem;'>No Predictions Yet</h2>
                <p style='font-size: 1.1rem; color: #64748b;'>Start by making your first prediction to begin tracking student performance</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        total = len(history_df)
        at_risk = len(history_df[history_df["risk_status"] == "At Risk"])
        avg_score = history_df["predicted_score"].mean()

        # Summary Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Total Predictions</div>
                    <div class='metric-value'>{total}</div>
                    <div style='color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;'>All time</div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Students At Risk</div>
                    <div class='metric-value' style='background: linear-gradient(135deg, #f43f5e 0%, #dc2626 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{at_risk}</div>
                    <div style='color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;'>{at_risk/total*100:.1f}% of total</div>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Average Score</div>
                    <div class='metric-value'>{avg_score:.1f}</div>
                    <div style='color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;'>Mean across all predictions</div>
                </div>
            """, unsafe_allow_html=True)

        # Recent Predictions
        st.markdown("<div class='section-header'><span class='section-icon'>📊</span> Recent Predictions</div>", unsafe_allow_html=True)

        display_df = history_df[["timestamp", "predicted_score", "risk_status", "hours_studied", "attendance", "threshold"]].copy()
        display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime('%Y-%m-%d %H:%M')
        display_df.columns = ["Timestamp", "Predicted Score", "Risk Status", "Study Hours", "Attendance %", "Threshold"]

        st.dataframe(display_df, use_container_width=True, height=500)

        # Actions
        col1, col2 = st.columns(2)

        with col1:
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete History",
                data=csv,
                file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            if st.button("🗑️ Clear All History", use_container_width=True, type="secondary"):
                conn = sqlite3.connect(DB_PATH)
                conn.execute("DELETE FROM predictions")
                conn.commit()
                conn.close()
                st.success("✅ History cleared successfully!")
                st.rerun()
