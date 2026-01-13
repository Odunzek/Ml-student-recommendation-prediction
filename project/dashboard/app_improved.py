import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
import io

# ====================================================
# CONFIG
# ====================================================
PIPELINE_PATH = "student_regression_pipeline.pkl"
DB_PATH = "predictions.db"

st.set_page_config(
    page_title="Student Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================================================
# CSS STYLING
# ====================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #ff6cab 0%, #7366ff 50%, #20e3b2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #d0d3ff;
        font-size: 1.0rem;
        margin-bottom: 1.8rem;
    }
    .section-label {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.6rem;
        color: #f5f5ff;
    }
    .risk-card-high {
        padding: 1.3rem 1.5rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #ff4b5c, #ff9966);
        color: white;
    }
    .risk-card-low {
        padding: 1.3rem 1.5rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #20e3b2, #2cccff);
        color: #05222a;
    }
    .metric-pill {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.3rem;
        background: rgba(0,0,0,0.2);
        color: #fdfdff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ====================================================
# DATABASE SETUP
# ====================================================
def init_db():
    """Initialize SQLite database for prediction history"""
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
    try:
        with open(PIPELINE_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Pipeline file not found at {PIPELINE_PATH}")
        st.info("Please run the notebook cell to create the pipeline first")
        st.stop()

pipeline = load_pipeline()

# ====================================================
# PREPROCESSING FUNCTION
# ====================================================
def preprocess_input(input_dict):
    """
    Encode categorical variables to numeric.
    Scaling is handled by the pipeline automatically.
    """
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

    # Column order must match training data
    expected_cols = [
        "Hours_Studied",
        "Attendance",
        "Parental_Involvement",
        "Access_to_Resources",
        "Extracurricular_Activities",
        "Sleep_Hours",
        "Previous_Scores",
        "Motivation_Level",
        "Internet_Access",
        "Tutoring_Sessions",
        "Family_Income",
        "Teacher_Quality",
        "School_Type",
        "Peer_Influence",
        "Physical_Activity",
        "Learning_Disabilities",
        "Distance_from_Home",
        "Gender",
        "Parental_Education_Level_High School",
        "Parental_Education_Level_Postgraduate",
        "Parental_Education_Level_Unknown",
    ]

    df = df.reindex(columns=expected_cols, fill_value=0)
    return df


# ====================================================
# GAUGE CHART
# ====================================================
def score_gauge(score: float, threshold: float):
    score = max(0, min(100, score))
    threshold = max(0, min(100, threshold))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100", "font": {"size": 26}},
            title={"text": "Predicted Exam Score", "font": {"size": 16, "color": "#f5f5ff"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ffd166"},
                "steps": [
                    {"range": [0, 50], "color": "#ff4b5c"},
                    {"range": [50, 70], "color": "#ffb347"},
                    {"range": [70, 85], "color": "#4ecdc4"},
                    {"range": [85, 100], "color": "#2b9df4"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": threshold,
                },
            },
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f5f5ff"),
    )
    return fig


# ====================================================
# SIDEBAR
# ====================================================
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Choose Page", ["Single Student Prediction", "Batch Upload", "Prediction History"])

    st.markdown("---")
    st.markdown("### Settings")
    risk_threshold = st.slider(
        "Risk threshold (exam score)",
        min_value=40,
        max_value=90,
        value=65,
        step=1,
        help="If predicted exam score is below this value, the student is flagged as 'At Risk'.",
    )

# ====================================================
# PAGE: SINGLE STUDENT PREDICTION
# ====================================================
if page == "Single Student Prediction":
    st.markdown("<p class='main-title'>Student Success Predictor</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Predict exam scores and identify at-risk students early with AI-powered insights.</p>",
        unsafe_allow_html=True,
    )

    # Input form
    st.markdown("<div class='section-label'>Academic Performance Metrics</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        hours_studied = st.number_input("Hours Studied / Week", 0, 168, 15)
        attendance = st.slider("Attendance (%)", 0, 100, 80)

    with c2:
        previous_scores = st.slider("Previous Scores (avg %)", 0, 100, 70)
        tutoring_sessions = st.number_input("Tutoring Sessions / Month", 0, 30, 2)

    with c3:
        sleep_hours = st.slider("Sleep Hours / Night", 0, 12, 7)
        physical_activity = st.number_input("Physical Activity Hours / Week", 0, 20, 3)

    st.markdown("<div class='section-label'>Family & Support System</div>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
        parental_education = st.selectbox("Parental Education", ["High School", "Postgraduate", "Unknown"])

    with f2:
        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
        access_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])

    with f3:
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        peer_influence = st.selectbox("Peer Influence", ["Positive", "Neutral", "Negative"])

    with f4:
        motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
        extra_curricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

    st.markdown("<div class='section-label'>School Environment</div>", unsafe_allow_html=True)
    e1, e2, e3, e4 = st.columns(4)

    with e1:
        school_type = st.selectbox("School Type", ["Public", "Private"])

    with e2:
        teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])

    with e3:
        distance_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])

    with e4:
        learning_disabilities = st.selectbox("Learning Disabilities", ["No", "Yes"])
        gender = st.selectbox("Gender", ["Male", "Female"])

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

    st.markdown("---")

    # Prediction button
    center_col = st.columns([1, 2, 1])[1]
    with center_col:
        predict_btn = st.button("Analyze Student Risk", use_container_width=True, type="primary")

    if predict_btn:
        try:
            # Preprocess and predict
            X_input = preprocess_input(input_data)
            pred_score = float(pipeline.predict(X_input)[0])

            is_risk = pred_score < risk_threshold
            risk_label = "AT RISK" if is_risk else "NOT AT RISK"
            risk_class = "risk-card-high" if is_risk else "risk-card-low"
            emoji = "⚠️" if is_risk else "✅"

            # Save to database
            save_prediction(pred_score, is_risk, input_data, risk_threshold)

            # Display results
            g1, g2 = st.columns([1.2, 1])

            with g1:
                fig = score_gauge(pred_score, risk_threshold)
                st.plotly_chart(fig, use_container_width=True)

            with g2:
                st.markdown(
                    f"""
                    <div class="{risk_class}">
                        <div style="font-size:1.4rem; font-weight:800; margin-bottom:0.4rem;">
                            {emoji} {risk_label}
                        </div>
                        <div class="small-label">Predicted exam score</div>
                        <div style="font-size:2.0rem; font-weight:800; margin-bottom:0.5rem;">
                            {pred_score:.1f} / 100
                        </div>
                        <span class="metric-pill">Threshold: {risk_threshold}</span>
                        <span class="metric-pill">Gap: {pred_score - risk_threshold:+.1f}</span>
                        <p style="margin-top:0.8rem; font-size:0.9rem;">
                            Scores below the threshold are flagged as at-risk for early intervention.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # SHAP Explainability
            st.markdown("### Why This Prediction?")
            st.markdown("Understanding which factors influenced this prediction:")

            with st.spinner("Generating feature importance..."):
                # Get the regressor from the pipeline
                regressor = pipeline.named_steps['regressor']

                # Transform input through preprocessing only
                X_transformed = pipeline.named_steps['preprocessor'].transform(X_input)

                # Create SHAP explainer
                explainer = shap.TreeExplainer(regressor)
                shap_values = explainer.shap_values(X_transformed)

                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_transformed, feature_names=X_input.columns.tolist(),
                                plot_type="bar", show=False)
                st.pyplot(fig)
                plt.close()

            # Recommendations
            st.markdown("### Insights & Recommendations")

            reasons = []
            recs = []

            if attendance < 75:
                reasons.append("Attendance is below 75%, linked to lower performance.")
                recs.append("Try to keep attendance above 85%.")
            elif attendance >= 90:
                reasons.append("Strong attendance is a positive factor.")

            if hours_studied < 8:
                reasons.append("Study hours are on the low side.")
                recs.append("Aim for at least 10-15 focused study hours per week.")
            elif hours_studied >= 15:
                reasons.append("Healthy study routine detected.")

            if sleep_hours < 6:
                reasons.append("Limited sleep can hurt focus and memory.")
                recs.append("Try to get 7-8 hours of sleep consistently.")
            elif 7 <= sleep_hours <= 9:
                reasons.append("Sleep duration looks optimal.")

            if motivation_level == "Low":
                recs.append("Work on small, achievable goals to boost motivation.")

            if tutoring_sessions < 1 and pred_score < risk_threshold:
                recs.append("Consider 1-2 tutoring sessions per month for extra support.")

            if extra_curricular == "No":
                recs.append("A balanced life helps. Consider one activity the student enjoys.")

            if reasons:
                st.markdown("**Key Factors:**")
                for r in reasons:
                    st.markdown(f"- {r}")

            if recs:
                st.markdown("**Recommendations:**")
                for r in recs:
                    st.markdown(f"- {r}")
            else:
                st.markdown("The profile looks solid. Keep monitoring and supporting the student's routine.")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            st.code(traceback.format_exc())

# ====================================================
# PAGE: BATCH UPLOAD
# ====================================================
elif page == "Batch Upload":
    st.markdown("<p class='main-title'>Batch Student Analysis</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Upload a CSV file to analyze multiple students at once.</p>", unsafe_allow_html=True)

    st.markdown("### Upload CSV File")
    st.markdown("Your CSV should contain the following columns:")

    expected_columns = [
        "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
        "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores", "Motivation_Level",
        "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
        "School_Type", "Peer_Influence", "Physical_Activity", "Learning_Disabilities",
        "Distance_from_Home", "Gender", "Parental_Education_Level"
    ]

    st.code(", ".join(expected_columns), language="text")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df_upload)} students")

            st.markdown("### Preview")
            st.dataframe(df_upload.head())

            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Processing..."):
                    # Process each row
                    predictions = []
                    for idx, row in df_upload.iterrows():
                        input_dict = row.to_dict()
                        X_input = preprocess_input(input_dict)
                        pred_score = float(pipeline.predict(X_input)[0])
                        is_risk = pred_score < risk_threshold

                        predictions.append({
                            "Predicted_Score": round(pred_score, 1),
                            "Risk_Status": "At Risk" if is_risk else "Not At Risk"
                        })

                    # Add predictions to dataframe
                    results_df = df_upload.copy()
                    results_df["Predicted_Score"] = [p["Predicted_Score"] for p in predictions]
                    results_df["Risk_Status"] = [p["Risk_Status"] for p in predictions]

                    st.success("Batch prediction complete!")

                    # Show summary
                    at_risk_count = sum(1 for p in predictions if p["Risk_Status"] == "At Risk")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Students", len(predictions))
                    col2.metric("At Risk", at_risk_count, delta=f"{at_risk_count/len(predictions)*100:.1f}%")
                    col3.metric("Not At Risk", len(predictions) - at_risk_count)

                    # Show results
                    st.markdown("### Results")
                    st.dataframe(results_df)

                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name=f"student_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            import traceback
            st.code(traceback.format_exc())

# ====================================================
# PAGE: PREDICTION HISTORY
# ====================================================
elif page == "Prediction History":
    st.markdown("<p class='main-title'>Prediction History</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>View past predictions and analytics.</p>", unsafe_allow_html=True)

    # Load from database
    conn = sqlite3.connect(DB_PATH)
    history_df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()

    if len(history_df) == 0:
        st.info("No predictions yet. Make your first prediction!")
    else:
        # Summary metrics
        total = len(history_df)
        at_risk = len(history_df[history_df["risk_status"] == "At Risk"])
        avg_score = history_df["predicted_score"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", total)
        col2.metric("At Risk Count", at_risk, delta=f"{at_risk/total*100:.1f}%")
        col3.metric("Average Score", f"{avg_score:.1f}")

        # Recent predictions table
        st.markdown("### Recent Predictions")
        display_df = history_df[["timestamp", "predicted_score", "risk_status", "hours_studied", "attendance", "threshold"]]
        display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(display_df, use_container_width=True)

        # Download history
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="Download Full History",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        # Clear history button
        if st.button("Clear All History", type="secondary"):
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM predictions")
            conn.commit()
            conn.close()
            st.success("History cleared!")
            st.rerun()
