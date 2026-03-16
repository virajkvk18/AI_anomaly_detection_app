import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.ensemble import IsolationForest
import psutil
from pathlib import Path
import json
import plotly.express as px

pd.set_option("styler.render.max_elements", 2500000)

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Anomaly Detection Platform",
    page_icon="🚨",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>

.main {
    background-color:#0E1117;
}

h1, h2, h3 {
    color:#00C8FF;
}

.stButton>button {
    background-color:#00C8FF;
    color:white;
    border-radius:10px;
    padding:10px 20px;
}

[data-testid="stSidebar"] {
    background-color:#111827;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model_path = "model/anomaly_model.pkl"
pretrained_model = load(model_path)

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Navigation")

page = st.sidebar.radio(
    "Go To",
    ["🏠 Dashboard", "📂 Upload Dataset", "🖥 System Monitor", "📜 Prediction Logs"]
)

# ================= DASHBOARD =================
if page == "🏠 Dashboard":

    st.title("🚨 AI-Based Anomaly Detection Platform")

    st.markdown("""
    Detect anomalies in **large datasets** or monitor **real-time system metrics**  
    using **Machine Learning (Isolation Forest)**.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Type", "Isolation Forest")
    col2.metric("Supported Files", "CSV / XLSX")
    col3.metric("Monitoring", "CPU / Memory / Disk")

    st.divider()

    st.subheader("📊 Example Anomaly Visualization")

    sample = np.random.randn(200)

    fig = px.scatter(
        x=list(range(len(sample))),
        y=sample,
        title="Sample Anomaly Visualization"
    )

    st.plotly_chart(fig, use_container_width=True)


# ================= FILE UPLOAD PAGE =================
elif page == "📂 Upload Dataset":

    st.title("📂 Upload Dataset for Anomaly Detection")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx"]
    )

    if uploaded_file:

        with st.spinner("Reading dataset..."):

            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

        st.success("Dataset uploaded successfully!")

        tab1, tab2, tab3 = st.tabs(
            ["📄 Data Preview", "📊 Visualization", "🤖 Run Detection"]
        )

        # ================= DATA PREVIEW =================
        with tab1:

            st.subheader("Dataset Preview")

            st.dataframe(df.head(100))

            st.write("Shape:", df.shape)

        # ================= VISUALIZATION =================
        with tab2:

            numeric_df = df.select_dtypes(include=["number"])

            if numeric_df.empty:
                st.warning("No numeric columns available for visualization")
            else:

                column = st.selectbox(
                    "Select column for visualization",
                    numeric_df.columns
                )

                fig = px.histogram(
                    numeric_df,
                    x=column,
                    title=f"Distribution of {column}"
                )

                st.plotly_chart(fig, use_container_width=True)

        # ================= RUN DETECTION =================
        with tab3:

            numeric_df = df.select_dtypes(include=["number"])

            if numeric_df.empty:

                st.error("No numeric columns found")

            else:

                if st.button("🚀 Run Anomaly Detection"):

                    with st.spinner("Running Isolation Forest..."):

                        iso_model = IsolationForest(
                            contamination=0.05,
                            random_state=42
                        )

                        predictions = iso_model.fit_predict(numeric_df)

                        numeric_df["Prediction"] = pd.Series(predictions).map(
                            {1: "NORMAL", -1: "ANOMALY"}
                        )

                        numeric_df["Anomaly Score"] = iso_model.decision_function(
                            numeric_df.select_dtypes(include=['number']).values
                        )

                        result_df = numeric_df

                    st.success("Detection Completed!")

                    anomalies_df = result_df[
                        result_df["Prediction"] == "ANOMALY"
                    ]

                    col1, col2 = st.columns(2)

                    col1.metric("Total Records", len(result_df))
                    col2.metric("Anomalies Detected", len(anomalies_df))

                    st.subheader("🚨 Detected Anomalies")

                    if not anomalies_df.empty:
                        st.dataframe(anomalies_df)
                    else:
                        st.write("No anomalies detected")

                    st.subheader("📊 All Predictions")

                    st.dataframe(result_df.tail(500))

                    st.download_button(
                        "📥 Download Results",
                        data=result_df.to_csv(index=False),
                        file_name="anomaly_results.csv",
                        mime="text/csv"
                    )

                    # ================= LOGGING =================
                    log_file = Path("logs/prediction_log.json")
                    log_file.parent.mkdir(parents=True, exist_ok=True)

                    if not log_file.exists():
                        log_file.write_text("[]")

                    with log_file.open("r") as f:
                        logs = json.load(f)

                    logs.append({
                        "total_records": len(result_df),
                        "total_anomalies": int(
                            (result_df["Prediction"] == "ANOMALY").sum()
                        ),
                        "total_normal": int(
                            (result_df["Prediction"] == "NORMAL").sum()
                        )
                    })

                    with log_file.open("w") as f:
                        json.dump(logs, f, indent=4)


# ================= SYSTEM MONITOR =================
elif page == "🖥 System Monitor":

    st.title("🖥 Real-Time System Monitoring")

    if st.button("📊 Analyze Current System Metrics"):

        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent

        try:
            disk = psutil.disk_usage('/').percent
        except:
            disk = 0

        col1, col2, col3 = st.columns(3)

        col1.metric("CPU Usage", f"{cpu}%")
        col2.metric("Memory Usage", f"{memory}%")
        col3.metric("Disk Usage", f"{disk}%")

        x = np.array([[cpu, memory, disk]])

        pred = pretrained_model.predict(x)[0]

        try:
            score = pretrained_model.decision_function(x)[0]
        except:
            score = None

        status = "🚨 ANOMALY" if pred == -1 else "✔ NORMAL"

        st.subheader(f"Prediction: {status}")

        if score is not None:
            st.write(f"Anomaly Score: {score:.4f}")

        # Logging
        log_file = Path("logs/prediction_log.json")

        log_file.parent.mkdir(parents=True, exist_ok=True)

        if not log_file.exists():
            log_file.write_text("[]")

        with log_file.open("r") as f:
            logs = json.load(f)

        logs.append({
            "features": [cpu, memory, disk],
            "prediction": status,
            "score": float(score) if score else None
        })

        with log_file.open("w") as f:
            json.dump(logs, f, indent=4)


# ================= PREDICTION LOG =================
elif page == "📜 Prediction Logs":

    st.title("📜 Recent Prediction Logs")

    log_file = Path("logs/prediction_log.json")

    if log_file.exists():

        with log_file.open() as f:
            logs = json.load(f)

        if logs:

            df_logs = pd.DataFrame(logs)

            st.dataframe(df_logs.tail(20))

        else:

            st.write("No logs available yet")

    else:

        st.write("No prediction log found")