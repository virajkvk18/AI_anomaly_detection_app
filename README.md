🚨 AI-Based Anomaly Detection Platform
An AI-powered anomaly detection platform that detects unusual patterns in datasets and real-time system metrics (CPU, Memory, Disk).
Built with Python, Machine Learning, and Streamlit, the platform provides an interactive dashboard for anomaly analysis and system monitoring.

📌 Features

✅ Upload CSV / Excel datasets for anomaly detection
✅ Automatic anomaly detection using Isolation Forest
✅ Interactive data visualization with Plotly
✅ Real-time system metrics monitoring (CPU, Memory, Disk)
✅ Download anomaly detection results
✅ Prediction logging system
✅ Modern Streamlit dashboard UI with sidebar navigation

🧠 How It Works

The system uses the Isolation Forest algorithm, an unsupervised machine learning model designed to detect outliers.

Workflow

1️⃣ User uploads a dataset
2️⃣ System extracts numeric features
3️⃣ Isolation Forest model analyzes data patterns
4️⃣ Outliers are classified as ANOMALY
5️⃣ Results are displayed with interactive visualizations

For system monitoring:

1️⃣ CPU, Memory, Disk metrics are collected using psutil
2️⃣ Pre-trained model predicts normal vs anomaly system behavior

🖥️ Dashboard Preview

(Add screenshots of your app here)

Example sections:

Dashboard

Dataset Upload

Data Visualization

Anomaly Detection Results

System Monitoring

Prediction Logs

Example:

screenshots/
dashboard.png
upload_page.png
anomaly_results.png
⚙️ Tech Stack
Technology	Purpose
Python	Core programming
Streamlit	Interactive web dashboard
Scikit-learn	Machine learning models
Pandas	Data processing
NumPy	Numerical operations
Plotly	Interactive visualizations
Joblib	Model loading
Psutil	System monitoring
📂 Project Structure
AI_ANOMALY_DETECTION_APP
│
├── api
│   └── app.py
│
├── data
│   ├── collect_metrics.py
│   ├── system_metrics.csv
│
├── logs
│   └── prediction_log.json
│
├── model
│   ├── anomaly_model.pkl
│   └── train_model.py
│
├── monitoring
│   └── prediction_log.json
│
├── tester
│   └── tester.py
│
├── streamlit_app.py
├── requirements.txt
└── README.md
