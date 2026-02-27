# 🛡️ ChurnGuard – Customer Churn Prediction System

A modern, end-to-end **Customer Churn Prediction Web App** built with **Streamlit** and **Scikit-learn**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-ChurnGuard-6366f1?style=for-the-badge)](https://churnguardx.streamlit.app/)

### 🌐 [**Live Demo → churnguardx.streamlit.app**](https://churnguardx.streamlit.app/)

---

## 📌 Features

- 📂 **CSV Upload** or built-in sample dataset (1,000 customers)
- 🧹 **Auto Preprocessing** – missing values, encoding, scaling
- 🤖 **ML Models** – Random Forest & Logistic Regression
- 📈 **Interactive Dashboard** – Pie chart, histograms, heatmap, ROC curve
- 🔮 **Real-time Prediction** – Single customer churn prediction with gauge chart
- 🎨 **Professional UI** – Dark gradient theme, styled metric cards, smooth animations

---

## 🖥️ Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | App intro, data loading (CSV upload or sample data) |
| 📊 **Data Exploration** | Dataset preview, statistics, visualizations |
| 🤖 **Model Training** | Train model, view accuracy, confusion matrix, feature importance, ROC |
| 🔮 **Prediction** | Customer input form → churn prediction with probability gauge |
| ℹ️ **About** | Project overview, tech stack |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/aryankule-08-a11y/-Churn-Guard.git
cd -Churn-Guard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Framework | Streamlit |
| ML | Scikit-learn (Random Forest, Logistic Regression) |
| Data | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |

---

## 📊 Model Metrics

- ✅ Accuracy & AUC-ROC Score
- 📉 Confusion Matrix (heatmap)
- 📋 Classification Report
- 🌟 Feature Importance / Coefficients
- 📈 ROC Curve

---

## 🔮 Prediction Input Fields

| Field | Type |
|-------|------|
| Gender | Male / Female |
| Age | 18–80 |
| Tenure | 0–72 months |
| Monthly Charges | $18–$120 |
| Contract Type | Month-to-month / One year / Two year |
| Internet Service | DSL / Fiber optic / No |
| Payment Method | Electronic check / Mailed check / Bank transfer / Credit card |

---

## 📁 Project Structure

```
ChurnGuard/
├── app.py                    # Main Streamlit application
├── generate_sample_data.py   # Sample dataset generator
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 👨‍💻 Author

Built with ❤️ by **Aryan Kule**

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
