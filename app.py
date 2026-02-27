"""
🛡️ ChurnGuard – Customer Churn Prediction System
A modern, end-to-end Streamlit web application for predicting customer churn
using Machine Learning (Logistic Regression & Random Forest).
"""

# ──────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve)
import warnings, io
warnings.filterwarnings('ignore')

# Local dataset generator
from generate_sample_data import generate_sample_dataset

# ──────────────────────────────────────────────
# 2. PAGE CONFIG & CUSTOM CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard – Churn Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* ---------- Global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #e2e8f0 !important;
}

/* ---------- Metric Cards ---------- */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e293b, #334155);
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,.25);
}
div[data-testid="stMetric"] label { color: #94a3b8 !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #f1f5f9 !important; }

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2.2rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(99,102,241,.45);
}

/* ---------- Headers ---------- */
h1 { background: linear-gradient(135deg, #6366f1, #a855f7);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     font-weight: 800 !important; }
h2 { color: #6366f1 !important; font-weight: 700 !important; }
h3 { color: #8b5cf6 !important; }

/* ---------- Dataframe ---------- */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ---------- Success / Error boxes ---------- */
div[data-testid="stAlert"] { border-radius: 12px; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 3. HELPER FUNCTIONS
# ──────────────────────────────────────────────

@st.cache_data
def load_sample_data():
    """Load the built-in sample churn dataset."""
    return generate_sample_dataset(1000)


def preprocess_data(df):
    """
    Preprocess the dataset:
      - Drop ID columns
      - Handle missing values
      - Encode categoricals
      - Scale numericals
    Returns: X_train, X_test, y_train, y_test, scaler, encoders, feature_names
    """
    data = df.copy()

    # Drop ID-like columns
    id_cols = [c for c in data.columns if 'id' in c.lower() or 'customerid' in c.lower()]
    data.drop(columns=id_cols, inplace=True, errors='ignore')

    # Ensure Churn column is int
    if 'Churn' not in data.columns:
        st.error("❌ Dataset must contain a **Churn** column (0/1 or Yes/No).")
        return None
    if data['Churn'].dtype == 'object':
        data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})

    # Fill missing values
    for col in data.select_dtypes(include='number').columns:
        data[col].fillna(data[col].median(), inplace=True)
    for col in data.select_dtypes(include='object').columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Encode categorical features
    encoders = {}
    cat_cols = data.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    # Separate features & target
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    feature_names = X.columns.tolist()

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler, encoders, feature_names


def train_model(X_train, y_train, model_type='Random Forest'):
    """Train and return the selected ML model."""
    if model_type == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model


# ──────────────────────────────────────────────
# 4. SIDEBAR NAVIGATION
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ ChurnGuard")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Data Exploration", "🤖 Model Training",
         "🔮 Prediction", "ℹ️ About"],
        index=0,
    )
    st.markdown("---")
    st.caption("© 2026 ChurnGuard · Built with Streamlit")

# ──────────────────────────────────────────────
# 5. DATA LOADING (persisted in session state)
# ──────────────────────────────────────────────
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'results' not in st.session_state:
    st.session_state.results = {}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE: HOME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == "🏠 Home":
    st.markdown("# 🛡️ ChurnGuard")
    st.markdown("### Customer Churn Prediction System")
    st.markdown("---")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        > **Predict which customers are likely to leave** — before they do.

        ChurnGuard uses advanced **Machine Learning** to analyze customer
        behaviour and predict churn with high accuracy. Upload your own
        dataset or use our built-in sample data to get started instantly.

        **✨ Key Features**
        - 📂 Upload CSV *or* use sample data
        - 🧹 Automated data preprocessing
        - 🤖 Logistic Regression & Random Forest models
        - 📈 Interactive charts & dashboards
        - 🔮 Real-time single customer prediction
        """)
    with col2:
        st.markdown("#### 🚀 Quick Start")
        st.info("1️⃣  Load data on the **Data Exploration** page\n\n"
                "2️⃣  Train a model on the **Model Training** page\n\n"
                "3️⃣  Get predictions on the **Prediction** page")

    # ----- Data loading section -----
    st.markdown("---")
    st.markdown("## 📂 Load Your Dataset")
    load_col1, load_col2 = st.columns(2)

    with load_col1:
        st.markdown("#### Upload CSV")
        uploaded = st.file_uploader("Choose a CSV file", type=['csv'])
        if uploaded:
            st.session_state.df = pd.read_csv(uploaded)
            st.success(f"✅ Uploaded **{uploaded.name}** — {len(st.session_state.df):,} rows")

    with load_col2:
        st.markdown("#### Or Use Sample Data")
        if st.button("🎲 Load Sample Dataset", use_container_width=True):
            st.session_state.df = load_sample_data()
            st.success(f"✅ Sample dataset loaded — {len(st.session_state.df):,} rows")

    # Quick stats if data is loaded
    if st.session_state.df is not None:
        st.markdown("---")
        df = st.session_state.df
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Customers", f"{len(df):,}")
        m2.metric("Features", f"{df.shape[1] - 1}")
        churn_rate = df['Churn'].mean() * 100 if 'Churn' in df.columns else 0
        m3.metric("Churn Rate", f"{churn_rate:.1f}%")
        m4.metric("Retained", f"{100 - churn_rate:.1f}%")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE: DATA EXPLORATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "📊 Data Exploration":
    st.markdown("# 📊 Data Exploration")
    st.markdown("---")

    if st.session_state.df is None:
        st.warning("⚠️ No dataset loaded. Go to **🏠 Home** to load data first.")
        st.stop()

    df = st.session_state.df

    # ---- Dataset Preview ----
    tab1, tab2, tab3 = st.tabs(["📋 Preview", "📈 Statistics", "📊 Visualizations"])

    with tab1:
        st.dataframe(df.head(20), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]}")
        c3.metric("Missing Values", f"{df.isnull().sum().sum()}")

    with tab2:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)
        st.markdown("#### Data Types")
        dtype_df = pd.DataFrame({'Column': df.dtypes.index, 'Type': df.dtypes.values.astype(str),
                                 'Non-Null': df.notnull().sum().values, 'Nulls': df.isnull().sum().values})
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("### 📊 Dashboard")
        viz1, viz2 = st.columns(2)

        with viz1:
            # Churn Distribution Pie Chart
            if 'Churn' in df.columns:
                churn_counts = df['Churn'].value_counts().reset_index()
                churn_counts.columns = ['Churn', 'Count']
                churn_counts['Label'] = churn_counts['Churn'].map({0: 'Retained', 1: 'Churned',
                                                                    'No': 'Retained', 'Yes': 'Churned'})
                fig_pie = px.pie(churn_counts, values='Count', names='Label',
                                 color_discrete_sequence=['#6366f1', '#f43f5e'],
                                 title='🔄 Churn vs Retained',
                                 hole=0.45)
                fig_pie.update_traces(textinfo='percent+label', textfont_size=14)
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      font=dict(color='#e2e8f0'), title_font_size=18)
                st.plotly_chart(fig_pie, use_container_width=True)

        with viz2:
            # Monthly Charges vs Churn
            if 'MonthlyCharges' in df.columns and 'Churn' in df.columns:
                temp = df.copy()
                temp['Churn_Label'] = temp['Churn'].map({0: 'Retained', 1: 'Churned', 'No': 'Retained', 'Yes': 'Churned'})
                fig_bar = px.histogram(temp, x='MonthlyCharges', color='Churn_Label',
                                        barmode='overlay', nbins=30,
                                        color_discrete_map={'Retained': '#6366f1', 'Churned': '#f43f5e'},
                                        title='💰 Monthly Charges Distribution by Churn')
                fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                       font=dict(color='#e2e8f0'), title_font_size=18,
                                       xaxis=dict(gridcolor='#334155'), yaxis=dict(gridcolor='#334155'))
                st.plotly_chart(fig_bar, use_container_width=True)

        # Correlation Heatmap
        st.markdown("### 🔥 Feature Correlations")
        numeric_df = df.select_dtypes(include='number')
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            fig_heat = px.imshow(corr, text_auto='.2f',
                                  color_continuous_scale='RdBu_r',
                                  title='Correlation Matrix')
            fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'),
                                    title_font_size=18, height=500)
            st.plotly_chart(fig_heat, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE: MODEL TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🤖 Model Training":
    st.markdown("# 🤖 Model Training")
    st.markdown("---")

    if st.session_state.df is None:
        st.warning("⚠️ No dataset loaded. Go to **🏠 Home** to load data first.")
        st.stop()

    # Model selection
    col_a, col_b = st.columns([2, 3])
    with col_a:
        model_choice = st.selectbox("Select Algorithm", ['Random Forest', 'Logistic Regression'], index=0)
    with col_b:
        st.info(f"📌 **{model_choice}** — {'Ensemble tree-based model with high accuracy.' if model_choice == 'Random Forest' else 'Linear model ideal for binary classification.'}")

    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner("⏳ Preprocessing data & training model..."):
            result = preprocess_data(st.session_state.df)
            if result is None:
                st.stop()
            X_train, X_test, y_train, y_test, scaler, encoders, feature_names = result
            model = train_model(X_train, y_train, model_choice)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred, output_dict=True)
            auc = roc_auc_score(y_test, y_proba)

            # Save to session
            st.session_state.model = model
            st.session_state.results = {
                'accuracy': acc, 'cm': cm, 'cr': cr, 'auc': auc,
                'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba,
                'scaler': scaler, 'encoders': encoders,
                'feature_names': feature_names, 'model_type': model_choice,
                'X_test': X_test
            }

        st.success(f"✅ **{model_choice}** trained successfully!")

        # ---- Metrics Row ----
        st.markdown("### 📈 Performance Metrics")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{acc:.2%}")
        k2.metric("AUC-ROC", f"{auc:.4f}")
        k3.metric("Precision (Churn)", f"{cr['1']['precision']:.2%}")
        k4.metric("Recall (Churn)", f"{cr['1']['recall']:.2%}")

    # Show saved results if available
    if st.session_state.results:
        r = st.session_state.results
        st.markdown("---")
        tab_cm, tab_cr, tab_fi, tab_roc = st.tabs(
            ["📉 Confusion Matrix", "📋 Classification Report",
             "🌟 Feature Importance", "📈 ROC Curve"])

        with tab_cm:
            labels = ['Retained (0)', 'Churned (1)']
            fig_cm = px.imshow(r['cm'], text_auto=True,
                               x=labels, y=labels,
                               color_continuous_scale=['#1e293b', '#6366f1', '#f43f5e'],
                               title='Confusion Matrix')
            fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'),
                                  title_font_size=18, height=450,
                                  xaxis_title='Predicted', yaxis_title='Actual')
            st.plotly_chart(fig_cm, use_container_width=True)

        with tab_cr:
            cr_df = pd.DataFrame(r['cr']).T.round(3)
            st.dataframe(cr_df, use_container_width=True)

        with tab_fi:
            model = st.session_state.model
            if hasattr(model, 'feature_importances_'):
                fi = pd.DataFrame({'Feature': r['feature_names'],
                                   'Importance': model.feature_importances_})
                fi = fi.sort_values('Importance', ascending=True)
                fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h',
                                color='Importance', color_continuous_scale=['#6366f1', '#a855f7', '#f43f5e'],
                                title='🌟 Feature Importance')
                fig_fi.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      font=dict(color='#e2e8f0'), title_font_size=18, height=450,
                                      xaxis=dict(gridcolor='#334155'), yaxis=dict(gridcolor='#334155'))
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                coefs = pd.DataFrame({'Feature': r['feature_names'],
                                      'Coefficient': np.abs(model.coef_[0])})
                coefs = coefs.sort_values('Coefficient', ascending=True)
                fig_co = px.bar(coefs, x='Coefficient', y='Feature', orientation='h',
                                color='Coefficient', color_continuous_scale=['#6366f1', '#a855f7', '#f43f5e'],
                                title='🌟 Feature Coefficients (absolute)')
                fig_co.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      font=dict(color='#e2e8f0'), title_font_size=18, height=450)
                st.plotly_chart(fig_co, use_container_width=True)

        with tab_roc:
            fpr, tpr, _ = roc_curve(r['y_test'], r['y_proba'])
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC = {r['auc']:.4f}",
                                          line=dict(color='#6366f1', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                                          line=dict(color='#475569', dash='dash')))
            fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate',
                                   yaxis_title='True Positive Rate',
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   font=dict(color='#e2e8f0'), title_font_size=18, height=450,
                                   xaxis=dict(gridcolor='#334155'), yaxis=dict(gridcolor='#334155'))
            st.plotly_chart(fig_roc, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE: PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "🔮 Prediction":
    st.markdown("# 🔮 Single Customer Prediction")
    st.markdown("---")

    if st.session_state.model is None:
        st.warning("⚠️ No trained model found. Go to **🤖 Model Training** to train a model first.")
        st.stop()

    r = st.session_state.results

    st.markdown("#### Enter Customer Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("👤 Gender", ['Male', 'Female'])
        age = st.slider("🎂 Age", 18, 80, 35)
        tenure = st.slider("📅 Tenure (months)", 0, 72, 12)

    with col2:
        monthly_charges = st.number_input("💰 Monthly Charges ($)", 18.0, 120.0, 50.0, step=1.0)
        contract = st.selectbox("📝 Contract Type", ['Month-to-month', 'One year', 'Two year'])

    with col3:
        internet = st.selectbox("🌐 Internet Service", ['DSL', 'Fiber optic', 'No'])
        payment = st.selectbox("💳 Payment Method",
                               ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])

    total_charges = monthly_charges * tenure

    st.markdown("---")
    if st.button("🔮 Predict Churn", use_container_width=True):
        try:
            # Build input dict
            input_data = {
                'Gender': gender, 'Age': age, 'Tenure': tenure,
                'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
                'ContractType': contract, 'InternetService': internet,
                'PaymentMethod': payment
            }

            # Encode categorical features
            for col, val in input_data.items():
                if col in r['encoders']:
                    le = r['encoders'][col]
                    if val in le.classes_:
                        input_data[col] = le.transform([val])[0]
                    else:
                        input_data[col] = 0

            # Create DataFrame in correct feature order
            input_df = pd.DataFrame([input_data])
            # Ensure all features exist
            for feat in r['feature_names']:
                if feat not in input_df.columns:
                    input_df[feat] = 0
            input_df = input_df[r['feature_names']]

            # Scale
            input_scaled = pd.DataFrame(r['scaler'].transform(input_df), columns=r['feature_names'])

            # Predict
            prediction = st.session_state.model.predict(input_scaled)[0]
            probability = st.session_state.model.predict_proba(input_scaled)[0]

            churn_prob = probability[1] * 100
            retain_prob = probability[0] * 100

            st.markdown("---")
            st.markdown("### 📊 Prediction Result")

            res1, res2 = st.columns([2, 1])

            with res1:
                if prediction == 1:
                    st.error(f"## ⚠️ Customer Will Churn")
                    st.markdown(f"**Churn Probability: `{churn_prob:.1f}%`**")
                    st.markdown("🔴 This customer is at **high risk** of leaving. "
                                "Consider offering retention incentives.")
                else:
                    st.success(f"## ✅ Customer Will Not Churn")
                    st.markdown(f"**Retention Probability: `{retain_prob:.1f}%`**")
                    st.markdown("🟢 This customer is **likely to stay**. Keep up the good service!")

            with res2:
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=churn_prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Risk", 'font': {'color': '#e2e8f0'}},
                    number={'suffix': '%', 'font': {'color': '#f1f5f9'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#94a3b8'},
                        'bar': {'color': '#6366f1'},
                        'bgcolor': '#1e293b',
                        'steps': [
                            {'range': [0, 30], 'color': '#22c55e'},
                            {'range': [30, 70], 'color': '#eab308'},
                            {'range': [70, 100], 'color': '#ef4444'}
                        ],
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'),
                                         height=280, margin=dict(t=60, b=0))
                st.plotly_chart(fig_gauge, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE: ABOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About ChurnGuard")
    st.markdown("---")

    st.markdown("""
    ### 🎯 Project Overview
    **ChurnGuard** is an intelligent customer churn prediction system designed
    to help businesses identify customers who are at risk of leaving.
    By leveraging machine learning algorithms, ChurnGuard provides
    actionable insights to reduce customer attrition and improve retention.

    ### 🔧 How It Works
    1. **Data Ingestion** — Upload your own CSV or use the built-in sample dataset
    2. **Preprocessing** — Automatic handling of missing values, encoding, and scaling
    3. **Model Training** — Choose between Logistic Regression or Random Forest
    4. **Evaluation** — Review accuracy, confusion matrix, classification report & ROC curve
    5. **Prediction** — Enter individual customer details to get real-time churn predictions
    """)

    st.markdown("### 🛠️ Technologies Used")
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    with tech_col1:
        st.markdown("""
        **Languages & Frameworks**
        - 🐍 Python 3.10+
        - 🎈 Streamlit
        """)
    with tech_col2:
        st.markdown("""
        **Data & ML**
        - 🐼 Pandas & NumPy
        - 🤖 Scikit-learn
        """)
    with tech_col3:
        st.markdown("""
        **Visualization**
        - 📊 Plotly
        - 🎨 Matplotlib & Seaborn
        """)

    st.markdown("---")
    st.markdown("""
    ### 📌 Models Available
    | Model | Type | Strengths |
    |---|---|---|
    | **Random Forest** | Ensemble | High accuracy, handles non-linearity |
    | **Logistic Regression** | Linear | Fast, interpretable, good baseline |
    """)

    st.markdown("---")
    st.markdown("### 👨‍💻 Developed By")
    st.info("Built with ❤️ using **Python** and **Streamlit** · ChurnGuard © 2026")
