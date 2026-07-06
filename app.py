"""
===========================================================================
 🏗️ Concrete Compressive Strength Prediction using Machine Learning
 A production-grade Streamlit application for ML-powered prediction
===========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="Concrete Strength Predictor | AI Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================== PATHS ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Concrete_Data.xls")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ======================== CUSTOM CSS ========================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --primary: #1E3A8A;
        --secondary: #475569;
        --accent: #F97316;
        --bg: #F8FAFC;
        --card: #FFFFFF;
        --text-primary: #0F172A;
        --text-secondary: #64748B;
        --success: #10B981;
        --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04);
        --shadow-xl: 0 20px 25px -5px rgba(0,0,0,0.08), 0 8px 10px -6px rgba(0,0,0,0.04);
        --radius: 12px;
        --radius-lg: 16px;
    }

    .stApp {
        background: var(--bg) !important;
        font-family: 'Inter', sans-serif !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent !important;}

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
    }
    section[data-testid="stSidebar"] * {
        color: #CBD5E1 !important;
    }
    .sidebar-brand { text-align: center; margin-bottom: 20px; }
    .sidebar-brand h2 { margin: 0; color: white !important; font-weight: 800; }
    .sidebar-brand p { margin: 0; font-size: 13px; color: #94A3B8 !important; }

    /* ---- Cards ---- */
    .premium-card {
        background: var(--card);
        border: 1px solid #E2E8F0;
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .premium-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    .metric-card {
        background: var(--card);
        border: 1px solid #E2E8F0;
        border-radius: var(--radius);
        padding: 1.25rem;
        text-align: center;
        box-shadow: var(--shadow);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    .metric-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--primary);
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.8rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        font-weight: 600;
    }

    /* ---- Hero Header ---- */
    .hero-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 50%, #1E40AF 100%);
        border-radius: var(--radius-lg);
        padding: 2.5rem;
        color: white;
        box-shadow: var(--shadow-xl);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-title { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; line-height: 1.2;}
    .hero-subtitle { font-size: 1.15rem; opacity: 0.9; margin-top: 0.5rem; }

    /* ---- Prediction Result ---- */
    .prediction-result {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #6EE7B7;
        border-radius: var(--radius-lg);
        padding: 2rem;
        text-align: center;
        box-shadow: var(--shadow-lg);
        animation: fadeIn 0.5s ease-out;
    }
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 900;
        color: #065F46;
        font-family: 'JetBrains Mono', monospace;
    }
    .prediction-unit { color: #047857; font-weight: 600; font-size: 1.2rem; }

    .strength-badge {
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
        margin-top: 1rem;
    }
    .badge-low { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
    .badge-medium { background: #DBEAFE; color: #1E40AF; border: 1px solid #93C5FD; }
    .badge-high { background: #D1FAE5; color: #065F46; border: 1px solid #6EE7B7; }
    .badge-ultra { background: #EDE9FE; color: #5B21B6; border: 1px solid #C4B5FD; }

    /* ---- Workflow and Explanations ---- */
    .workflow-box {
        background: var(--primary);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        margin: 5px auto;
        width: fit-content;
        box-shadow: var(--shadow-md);
    }
    .workflow-arrow { text-align: center; font-size: 24px; color: var(--secondary); margin: 2px 0;}
    
    .feature-card {
        background: white;
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: var(--shadow);
        transition: transform 0.2s;
    }
    .feature-card:hover { transform: translateX(5px); }
    .feature-title { font-weight: 700; font-size: 1.1rem; color: var(--primary); margin-bottom: 5px; }

    .prediction-intro {
        background: #FFFFFF;
        border: 1px solid #D7DEE8;
        border-left: 5px solid var(--accent);
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow);
    }
    .prediction-intro h3 {
        margin: 0 0 0.35rem 0;
        color: var(--text-primary);
        font-size: 1.2rem;
        font-weight: 800;
    }
    .prediction-intro p {
        margin: 0;
        color: var(--text-secondary);
        font-size: 0.95rem;
        line-height: 1.55;
    }
    .input-block {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.9rem 0.9rem 0.6rem;
        margin-bottom: 0.95rem;
        min-height: 150px;
    }
    .input-title {
        color: var(--text-primary);
        font-size: 0.95rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .input-help {
        color: var(--text-secondary);
        font-size: 0.78rem;
        line-height: 1.4;
        min-height: 34px;
        margin-bottom: 0.45rem;
    }
    .input-unit {
        display: inline-block;
        background: #E0F2FE;
        color: #075985;
        border: 1px solid #BAE6FD;
        border-radius: 999px;
        padding: 2px 9px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .summary-chip {
        display: inline-block;
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        color: #334155;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 700;
        margin: 0 6px 8px 0;
    }
    .about-step {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        min-height: 130px;
        box-shadow: var(--shadow);
    }
    .about-step h4 {
        margin: 0 0 0.45rem 0;
        color: var(--primary);
        font-size: 1rem;
    }
    .about-step p {
        margin: 0;
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--text-primary);
        border-bottom: 3px solid var(--primary);
        display: inline-block;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }

    /* ---- Button Override ---- */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1E3A8A, #3B82F6) !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.4) !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# ======================== DATA & MODEL LOADING ========================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_excel(DATA_PATH)
    df.columns = [
        "Cement", "Blast Furnace Slag", "Fly Ash", "Water",
        "Superplasticizer", "Coarse Aggregate", "Fine Aggregate",
        "Age", "Compressive Strength"
    ]
    return df

@st.cache_resource(show_spinner=False)
def load_model(model_name):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_scaler():
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        return None
    with open(scaler_path, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def get_model_metrics():
    df = load_data()
    X = df.drop("Compressive Strength", axis=1)
    y = df["Compressive Strength"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    metrics = {}
    for name in ["random_forest", "linear_reg"]:
        model = load_model(name)
        if model is None:
            continue
        
        # Ensure test set has the exact column names the model was trained on
        X_test_model = X_test_s.copy()
        if hasattr(model, 'feature_names_in_'):
            X_test_model.columns = model.feature_names_in_
            
        y_pred = model.predict(X_test_model)
        metrics[name] = {
            "accuracy": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "y_test": y_test.values,
            "y_pred": y_pred,
        }
    return metrics, X_test_s, y_test

def get_strength_category(strength):
    if strength < 20: return "Low Strength", "badge-low", "⚠️"
    elif strength < 40: return "Medium Strength", "badge-medium", "🔵"
    elif strength < 60: return "High Strength", "badge-high", "✅"
    else: return "Ultra High Strength", "badge-ultra", "🟣"

def get_confidence(strength, r2):
    base = r2 * 100
    if strength < 10 or strength > 75: base *= 0.92
    return min(round(base, 1), 99.9)

def generate_report_csv(inputs, prediction, category):
    report = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **inputs,
        "Predicted Strength (MPa)": round(prediction, 2),
        "Category": category,
    }])
    return report.to_csv(index=False).encode("utf-8")

# ======================== SIDEBAR ========================
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <h2>🏗️ Concrete AI</h2>
            <p>Strength Prediction System</p>
        </div>
        <hr style="border-color:#334155; margin-top:0;">
        """, unsafe_allow_html=True)
        
        st.markdown("### 📍 Navigation")
        page = st.radio("Go to", [
            "🏠 Dashboard", 
            "❓ What Is This Project?",
            "📖 How It Works", 
            "🔮 Prediction", 
            "📊 Dataset Analysis", 
            "📈 Model Performance", 
            "🎯 Feature Importance",
            "ℹ️ About Project"
        ], label_visibility="collapsed")
        
        st.markdown("<hr style='border-color:#334155;'>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Settings")
        active_model = st.selectbox("Active Model", ["random_forest", "linear_reg"], format_func=lambda x: "🌲 Random Forest" if x == "random_forest" else "📐 Linear Regression")
        
        st.markdown("<hr style='border-color:#334155;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; color:#94A3B8; font-size:12px;">
            <p>🌓 Theme: System</p>
            <p>Version 2.0.0</p>
            <p>Developed by <b>ABHAY MISHRA</b></p>
        </div>
        """, unsafe_allow_html=True)
    return page, active_model

# ======================== HEADER / OVERVIEW ========================
def render_dashboard(active_model):
    df = load_data()
    metrics, _, _ = get_model_metrics()
    now = datetime.now()
    model_label = "🌲 Random Forest" if active_model == "random_forest" else "📐 Linear Regression"

    st.markdown(f"""
    <div class="hero-header">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div style="max-width: 75%;">
                <div class="hero-title">Concrete Compressive Strength Prediction using Machine Learning</div>
                <div class="hero-subtitle">AI-powered prediction system for civil engineering applications</div>
            </div>
            <div style="text-align:right; font-size:14px; opacity:0.9;">
                <div>{now.strftime('%B %d, %Y')}</div>
                <div style="margin-top:5px;">🟢 System Online</div>
                <div style="margin-top:5px; font-weight:bold;">{model_label}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔍 What is this project?</div>', unsafe_allow_html=True)
    st.info("This project predicts the compressive strength of concrete using machine learning algorithms based on the material composition and curing age of concrete. It is designed to assist civil engineers and researchers in optimizing concrete mixtures without waiting for 28-day physical tests.")
    
    c1, c2, c3, c4 = st.columns(4)
    acc = metrics.get(active_model, {}).get("accuracy", 0)
    
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-icon">📦</div><div class="metric-value">{len(df):,}</div><div class="metric-label">Dataset Size</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-icon">🧩</div><div class="metric-value">{len(df.columns)-1}</div><div class="metric-label">Number of Features</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-icon">🎯</div><div class="metric-value">{acc*100:.1f}%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-icon">🤖</div><div class="metric-value" style="font-size:1.3rem;">{model_label}</div><div class="metric-label">Active Algorithm</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🌍 Real World Applications</div>', unsafe_allow_html=True)
    
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("""<div class="feature-card"><div class="feature-title">🏗️ Civil Engineering & Structural Design</div>
        <p style="font-size:14px; color:#64748B;">Allows structural engineers to estimate strength rapidly before actual deployment, ensuring safety and compliance.</p></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="feature-card"><div class="feature-title">🧪 Mix Design Optimization</div>
        <p style="font-size:14px; color:#64748B;">Helps material scientists find the perfect balance of ingredients to maximize strength while minimizing costs.</p></div>""", unsafe_allow_html=True)
    with a2:
        st.markdown("""<div class="feature-card"><div class="feature-title">🏢 Construction Industry</div>
        <p style="font-size:14px; color:#64748B;">Accelerates construction timelines by reducing dependency on slow, traditional 28-day crushing tests.</p></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="feature-card"><div class="feature-title">✅ Quality Control</div>
        <p style="font-size:14px; color:#64748B;">Provides real-time validation of concrete batches arriving at construction sites.</p></div>""", unsafe_allow_html=True)
    with a3:
        st.markdown("""<div class="feature-card"><div class="feature-title">💰 Cost Reduction</div>
        <p style="font-size:14px; color:#64748B;">Minimizes material waste by preventing the over-engineering of concrete mixtures with excessive cement.</p></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="feature-card"><div class="feature-title">♻️ Sustainable Construction</div>
        <p style="font-size:14px; color:#64748B;">Encourages the use of eco-friendly replacements like Fly Ash and Slag by modeling their exact impact.</p></div>""", unsafe_allow_html=True)

# ======================== WHAT IS THIS PROJECT? ========================
def render_what_is_this():
    st.markdown('<div class="section-header">❓ What Is This Project?</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="prediction-intro">
        <h3>🏗️ What is Concrete Compressive Strength?</h3>
        <p>
            Concrete compressive strength represents <b>how much load (pressure) concrete can withstand before it cracks or fails</b>.
            It is measured in <b>Megapascals (MPa)</b>. The higher the MPa value, the stronger the concrete.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="prediction-intro">
        <h3>🤖 What Does This AI System Do?</h3>
        <p>
            Instead of waiting <b>28 days</b> for a laboratory compression test, this AI model predicts the expected strength 
            of concrete <b>instantly</b> — just by analyzing the ingredients and curing age of the concrete mixture. 
            Think of it as a "smart calculator" for concrete quality.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🧪 How Is Concrete Tested Traditionally?</div>', unsafe_allow_html=True)
    
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
        <div class="about-step">
            <h4>1️⃣ Mix & Pour</h4>
            <p>Concrete is mixed with cement, water, aggregates, and admixtures, then poured into cylindrical molds.</p>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="about-step">
            <h4>2️⃣ Cure for 28 Days</h4>
            <p>The samples are cured (kept moist) for up to 28 days to allow the chemical hydration process to complete.</p>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class="about-step">
            <h4>3️⃣ Crush & Measure</h4>
            <p>A compression testing machine crushes the sample. The maximum load it bears (in MPa) is the compressive strength.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="prediction-intro">
        <h3>⚡ Why Use Machine Learning Instead?</h3>
        <p>
            Traditional testing is <b>slow (28 days)</b>, <b>expensive</b>, and <b>destructive</b> (the sample is destroyed). 
            With Machine Learning, we can predict the result <b>in seconds</b> based on the mix design, 
            helping engineers optimize concrete recipes faster, cheaper, and without wasting materials.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_how_it_works():
    st.markdown('<div class="section-header">⚙️ How Concrete Strength Prediction Works</div>', unsafe_allow_html=True)
    
    col_w1, col_w2 = st.columns([1, 1.5])
    with col_w1:
        st.markdown("<h4 style='color:#1E3A8A; text-align:center;'>System Workflow</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; margin-top:20px;">
            <div class="workflow-box">✍️ User Inputs</div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-box">🛡️ Data Validation</div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-box">🔄 Feature Processing (StandardScaler)</div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-box">🤖 Machine Learning Model</div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-box">🔮 Strength Prediction</div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-box">📊 Engineering Analysis</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_w2:
        st.markdown("<h4 style='color:#1E3A8A; text-align:center;'>Machine Learning Workflow</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; margin-top:20px;">
            <div class="workflow-box" style="background:#F97316;">📁 Dataset</div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-box" style="background:#F97316;">🧹 Data Cleaning</div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-box" style="background:#F97316;">📈 Exploratory Data Analysis (EDA)</div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-box" style="background:#F97316;">⚙️ Feature Engineering</div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-box" style="background:#F97316;">✂️ Train-Test Split (80/20)</div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-box" style="background:#F97316;">🧠 Train: Linear Regression & Random Forest</div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-box" style="background:#F97316;">📉 Model Evaluation (R², RMSE, MAE)</div>
            <div class="workflow-arrow">↓</div>
            <div class="workflow-box" style="background:#F97316;">💾 Model Saving (Pickle)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<br><div class="section-header">🧪 Feature Explanation Section</div>', unsafe_allow_html=True)
    st.write("Understanding the material inputs used in the concrete mixture:")
    
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown('<div class="feature-card" title="Primary binding material. Directly increases concrete strength."><div class="feature-title">🧱 Cement</div><p style="font-size:13px; color:#64748B;">Primary binding material. Directly increases concrete strength.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-card" title="Improves workability. Reduces water requirement."><div class="feature-title">🧪 Superplasticizer</div><p style="font-size:13px; color:#64748B;">Improves workability. Reduces water requirement.</p></div>', unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="feature-card" title="Cement replacement material. Improves durability."><div class="feature-title">🔩 Blast Furnace Slag</div><p style="font-size:13px; color:#64748B;">Cement replacement material. Improves durability.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-card" title="Provides structural strength."><div class="feature-title">🪨 Coarse Aggregate</div><p style="font-size:13px; color:#64748B;">Provides structural strength.</p></div>', unsafe_allow_html=True)
    with f3:
        st.markdown('<div class="feature-card" title="Enhances workability. Improves long-term strength."><div class="feature-title">🌫️ Fly Ash</div><p style="font-size:13px; color:#64748B;">Enhances workability. Improves long-term strength.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-card" title="Fills void spaces. Improves compactness."><div class="feature-title">🏖️ Fine Aggregate</div><p style="font-size:13px; color:#64748B;">Fills void spaces. Improves compactness.</p></div>', unsafe_allow_html=True)
    with f4:
        st.markdown('<div class="feature-card" title="Required for hydration. Excess water decreases strength."><div class="feature-title">💧 Water</div><p style="font-size:13px; color:#64748B;">Required for hydration. Excess water decreases strength.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="feature-card" title="Longer curing generally increases strength."><div class="feature-title">📅 Age</div><p style="font-size:13px; color:#64748B;">Longer curing generally increases strength.</p></div>', unsafe_allow_html=True)

# ======================== DATA ANALYSIS ========================
def render_data_analysis():
    df = load_data()
    st.markdown('<div class="section-header">📊 Dataset Analysis</div>', unsafe_allow_html=True)
    
    st.write("Overview of the dataset variables, ranges, and statistical properties.")
    st.dataframe(df.describe().style.format("{:.2f}"), use_container_width=True)
    
    st.markdown('<div class="section-header">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
    st.write("Displays the linear relationship between variables. Values closer to 1 or -1 indicate strong relationships.")
    
    fig = px.imshow(df.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
    fig.update_layout(height=600, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-header">📈 Prediction Distribution</div>', unsafe_allow_html=True)
    fig2 = px.histogram(df, x="Compressive Strength", nbins=40, color_discrete_sequence=["#1E3A8A"], template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# ======================== MODEL PERFORMANCE ========================
def render_model_performance(active_model):
    metrics, X_test, y_test = get_model_metrics()
    
    st.markdown('<div class="section-header">⚔️ Model Comparison</div>', unsafe_allow_html=True)
    rf_m = metrics.get("random_forest", {})
    lr_m = metrics.get("linear_reg", {})

    c1, c2 = st.columns(2)
    with c1:
        is_best = rf_m.get("accuracy",0) >= lr_m.get("accuracy",0)
        border = "border:3px solid #10B981;" if is_best else ""
        badge = "<span style='background:#10B981;color:white;padding:3px 10px;border-radius:10px;font-size:12px;'>⭐ Best Model</span>" if is_best else ""
        st.markdown(f'<div class="premium-card" style="{border}"><h3 style="margin:0;color:#1E3A8A;">🌲 Random Forest {badge}</h3><hr><h2 style="color:#1E3A8A;">Accuracy (R²): {rf_m.get("accuracy", 0)*100:.2f}%</h2><h4 style="color:#475569;">RMSE: {rf_m.get("rmse", 0):.2f} MPa</h4><h4 style="color:#475569;">MAE: {rf_m.get("mae", 0):.2f} MPa</h4></div>', unsafe_allow_html=True)
    with c2:
        is_best = lr_m.get("accuracy",0) > rf_m.get("accuracy",0)
        border = "border:3px solid #10B981;" if is_best else ""
        badge = "<span style='background:#10B981;color:white;padding:3px 10px;border-radius:10px;font-size:12px;'>⭐ Best Model</span>" if is_best else ""
        st.markdown(f'<div class="premium-card" style="{border}"><h3 style="margin:0;color:#1E3A8A;">📐 Linear Regression {badge}</h3><hr><h2 style="color:#1E3A8A;">Accuracy (R²): {lr_m.get("accuracy", 0)*100:.2f}%</h2><h4 style="color:#475569;">RMSE: {lr_m.get("rmse", 0):.2f} MPa</h4><h4 style="color:#475569;">MAE: {lr_m.get("mae", 0):.2f} MPa</h4></div>', unsafe_allow_html=True)

    st.markdown('<br><div class="section-header">🎯 Actual vs Predicted Scatter Plot</div>', unsafe_allow_html=True)
    st.write(f"Visualizing the performance of **{active_model.replace('_',' ').title()}**. Points closer to the red dashed line indicate perfectly accurate predictions.")
    y_pred = metrics.get(active_model, {}).get("y_pred", [])
    y_actual = metrics.get(active_model, {}).get("y_test", [])
    if len(y_pred) > 0:
        fig = px.scatter(x=y_actual, y=y_pred, labels={"x": "Actual Strength (MPa)", "y": "Predicted Strength (MPa)"}, template="plotly_white")
        fig.add_shape(type="line", x0=min(y_actual), y0=min(y_actual), x1=max(y_actual), y1=max(y_actual), line=dict(color="Red", dash="dash"))
        fig.update_traces(marker=dict(color="#3B82F6", size=8, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)

# ======================== FEATURE IMPORTANCE ========================
def render_feature_importance():
    st.markdown('<div class="section-header">🎯 Feature Importance</div>', unsafe_allow_html=True)
    st.write("This chart shows which material components have the most significant impact on the final concrete strength according to the Random Forest algorithm.")
    model = load_model("random_forest")
    if model:
        importances = model.feature_importances_
        features = ["Cement", "Blast Furnace Slag", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age"]
        df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance")
        fig = px.bar(df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Blues", template="plotly_white")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Random Forest model not found.")


# ======================== FOOTER ========================
def render_footer():
    st.markdown("""
    <div style="text-align:center; padding:25px; color:#64748B; border-top:1px solid #E2E8F0; margin-top:50px;">
        <p style="font-weight:bold; font-size:16px;">Built with Streamlit + Machine Learning</p>
        <p style="font-size:24px;">
            <a href="#" style="text-decoration:none; margin:0 10px;">🐙</a>
            <a href="#" style="text-decoration:none; margin:0 10px;">💼</a>
        </p>
        <p>Version 2.0.0 | © 2026 Concrete AI Prediction System</p>
        <p>Designed and Developed by <b>ABHAY MISHRA</b></p>
    </div>
    """, unsafe_allow_html=True)


# ======================== UPDATED PREDICTION UI ========================
def render_prediction(active_model):
    st.markdown('<div class="section-header">Concrete Strength Prediction</div>', unsafe_allow_html=True)

    model_label = "Random Forest" if active_model == "random_forest" else "Linear Regression"
    st.markdown(f"""
    <div class="prediction-intro">
        <h3>Add concrete mix values</h3>
        <p>
            Niche diye gaye inputs mein concrete ke ingredients ki quantity add karein.
            Har box ke upar clearly mention hai ki value Cement, Fly Ash, Water, Aggregate ya Age ke liye hai.
            Selected model <b>{model_label}</b> in values ko analyze karke expected compressive strength MPa mein predict karta hai.
        </p>
    </div>
    """, unsafe_allow_html=True)

    def ingredient_input(title, desc, unit, label, min_value, max_value, default, step=1.0, is_int=False):
        st.markdown(f"""
        <div class="input-block">
            <div class="input-title">{title}</div>
            <div class="input-help">{desc}</div>
            <span class="input-unit">{unit}</span>
        """, unsafe_allow_html=True)

        if is_int:
            value = st.number_input(
                label,
                int(min_value),
                int(max_value),
                int(default),
                step=int(step),
                label_visibility="collapsed",
                help=desc,
            )
        else:
            value = st.number_input(
                label,
                float(min_value),
                float(max_value),
                float(default),
                step=float(step),
                label_visibility="collapsed",
                help=desc,
            )

        st.markdown('</div>', unsafe_allow_html=True)
        return value

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cement = ingredient_input(
            "Cement",
            "Main binding material. Cement zyada hone se strength generally improve hoti hai.",
            "kg per m3",
            "Cement value",
            100.0,
            600.0,
            280.0,
            5.0,
        )
        water = ingredient_input(
            "Water",
            "Hydration ke liye required. Excess water concrete strength ko reduce kar sakta hai.",
            "kg per m3",
            "Water value",
            120.0,
            250.0,
            180.0,
            1.0,
        )
    with c2:
        slag = ingredient_input(
            "Blast Furnace Slag",
            "Cement replacement material. Durability aur long-term performance improve karta hai.",
            "kg per m3",
            "Blast furnace slag value",
            0.0,
            360.0,
            70.0,
            5.0,
        )
        superplast = ingredient_input(
            "Superplasticizer",
            "Workability improve karta hai aur water requirement kam karne mein help karta hai.",
            "kg per m3",
            "Superplasticizer value",
            0.0,
            33.0,
            6.0,
            0.5,
        )
    with c3:
        fly_ash = ingredient_input(
            "Fly Ash",
            "Industrial by-product jo workability aur long-term strength ko improve kar sakta hai.",
            "kg per m3",
            "Fly ash value",
            0.0,
            210.0,
            50.0,
            5.0,
        )
        coarse_agg = ingredient_input(
            "Coarse Aggregate",
            "Large aggregate particles jo concrete ko volume aur structural skeleton dete hain.",
            "kg per m3",
            "Coarse aggregate value",
            800.0,
            1150.0,
            970.0,
            5.0,
        )
    with c4:
        fine_agg = ingredient_input(
            "Fine Aggregate",
            "Sand/fine particles jo voids fill karke concrete ko compact banate hain.",
            "kg per m3",
            "Fine aggregate value",
            590.0,
            1000.0,
            775.0,
            5.0,
        )
        age = ingredient_input(
            "Age",
            "Concrete curing age. Time badhne ke saath strength generally increase hoti hai.",
            "days",
            "Age value",
            1,
            365,
            28,
            1,
            is_int=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:12px;">
        <span class="summary-chip">Cement: binder</span>
        <span class="summary-chip">Fly Ash/Slag: replacement materials</span>
        <span class="summary-chip">Water: hydration</span>
        <span class="summary-chip">Aggregates: structure</span>
        <span class="summary-chip">Age: curing time</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict Concrete Strength", use_container_width=True):
        model = load_model(active_model)
        scaler = load_scaler()
        if not model or not scaler:
            st.error("Model/Scaler missing. Run train_model.py first.")
            return

        progress_text = "Running AI Prediction Model..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        my_bar.empty()

        features = pd.DataFrame(
            [[cement, slag, fly_ash, water, superplast, coarse_agg, fine_agg, age]],
            columns=[
                "Cement (component 1)(kg in a m^3 mixture)",
                "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
                "Fly Ash (component 3)(kg in a m^3 mixture)",
                "Water  (component 4)(kg in a m^3 mixture)",
                "Superplasticizer (component 5)(kg in a m^3 mixture)",
                "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
                "Fine Aggregate (component 7)(kg in a m^3 mixture)",
                "Age (day)",
            ],
        )
        features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)
        pred = model.predict(features_scaled)[0]

        category, badge_class, icon = get_strength_category(pred)
        r2 = get_model_metrics()[0].get(active_model, {}).get("accuracy", 0.85)
        conf = get_confidence(pred, r2)

        st.success("Prediction complete!")
        st.markdown("#### Entered mix values")
        st.markdown(f"""
        <div>
            <span class="summary-chip">Cement: {cement:.1f} kg/m3</span>
            <span class="summary-chip">Blast Furnace Slag: {slag:.1f} kg/m3</span>
            <span class="summary-chip">Fly Ash: {fly_ash:.1f} kg/m3</span>
            <span class="summary-chip">Water: {water:.1f} kg/m3</span>
            <span class="summary-chip">Superplasticizer: {superplast:.1f} kg/m3</span>
            <span class="summary-chip">Coarse Aggregate: {coarse_agg:.1f} kg/m3</span>
            <span class="summary-chip">Fine Aggregate: {fine_agg:.1f} kg/m3</span>
            <span class="summary-chip">Age: {age} days</span>
        </div>
        """, unsafe_allow_html=True)

        r1, r2_col = st.columns([1, 1])
        with r1:
            st.markdown(f"""
            <div class="prediction-result">
                <div style="color:#047857; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;">Predicted Compressive Strength</div>
                <div class="prediction-value">{pred:.2f}</div>
                <div class="prediction-unit">Megapascals (MPa)</div>
                <div><span class="strength-badge {badge_class}">{icon} {category}</span></div>
            </div>
            """, unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                title={'text': "Strength Gauge"},
                gauge={
                    'axis': {'range': [0, 90]},
                    'bar': {'color': "#1E3A8A"},
                    'steps': [
                        {'range': [0, 20], 'color': "#FEF3C7"},
                        {'range': [20, 40], 'color': "#DBEAFE"},
                        {'range': [40, 60], 'color': "#D1FAE5"},
                        {'range': [60, 90], 'color': "#EDE9FE"},
                    ],
                },
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with r2_col:
            ratio = cement / max(water, 1)
            if ratio > 1.8:
                ratio_text = "<em>Good:</em> High cement content relative to water typically supports higher strength."
            else:
                ratio_text = "<em>Warning:</em> Low ratio. Excess water can reduce concrete strength."

            if pred >= 60:
                recommendation = "Excellent strength. Suitable for high-rise buildings and heavy load-bearing structures."
            elif pred >= 40:
                recommendation = "Good strength. Suitable for standard residential and commercial construction."
            else:
                recommendation = "Moderate to low strength. Consider reducing water content or increasing cement/curing age."

            st.markdown(f"""
<div class="premium-card">
    <h4 style="color:#1E3A8A; margin-top:0;">AI Insights</h4>
    <hr style="border:0; border-top:1px solid #E2E8F0; margin:10px 0 15px 0;">
    
    <div style="margin-bottom:15px;">
        <strong style="color:var(--text-primary); font-size:0.95rem;">Prediction Confidence:</strong><br>
        <span style="color:var(--text-secondary); font-size:0.9rem;">The model is <strong>{conf}%</strong> confident in this prediction based on historical data patterns.</span>
    </div>
    
    <div style="margin-bottom:15px;">
        <strong style="color:var(--text-primary); font-size:0.95rem;">Material Composition Analysis:</strong><br>
        <span style="color:var(--text-secondary); font-size:0.9rem;">Your Cement-to-Water ratio is <strong>{ratio:.2f}</strong>.<br>{ratio_text}</span>
    </div>
    
    <div>
        <strong style="color:var(--text-primary); font-size:0.95rem;">Engineering Recommendation:</strong><br>
        <span style="color:var(--text-secondary); font-size:0.9rem;">{recommendation}</span>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Export Section</div>', unsafe_allow_html=True)

        c_ex1, c_ex2 = st.columns(2)
        csv_data = generate_report_csv(
            {
                "Cement": cement,
                "Blast Furnace Slag": slag,
                "Fly Ash": fly_ash,
                "Water": water,
                "Superplasticizer": superplast,
                "Coarse Aggregate": coarse_agg,
                "Fine Aggregate": fine_agg,
                "Age": age,
            },
            pred,
            category,
        )
        with c_ex1:
            st.download_button("Export Prediction to CSV", csv_data, "prediction_report.csv", "text/csv", use_container_width=True)
        with c_ex2:
            st.download_button("Export Report to TXT", csv_data, "prediction_report.txt", "text/plain", use_container_width=True)


# ======================== UPDATED ABOUT SECTION ========================
def render_about():
    st.markdown('<div class="section-header">ℹ️ About Project</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); border-radius: 15px; padding: 35px; color: white; margin-bottom: 30px; box-shadow: 0 10px 20px -5px rgba(30, 58, 138, 0.4); position: relative; overflow: hidden;">
        <div style="position: relative; z-index: 2;">
            <h2 style="margin:0 0 15px 0; font-weight:800; font-size: 2.2rem; letter-spacing: -0.5px;">🏗️ Concrete AI Estimator</h2>
            <p style="font-size: 1.15rem; line-height: 1.7; opacity: 0.95; margin: 0; max-width: 90%;">
                This project leverages <b>Machine Learning</b> to predict the compressive strength of concrete instantly. 
                By bypassing the traditional, time-consuming 28-day physical destruction testing, this AI tool empowers civil engineers, researchers, and students to <b>rapidly prototype and optimize concrete mix designs</b> with high accuracy.
            </p>
        </div>
        <div style="position: absolute; right: -20px; bottom: -40px; font-size: 12rem; opacity: 0.1; z-index: 1;">
            ⚙️
        </div>
    </div>
    
    <h3 style="color:#1E3A8A; font-weight:700; margin-bottom:20px;">🔄 How The System Processes Data</h3>
    <div style="display:flex; justify-content:space-between; gap:20px; margin-bottom: 40px; flex-wrap: wrap;">
        <div style="flex:1; background: white; border-top: 5px solid #F97316; padding: 25px 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); text-align:center; transition: transform 0.3s ease;">
            <div style="font-size:2.5rem; margin-bottom:15px;">📥</div>
            <h4 style="color:#0F172A; margin:0 0 10px 0; font-size:1.1rem;">1. Input Layer</h4>
            <p style="font-size:0.9rem; color:#64748B; margin:0; line-height:1.5;">User provides the exact quantities of 7 distinct materials and the concrete curing age.</p>
        </div>
        <div style="flex:1; background: white; border-top: 5px solid #3B82F6; padding: 25px 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); text-align:center; transition: transform 0.3s ease;">
            <div style="font-size:2.5rem; margin-bottom:15px;">⚙️</div>
            <h4 style="color:#0F172A; margin:0 0 10px 0; font-size:1.1rem;">2. Scaling</h4>
            <p style="font-size:0.9rem; color:#64748B; margin:0; line-height:1.5;">Data is passed through a <i>StandardScaler</i> to match the statistical distribution of the training dataset.</p>
        </div>
        <div style="flex:1; background: white; border-top: 5px solid #10B981; padding: 25px 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); text-align:center; transition: transform 0.3s ease;">
            <div style="font-size:2.5rem; margin-bottom:15px;">🧠</div>
            <h4 style="color:#0F172A; margin:0 0 10px 0; font-size:1.1rem;">3. AI Inference</h4>
            <p style="font-size:0.9rem; color:#64748B; margin:0; line-height:1.5;">The active machine learning algorithm analyzes the scaled data to calculate expected strength.</p>
        </div>
        <div style="flex:1; background: white; border-top: 5px solid #8B5CF6; padding: 25px 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); text-align:center; transition: transform 0.3s ease;">
            <div style="font-size:2.5rem; margin-bottom:15px;">📊</div>
            <h4 style="color:#0F172A; margin:0 0 10px 0; font-size:1.1rem;">4. Final Output</h4>
            <p style="font-size:0.9rem; color:#64748B; margin:0; line-height:1.5;">The predicted MPa value is displayed alongside intelligent engineering recommendations.</p>
        </div>
    </div>
    
    <div style="background: white; border-radius: 16px; padding: 35px; border: 1px solid #E2E8F0; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05);">
        <h3 style="color:#1E3A8A; margin-top:0; border-bottom: 2px solid #F1F5F9; padding-bottom: 15px; font-weight:800;">🤖 Models Under The Hood</h3>
    </div>
    """, unsafe_allow_html=True)
    
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("""
        <div style="background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%); padding:25px; border-radius:12px; border-left: 6px solid #3B82F6; box-shadow: inset 0 2px 4px rgba(255,255,255,0.5); height:100%;">
            <div style="display:flex; align-items:center; gap:15px; margin-bottom:15px;">
                <div style="font-size:2.5rem; background:white; padding:10px; border-radius:10px; box-shadow:0 2px 5px rgba(0,0,0,0.05);">📐</div>
                <div>
                    <h4 style="color:#0F172A; margin:0; font-size:1.2rem; font-weight:800;">Linear Regression</h4>
                    <span style="background:#DBEAFE; color:#1E40AF; padding:3px 8px; border-radius:5px; font-size:0.75rem; font-weight:700;">BASELINE MODEL</span>
                </div>
            </div>
            <p style="font-size:0.95rem; color:#475569; margin:0; line-height:1.7;">
                It attempts to draw a best-fit straight line through the multidimensional data to find simple, direct linear relationships between ingredients and the final concrete strength.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        st.markdown("""
        <div style="background: linear-gradient(180deg, #FFF7ED 0%, #FFEDD5 100%); padding:25px; border-radius:12px; border-left: 6px solid #F97316; box-shadow: inset 0 2px 4px rgba(255,255,255,0.5); height:100%;">
            <div style="display:flex; align-items:center; gap:15px; margin-bottom:15px;">
                <div style="font-size:2.5rem; background:white; padding:10px; border-radius:10px; box-shadow:0 2px 5px rgba(0,0,0,0.05);">🌲</div>
                <div>
                    <h4 style="color:#9A3412; margin:0; font-size:1.2rem; font-weight:800;">Random Forest Regressor</h4>
                    <span style="background:#FFEDD5; color:#9A3412; padding:3px 8px; border-radius:5px; font-size:0.75rem; font-weight:700;">HIGH ACCURACY</span>
                </div>
            </div>
            <p style="font-size:0.95rem; color:#78350F; margin:0; line-height:1.7;">
                An advanced ensemble method that creates hundreds of decision trees. By averaging their predictions, it excels at capturing highly complex, non-linear patterns in the mixture.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: white; border-radius: 16px; padding: 35px; border: 1px solid #E2E8F0; border-top: none; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05);">
        <h3 style="color:#1E3A8A; margin-top:0; border-bottom: 2px solid #F1F5F9; padding-bottom: 15px; font-weight:800;">🛠️ Technology Stack</h3>
        <p style="color:#64748B; font-size:0.95rem; margin-bottom: 20px;">This application is entirely built using modern Python frameworks for data science and web deployment.</p>
        <div style="display:flex; flex-wrap:wrap; gap:15px;">
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:10px 20px; border-radius:30px; font-weight:700; color:#0F172A; display:flex; align-items:center; gap:8px;">
                <span style="font-size:1.2rem;">🐍</span> Python 3
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:10px 20px; border-radius:30px; font-weight:700; color:#0F172A; display:flex; align-items:center; gap:8px;">
                <span style="font-size:1.2rem;">👑</span> Streamlit
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:10px 20px; border-radius:30px; font-weight:700; color:#0F172A; display:flex; align-items:center; gap:8px;">
                <span style="font-size:1.2rem;">🐼</span> Pandas
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:10px 20px; border-radius:30px; font-weight:700; color:#0F172A; display:flex; align-items:center; gap:8px;">
                <span style="font-size:1.2rem;">🔢</span> NumPy
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:10px 20px; border-radius:30px; font-weight:700; color:#0F172A; display:flex; align-items:center; gap:8px;">
                <span style="font-size:1.2rem;">🤖</span> Scikit-Learn
            </div>
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:10px 20px; border-radius:30px; font-weight:700; color:#0F172A; display:flex; align-items:center; gap:8px;">
                <span style="font-size:1.2rem;">📊</span> Plotly
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    inject_css()
    page, active_model = render_sidebar()

    if page == "🏠 Dashboard": render_dashboard(active_model)
    elif page == "❓ What Is This Project?": render_what_is_this()
    elif page == "📖 How It Works": render_how_it_works()
    elif page == "🔮 Prediction": render_prediction(active_model)
    elif page == "📊 Dataset Analysis": render_data_analysis()
    elif page == "📈 Model Performance": render_model_performance(active_model)
    elif page == "🎯 Feature Importance": render_feature_importance()
    elif page == "ℹ️ About Project": render_about()
    
    render_footer()

if __name__ == "__main__":
    main()
