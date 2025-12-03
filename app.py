import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import time

# --- 1. PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="NeuroSignal Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a Medical/Clinical look
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f7;
    }
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)


# --- 2. LOAD RESOURCES ---
@st.cache_data
def load_data():
    return pd.read_csv('data/data_clean.csv')


@st.cache_resource
def load_model():
    return joblib.load('seizure_model.pkl')


try:
    df = load_data()
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Critical Error: Run 'train_model.py' first to generate the model!")
    st.stop()


# --- 3. HELPER FUNCTIONS ---
def extract_features(row, fs=178):
    # Time Domain
    mean_val = np.mean(row)
    std_val = np.std(row)
    max_val = np.max(row)
    min_val = np.min(row)
    skew_val = skew(row)
    kurt_val = kurtosis(row)

    # Frequency Domain (The "Neuro" part)
    freqs, psd = welch(row, fs, nperseg=fs)
    delta = np.sum(psd[(freqs >= 0.5) & (freqs < 4)])
    theta = np.sum(psd[(freqs >= 4) & (freqs < 8)])
    alpha = np.sum(psd[(freqs >= 8) & (freqs < 13)])
    beta = np.sum(psd[(freqs >= 13) & (freqs <= 30)])

    return pd.DataFrame([[mean_val, std_val, max_val, min_val, skew_val, kurt_val, delta, theta, alpha, beta]],
                        columns=['mean', 'std', 'max', 'min', 'skew', 'kurtosis', 'delta', 'theta', 'alpha', 'beta'])


# --- 4. SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
st.sidebar.title("NeuroSignal AI")
st.sidebar.markdown("**Patient Data Portal**")

# Smart Selector: Separate Seizure vs Normal IDs for easy demo
st.sidebar.divider()
demo_mode = st.sidebar.radio("Simulation Mode:", ["Random Patient", "Known Seizure Case", "Known Normal Case"])

if demo_mode == "Known Seizure Case":
    # Filter for rows where label is 1 (Seizure)
    # We use original CSV logic (1=Seizure)
    subset = df[df['y'] == 1].index
    sample_index = st.sidebar.selectbox("Select Seizure ID:", subset[:10])  # Show first 10
elif demo_mode == "Known Normal Case":
    subset = df[df['y'] != 1].index
    sample_index = st.sidebar.selectbox("Select Normal ID:", subset[:10])
else:
    sample_index = st.sidebar.number_input("Enter Row ID", min_value=0, max_value=len(df) - 1, value=0)

# Load selected patient data
row_data = df.iloc[sample_index].drop('y').values.astype(float)
actual_label = df.iloc[sample_index]['y']
actual_text = "SEIZURE (Ictal)" if actual_label == 1 else "NORMAL (Inter-ictal)"

# --- 5. MAIN DASHBOARD ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header(f"Patient ID: {sample_index}")
    st.markdown(f"**Clinical Status:** `{actual_text}`")

with col2:
    if st.button("🚀 Run AI Diagnosis", type="primary", use_container_width=True):
        processing = True
    else:
        processing = False

# TABBED INTERFACE
tab1, tab2 = st.tabs(["📉 Live EEG Signal", "🧠 Spectral Analysis"])

with tab1:
    # INTERACTIVE PLOTLY CHART
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=row_data, mode='lines', name='EEG Signal',
                             line=dict(color='#2E86C1', width=2)))
    fig.update_layout(
        title="Raw Electroencephalogram (1 second window)",
        xaxis_title="Time (samples)",
        yaxis_title="Amplitude (uV)",
        template="plotly_white",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Frequency Analysis Visualization
    freqs, psd = welch(row_data, 178, nperseg=178)
    fig_psd = px.line(x=freqs, y=psd, title="Power Spectral Density (Brainwave Frequencies)",
                      labels={'x': 'Frequency (Hz)', 'y': 'Power'})
    # Highlight Alpha/Beta/Delta ranges
    fig_psd.add_vrect(x0=0.5, x1=4, fillcolor="red", opacity=0.1, annotation_text="Delta")
    fig_psd.add_vrect(x0=8, x1=13, fillcolor="green", opacity=0.1, annotation_text="Alpha")
    fig_psd.update_layout(height=350)
    st.plotly_chart(fig_psd, use_container_width=True)

# --- 6. AI ANALYSIS SECTION ---
st.divider()

if processing:
    with st.spinner("Processing Signal... Extracting Biomarkers..."):
        time.sleep(1)  # Fake delay for dramatic effect
        features = extract_features(row_data)
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]  # Probability of Seizure

    # DYNAMIC METRICS ROW
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Delta Power (Sleep/Pathology)", f"{features['delta'][0]:.1f}")
    m2.metric("Beta Power (Active)", f"{features['beta'][0]:.1f}")
    m3.metric("Signal Volatility (Std)", f"{features['std'][0]:.1f}")
    m4.metric("Kurtosis (Spikiness)", f"{features['kurtosis'][0]:.2f}")

    # RESULT COLUMNS
    r1, r2 = st.columns([1, 2])

    with r1:
        # GAUGE CHART
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Seizure Probability"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "red" if prob > 0.5 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "lightcoral"}],
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with r2:
        st.subheader("Diagnostic Report")
        if prediction == 1:
            st.error(f"### 🚨 SEIZURE DETECTED")
            st.write(
                "The model detected abnormal high-amplitude spikes and synchronized firing patterns consistent with an Ictal state.")
            st.markdown(f"**Confidence:** `{prob * 100:.2f}%`")
        else:
            st.success(f"### ✅ Normal Brain Activity")
            st.write("Signal amplitude and frequency distribution are within healthy ranges (Inter-ictal state).")
            st.markdown(f"**Confidence:** `{(1 - prob) * 100:.2f}%`")

else:
    st.info("👆 Click 'Run AI Diagnosis' to analyze this patient's signal.")