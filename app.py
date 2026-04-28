import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
import urllib.request
import os

# ==========================================
# 1. SYSTEM ARCHITECTURE & UTILS [cite: 349-407]
# ==========================================
class AEGIS(nn.Module):
    def __init__(self, window_size=1024):
        super().__init__()
        self.window_size = window_size
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 7, stride=2, padding=3, output_padding=1), nn.BatchNorm1d(1), nn.ReLU()
        )
        self.output_layer = nn.Sequential(nn.Conv1d(1, 1, 1), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)[:, :, :self.window_size]
        return self.output_layer(xhat)

def zscore_normalise(window):
    """P7: Per-window z-score (removes load amplitude bias) [cite: 324-327, 1514]."""
    return (window - window.mean()) / (window.std() + 1e-8)

@st.cache_resource
def load_assets():
    """Load model with CPU mapping for cloud stability [cite: 1523-1541]."""
    try:
        ckpt = torch.load("aegis_model_improved.pt", map_location=torch.device('cpu'))
        model = AEGIS(window_size=ckpt.get('window_size', 1024))
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt['threshold']
    except: return None, 1.142 # Fallback threshold

@st.cache_data
def fetch_data(file_num):
    """One-Click Demo: Fetch directly from Case Western servers [cite: 87, 292-306]."""
    url = f"https://engineering.case.edu/sites/default/files/{file_num}.mat"
    fpath = f"{file_num}.mat"
    if not os.path.exists(fpath):
        urllib.request.urlretrieve(url, fpath)
    mat = loadmat(fpath)
    key = [k for k in mat if "DE_time" in k and not k.startswith("_")][0]
    return mat[key].flatten().astype(np.float32)

# ==========================================
# 2. APP LAYOUT
# ==========================================
st.set_page_config(page_title="AEGIS: 2026 IAS Challenge", layout="wide")
tab1, tab2 = st.tabs(["🚀 One-Click Diagnostic", "📊 Benchmark Comparison"])

# --- TAB 1: ONE-CLICK DIAGNOSTIC ---
with tab1:
    st.header("Predictive Bearing Fault Diagnosis")
    st.sidebar.header("📂 Demo Scenarios")
    
    # Selection for Judges [cite: 88-286]
    demos = {
        "Normal (0 HP)": 97,
        "Inner Race Fault (0.007\")": 105,
        "Outer Race Fault (0.014\")": 197,
        "Ball Fault (0.021\")": 222
    }
    mode = st.sidebar.selectbox("Choose Scenario", list(demos.keys()))
    model, threshold = load_assets()
    
    if st.sidebar.button("Run Diagnostic"):
        signal = fetch_data(demos[mode])
        windows = [zscore_normalise(signal[i*1024:(i+1)*1024]) for i in range(len(signal)//1024)]
        batch = torch.FloatTensor(np.array(windows)).unsqueeze(1)
        
        with torch.no_grad():
            reconstructions = model(batch)
            mse_scores = torch.mean((batch - reconstructions)**2, dim=(1,2)).numpy()
        
        # P3 Metrics [cite: 701, 1505, 1596]
        actual_fpr = np.mean(mse_scores > threshold)
        fa_per_hr = actual_fpr * (12000 / 1024 * 3600)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Status", "ANOMALY" if np.any(mse_scores > threshold) else "NORMAL")
        c2.metric("Mean Score", f"{np.mean(mse_scores):.6f}")
        c3.metric("Est. False Alarms/Hr", f"{fa_per_hr:,.0f}")

        # P6 Visualization [cite: 1307-1333, 1712]
        st.divider()
        st.subheader("🔬 Saliency & Frequency Analysis (P6)")
        idx = st.slider("Inspect Window", 0, len(windows)-1, 0)
        resid = np.abs(windows[idx] - reconstructions[idx].squeeze().numpy())
        fft_res = np.abs(np.fft.rfft(resid, n=2048))
        freqs = np.fft.rfftfreq(2048, d=1.0/12000)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        ax1.fill_between(range(1024), resid, color="#e67e22", label="Saliency")
        ax1.legend(); ax2.plot(freqs, fft_res, color="#e74c3c"); ax2.set_xlim(0, 6000)
        st.pyplot(fig)

# --- TAB 2: BENCHMARK COMPARISON ---
with tab2:
    st.header("Performance Benchmarks (5-Seed Summary)")
    st.write("Comparison of AEGIS against modern and classical baselines [cite: 1647-1663, 1805-1807].")
    
    # Multi-seed results data from notebook [cite: 1661-1663, 1775-1776]
    comparison_data = {
        "Method": ["AEGIS (Ours)", "Deep-SVDD", "LSTM-AE", "FC-AE"],
        "Mean F1": [0.9789, 0.9138, 0.6734, 0.8631], # [cite: 1661-1663, 1805]
        "Std Dev": [0.0421, 0.1021, 0.1197, 0.104]   # [cite: 1661-1663, 1805]
    }
    st.table(comparison_data)

    st.subheader("Statistical Significance (P4)")
    st.markdown("""
    * **AEGIS vs LSTM-AE:** Significant at ${\alpha}=0.05$ ($p=0.0312$, Cliff's delta=1.0)[cite: 1665].
    * **AEGIS vs Deep-SVDD:** Not significant at ${\alpha}=0.05$ ($p=0.3125$). No superiority claimed [cite: 1667-1669, 1821].
    """)
    
    st.divider()
    st.subheader("⚠️ Section VII: Honest Limitations")
    st.info("""
    1. **Effective Sample Size (P2):** Test windows are not independent ($ESS \approx 5-10$) [cite: 1501-1502, 1813].
    2. **False Alarms (P3):** Without post-processing, AEGIS generates ~2,600 FA/hr at threshold[cite: 1505, 1816].
    3. **Ball Fault Gap (P5):** No method (including SVDD) exceeds $F1 > 0.86$ on Ball Faults[cite: 1512, 1823].
    """)