import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import loadmat

# ==========================================
# 1. AEGIS ARCHITECTURE [cite: 349-407]
# ==========================================
class Conv1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, pool=2, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(pool),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.block(x)

class ConvTranspose1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, output_padding=stride - 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )
    def forward(self, x): return self.block(x)

class AEGIS(nn.Module):
    def __init__(self, window_size=1024):
        super().__init__()
        self.window_size = window_size
        self.encoder = nn.Sequential(
            Conv1DBlock(1, 32, 7, pool=2),
            Conv1DBlock(32, 64, 5, pool=2),
            Conv1DBlock(64, 128, 3, pool=2)
        )
        self.decoder = nn.Sequential(
            ConvTranspose1DBlock(128, 64, 3),
            ConvTranspose1DBlock(64, 32, 5),
            ConvTranspose1DBlock(32, 1, 7)
        )
        self.output_layer = nn.Sequential(nn.Conv1d(1, 1, 1), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)[:, :, :self.window_size]
        return self.output_layer(xhat)

# ==========================================
# 2. UTILITY FUNCTIONS [cite: 324-338]
# ==========================================
def zscore_normalise(window):
    """P7: Per-window z-score normalisation [cite: 324-327]."""
    return (window - window.mean()) / (window.std() + 1e-8)

@st.cache_resource
def load_assets():
    """
    CRITICAL FIX: Explicit CPU mapping for cloud deployment.
    Prevents blank page/silent crash on CPU-only servers.
    """
    try:
        # Force CPU mapping for weights originally trained on CUDA 
        ckpt = torch.load("aegis_model_improved.pt", map_location=torch.device('cpu'))
        model = AEGIS(window_size=ckpt.get('window_size', 1024))
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt['threshold']
    except Exception as e:
        st.error(f"Failed to load aegis_model_improved.pt: {e}")
        return None, None

# ==========================================
# 3. STREAMLIT INTERFACE
# ==========================================
st.set_page_config(page_title="AEGIS Improved PoC", layout="wide")
st.title("AEGIS Improved: Predictive Bearing Fault Diagnosis")
st.caption("2026 IEEE IAS Generative AI Challenge")

# Load assets immediately
model, threshold = load_assets()

# Sidebar: Metadata and Limitations [cite: 1808-1830]
with st.sidebar:
    st.header("📋 System Context")
    if threshold:
        st.write(f"**Diagnostic Threshold:** `{threshold:.6f}`")
    st.write("**Sampling Rate:** 12,000 Hz")
    
    st.divider()
    st.header("⚠️ Section VII: Limitations")
    st.info("""
    - **P1:** Cross-load validity only established within CWRU[cite: 1810].
    - **P2:** Low Effective Sample Size (~5-10); window-level CIs are optimistic[cite: 1813].
    - **P3:** High False Alarm rate (2k+/hr) without post-processing[cite: 1816].
    """)

# Main Workflow
uploaded_file = st.file_uploader("Upload Vibration Data (.mat)", type="mat")

if uploaded_file and model:
    # 3.1 Data Processing [cite: 301-305]
    mat = loadmat(uploaded_file)
    de_key = [k for k in mat if "DE_time" in k and not k.startswith("_")][0]
    signal = mat[de_key].flatten().astype(np.float32)
    
    # Windowing
    win_size = 1024
    num_windows = len(signal) // win_size
    windows = [zscore_normalise(signal[i*win_size:(i+1)*win_size]) for i in range(num_windows)]
    input_batch = torch.FloatTensor(np.array(windows)).unsqueeze(1)
    
    # 3.2 Inference
    with torch.no_grad():
        reconstructions = model(input_batch)
        mse_scores = torch.mean((input_batch - reconstructions)**2, dim=(1,2)).numpy()
    
    # 3.3 Deployment Metrics (P3) [cite: 685-687, 1596]
    st.header("📡 P3: Deployment Metrics")
    actual_fpr = np.mean(mse_scores > threshold)
    fa_per_hr = actual_fpr * (12000 / 1024 * 3600)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Status", "ANOMALY" if np.any(mse_scores > threshold) else "NORMAL")
    col2.metric("Mean MSE", f"{np.mean(mse_scores):.6f}")
    col3.metric("Est. False Alarms/Hr", f"{fa_per_hr:,.0f}")

    # 3.4 Diagnostic Saliency (P6) [cite: 1307-1333, 1712]
    st.divider()
    st.header("🔬 P6: Fault Interpretability")
    idx = st.select_slider("Select Window Index", options=range(num_windows))
    
    raw_win = windows[idx]
    rec_win = reconstructions[idx].squeeze().numpy()
    residual = np.abs(raw_win - rec_win)
    
    # Frequency Spectrum of Residual [cite: 1253-1254, 1719]
    n_fft = 2048
    fft_res = np.abs(np.fft.rfft(residual, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0/12000)

    # Plotting
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    # Time Domain Saliency
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(raw_win, color="#2ecc71", alpha=0.5, label="Raw Signal")
    ax1.fill_between(range(win_size), residual, color="#e67e22", alpha=0.8, label="Saliency (Error)")
    ax1.set_title("Reconstruction Saliency Map")
    ax1.legend()
    
    # Frequency Residual
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(freqs, fft_res, color="#e74c3c")
    ax2.set_title("FFT of Residual (Fault Frequency Analysis)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_xlim(0, 6000)
    
    plt.tight_layout()
    st.pyplot(fig)

elif not model:
    st.warning("Model failed to load. Check the application logs for PyTorch errors.")
else:
    st.info("Upload a .mat file to begin diagnostic assessment.")