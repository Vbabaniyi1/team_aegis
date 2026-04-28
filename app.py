import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import urllib.request

# ===== AEGIS Model Definition (matching notebook) =====
class Conv1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, pool=2, dropout=0.1):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(pool),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.block(x)

class ConvTranspose1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=2):
        super().__init__()
        pad = kernel // 2
        out_pad = stride - 1
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel,
                               stride=stride, padding=pad,
                               output_padding=out_pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.block(x)

class AEGIS(nn.Module):
    def __init__(self, window_size=1024):
        super().__init__()
        self.window_size = window_size
        self.encoder = nn.Sequential(
            Conv1DBlock(1, 32, 7, pool=2, dropout=0.1),
            Conv1DBlock(32, 64, 5, pool=2, dropout=0.1),
            Conv1DBlock(64, 128, 3, pool=2, dropout=0.1),
        )
        self.decoder = nn.Sequential(
            ConvTranspose1DBlock(128, 64, 3, stride=2),
            ConvTranspose1DBlock(64, 32, 5, stride=2),
            ConvTranspose1DBlock(32, 1, 7, stride=2),
        )
        self.output_layer = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return self.output_layer(xhat)
    def anomaly_score(self, x):
        with torch.no_grad():
            return torch.mean((x - self.forward(x)) ** 2, dim=(1, 2))

# ===== CWRU Dataset Map =====
CWRU_BASE_URL = "https://engineering.case.edu/sites/default/files/"
CWRU_FILE_MAP = {
    "Normal 0HP": (97, "Normal"),
    "Normal 1HP": (98, "Normal"),
    "Normal 2HP": (99, "Normal"),
    "Normal 3HP": (100, "Normal"),
    "IR 0.007\" 0HP": (105, "IR"),
    "IR 0.007\" 1HP": (106, "IR"),
    "IR 0.007\" 2HP": (107, "IR"),
    "IR 0.007\" 3HP": (108, "IR"),
    "IR 0.014\" 0HP": (169, "IR"),
    "IR 0.014\" 1HP": (170, "IR"),
    "IR 0.014\" 2HP": (171, "IR"),
    "IR 0.014\" 3HP": (172, "IR"),
    "IR 0.021\" 0HP": (209, "IR"),
    "IR 0.021\" 1HP": (210, "IR"),
    "IR 0.021\" 2HP": (211, "IR"),
    "IR 0.021\" 3HP": (212, "IR"),
    "OR 0.007\" 0HP": (130, "OR"),
    "OR 0.007\" 1HP": (131, "OR"),
    "OR 0.007\" 2HP": (132, "OR"),
    "OR 0.007\" 3HP": (133, "OR"),
    "OR 0.014\" 0HP": (197, "OR"),
    "OR 0.014\" 1HP": (198, "OR"),
    "OR 0.014\" 2HP": (199, "OR"),
    "OR 0.014\" 3HP": (200, "OR"),
    "OR 0.021\" 0HP": (234, "OR"),
    "OR 0.021\" 1HP": (235, "OR"),
    "OR 0.021\" 2HP": (236, "OR"),
    "OR 0.021\" 3HP": (237, "OR"),
    "BA 0.007\" 0HP": (118, "BA"),
    "BA 0.007\" 1HP": (119, "BA"),
    "BA 0.007\" 2HP": (120, "BA"),
    "BA 0.007\" 3HP": (121, "BA"),
    "BA 0.014\" 0HP": (185, "BA"),
    "BA 0.014\" 1HP": (186, "BA"),
    "BA 0.014\" 2HP": (187, "BA"),
    "BA 0.014\" 3HP": (188, "BA"),
    "BA 0.021\" 0HP": (222, "BA"),
    "BA 0.021\" 1HP": (223, "BA"),
    "BA 0.021\" 2HP": (224, "BA"),
    "BA 0.021\" 3HP": (225, "BA"),
}

# ===== Download & Load .mat file =====
@st.cache_data(show_spinner="Downloading signal from CWRU server...")
def download_and_load(file_num: int) -> np.ndarray:
    fname = f"cwru_{file_num}.mat"
    if not os.path.exists(fname):
        url = f"{CWRU_BASE_URL}{file_num}.mat"
        urllib.request.urlretrieve(url, fname)
    mat = loadmat(fname)
    de_keys = [k for k in mat if "DE_time" in k and not k.startswith("_")]
    if not de_keys:
        raise ValueError(f"No DE_time variable found in {file_num}.mat")
    return mat[de_keys[0]].flatten().astype(np.float64)

# ===== Preprocessing =====
WINDOW_SIZE = 1024

def extract_windows(signal: np.ndarray, window_size=WINDOW_SIZE):
    windows = []
    for start in range(0, len(signal) - window_size + 1, window_size):
        w = signal[start:start + window_size].astype(np.float32)
        mu, sigma = w.mean(), w.std()
        w = (w - mu) / (sigma + 1e-8)
        windows.append(w)
    return np.array(windows, dtype=np.float32) if windows else np.empty((0, window_size), dtype=np.float32)

# ===== Load Model (cached) =====
@st.cache_resource
def load_model(checkpoint_path="aegis_model_improved.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AEGIS(window_size=WINDOW_SIZE).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    threshold = checkpoint["threshold"]
    return model, threshold, device

# ===== Streamlit UI =====
st.set_page_config(page_title="AEGIS Fault Detector – Live CWRU Demo", layout="wide")
st.title("🛠️ AEGIS Bearing Fault Detection (CWRU Dataset)")
st.markdown("Select any **CWRU bearing signal** from the list below. The model will instantly tell you whether it is **Normal** or **Faulty** (no file upload needed).")

col1, col2 = st.columns([2, 1])
with col1:
    scenario = st.selectbox("Choose a test signal", list(CWRU_FILE_MAP.keys()))
    label = CWRU_FILE_MAP[scenario][1]  # "Normal" or fault type
    file_num = CWRU_FILE_MAP[scenario][0]
with col2:
    st.write("")
    st.write("")
    run_button = st.button("🔍 Analyze Signal")

# Option to upload custom .mat as fallback
with st.expander("...or upload your own .mat file"):
    uploaded_file = st.file_uploader("Drag .mat file here", type=["mat"])
    if uploaded_file:
        try:
            mat = loadmat(uploaded_file)
            de_keys = [k for k in mat if "DE_time" in k and not k.startswith("_")]
            if de_keys:
                signal = mat[de_keys[0]].flatten().astype(np.float64)
                run_button = True
                label = "Uploaded signal"
            else:
                st.error("No DE_time signal found.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

if run_button:
    # Load signal (either from CWRU or upload)
    if uploaded_file:
        # already loaded
        pass
    else:
        try:
            signal = download_and_load(file_num)
        except Exception as e:
            st.error(f"Failed to download/load {scenario}: {e}")
            st.stop()

    # Preprocess
    windows = extract_windows(signal)
    if len(windows) == 0:
        st.error("Signal too short for 1024-point windows.")
    else:
        model, threshold, device = load_model()
        t_windows = torch.FloatTensor(windows).unsqueeze(1).to(device)
        scores = model.anomaly_score(t_windows).cpu().numpy()
        mean_score = scores.mean()

        # Determine result
        pred_class = "Fault" if mean_score > threshold else "Normal"
        true_label = label  # from the map (Normal, IR, OR, BA)

        # Display
        c1, c2, c3 = st.columns(3)
        with c1:
            if pred_class == "Fault":
                st.error(f"🚨 **Model says: FAULT**")
            else:
                st.success(f"✅ **Model says: NORMAL**")
        with c2:
            st.metric("Anomaly Score", f"{mean_score:.6f}")
        with c3:
            st.metric("Threshold", f"{threshold:.6f}")
            st.caption(f"True label according to CWRU: **{true_label}**")

        # Plot
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(signal[:min(5000, len(signal))], color='black', linewidth=0.5)
        ax.set_title(f"First 5000 samples – {scenario}")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Amplitude")
        if pred_class == "Fault":
            ax.annotate("FAULT", xy=(0.5, 0.9), xycoords='axes fraction',
                        ha='center', fontsize=20, color='red', alpha=0.3)
        st.pyplot(fig)