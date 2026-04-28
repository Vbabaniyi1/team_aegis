import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
from scipy.io import loadmat

# --- [Keep your AEGIS class definition here] ---

@st.cache_resource
def load_assets():
    """Load model with explicit path and device mapping ."""
    model_path = "aegis_model_improved.pt"
    
    # 1. Check if the file actually exists in the root
    if not os.path.exists(model_path):
        return None, 1.142

    try:
        # 2. Force CPU mapping for cloud stability
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        model = AEGIS(window_size=ckpt.get('window_size', 1024))
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt.get('threshold', 1.142)
    except Exception as e:
        # Log the specific error to the Streamlit UI for debugging
        st.sidebar.error(f"Error loading checkpoint: {e}")
        return None, 1.142

# --- [UI Logic] ---
model, threshold = load_assets()

if model is None:
    st.error("❌ **Model Load Failed:** `aegis_model_improved.pt` not found or corrupted.")
    st.info("Ensure the model file is in the root directory of your GitHub repo (not in a subfolder).")
else:
    # --- [One-Click Demo Logic] ---
    if st.sidebar.button("Run Diagnostic"):
        # ... fetch_data logic ...
        
        # Guard against calling NoneType
        batch_np = np.array(windows, dtype=np.float32)
        batch = torch.from_numpy(batch_np).unsqueeze(1)
        
        with torch.no_grad():
            # Line 88 Fix: Only runs if model is valid
            reconstructions = model(batch) 
            mse_scores = torch.mean((batch - reconstructions)**2, dim=(1,2)).numpy()
        
        # ... rest of metrics ...