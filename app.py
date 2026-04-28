import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os

# --- Ensure the Model is checked before use ---
@st.cache_resource
def load_assets():
    """Load model with strict path checking ."""
    model_path = "aegis_model_improved.pt"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found in root directory.")
        return None, 1.142 # Fallback threshold [cite: 1599]

    try:
        # Map location to CPU for cloud stability [cite: 1523]
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Initialize model with window size from checkpoint [cite: 1526-1528]
        model = AEGIS(window_size=ckpt.get('window_size', 1024))
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        return model, ckpt.get('threshold', 1.142)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, 1.142