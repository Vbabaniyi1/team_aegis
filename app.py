import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# --- AEGIS Architecture (Re-implemented from Page 5) ---
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

def zscore_normalise(window):
    """Per-window z-score as justified by P7 ablation[cite: 1218]."""
    return (window - window.mean()) / (window.std() + 1e-8)