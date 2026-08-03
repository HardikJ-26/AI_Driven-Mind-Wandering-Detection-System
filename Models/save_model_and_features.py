"""
Script to extract and save the trained model weights, biases, and feature names as .pkl files.
This script loads the trained model and feature information and saves them for use in cross-validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import pickle
import warnings
import sys

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# Import or define the model class
class ModalityEncoder(nn.Module):
    """Encodes a single modality's time-series data using 1D CNN and Transformer."""
    def __init__(self, in_channels, d_model=64, nhead=4, num_layers=2):
        super(ModalityEncoder, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = F.relu(x)
        x = x.transpose(1, 2)
        return self.transformer(x)


class ADHDRegressionModel(nn.Module):
    """Cross-Attention Fusion Regression Model for ADHD assessment."""
    def __init__(self, neural_dim=64, ocular_dim=4, physio_dim=2, d_model=64):
        super(ADHDRegressionModel, self).__init__()
        
        self.neural_encoder = ModalityEncoder(in_channels=neural_dim, d_model=d_model)
        self.ocular_encoder = ModalityEncoder(in_channels=ocular_dim, d_model=d_model)
        self.physio_encoder = ModalityEncoder(in_channels=physio_dim, d_model=d_model)
        
        self.cross_attn_neural_ocular = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.cross_attn_neural_physio = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        
        self.regression_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, neural_seq, ocular_seq, physio_seq):
        neural_encoded = self.neural_encoder(neural_seq)
        ocular_encoded = self.ocular_encoder(ocular_seq)
        physio_encoded = self.physio_encoder(physio_seq)
        
        attn_out_ocular, _ = self.cross_attn_neural_ocular(query=neural_encoded, key=ocular_encoded, value=ocular_encoded)
        attn_out_physio, _ = self.cross_attn_neural_physio(query=neural_encoded, key=physio_encoded, value=physio_encoded)
        
        fused_representation = torch.cat((attn_out_ocular, attn_out_physio), dim=-1)
        pooled_representation = fused_representation.mean(dim=1)
        return self.regression_mlp(pooled_representation)


def extract_feature_names():
    """Extract the list of features used in the model."""
    features = [
        'hr_mean', 'hr_std',
        'br_mean', 'br_std',
        'pupil_mean', 'pupil_std',
        'blink_rate_mean', 'blink_rate_std',
        'saccade_rate_mean', 'saccade_rate_std',
        'eeg_gfp_mean', 'eeg_gfp_std'
    ]
    return features


def save_model_and_features():
    """Load trained model and save weights/biases and features as .pkl files."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("Saving Model Weights, Biases, and Features")
    print("=" * 60)
    
    # ========== STEP 1: Initialize and load model weights ==========
    print("\n[1/3] Initializing model...")
    model = ADHDRegressionModel(neural_dim=64, ocular_dim=4, physio_dim=2, d_model=64)
    
    # Try to load pre-trained weights if they exist
    model_weights_path = os.path.join(current_dir, 'webapp', 'model_weights.pth')
    if os.path.exists(model_weights_path):
        print(f"  [OK] Loading model weights from: {model_weights_path}")
        model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
    else:
        print(f"  [WARN] Model weights not found at {model_weights_path}")
        print(f"  --> Saving model with initialized (untrained) weights")
    
    model.eval()
    
    # ========== STEP 2: Extract weights and biases ==========
    print("\n[2/3] Extracting weights and biases...")
    model_data = {
        'model_class': 'ADHDRegressionModel',
        'state_dict': model.state_dict(),
        'model_config': {
            'neural_dim': 64,
            'ocular_dim': 4,
            'physio_dim': 2,
            'd_model': 64
        },
        'architecture_summary': str(model)
    }
    
    # Extract layer-wise weights and biases
    weights_biases = {}
    for name, param in model.named_parameters():
        weights_biases[name] = param.data.cpu().numpy()
    
    model_data['weights_biases'] = weights_biases
    
    print(f"  [OK] Extracted {len(weights_biases)} layer parameters")
    for name in list(weights_biases.keys())[:5]:
        shape = weights_biases[name].shape
        print(f"    - {name}: {shape}")
    if len(weights_biases) > 5:
        print(f"    ... and {len(weights_biases) - 5} more layers")
    
    # ========== STEP 3: Save model as .pkl ==========
    print("\n[3/3] Saving model and features...")
    
    model_pkl_path = os.path.join(current_dir, 'cross_adhd_model.pkl')
    with open(model_pkl_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"  [OK] Model saved to: {model_pkl_path}")
    
    # ========== STEP 4: Extract and save feature names ==========
    features = extract_feature_names()
    
    feature_pipeline_data = {
        'features': features,
        'num_features': len(features),
        'feature_types': {
            'hr_features': ['hr_mean', 'hr_std'],
            'br_features': ['br_mean', 'br_std'],
            'pupil_features': ['pupil_mean', 'pupil_std'],
            'blink_features': ['blink_rate_mean', 'blink_rate_std'],
            'saccade_features': ['saccade_rate_mean', 'saccade_rate_std'],
            'eeg_features': ['eeg_gfp_mean', 'eeg_gfp_std']
        },
        'modalities': {
            'neural': 64,      # EEG feature dimension
            'ocular': 4,       # Eye-tracking feature dimension
            'physio': 2        # Physiological feature dimension
        }
    }
    
    features_pkl_path = os.path.join(current_dir, 'feature_pipeline.pkl')
    with open(features_pkl_path, 'wb') as f:
        pickle.dump(feature_pipeline_data, f)
    print(f"  [OK] Features saved to: {features_pkl_path}")
    
    # ========== VERIFICATION ==========
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Verify model file
    if os.path.exists(model_pkl_path):
        with open(model_pkl_path, 'rb') as f:
            loaded_model = pickle.load(f)
        print(f"\n[OK] Model file verified:")
        print(f"  - File size: {os.path.getsize(model_pkl_path) / 1024:.2f} KB")
        print(f"  - Contains keys: {list(loaded_model.keys())}")
        print(f"  - Number of parameters: {len(loaded_model['weights_biases'])}")
    
    # Verify features file
    if os.path.exists(features_pkl_path):
        with open(features_pkl_path, 'rb') as f:
            loaded_features = pickle.load(f)
        print(f"\n[OK] Features file verified:")
        print(f"  - File size: {os.path.getsize(features_pkl_path) / 1024:.2f} KB")
        print(f"  - Features: {loaded_features['features']}")
        print(f"  - Total features: {loaded_features['num_features']}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: Saved model and features as .pkl files")
    print("=" * 60)
    
    return model_pkl_path, features_pkl_path


if __name__ == "__main__":
    save_model_and_features()
