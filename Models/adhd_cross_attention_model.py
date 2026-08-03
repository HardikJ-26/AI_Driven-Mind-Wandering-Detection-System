import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalityEncoder(nn.Module):
    """
    Encodes a single modality's time-series data using 1D CNN for local feature 
    extraction and a Transformer Encoder for long-range temporal dependencies.
    """
    def __init__(self, in_channels, d_model=64, nhead=4, num_layers=2):
        super(ModalityEncoder, self).__init__()
        # Project raw features to d_model space using 1D Convolution
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=d_model, kernel_size=3, padding=1)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, in_channels)
        # Conv1d expects (batch_size, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = F.relu(x)
        x = x.transpose(1, 2) # Back to (batch_size, seq_len, d_model)
        
        # Transformer expects (batch_size, seq_len, d_model) when batch_first=True
        encoded_x = self.transformer(x)
        return encoded_x

class ADHDRegressionModel(nn.Module):
    def __init__(self, neural_dim=64, ocular_dim=4, physio_dim=2, d_model=64):
        super(ADHDRegressionModel, self).__init__()
        
        # 1. Modality-Specific Temporal Encoders
        self.neural_encoder = ModalityEncoder(in_channels=neural_dim, d_model=d_model)
        self.ocular_encoder = ModalityEncoder(in_channels=ocular_dim, d_model=d_model)
        self.physio_encoder = ModalityEncoder(in_channels=physio_dim, d_model=d_model)
        
        # 2. Cross-Attention Fusion
        # We use Neural as the Query, and Ocular/Physiological as Key/Values
        self.cross_attn_neural_ocular = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.cross_attn_neural_physio = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        
        # 3. Regression Head
        # Output will be a combination of the two cross-attentions (concatenated, then pooled)
        self.regression_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # Bounds the output strictly between 0 and 1
        )

    def forward(self, neural_seq, ocular_seq, physio_seq):
        # neural_seq shape: (batch_size, seq_len, neural_dim)
        
        # Stage A: Encode individual modalities
        neural_encoded = self.neural_encoder(neural_seq)
        ocular_encoded = self.ocular_encoder(ocular_seq)
        physio_encoded = self.physio_encoder(physio_seq)
        
        # Stage B: Cross-Attention Fusion
        # Query: Neural, Key/Value: Ocular
        attn_out_ocular, _ = self.cross_attn_neural_ocular(query=neural_encoded, key=ocular_encoded, value=ocular_encoded)
        
        # Query: Neural, Key/Value: Physiological
        attn_out_physio, _ = self.cross_attn_neural_physio(query=neural_encoded, key=physio_encoded, value=physio_encoded)
        
        # Stage C: Feature Aggregation and Regression
        # Concatenate the fused representations along the feature dimension
        # Shape: (batch_size, seq_len, d_model * 2)
        fused_representation = torch.cat((attn_out_ocular, attn_out_physio), dim=-1)
        
        # Global Average Pooling over the temporal dimension (seq_len)
        # Shape: (batch_size, d_model * 2)
        pooled_representation = fused_representation.mean(dim=1)
        
        # Final prediction bounded between 0 and 1
        adhd_score = self.regression_mlp(pooled_representation)
        
        return adhd_score

# ==========================================
# MOCK DATA GENERATOR & VERIFICATION SCRIPT
# ==========================================

def generate_mock_data(batch_size=8, seq_len=120):
    """
    Generates mock random data mimicking the BBBD preprocessed time-series data.
    Assumes a 30-second window sampled at 4Hz = 120 sequence steps.
    """
    print(f"Generating mock data for {batch_size} subjects, sequence length {seq_len}...")
    
    # Neural (EEG): Assume 64 channels
    neural_dim = 64
    neural_data = torch.randn(batch_size, seq_len, neural_dim)
    
    # Ocular (Eye-tracking): Assume 4 channels (x, y, pupil size, blink_rate)
    ocular_dim = 4
    ocular_data = torch.randn(batch_size, seq_len, ocular_dim)
    
    # Physiological (ECG, Respiration): Assume 2 channels (Heart Rate, Breath Rate)
    physio_dim = 2
    physio_data = torch.randn(batch_size, seq_len, physio_dim)
    
    # Random target ADHD degrees between 0 and 1
    target_adhd_scores = torch.rand(batch_size, 1)
    
    return neural_data, ocular_data, physio_data, target_adhd_scores

if __name__ == "__main__":
    print("--- Initializing Cross-Attention Fusion Regression Model ---")
    # Instantiate the model
    model = ADHDRegressionModel(neural_dim=64, ocular_dim=4, physio_dim=2, d_model=64)
    
    # Generate mock datapoints
    neural_seq, ocular_seq, physio_seq, ground_truth = generate_mock_data(batch_size=4, seq_len=120)
    
    print("\n--- Input Tensor Shapes ---")
    print(f"Neural Input:      {neural_seq.shape}")
    print(f"Ocular Input:      {ocular_seq.shape}")
    print(f"Physiological Input: {physio_seq.shape}")
    
    print("\n--- Running Forward Pass ---")
    # Inference
    model.eval() # Set to evaluation mode
    with torch.no_grad():
        predicted_scores = model(neural_seq, ocular_seq, physio_seq)
    
    print("\n--- Verification Results ---")
    for i in range(len(predicted_scores)):
        print(f"Subject {i+1}:")
        print(f"   Ground Truth ADHD Degree: {ground_truth[i].item():.4f}")
        print(f"   Predicted ADHD Degree:    {predicted_scores[i].item():.4f}")
        
    print("\nSUCCESS: Model successfully generated a prediction strictly bounded between 0.0 and 1.0.")
