import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        # 1D Convolutional front-end to downsample time series and extract local patterns
        self.conv_proj = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            norm_first=True  # More stable for optimization
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 32), # x2 because we concatenate average and max pooling
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1) # Binary classification output (logits)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        # Conv1d expects [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        x = self.conv_proj(x)
        # Transpose back to [batch_size, seq_len_downsampled, d_model]
        x = x.transpose(1, 2)
        
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Combined Global Average and Max Pooling over time
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        x_pool = torch.cat([avg_pool, max_pool], dim=-1)
        
        logits = self.fc(x_pool)
        return logits.squeeze(-1)


class MambaBlock(nn.Module):
    """
    A simplified, pure PyTorch implementation of the selective SSM (Mamba) Block.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # 1D Convolution over sequence
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )

        # SSM parameters
        # A matrix: (d_inner, d_state)
        # We parameterize A as log(A) for stability
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.d_inner + 1, dtype=torch.float32).unsqueeze(1).repeat(1, d_state)))
        
        # Projections to B, C, Delta
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)
        self.dt_proj = nn.Linear(1, self.d_inner)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape
        
        # Project inputs
        projected = self.in_proj(x) # [batch_size, seq_len, d_inner * 2]
        x_inner, gate = projected.chunk(2, dim=-1) # Each is [batch_size, seq_len, d_inner]

        # Conv1d expects [batch_size, channels, seq_len]
        x_conv = x_inner.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len] # Crop padding
        x_conv = x_conv.transpose(1, 2) # [batch_size, seq_len, d_inner]
        
        x_conv = F.silu(x_conv)

        # SSM Projection
        # Project x_conv to B, C, and Delta
        # B: [batch_size, seq_len, d_state]
        # C: [batch_size, seq_len, d_state]
        # Delta: [batch_size, seq_len, d_inner]
        proj_out = self.x_proj(x_conv) # [batch_size, seq_len, d_state * 2 + 1]
        B, C, dt_raw = torch.split(proj_out, [self.d_state, self.d_state, 1], dim=-1)
        
        # Discretize Delta
        # dt_proj maps dt_raw to d_inner, then softplus for positive step size
        delta = F.softplus(self.dt_proj(dt_raw)) # [batch_size, seq_len, d_inner]

        # A matrix
        A = -torch.exp(self.A_log) # [d_inner, d_state] (negative for stability)

        # Discretize A and B
        # dA[b, t, i, s] = exp(delta[b, t, i] * A[i, s])
        # dB[b, t, i, s] = delta[b, t, i] * B[b, t, s]
        
        # Selective Scan Loop
        # h: [batch_size, d_inner, d_state]
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y = torch.zeros(batch_size, seq_len, self.d_inner, device=x.device, dtype=x.dtype)

        # Precompute dA
        # delta: [batch_size, seq_len, d_inner, 1]
        # A: [1, 1, d_inner, d_state]
        delta_unsqueeze = delta.unsqueeze(-1)
        A_unsqueeze = A.unsqueeze(0).unsqueeze(0)
        dA = torch.exp(delta_unsqueeze * A_unsqueeze) # [batch_size, seq_len, d_inner, d_state]

        # Loop through time to compute state updates
        for t in range(seq_len):
            # x_t: [batch_size, d_inner, 1]
            x_t = x_conv[:, t, :].unsqueeze(-1)
            # B_t: [batch_size, 1, d_state]
            B_t = B[:, t, :].unsqueeze(1)
            # dB_t = delta_t * B_t: [batch_size, d_inner, d_state]
            dB_t = delta[:, t, :].unsqueeze(-1) * B_t
            
            # Update state h_t = dA_t * h_{t-1} + dB_t * x_t
            h = dA[:, t, :, :] * h + dB_t * x_t
            
            # Output y_t = C_t * h_t
            # C_t: [batch_size, 1, d_state]
            C_t = C[:, t, :].unsqueeze(1)
            # y_t: [batch_size, d_inner]
            y[:, t, :] = torch.sum(h * C_t, dim=-1)

        # Multiplicative gating
        y = y * F.silu(gate)

        # Output projection
        out = self.out_proj(y) # [batch_size, seq_len, d_model]
        return out

class MambaClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, d_state=16, num_layers=2, dropout=0.1):
        super().__init__()
        self.conv_proj = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state)
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, 32), # x2 for combined avg + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1) # Binary classification output (logits)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = x.transpose(1, 2)
        x = self.conv_proj(x)
        x = x.transpose(1, 2)
        
        for layer in self.layers:
            x = layer(x) + x # Residual connection
            
        # Combined Global Average and Max Pooling over time
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        x_pool = torch.cat([avg_pool, max_pool], dim=-1)
        
        logits = self.fc(x_pool)
        return logits.squeeze(-1)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1, bidirectional=True):
        super().__init__()
        self.conv_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(gru_out_dim * 2, 32),  # x2 for combined avg + max pooling
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Binary classification output (logits)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = x.transpose(1, 2)
        x = self.conv_proj(x)
        x = x.transpose(1, 2)  # [batch_size, downsampled_seq_len, hidden_dim]

        out, _ = self.gru(x)  # out shape: [batch_size, downsampled_seq_len, gru_out_dim]

        # Combined Global Average and Max Pooling over time
        avg_pool = torch.mean(out, dim=1)
        max_pool, _ = torch.max(out, dim=1)
        x_pool = torch.cat([avg_pool, max_pool], dim=-1)

        logits = self.fc(x_pool)
        return logits.squeeze(-1)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.conv_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_proj(x)
        x = x.transpose(1, 2)

        out, _ = self.lstm(x)

        avg_pool = torch.mean(out, dim=1)
        max_pool, _ = torch.max(out, dim=1)
        x_pool = torch.cat([avg_pool, max_pool], dim=-1)

        logits = self.fc(x_pool)
        return logits.squeeze(-1)

