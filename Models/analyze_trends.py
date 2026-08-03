import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from src.models import TransformerClassifier
from src.data_preprocessing import get_combined_windowed_dataframe

class SimpleDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x_eeg = row['eeg']
        x_oc = row['ocular']
        x_ph = row['physio']
        x_concat = np.concatenate([x_eeg, x_oc, x_ph], axis=-1)
        return torch.tensor(x_concat, dtype=torch.float32), torch.tensor(row['label'], dtype=torch.float32)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for trend analysis: {device}")
    
    cache_file2 = r"f:\DATA C DRIVE\BBBD experiments\data_cache\exp2_raw_trials.pkl"
    if not os.path.exists(cache_file2):
        raise FileNotFoundError("Exp 2 Cache file not found. Run preprocessing first.")
        
    with open(cache_file2, 'rb') as f:
        trials = pickle.load(f)
        
    results_dir = r"f:\DATA C DRIVE\BBBD experiments\results"
    os.makedirs(results_dir, exist_ok=True)
    
    # We will use the 10s window size and Transformer Early Fusion model for trend analysis
    win_size = 10
    df_windowed = get_combined_windowed_dataframe(trials, window_size_sec=win_size, step_sec=2.0, target_fs=64.0) # 2s step for fine temporal resolution
    
    # Train a single master model on all Experiment 2 data to extract predictions
    input_dim = 76 # Early Fusion dimension (64 EEG + 11 ocular + 1 physio)
    master_model = TransformerClassifier(input_dim=input_dim).to(device)
    
    dataset = SimpleDataset(df_windowed)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    optimizer = torch.optim.AdamW(master_model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print("Training master Transformer model for trend extraction...")
    master_model.train()
    for epoch in range(10):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(master_model(x), y)
            loss.backward()
            optimizer.step()
            
    # Run evaluation to get prediction probabilities
    master_model.eval()
    all_probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = master_model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            
    # Add prediction probabilities to the windowed DataFrame
    df_windowed['attentive_prob'] = 1.0 - np.array(all_probs)
    df_windowed['pred_label'] = (np.array(all_probs) > 0.5).astype(int)
    
    df = df_windowed
    
    # 1. Attention trends over time (0s to 300s)
    # Group by session (ses-01 = Attentive, ses-02 = Distracted) and start_time
    trends = df.groupby(['session', 'start_time'])['attentive_prob'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    for ses, label, color in [('ses-01', 'Attentive Session (Target: Watch normally)', 'blue'), 
                               ('ses-02', 'Distracted Session (Target: Count backwards)', 'red')]:
        ses_data = trends[trends['session'] == ses]
        plt.plot(ses_data['start_time'], ses_data['attentive_prob'], label=label, color=color, linewidth=2)
        
    plt.xlabel('Time into Video (seconds)')
    plt.ylabel('Model Attention Score (Probability of Attentive State)')
    plt.title('Attention Retention Curves during Video Viewing')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "attention_retention_curves.png"))
    plt.close()
    print("Saved attention retention curves plot.")
    
    # 2. Mind Wandering Analysis in the Attentive Session (ses-01)
    # Define mind wandering: consecutive windows classified as Distracted (label=1)
    # We will compute the percentage of windows predicted as "Distracted" for each subject during ses-01
    df_attentive = df[df['session'] == 'ses-01']
    
    subject_wandering = []
    for sub, sub_df in df_attentive.groupby('subject'):
        total_windows = len(sub_df)
        distracted_windows = (sub_df['pred_label'] == 1).sum()
        pct_wandering = (distracted_windows / total_windows) * 100 if total_windows > 0 else 0.0
        
        # Get memory score (constant for a subject/task combo, average across tasks)
        avg_memory_score = sub_df['memory_score'].mean()
        
        subject_wandering.append({
            'subject': sub,
            'pct_wandering': pct_wandering,
            'memory_score': avg_memory_score
        })
        
    df_wandering = pd.DataFrame(subject_wandering)
    
    # Compute correlation
    r_val, p_val = pearsonr(df_wandering['pct_wandering'], df_wandering['memory_score'])
    print(f"Correlation between Mind Wandering (%) and Memory Test Score: r = {r_val:.4f}, p = {p_val:.4f}")
    
    # Save correlation metadata
    with open(os.path.join(results_dir, "mind_wandering_stats.json"), 'w') as f:
        json.dump({
            'pearson_r': r_val,
            'p_value': p_val
        }, f, indent=4)
        
    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df_wandering['pct_wandering'], df_wandering['memory_score'] * 100, color='purple', alpha=0.7, edgecolors='k', s=80)
    
    # Draw trendline
    m, b = np.polyfit(df_wandering['pct_wandering'], df_wandering['memory_score'] * 100, 1)
    plt.plot(df_wandering['pct_wandering'], m*df_wandering['pct_wandering'] + b, color='grey', linestyle='--', label=f'Trendline (r={r_val:.2f})')
    
    plt.xlabel('Mind Wandering Time (% of Session)')
    plt.ylabel('Memory Test Score (%)')
    plt.title('Relationship Between Predicted Mind Wandering and Learning Outcomes')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "mind_wandering_vs_learning.png"))
    plt.close()
    print("Saved mind wandering scatter plot.")

if __name__ == "__main__":
    main()
