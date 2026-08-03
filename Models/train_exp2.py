"""
train_exp2.py — Train Transformer & Mamba on Experiment 2
──────────────────────────────────────────────────────────
Window sizes  : 4 s, 10 s, 15 s
Modalities    : EEG | Ocular | Physio | Early Fusion | Late Fusion
Validation    : 5-fold GroupKFold (grouped by subject)
Results saved : results/  (metrics_exp2.json + confusion matrices + bar charts)
"""

import os
import json
import pickle
import numpy as np

from src.data_preprocessing import get_combined_windowed_dataframe
from src.trainer import run_experiment, save_summary_chart

BASE        = r"f:\DATA C DRIVE\BBBD experiments"
CACHE_FILE  = os.path.join(BASE, "data_cache", "exp2_raw_trials.pkl")
RESULTS_DIR = os.path.join(BASE, "results")
WINDOW_SIZES = [20]
MODELS       = ['gru', 'lstm']
EPOCHS       = 10

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(CACHE_FILE):
        raise FileNotFoundError(
            f"Cache not found: {CACHE_FILE}\n"
            "Run  python -m src.data_preprocessing  first."
        )

    with open(CACHE_FILE, 'rb') as f:
        trials = pickle.load(f)

    # Infer EEG feature dimension from the first trial
    eeg_dim = trials[0]['eeg'].shape[1]
    print(f"[Exp 2] EEG dim={eeg_dim}  Total trials={len(trials)}")

    all_metrics = {}

    for win in WINDOW_SIZES:
        print(f"\n{'─'*60}")
        print(f"  Window = {win} s")
        print(f"{'─'*60}")

        df_windowed = get_combined_windowed_dataframe(trials, window_size_sec=win, target_fs=64.0)
        print(f"  Total windows: {len(df_windowed)}")

        metrics = run_experiment(
            exp_label    = f"exp2_win{win}",
            df_windowed  = df_windowed,
            eeg_dim      = eeg_dim,
            models_list  = MODELS,
            results_dir  = RESULTS_DIR,
            epochs       = EPOCHS,
        )
        all_metrics[f"win{win}"] = metrics

    # Save JSON metrics
    out_json = os.path.join(RESULTS_DIR, "metrics_exp2.json")
    with open(out_json, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\nSaved metrics → {out_json}")

    # Summary bar charts
    save_summary_chart(all_metrics, WINDOW_SIZES, "Exp2", RESULTS_DIR)
    print("Saved summary charts.")

if __name__ == "__main__":
    main()
