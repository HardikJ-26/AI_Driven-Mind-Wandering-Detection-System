"""
evaluate_cross.py — Cross-Experiment Evaluation
─────────────────────────────────────────────────
Train on Experiment 2 → Test on Experiment 3  (and vice-versa)
Common modalities: EEG | Ocular | Physio | Early Fusion | Late Fusion
Window sizes: 4 s, 10 s, 15 s
Results saved: results/metrics_cross.json + confusion matrices + comparison plots
"""

import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data_preprocessing import get_combined_windowed_dataframe
from src.trainer import (MultimodalDataset, build_model,
                          train_epoch, evaluate, save_cm, train_and_eval)
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

BASE        = r"f:\DATA C DRIVE\BBBD experiments"
RESULTS_DIR = os.path.join(BASE, "results")
WINDOW_SIZES = [20]
MODELS       = ['gru', 'lstm']
EPOCHS       = 10
BATCH_SIZE   = 64
OCULAR_DIM   = 11
PHYSIO_DIM   =  1


def full_train_eval(model_name, train_df, test_df, mod, eeg_dim, device):
    """Train on train_df (with 10% validation split), evaluate on test_df."""
    input_dims = {
        'eeg'          : eeg_dim,
        'ocular'       : OCULAR_DIM,
        'physio'       : PHYSIO_DIM,
        'early_fusion' : eeg_dim + OCULAR_DIM + PHYSIO_DIM,
    }
    inp_dim = input_dims[mod]

    # Split 10% of training data for validation
    train_actual_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

    tr_ds = MultimodalDataset(train_actual_df, mod, eeg_dim)
    val_ds = MultimodalDataset(val_df, mod, eeg_dim)
    te_ds = MultimodalDataset(test_df,  mod, eeg_dim)

    preds, probs, tgts, _, _ = train_and_eval(
        model_name, tr_ds, val_ds, te_ds, inp_dim, device, EPOCHS, BATCH_SIZE
    )
    return preds, probs, tgts


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cross-Eval device: {device}")

    cache2 = os.path.join(BASE, "data_cache", "exp2_raw_trials.pkl")
    cache3 = os.path.join(BASE, "data_cache", "exp3_raw_trials.pkl")
    if not os.path.exists(cache2) or not os.path.exists(cache3):
        raise FileNotFoundError("Cache files not found. Run python -m src.data_preprocessing first.")

    with open(cache2, 'rb') as f:
        trials2 = pickle.load(f)
    with open(cache3, 'rb') as f:
        trials3 = pickle.load(f)

    eeg_dim2 = trials2[0]['eeg'].shape[1]
    eeg_dim3 = trials3[0]['eeg'].shape[1]
    # Use the minimum common EEG dim (in case channel count differs)
    eeg_dim  = min(eeg_dim2, eeg_dim3)
    print(f"EEG dims: Exp2={eeg_dim2}, Exp3={eeg_dim3}, using common={eeg_dim}")

    # Trim EEG to common dimension for cross-experiment fairness
    def trim_eeg(trials, dim):
        for t in trials:
            t['eeg'] = t['eeg'][:, :dim]
        return trials

    trials2 = trim_eeg(trials2, eeg_dim)
    trials3 = trim_eeg(trials3, eeg_dim)

    cross_metrics = {}

    for win in WINDOW_SIZES:
        print(f"\n{'─'*60}\n  Cross-Eval Window = {win} s\n{'─'*60}")
        win2 = get_combined_windowed_dataframe(trials2, window_size_sec=win, target_fs=64.0)
        win3 = get_combined_windowed_dataframe(trials3, window_size_sec=win, target_fs=64.0)

        win_key = f"win{win}"
        cross_metrics[win_key] = {}

        for model_name in MODELS:
            cross_metrics[win_key][model_name] = {}

            # Combos: (train_label, test_label, train_data, test_data)
            combos = [
                ("exp2→exp3", win2, win3),
                ("exp3→exp2", win3, win2),
            ]

            for combo_label, train_data, test_data in combos:
                cross_metrics[win_key][model_name][combo_label] = {}

                # ── Single modalities + early fusion ──────────────────────
                for mod in ['eeg', 'ocular', 'physio', 'early_fusion']:
                    print(f"  {combo_label} | {model_name.upper()} | {mod} ...")
                    preds, probs, tgts = full_train_eval(
                        model_name, train_data, test_data, mod, eeg_dim, device)

                    acc = accuracy_score(tgts, preds)
                    f1  = f1_score(tgts, preds, zero_division=0)
                    auc = roc_auc_score(tgts, probs)
                    print(f"    Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

                    cross_metrics[win_key][model_name][combo_label][mod] = {
                        'accuracy': acc, 'f1': f1, 'auc': auc}

                    cm_file = os.path.join(
                        RESULTS_DIR,
                        f"cm_cross_win{win}_{model_name}_{mod}_{combo_label.replace('→','_to_')}.png")
                    save_cm(tgts, preds, cm_file,
                            f"Cross ({combo_label}): {model_name.upper()} | {mod}")

                # ── Late fusion ────────────────────────────────────────────
                print(f"  {combo_label} | {model_name.upper()} | late_fusion ...")
                mod_probs_list = []
                tgts = None
                for mod in ['eeg', 'ocular', 'physio']:
                    _, probs, tgts = full_train_eval(
                        model_name, train_data, test_data, mod, eeg_dim, device)
                    mod_probs_list.append(probs)

                late_probs = np.mean(mod_probs_list, axis=0)
                late_preds = (late_probs > 0.5).astype(float)
                acc = accuracy_score(tgts, late_preds)
                f1  = f1_score(tgts, late_preds, zero_division=0)
                auc = roc_auc_score(tgts, late_probs)
                print(f"    Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

                cross_metrics[win_key][model_name][combo_label]['late_fusion'] = {
                    'accuracy': acc, 'f1': f1, 'auc': auc}

                cm_file = os.path.join(
                    RESULTS_DIR,
                    f"cm_cross_win{win}_{model_name}_late_fusion_{combo_label.replace('→','_to_')}.png")
                save_cm(tgts, late_preds, cm_file,
                        f"Cross ({combo_label}): {model_name.upper()} | late_fusion")

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, "metrics_cross.json")
    with open(out_json, 'w') as f:
        json.dump(cross_metrics, f, indent=4)
    print(f"\nSaved cross metrics → {out_json}")

    # Comparison line plot: Accuracy vs Window for each model & combo
    _comparison_plot(cross_metrics, RESULTS_DIR)


def _comparison_plot(metrics, results_dir):
    """Line chart comparing Transformer vs Mamba across window sizes for cross eval."""
    combos     = ["exp2→exp3", "exp3→exp2"]
    modalities = ['early_fusion', 'late_fusion']
    colors     = {'transformer': '#4C72B0', 'mamba': '#DD8452', 'gru': '#55A868', 'lstm': '#C44E52'}
    styles     = {'early_fusion': '-', 'late_fusion': '--'}

    for combo in combos:
        fig, ax = plt.subplots(figsize=(9, 5))
        for model_name in MODELS:
            for mod in modalities:
                accs = []
                wins = []
                for win in WINDOW_SIZES:
                    key  = f"win{win}"
                    val  = (metrics.get(key, {})
                                   .get(model_name, {})
                                   .get(combo, {})
                                   .get(mod, {})
                                   .get('accuracy', None))
                    if val is not None:
                        accs.append(val)
                        wins.append(win)
                if accs:
                    ax.plot(wins, accs,
                            marker='o',
                            linestyle=styles[mod],
                            color=colors[model_name],
                            label=f"{model_name.capitalize()} | {mod}")

        ax.set_xticks(WINDOW_SIZES)
        ax.set_xlabel("Window Size (s)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Cross-Experiment ({combo}) — Accuracy vs Window Size")
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        out = os.path.join(results_dir,
                           f"cross_comparison_{combo.replace('→','_to_')}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Saved comparison plot → {out}")


if __name__ == "__main__":
    main()
