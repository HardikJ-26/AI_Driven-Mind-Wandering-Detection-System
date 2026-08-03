"""
src/trainer.py
──────────────
Shared training, evaluation, and result-saving utilities for all train_expN scripts.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix)
import pickle
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for saving figures
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = None

try:
    from .models import TransformerClassifier, MambaClassifier, GRUClassifier, LSTMClassifier
except ImportError:
    from models import TransformerClassifier, MambaClassifier, GRUClassifier, LSTMClassifier


# ── Input dimensions ─────────────────────────────────────────────────────────
# These match the original/raw feature shapes.
OCULAR_DIM = 11
PHYSIO_DIM =  1


# ── Dataset ──────────────────────────────────────────────────────────────────
class MultimodalDataset(Dataset):
    def __init__(self, df, modality, eeg_dim):
        self.df = df.reset_index(drop=True)
        self.modality  = modality
        self.eeg_dim   = eeg_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['label']

        if self.modality == 'eeg':
            x = row['eeg'][:, :self.eeg_dim]
        elif self.modality == 'ocular':
            x = row['ocular']
        elif self.modality == 'physio':
            x = row['physio']
        elif self.modality == 'early_fusion':
            eeg_part = row['eeg'][:, :self.eeg_dim]
            x = np.concatenate([eeg_part, row['ocular'], row['physio']], axis=-1)
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

        return (
            torch.tensor(x,            dtype=torch.float32),
            torch.tensor(label,        dtype=torch.float32),
        )


# ── Build model ──────────────────────────────────────────────────────────────
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(model_name, input_dim, device, hyperparams=None):
    hp = hyperparams or {}
    if model_name == 'transformer':
        return TransformerClassifier(input_dim=input_dim).to(device)
    if model_name == 'mamba':
        return MambaClassifier(input_dim=input_dim).to(device)
    if model_name == 'gru':
        return GRUClassifier(
            input_dim=input_dim,
            hidden_dim=hp.get('hidden_dim', 64),
            num_layers=hp.get('num_layers', 2),
            dropout=hp.get('dropout', 0.1),
            bidirectional=hp.get('bidirectional', True),
        ).to(device)
    if model_name == 'lstm':
        return LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=hp.get('hidden_dim', 64),
            num_layers=hp.get('num_layers', 2),
            dropout=hp.get('dropout', 0.1),
        ).to(device)
    raise ValueError(f"Unknown model: {model_name}")


def _progress_iter(iterable, total, desc):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, leave=False)

    for i, item in enumerate(iterable, 1):
        filled = int(30 * i / total) if total else 30
        bar = '#' * filled + '-' * (30 - filled)
        print(f"\r{desc} [{bar}] {i}/{total}", end='', flush=True)
        yield item
    print()


# ── Training loop ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_batches = len(loader)
    for batch_idx, (x, y) in enumerate(_progress_iter(loader, total_batches, '  Training batches')):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    preds, probs, targets = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = torch.sigmoid(model(x))
            preds.extend((p > 0.5).float().cpu().numpy())
            probs.extend(p.cpu().numpy())
            targets.extend(y.numpy())
    return np.array(preds), np.array(probs), np.array(targets)


def _get_pos_weight(labels):
    labels = np.asarray(labels, dtype=np.float32)
    pos = labels.sum()
    neg = len(labels) - pos
    if pos <= 0 or neg <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(neg / pos, dtype=torch.float32)


# ── Train + Evaluate a single model ──────────────────────────────────────────
def train_and_eval(model_name, train_ds, val_ds, test_ds, input_dim, device,
                   epochs=10, batch_size=64, hyperparams=None):
    set_seed(42)
    model = build_model(model_name, input_dim, device, hyperparams=hyperparams)
    batch_size = hyperparams.get('batch_size', batch_size) if hyperparams else batch_size
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, drop_last=False)
    te_loader = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)
    
    lr = hyperparams.get('lr', 1e-3) if hyperparams else 1e-3
    weight_decay = hyperparams.get('weight_decay', 1e-3) if hyperparams else 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    pos_weight = _get_pos_weight(train_ds.df['label'].values)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    best_val_acc = -1.0
    best_model_state = None
    epochs_no_improve = 0
    patience = hyperparams.get('patience', 4) if hyperparams else 4
    
    for epoch in _progress_iter(range(epochs), epochs, 'Epoch'):
        train_epoch(model, tr_loader, optimizer, criterion, device)
        
        # Check validation accuracy to save best model state
        val_preds, _, val_tgts = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_tgts, val_preds)
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_model_state = pickle.loads(pickle.dumps(model.state_dict()))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
            
    # Load the best model state if found
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    preds, probs, tgts = evaluate(model, te_loader, device)
    return preds, probs, tgts, model, best_val_acc


def tune_gru_hyperparams(model_name, train_ds, val_ds, test_ds, input_dim, device,
                         epochs=10, batch_size=64):
    if model_name == 'lstm':
        print("    [LSTM tuning] using compact defaults")
        return train_and_eval(model_name, train_ds, val_ds, test_ds, input_dim, device,
                              epochs=epochs, batch_size=batch_size, hyperparams={
                                  'lr': 1e-3,
                                  'weight_decay': 1e-4,
                                  'hidden_dim': 64,
                                  'num_layers': 2,
                                  'dropout': 0.1,
                                  'batch_size': 64,
                                  'patience': 4,
                              })

    if model_name not in {'gru', 'lstm'}:
        return train_and_eval(model_name, train_ds, val_ds, test_ds, input_dim, device,
                              epochs=epochs, batch_size=batch_size)

    candidates = [
        {'lr': 1e-3, 'weight_decay': 1e-4, 'hidden_dim': 64, 'num_layers': 2,
         'dropout': 0.1, 'bidirectional': True, 'batch_size': 64, 'patience': 4},
        {'lr': 3e-4, 'weight_decay': 1e-4, 'hidden_dim': 128, 'num_layers': 1,
         'dropout': 0.1, 'bidirectional': False, 'batch_size': 32, 'patience': 4},
        {'lr': 1e-4, 'weight_decay': 1e-4, 'hidden_dim': 64, 'num_layers': 1,
         'dropout': 0.0, 'bidirectional': False, 'batch_size': 64, 'patience': 4},
        {'lr': 5e-4, 'weight_decay': 1e-3, 'hidden_dim': 32, 'num_layers': 2,
         'dropout': 0.2, 'bidirectional': True, 'batch_size': 32, 'patience': 4},
    ]

    best_result = None
    for hp in candidates:
        print(f"    [GRU tuning] trying {hp}")
        result = train_and_eval(model_name, train_ds, val_ds, test_ds, input_dim, device,
                                epochs=epochs, batch_size=batch_size, hyperparams=hp)
        preds, probs, tgts, model, val_acc = result
        if best_result is None or val_acc > best_result[4]:
            best_result = (preds, probs, tgts, model, val_acc, hp)

    if best_result is None:
        return train_and_eval(model_name, train_ds, val_ds, test_ds, input_dim, device,
                              epochs=epochs, batch_size=batch_size)

    preds, probs, tgts, model, val_acc, best_hp = best_result
    print(f"    [GRU tuning] best config: {best_hp} | val_acc={val_acc:.4f}")
    return preds, probs, tgts, model, val_acc


# ── Confusion matrix plot ────────────────────────────────────────────────────
def save_cm(targets, preds, filepath, title):
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Attentive', 'Distracted'])
    ax.set_yticklabels(['Attentive', 'Distracted'])
    thresh = cm.max() / 2.0
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha='center', va='center',
                color='white' if val > thresh else 'black')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filepath, dpi=120)
    plt.close(fig)


# ── Core CV/Split loop ───────────────────────────────────────────────────────
def run_experiment(exp_label, df_windowed, eeg_dim, models_list,
                   results_dir, epochs=10, batch_size=64):
    """
    Splits the combined windowed DataFrame into stratified Train (80%), Val (10% of train),
    and Test (20%) sets, then trains and evaluates the selected models.

    Returns a nested metrics dict.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    input_dims = {
        'eeg'          : eeg_dim,
        'ocular'       : OCULAR_DIM,
        'physio'       : PHYSIO_DIM,
        'early_fusion' : eeg_dim + OCULAR_DIM + PHYSIO_DIM,
    }

    def safe_split(df, test_size, random_state=42):
        if len(np.unique(df['label'])) < 2:
            return df.iloc[:int(len(df) * (1 - test_size))].copy(), df.iloc[int(len(df) * (1 - test_size)):].copy()
        try:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])
            return train_df, test_df
        except ValueError:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
            return train_df, test_df

    # Stratified 80-20 train-test split
    train_val_df, test_df = safe_split(df_windowed, test_size=0.2, random_state=42)
    # Stratified split to get 10% validation from training data
    train_df, val_df = safe_split(train_val_df, test_size=0.1 / 0.8, random_state=42)

    print(f"  Split sizes: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
    print(f"  Label distribution: train={train_df['label'].value_counts().to_dict()} val={val_df['label'].value_counts().to_dict()} test={test_df['label'].value_counts().to_dict()}")

    metrics = {}

    for model_name in models_list:
        metrics[model_name] = {}

        # ── single modalities + early fusion ──────────────────────────────
        for mod in ['eeg', 'ocular', 'physio', 'early_fusion']:
            print(f"  [{exp_label}] {model_name.upper()} | {mod} ...")
            
            tr_ds = MultimodalDataset(train_df, mod, eeg_dim)
            val_ds = MultimodalDataset(val_df, mod, eeg_dim)
            te_ds = MultimodalDataset(test_df, mod, eeg_dim)

            if model_name in {'gru', 'lstm'}:
                preds, probs, tgts, _, _ = tune_gru_hyperparams(
                    model_name, tr_ds, val_ds, te_ds, input_dims[mod],
                    device, epochs=epochs, batch_size=batch_size)
            else:
                preds, probs, tgts, _, _ = train_and_eval(
                    model_name, tr_ds, val_ds, te_ds, input_dims[mod],
                    device, epochs, batch_size)

            acc = accuracy_score(tgts, preds)
            f1  = f1_score(tgts, preds, zero_division=0)
            auc = roc_auc_score(tgts, probs)
            print(f"    → Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

            metrics[model_name][mod] = {'accuracy': acc, 'f1': f1, 'auc': auc}
            save_cm(tgts, preds,
                    os.path.join(results_dir, f"cm_{exp_label}_{model_name}_{mod}.png"),
                    f"{exp_label}: {model_name.upper()} | {mod}")

        # ── late fusion ───────────────────────────────────────────────────
        print(f"  [{exp_label}] {model_name.upper()} | late_fusion ...")
        mod_probs_list = []
        tgts = None

        for mod in ['eeg', 'ocular', 'physio']:
            tr_ds = MultimodalDataset(train_df, mod, eeg_dim)
            val_ds = MultimodalDataset(val_df, mod, eeg_dim)
            te_ds = MultimodalDataset(test_df, mod, eeg_dim)
            
            if model_name in {'gru', 'lstm'}:
                _, probs, tgts, _, _ = tune_gru_hyperparams(
                    model_name, tr_ds, val_ds, te_ds, input_dims[mod],
                    device, epochs=epochs, batch_size=batch_size)
            else:
                _, probs, tgts, _, _ = train_and_eval(
                    model_name, tr_ds, val_ds, te_ds, input_dims[mod],
                    device, epochs, batch_size)
            mod_probs_list.append(probs)

        late_probs = np.mean(mod_probs_list, axis=0)
        late_preds = (late_probs > 0.5).astype(float)

        acc = accuracy_score(tgts, late_preds)
        f1  = f1_score(tgts, late_preds, zero_division=0)
        auc = roc_auc_score(tgts, late_probs)
        print(f"    → Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
        metrics[model_name]['late_fusion'] = {'accuracy': acc, 'f1': f1, 'auc': auc}
        save_cm(tgts, late_preds,
                os.path.join(results_dir, f"cm_{exp_label}_{model_name}_late_fusion.png"),
                f"{exp_label}: {model_name.upper()} | late_fusion")

    return metrics


# ── Summary bar chart ─────────────────────────────────────────────────────────
def save_summary_chart(all_metrics, window_sizes, exp_label, results_dir):
    """
    Saves a grouped bar chart comparing Accuracy across all modalities,
    models, and window sizes for one experiment.
    """
    modalities = ['eeg', 'ocular', 'physio', 'early_fusion', 'late_fusion']
    # Infer present model names from all_metrics or fallback to default list
    present_models = set()
    for win_data in all_metrics.values():
        present_models.update(win_data.keys())
    model_names = [m for m in ['transformer', 'mamba', 'gru', 'lstm'] if m in present_models] or ['transformer', 'mamba', 'gru', 'lstm']

    colors = {'transformer': '#4C72B0', 'mamba': '#DD8452', 'gru': '#55A868', 'lstm': '#C44E52'}

    for win in window_sizes:
        key = f"win{win}"
        if key not in all_metrics:
            continue
        wm = all_metrics[key]

        x = np.arange(len(modalities))
        num_models = len(model_names)
        width = 0.8 / max(num_models, 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        for offset, mname in enumerate(model_names):
            accs = [wm.get(mname, {}).get(mod, {}).get('accuracy', 0) for mod in modalities]
            ax.bar(x + offset * width, accs, width, label=mname.upper(),
                   color=colors.get(mname, '#333333'), alpha=0.85)

        ax.set_xticks(x + (width * (num_models - 1)) / 2)

        ax.set_xticklabels(modalities, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{exp_label} — Window {win}s')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"summary_{exp_label}_win{win}.png"), dpi=120)
        plt.close(fig)
