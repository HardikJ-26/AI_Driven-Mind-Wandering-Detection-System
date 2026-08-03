"""Train and evaluate the multimodal regression models on Exp4 derivatives.

Each sample is a participant/session/stimulus recording. Inputs are built
directly from the derivative files rather than from the 12 summary features:
  * neural: 64 EEG channels from the filtered BDF recording
  * ocular: gaze x/y, pupil diameter, and blink rate
  * physio: heart rate and breath rate
All streams are resampled to SEQUENCE_LENGTH points per recording.
The target is the normalized ADHD score (dichotomous ASRS score / 6.0).
"""

import os
import warnings
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

WINDOW_SECONDS = 10
RECORDING_SECONDS = 300
WINDOWS_PER_RECORDING = RECORDING_SECONDS // WINDOW_SECONDS
TIMESTEPS_PER_WINDOW = 40  # 4 Hz representation within every 10-second window
SEQUENCE_LENGTH = WINDOWS_PER_RECORDING * TIMESTEPS_PER_WINDOW
EPOCHS = int(os.environ.get("EPOCHS", "25"))
BATCH_SIZE = 16
RANDOM_STATE = 42
# Set MAX_SAMPLES for a quick smoke test
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0")) or None
MODEL_TYPE = os.environ.get("MODEL_TYPE", "transformer").lower()


class ModalityEncoder(nn.Module):
    def __init__(self, in_channels, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(F.relu(self.conv(x.transpose(1, 2))).transpose(1, 2))


class ADHDRegressionModel(nn.Module):
    """Transformer-based multimodal regression model."""
    def __init__(self, neural_dim=64, ocular_dim=4, physio_dim=2, d_model=64):
        super().__init__()
        self.neural_encoder = ModalityEncoder(neural_dim, d_model)
        self.ocular_encoder = ModalityEncoder(ocular_dim, d_model)
        self.physio_encoder = ModalityEncoder(physio_dim, d_model)
        self.cross_attn_neural_ocular = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.cross_attn_neural_physio = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.regression_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, neural_seq, ocular_seq, physio_seq):
        neural = self.neural_encoder(neural_seq)
        ocular = self.ocular_encoder(ocular_seq)
        physio = self.physio_encoder(physio_seq)
        ocular_attn, _ = self.cross_attn_neural_ocular(neural, ocular, ocular)
        physio_attn, _ = self.cross_attn_neural_physio(neural, physio, physio)
        return self.regression_mlp(torch.cat((ocular_attn, physio_attn), dim=-1).mean(dim=1))


class ADHDLSTMModel(nn.Module):
    """LSTM-based multimodal regression model."""
    def __init__(self, neural_dim=64, ocular_dim=4, physio_dim=2, hidden_size=64):
        super().__init__()
        self.neural_lstm = nn.LSTM(neural_dim, hidden_size, batch_first=True)
        self.ocular_lstm = nn.LSTM(ocular_dim, hidden_size, batch_first=True)
        self.physio_lstm = nn.LSTM(physio_dim, hidden_size, batch_first=True)
        self.regression_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, neural_seq, ocular_seq, physio_seq):
        neural = self.neural_lstm(neural_seq)[0][:, -1]
        ocular = self.ocular_lstm(ocular_seq)[0][:, -1]
        physio = self.physio_lstm(physio_seq)[0][:, -1]
        return self.regression_mlp(torch.cat((neural, ocular, physio), dim=1))


def resample_signal(values, target_length=SEQUENCE_LENGTH):
    """Linearly resample a 1-D/2-D recording to a common sequence length."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.size == 0:
        raise ValueError("empty signal")
    values = pd.DataFrame(values).interpolate(limit_direction="both").ffill().bfill().to_numpy(np.float32)
    source = np.linspace(0.0, 1.0, values.shape[0])
    target = np.linspace(0.0, 1.0, target_length)
    return np.column_stack([np.interp(target, source, values[:, col]) for col in range(values.shape[1])])


def read_tsv(path, expected_columns=None):
    values = np.loadtxt(path, delimiter="\t", ndmin=2)
    if expected_columns is not None:
        if values.shape[1] < expected_columns:
            raise ValueError(f"expected {expected_columns} columns, found {values.shape[1]}")
        values = values[:, :expected_columns]
    return values


def apply_interpolation_timestamp_mask(values, timestamp_path):
    """Honor eye tracking interpolation timestamp intervals before windowing."""
    if not timestamp_path.exists():
        return values
    intervals = pd.read_csv(timestamp_path, sep="\t")
    if not {"start_time", "end_time"}.issubset(intervals.columns):
        return values
    masked = np.asarray(values, dtype=np.float32).copy()
    sample_times = np.linspace(0.0, RECORDING_SECONDS, len(masked), endpoint=False)
    for interval in intervals[["start_time", "end_time"]].itertuples(index=False):
        masked[(sample_times >= interval.start_time) & (sample_times <= interval.end_time)] = np.nan
    return masked


def derivative_path(root, participant, session, modality, stimulus, description):
    filename = f"{participant}_{session}_task-stim{stimulus:02d}_desc-{description}"
    return root / participant / session / modality / filename


def load_recording(derivatives_root, participant, session, stimulus):
    """Load one recording's raw derivative streams as model-ready sequences."""
    eeg_path = derivative_path(derivatives_root, participant, session, "eeg", stimulus, "eeg.bdf")
    gaze_path = derivative_path(derivatives_root, participant, session, "eyetrack", stimulus, "gaze_visualangle_eyetrack.tsv")
    pupil_path = derivative_path(derivatives_root, participant, session, "eyetrack", stimulus, "pupil_eyetrack.tsv")
    gaze_timestamps = derivative_path(derivatives_root, participant, session, "eyetrack", stimulus, "gaze_interpolation_timestamps.tsv")
    pupil_timestamps = derivative_path(derivatives_root, participant, session, "eyetrack", stimulus, "pupil_interpolation_timestamps.tsv")
    blink_path = derivative_path(derivatives_root, participant, session, "eyetrack", stimulus, "blinkrate.tsv")
    heart_path = derivative_path(derivatives_root, participant, session, "beh", stimulus, "heartrate.tsv")
    breath_path = derivative_path(derivatives_root, participant, session, "beh", stimulus, "breathrate.tsv")
    required = (eeg_path, gaze_path, pupil_path, blink_path, heart_path, breath_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[0])

    raw = mne.io.read_raw_bdf(eeg_path, preload=True, verbose=False)
    neural = resample_signal(raw.get_data().T)
    if neural.shape[1] != 64:
        raise ValueError(f"expected 64 EEG channels, found {neural.shape[1]}")

    gaze = apply_interpolation_timestamp_mask(read_tsv(gaze_path, expected_columns=2), gaze_timestamps)
    pupil = apply_interpolation_timestamp_mask(read_tsv(pupil_path)[:, 0], pupil_timestamps)
    blink = read_tsv(blink_path)[:, 0]
    ocular = np.column_stack((
        resample_signal(gaze)[:, :2],
        resample_signal(pupil)[:, 0],
        resample_signal(blink)[:, 0],
    ))
    physio = np.column_stack((
        resample_signal(read_tsv(heart_path)[:, 0])[:, 0],
        resample_signal(read_tsv(breath_path)[:, 0])[:, 0],
    ))
    return neural, ocular, physio


def load_derivative_dataset(labels, derivatives_root):
    neural, ocular, physio, kept_rows, failures = [], [], [], [], []
    for recording_id, (_, row) in enumerate(labels.iterrows()):
        participant, session = str(row["participant_id"]), str(row["session"])
        stimulus = int(row["stimulus_no"])
        try:
            modalities = load_recording(derivatives_root, participant, session, stimulus)
            # Reshape into 30 non-overlapping 10-second windows.
            neural_windows = modalities[0].reshape(WINDOWS_PER_RECORDING, TIMESTEPS_PER_WINDOW, 64)
            ocular_windows = modalities[1].reshape(WINDOWS_PER_RECORDING, TIMESTEPS_PER_WINDOW, 4)
            physio_windows = modalities[2].reshape(WINDOWS_PER_RECORDING, TIMESTEPS_PER_WINDOW, 2)
            for window_index in range(WINDOWS_PER_RECORDING):
                neural.append(neural_windows[window_index])
                ocular.append(ocular_windows[window_index])
                physio.append(physio_windows[window_index])
                window_row = row.copy()
                window_row["recording_id"] = recording_id
                window_row["window_index"] = window_index
                window_row["window_start_seconds"] = window_index * WINDOW_SECONDS
                window_row["window_end_seconds"] = (window_index + 1) * WINDOW_SECONDS
                kept_rows.append(window_row)
        except Exception as error:
            failures.append(f"{participant}/{session}/stim{stimulus:02d}: {error}")

        if MAX_SAMPLES and len(kept_rows) >= MAX_SAMPLES:
            break

    if not kept_rows:
        raise RuntimeError("No complete multi-modal derivative recordings were found.")
    if failures:
        print(f"[WARN] Skipped {len(failures)} incomplete recordings; first: {failures[0]}")
    return (np.stack(neural), np.stack(ocular), np.stack(physio),
            pd.DataFrame(kept_rows).reset_index(drop=True))


def standardize(train, test):
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-7] = 1.0
    train = ((train - mean) / std).astype(np.float32)
    test = ((test - mean) / std).astype(np.float32)
    return train, test, mean, std


def predict(model, neural, ocular, physio):
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(neural), torch.from_numpy(ocular), torch.from_numpy(physio))
    return scores.squeeze(1).cpu().numpy()


# --- Main ---
base_dir = Path(__file__).resolve().parent.parent.parent
derivatives_root = base_dir / "experiment4" / "derivatives"
labels_path = base_dir / "experiment4" / "RESULTS" / "final_extracted_features.csv"
phenotype_path = base_dir / "experiment4" / "phenotype" / "asrs_questionnaire.tsv"
output_dir = base_dir / "experiment2" / "cross_model_regression"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"REGRESSION CROSS-VALIDATION: {MODEL_TYPE.upper()} on Exp4 Derivative Windows")
print("=" * 70)
if not derivatives_root.exists() or not labels_path.exists() or not phenotype_path.exists():
    raise FileNotFoundError("Required data files (derivatives, feature table, or phenotype table) are missing.")

# Load and merge normalized ADHD scores
asrs_df = pd.read_csv(phenotype_path, sep="\t")[["participant_id", "dichotomous_screener_score"]]
asrs_df["adhd_score_norm"] = asrs_df["dichotomous_screener_score"] / 6.0

labels = pd.read_csv(labels_path)
labels = labels.merge(asrs_df, on="participant_id", how="inner")

print(f"\nLoading original derivative data for {len(labels)} recordings...")
neural, ocular, physio, dataset = load_derivative_dataset(labels, derivatives_root)
y = dataset["adhd_score_norm"].to_numpy(dtype=np.float32)

print(f"[OK] Complete recordings: {dataset['recording_id'].nunique()} ({len(dataset)} windows)")
print(f"  Neural: {neural.shape}; Ocular: {ocular.shape}; Physio: {physio.shape}")
print(f"  Unique ADHD Scores distribution:\n{dataset['dichotomous_screener_score'].value_counts().to_string()}")

# Split whole recordings first, stratifying on discrete screener scores to ensure similar target distributions
recordings = dataset.drop_duplicates("recording_id")
train_records, test_records = train_test_split(
    recordings["recording_id"], test_size=0.20, random_state=RANDOM_STATE,
    stratify=recordings["dichotomous_screener_score"],
)
train_idx = np.flatnonzero(dataset["recording_id"].isin(train_records).to_numpy())
test_idx = np.flatnonzero(dataset["recording_id"].isin(test_records).to_numpy())

# Standardize inputs
neural_train, neural_test, neural_mean, neural_std = standardize(neural[train_idx], neural[test_idx])
ocular_train, ocular_test, ocular_mean, ocular_std = standardize(ocular[train_idx], ocular[test_idx])
physio_train, physio_test, physio_mean, physio_std = standardize(physio[train_idx], physio[test_idx])

y_train, y_test = y[train_idx], y[test_idx]

# Model and Optimization
torch.manual_seed(RANDOM_STATE)
if MODEL_TYPE not in {"transformer", "lstm"}:
    raise ValueError("MODEL_TYPE must be 'transformer' or 'lstm'.")

model = ADHDRegressionModel() if MODEL_TYPE == "transformer" else ADHDLSTMModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

best_val_loss = float("inf")
best_state = None

print(f"\nTraining {MODEL_TYPE.upper()} model for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    order = torch.randperm(len(train_idx))
    epoch_loss = 0.0
    for start in range(0, len(train_idx), BATCH_SIZE):
        batch = order[start:start + BATCH_SIZE].numpy()
        target = torch.from_numpy(y_train[batch]).unsqueeze(1)
        optimizer.zero_grad()
        output = model(
            torch.from_numpy(neural_train[batch]),
            torch.from_numpy(ocular_train[batch]),
            torch.from_numpy(physio_train[batch])
        )
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch)
    
    # Validation step
    val_probs = predict(model, neural_test, ocular_test, physio_test)
    val_mse = mean_squared_error(y_test, val_probs)
    val_mae = mean_absolute_error(y_test, val_probs)
    
    print(f"Epoch {epoch + 1:02d}/{EPOCHS}: loss={epoch_loss / len(train_idx):.6f} | Val MSE={val_mse:.6f} | Val MAE={val_mae:.6f}")
    
    if val_mse < best_val_loss:
        best_val_loss = val_mse
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [OK] Best model updated (Val MSE = {best_val_loss:.6f})")

if best_state is not None:
    model.load_state_dict(best_state)

# Evaluate best model on test set
predictions = predict(model, neural_test, ocular_test, physio_test)

metrics = {
    "MSE": float(mean_squared_error(y_test, predictions)),
    "MAE": float(mean_absolute_error(y_test, predictions)),
    "R2-Score": float(r2_score(y_test, predictions)),
}

print("\nTEST SET METRICS:")
for name, value in metrics.items():
    print(f"  {name}: {value:.6f}")

# Save outputs
results = dataset.iloc[test_idx][["participant_id", "session", "stimulus_no", "window_index",
                                 "window_start_seconds", "window_end_seconds", "dichotomous_screener_score"]].copy()
results["Actual"] = y_test
results["Predicted"] = predictions
results["Error"] = np.abs(y_test - predictions)
results.to_csv(output_dir / f"{MODEL_TYPE}_predictions_results.csv", index=False)

pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).to_csv(output_dir / f"{MODEL_TYPE}_performance_metrics.csv", index=False)

# Checkpoint packaging
checkpoint = {
    "model_type": MODEL_TYPE,
    "state_dict": model.state_dict(),
    "normalization": {
        "neural": {"mean": neural_mean, "std": neural_std},
        "ocular": {"mean": ocular_mean, "std": ocular_std},
        "physio": {"mean": physio_mean, "std": physio_std},
    },
    "window_seconds": WINDOW_SECONDS,
    "timesteps_per_window": TIMESTEPS_PER_WINDOW,
    "sampling_rate": 128,
    "modalities": {"neural": 64, "ocular": 4, "physio": 2},
    "metrics": metrics
}

checkpoint_name = f"exp4_{MODEL_TYPE}_regression_checkpoint.pkl"
with open(output_dir / checkpoint_name, "wb") as f:
    pickle.dump(checkpoint, f)

print(f"\nSaved checkpoint to: {output_dir / checkpoint_name}")

# Generate Plots
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Plot 1: Scatter plot
axes[0].scatter(y_test, predictions, alpha=0.6, color="steelblue", edgecolors="k")
axes[0].plot([0, 1], [0, 1], "--", color="red", linewidth=2)
axes[0].set_xlim(-0.05, 1.05)
axes[0].set_ylim(-0.05, 1.05)
axes[0].set(title="Predicted vs Actual ADHD Score", xlabel="Actual Normalized ADHD Score", ylabel="Predicted ADHD Score")
axes[0].grid(True, linestyle=":", alpha=0.6)

# Plot 2: Metrics Bar plot
axes[1].bar(list(metrics.keys()), list(metrics.values()), color=["dodgerblue", "orange", "lightgreen"], edgecolor="k")
axes[1].set_title("Test Set Metrics Summary")
axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)

# Plot 3: Residuals histogram
residuals = y_test - predictions
axes[2].hist(residuals, bins=15, color="mediumpurple", edgecolor="black", alpha=0.7)
axes[2].axvline(0.0, color="red", linestyle="--", linewidth=2)
axes[2].set(title="Residual Distribution (Actual - Predicted)", xlabel="Residual Error", ylabel="Count")
axes[2].grid(True, linestyle=":", alpha=0.6)

fig.suptitle(f"Experiment 4 Raw-Derivative {MODEL_TYPE.upper()} Regression Evaluation", fontweight="bold", fontsize=14)
fig.tight_layout()
fig.savefig(output_dir / f"{MODEL_TYPE}_regression_robustness_analysis.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[OK] Diagnostic plots saved to: {output_dir / f'{MODEL_TYPE}_regression_robustness_analysis.png'}")
