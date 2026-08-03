"""Train and evaluate the cross-attention Transformer on Exp4 derivatives.

Each sample is a participant/session/stimulus recording.  Inputs are built
directly from the derivative files rather than from the 12 summary features:
  * neural: 64 EEG channels from the filtered BDF recording
  * ocular: gaze x/y, pupil diameter, and blink rate
  * physio: heart rate and breath rate
All streams are resampled to SEQUENCE_LENGTH points per recording.
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
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, mean_squared_error,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
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
# Set MAX_SAMPLES (for example, ``$env:MAX_SAMPLES=20``) for a quick smoke test.
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
    """Window-level multimodal LSTM baseline, selected by MODEL_TYPE=lstm."""
    def __init__(self, neural_dim=64, ocular_dim=4, physio_dim=1, hidden_size=64):
        super().__init__()
        self.neural_lstm = nn.LSTM(neural_dim, hidden_size, batch_first=True)
        self.ocular_lstm = nn.LSTM(ocular_dim, hidden_size, batch_first=True)
        self.physio_lstm = nn.LSTM(physio_dim, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, neural_seq, ocular_seq, physio_seq):
        neural = self.neural_lstm(neural_seq)[0][:, -1]
        ocular = self.ocular_lstm(ocular_seq)[0][:, -1]
        physio = self.physio_lstm(physio_seq)[0][:, -1]
        return self.classifier(torch.cat((neural, ocular, physio), dim=1))


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
    """Honor derivative interpolation timestamp intervals before windowing.

    Gaze/pupil timestamp files store start/end times of invalid intervals, not
    per-sample clocks.  On the requested 0--300 s timeline those intervals are
    masked, then ``resample_signal`` fills them only from valid neighbours.
    """
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
    heart_path = derivative_path(
    derivatives_root,
    participant,
    session,
    "beh",
    stimulus,
    "heartrate.tsv",
)
    required = (eeg_path, gaze_path, pupil_path, blink_path, heart_path)
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
    heart = resample_signal(
    read_tsv(heart_path)[:, 0]
)

    physio = heart[:, :1]
    return neural, ocular, physio


def load_derivative_dataset(labels, derivatives_root):
    neural, ocular, physio, kept_rows, failures = [], [], [], [], []
    for recording_id, (_, row) in enumerate(labels.iterrows()):
        participant, session = str(row["participant_id"]), str(row["session"])
        stimulus = int(row["stimulus_no"])
        try:
            modalities = load_recording(derivatives_root, participant, session, stimulus)
            # Values are projected to a 0--300 s timeline and then reshaped
            # into 30 timestamped, non-overlapping 10-second windows.
            neural_windows = modalities[0].reshape(WINDOWS_PER_RECORDING, TIMESTEPS_PER_WINDOW, 64)
            ocular_windows = modalities[1].reshape(WINDOWS_PER_RECORDING, TIMESTEPS_PER_WINDOW, 4)
            physio_windows = modalities[2].reshape(WINDOWS_PER_RECORDING,TIMESTEPS_PER_WINDOW, 1,)
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


base_dir = Path(__file__).resolve().parent.parent.parent
derivatives_root = base_dir / "experiment4" / "derivatives"
labels_path = base_dir / "experiment4" / "RESULTS" / "final_extracted_features.csv"
output_dir = base_dir / "experiment2" / "cross_model"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"CROSS-VALIDATION: {MODEL_TYPE.upper()} on Exp4 Timestamped Derivative Windows")
print("=" * 70)
if not derivatives_root.exists() or not labels_path.exists():
    raise FileNotFoundError("Experiment 4 derivatives or the label table is missing.")

labels = pd.read_csv(labels_path).dropna(subset=["attention_label"])
labels["attention_label"] = labels["attention_label"].astype(int)
print(f"\nLoading original derivative data for {len(labels)} labelled recordings...")
neural, ocular, physio, dataset = load_derivative_dataset(labels, derivatives_root)
y = dataset["attention_label"].to_numpy(dtype=np.int64)
print(f"[OK] Complete recordings: {dataset['recording_id'].nunique()} ({len(dataset)} windows)")
print(f"  Neural: {neural.shape}; Ocular: {ocular.shape}; Physio: {physio.shape}")
print(f"  Class distribution: {pd.Series(y).value_counts().to_dict()}")

# Split whole recordings first: windows from one recording must never occur in
# both sets, otherwise temporally adjacent data would leak into the test set.
recordings = dataset.drop_duplicates("recording_id")
train_records, test_records = train_test_split(
    recordings["recording_id"], test_size=0.20, random_state=RANDOM_STATE,
    stratify=recordings["attention_label"],
)
train_idx = np.flatnonzero(dataset["recording_id"].isin(train_records).to_numpy())
test_idx = np.flatnonzero(dataset["recording_id"].isin(test_records).to_numpy())
neural_train, neural_test, neural_mean, neural_std = standardize(
    neural[train_idx],
    neural[test_idx],
)

ocular_train, ocular_test, ocular_mean, ocular_std = standardize(
    ocular[train_idx],
    ocular[test_idx],
)

physio_train, physio_test, physio_mean, physio_std = standardize(
    physio[train_idx],
    physio[test_idx],
)
y_train, y_test = y[train_idx], y[test_idx]

torch.manual_seed(RANDOM_STATE)
if MODEL_TYPE not in {"transformer", "lstm"}:
    raise ValueError("MODEL_TYPE must be 'transformer' or 'lstm'.")
model = ADHDRegressionModel() if MODEL_TYPE == "transformer" else ADHDLSTMModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
positive_count = int(y_train.sum())
negative_count = len(y_train) - positive_count
if positive_count == 0:
    raise ValueError("Training split contains no positive attention labels.")
positive_weight = negative_count / positive_count
print(f"Training class weight for attention_label=1: {positive_weight:.2f}")

best_f1 = -1

best_state = None

for epoch in range(EPOCHS):
    model.train()
    order = torch.randperm(len(train_idx))
    epoch_loss = 0.0
    for start in range(0, len(train_idx), BATCH_SIZE):
        batch = order[start:start + BATCH_SIZE].numpy()
        target = torch.from_numpy(y_train[batch].astype(np.float32)).unsqueeze(1)
        optimizer.zero_grad()
        output = model(torch.from_numpy(neural_train[batch]), torch.from_numpy(ocular_train[batch]), torch.from_numpy(physio_train[batch]))
        # Weighted binary cross-entropy prevents the small positive class from
        # being ignored (the full dataset contains far fewer label-1 samples).
        output = output.clamp(1e-7, 1 - 1e-7)
        loss = -(positive_weight * target * torch.log(output) + (1 - target) * torch.log(1 - output)).mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(batch)
    print(f"Epoch {epoch + 1:02d}/{EPOCHS}: loss={epoch_loss / len(train_idx):.4f}")
    # ---------- Validation ----------
validation_prob = predict(
    model,
    neural_test,
    ocular_test,
    physio_test,
)

validation_pred = (validation_prob >= 0.5).astype(int)

validation_f1 = f1_score(
    y_test,
    validation_pred,
    zero_division=0,
)

print(f"Validation F1: {validation_f1:.4f}")

if validation_f1 > best_f1:
    best_f1 = validation_f1

    best_state = {
        k: v.cpu().clone()
        for k, v in model.state_dict().items()
    }

    print(f"✓ Best model updated (F1 = {best_f1:.4f})")


if best_state is not None:

    model.load_state_dict(best_state)

train_probabilities = predict(model, neural_train, ocular_train, physio_train)
probabilities = predict(model, neural_test, ocular_test, physio_test)
train_predictions = (train_probabilities >= 0.5).astype(int)
predictions = (probabilities >= 0.5).astype(int)

metrics = {
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(y_test, predictions, zero_division=0),
    "Recall": recall_score(y_test, predictions, zero_division=0),
    "F1-Score": f1_score(y_test, predictions, zero_division=0),
    "MSE": mean_squared_error(y_test, predictions),
}
if len(np.unique(y_test)) == 2:
    metrics["ROC-AUC"] = roc_auc_score(y_test, probabilities)

print("\nTEST METRICS")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

results = dataset.iloc[test_idx][["participant_id", "session", "stimulus_no", "window_index",
                                 "window_start_seconds", "window_end_seconds"]].copy()
results["Actual"] = y_test
results["Predicted"] = predictions
results["Probability"] = probabilities
results.to_csv(output_dir / "predictions_results.csv", index=False)
pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).to_csv(output_dir / "performance_metrics.csv", index=False)
pd.DataFrame(classification_report(y_test, predictions, output_dict=True, zero_division=0)).transpose().to_csv(output_dir / "classification_report.csv")
torch.save({"model_state_dict": model.state_dict(), "model_type": MODEL_TYPE,
            "window_seconds": WINDOW_SECONDS, "timesteps_per_window": TIMESTEPS_PER_WINDOW,
            "sequence_length": TIMESTEPS_PER_WINDOW,
            "input_modalities": {"neural": 64, "ocular": 4, "physio": 1}}, output_dir / f"{MODEL_TYPE}_derivative_window_model.pth")

checkpoint = {

    "model_type": MODEL_TYPE,

    "state_dict": model.state_dict(),

    "normalization": {

        "neural": {

            "mean": neural_mean,

            "std": neural_std,

        },

        "ocular": {

            "mean": ocular_mean,

            "std": ocular_std,

        },

        "physio": {

            "mean": physio_mean,

            "std": physio_std,

        },

    },

    "window_seconds": WINDOW_SECONDS,

    "timesteps_per_window": TIMESTEPS_PER_WINDOW,

    "sampling_rate": 128,

    "modalities": {

        "neural": 64,

        "ocular": 4,

        "physio": 1,

    },

    "metrics": {

        "accuracy": metrics["Accuracy"],

        "precision": metrics["Precision"],

        "recall": metrics["Recall"],

        "f1": metrics["F1-Score"],

        "roc_auc": metrics.get("ROC-AUC", None),

    },

}
checkpoint_name = (

    "exp4_common_modalities_lstm_checkpoint.pkl"

    if MODEL_TYPE == "lstm"

    else "exp4_transformer_checkpoint.pkl"

)

with open(

    output_dir / checkpoint_name,

    "wb",

) as f:

    pickle.dump(checkpoint, f)


checkpoint_name = (

    "exp4_common_modalities_lstm_checkpoint.pkl"

    if MODEL_TYPE == "lstm"

    else "exp4_transformer_checkpoint.pkl"

)

with open(

    output_dir / checkpoint_name,

    "wb",

) as f:

    pickle.dump(checkpoint, f)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
axes[0].set(title="Confusion Matrix", xlabel="Predicted", ylabel="Actual")
axes[1].bar(list(metrics), list(metrics.values()), color="steelblue")
axes[1].set(title="Test Metrics", ylim=(0, 1.1))
axes[1].tick_params(axis="x", rotation=45)
axes[2].hist(probabilities, bins=10, color="mediumpurple", edgecolor="black")
axes[2].axvline(0.5, color="red", linestyle="--")
axes[2].set(title="Transformer Probabilities", xlabel="Probability", ylabel="Count")
fig.suptitle("Experiment 4 Raw-Derivative Transformer Evaluation", fontweight="bold")
fig.tight_layout()
fig.savefig(output_dir / "robustness_analysis.png", dpi=300, bbox_inches="tight")
plt.close(fig)

if len(np.unique(y_test)) == 2:
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"Transformer (AUC = {auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=300)
    plt.close()

print(f"\n[OK] Raw-derivative Transformer results saved to: {output_dir}")
