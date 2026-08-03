import os
import sys
import glob
import json
import pickle
import numpy as np
import pandas as pd
import mne

try:
    from .utils import read_tsv_gz, read_eeg_bdf, compute_band_powers, compute_rmssd
except ImportError:
    from utils import read_tsv_gz, read_eeg_bdf, compute_band_powers, compute_rmssd



def _safe_2d(arr, n_cols):
    """Ensure arr is 2-D with exactly n_cols columns; pad/trim or reshape as needed."""
    if arr.ndim == 1:
        arr = np.tile(arr.reshape(-1, 1), (1, n_cols))
    if arr.shape[1] < n_cols:
        pad = np.zeros((arr.shape[0], n_cols - arr.shape[1]))
        arr = np.concatenate([arr, pad], axis=1)
    return arr[:, :n_cols]


def _resolve_existing_path(directory, candidates):
    for candidate in candidates:
        path = os.path.join(directory, candidate)
        if os.path.exists(path):
            return path
    return None


def extract_raw_for_trial(sub, ses, task, exp_dir, exp_id, target_fs=64.0):
    """
    Loads preprocessed derivative signals for one (subject, session, stimulus) tuple,
    resamples to target_fs, standardizes (Z-score), and returns a trial dict.
    """
    sub_id  = f"sub-{sub:02d}"
    ses_id  = f"ses-{ses:02d}"
    task_id = f"task-stim{task:02d}"

    deriv_dir = os.path.join(exp_dir, "derivatives", sub_id, ses_id)

    # ---- file paths --------------------------------------------------------
    eeg_candidates = [
        f"{sub_id}_{ses_id}_{task_id}_desc-eeg.bdf",
        f"{sub_id}_{ses_id}_{task_id}_eeg.bdf",
    ]
    eeg_path = _resolve_existing_path(os.path.join(deriv_dir, "eeg"), eeg_candidates)
    pupil_path       = os.path.join(deriv_dir, "eyetrack", f"{sub_id}_{ses_id}_{task_id}_desc-pupil_eyetrack.tsv.gz")
    gaze_path        = os.path.join(deriv_dir, "eyetrack", f"{sub_id}_{ses_id}_{task_id}_desc-gaze_visualangle_eyetrack.tsv.gz")
    head_path        = os.path.join(deriv_dir, "eyetrack", f"{sub_id}_{ses_id}_{task_id}_desc-head_eyetrack.tsv.gz")
    blinkrate_path   = os.path.join(deriv_dir, "eyetrack", f"{sub_id}_{ses_id}_{task_id}_desc-blinkrate.tsv.gz")
    saccaderate_path = os.path.join(deriv_dir, "eyetrack", f"{sub_id}_{ses_id}_{task_id}_desc-saccaderate.tsv.gz")
    fixationrate_path= os.path.join(deriv_dir, "eyetrack", f"{sub_id}_{ses_id}_{task_id}_desc-fixationrate.tsv.gz")
    hr_path          = os.path.join(deriv_dir, "beh",      f"{sub_id}_{ses_id}_{task_id}_desc-heartrate.tsv.gz")

    # Skip if any critical file is missing
    for p in [eeg_path, pupil_path, gaze_path, head_path,
              blinkrate_path, saccaderate_path, fixationrate_path,
              hr_path]:
        if p is None or not os.path.exists(p):
            return None

    try:
        # ---- EEG -----------------------------------------------------------
        eeg_data, sfreq, ch_names = read_eeg_bdf(eeg_path, target_fs=target_fs)   # [n_ch, eeg_len]
        n_ch = eeg_data.shape[0]
        eeg_len = eeg_data.shape[1]

        # ---- Ocular --------------------------------------------------------
        df_pupil       = read_tsv_gz(pupil_path)
        df_gaze        = read_tsv_gz(gaze_path)
        df_head        = read_tsv_gz(head_path)
        df_blinkrate   = read_tsv_gz(blinkrate_path)
        df_saccaderate = read_tsv_gz(saccaderate_path)
        df_fixationrate= read_tsv_gz(fixationrate_path)

        # ---- Physio --------------------------------------------------------
        df_hr    = read_tsv_gz(hr_path)

        # ---- Resample Tabular Data to match target_fs ----------------------
        step = int(round(128.0 / target_fs))
        if step > 1:
            df_pupil        = df_pupil.iloc[::step]
            df_gaze         = df_gaze.iloc[::step]
            df_head         = df_head.iloc[::step]
            df_blinkrate    = df_blinkrate.iloc[::step]
            df_saccaderate  = df_saccaderate.iloc[::step]
            df_fixationrate = df_fixationrate.iloc[::step]
            df_hr           = df_hr.iloc[::step]

        # ---- Align all signal lengths --------------------------------------
        min_len = min(
            eeg_len,
            len(df_pupil), len(df_gaze), len(df_head),
            len(df_blinkrate), len(df_saccaderate), len(df_fixationrate),
            len(df_hr)
        )

        eeg_data         = eeg_data[:, :min_len]
        pupil_vals       = df_pupil.iloc[:min_len, 0].values.astype(float)
        gaze_vals        = _safe_2d(df_gaze.iloc[:min_len].values.astype(float),  4)   # X, Y, VisAngX, VisAngY
        head_vals        = _safe_2d(df_head.iloc[:min_len].values.astype(float),  3)   # X, Y, Z
        blinkrate_vals   = df_blinkrate.iloc[:min_len, 0].values.astype(float)
        saccaderate_vals = df_saccaderate.iloc[:min_len, 0].values.astype(float)
        fixationrate_vals= df_fixationrate.iloc[:min_len, 0].values.astype(float)
        hr_vals          = df_hr.iloc[:min_len, 0].values.astype(float)

        # ---- Z-Score Normalization per trial to standardize modalities ----
        eeg_mean = np.mean(eeg_data, axis=1, keepdims=True)
        eeg_std  = np.std(eeg_data, axis=1, keepdims=True) + 1e-8
        eeg_data = (eeg_data - eeg_mean) / eeg_std

        pupil_mean = np.mean(pupil_vals)
        pupil_std  = np.std(pupil_vals) + 1e-8
        pupil_vals = (pupil_vals - pupil_mean) / pupil_std

        gaze_mean = np.mean(gaze_vals, axis=0, keepdims=True)
        gaze_std  = np.std(gaze_vals, axis=0, keepdims=True) + 1e-8
        gaze_vals = (gaze_vals - gaze_mean) / gaze_std

        head_mean = np.mean(head_vals, axis=0, keepdims=True)
        head_std  = np.std(head_vals, axis=0, keepdims=True) + 1e-8
        head_vals = (head_vals - head_mean) / head_std

        blinkrate_vals   = (blinkrate_vals - np.mean(blinkrate_vals)) / (np.std(blinkrate_vals) + 1e-8)
        saccaderate_vals = (saccaderate_vals - np.mean(saccaderate_vals)) / (np.std(saccaderate_vals) + 1e-8)
        fixationrate_vals= (fixationrate_vals - np.mean(fixationrate_vals)) / (np.std(fixationrate_vals) + 1e-8)

        hr_vals = (hr_vals - np.mean(hr_vals)) / (np.std(hr_vals) + 1e-8)

        # Stack Ocular signals:
        # pupil(1), gaze(4), head(3), blinkrate(1), saccaderate(1), fixationrate(1) = 11 dimensions
        ocular_data = np.column_stack([
            pupil_vals, gaze_vals, head_vals, blinkrate_vals, saccaderate_vals, fixationrate_vals
        ])

        # Physio is raw heart rate (1 dimension)
        physio_data = hr_vals.reshape(-1, 1)

        # ---- Memory score (attentive session only) ----------------------
        memory_score = 0.0
        if ses == 1:
            score_file = os.path.join(exp_dir, "phenotype", "stimuli_questionnaire_scores.tsv")
            if os.path.exists(score_file):
                try:
                    df_scores = pd.read_csv(score_file, sep='\t')
                    row = df_scores[
                        (df_scores['participant_id'] == sub_id) &
                        (df_scores['stim_no'] == task)
                    ]
                    if not row.empty:
                        row0 = row.iloc[0]
                        mem = None
                        total = None
                        for col in ['memory_score', 'memory', 'score']:
                            if col in row0.index and pd.notna(row0[col]):
                                mem = float(row0[col])
                                break
                        for col in ['total_memory_questions', 'total_questions', 'total']:
                            if col in row0.index and pd.notna(row0[col]):
                                total = float(row0[col])
                                break
                        if mem is not None and total is not None:
                            memory_score = mem / total if total > 0 else 0.0
                        elif mem is not None:
                            memory_score = float(mem)
                except Exception:
                    # Missing or malformed phenotype values should not block training.
                    memory_score = 0.0

        return {
            'subject'     : sub_id,
            'session'     : ses_id,
            'task'        : task_id,
            'experiment'  : exp_id,
            'n_ch'        : n_ch,
            'eeg'         : eeg_data.T,         # Shape: [min_len, n_ch]
            'ocular'      : ocular_data,        # Shape: [min_len, 11]
            'physio'      : physio_data,        # Shape: [min_len, 1]
            'label'       : 0 if ses == 1 else 1,  # 0=Attentive, 1=Distracted
            'memory_score': memory_score,
        }

    except Exception as exc:
        print(f"  [WARN] Skipping {sub_id}_{ses_id}_{task_id}: {exc}")
        return None


def preprocess_experiment(exp_dir, exp_id, cache_file, target_fs=64.0):
    """Iterates over all subjects/sessions/tasks and caches the trial list with raw signals."""
    sub_folders = glob.glob(os.path.join(exp_dir, "sub-*"))
    subjects    = sorted([int(os.path.basename(f).split("-")[1]) for f in sub_folders])
    num_tasks   = 5 if exp_id == 2 else 6
    sessions    = [1, 2]
    trials      = []

    print(f"\n{'='*60}")
    print(f" Preprocessing Experiment {exp_id} (Extracting original/raw features)")
    print(f" Source  : {exp_dir}")
    print(f" Subjects: {subjects}")
    print(f"{'='*60}")

    for sub in subjects:
        for ses in sessions:
            for task in range(1, num_tasks + 1):
                trial = extract_raw_for_trial(sub, ses, task, exp_dir, exp_id, target_fs=target_fs)
                if trial is not None:
                    trials.append(trial)
                    print(f"  [OK] sub-{sub:02d}  ses-{ses:02d}  stim-{task:02d}"
                          f"  |  eeg={trial['eeg'].shape}  ocular={trial['ocular'].shape}  physio={trial['physio'].shape}")

    print(f"\nTotal trials cached: {len(trials)}")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(trials, f)
    print(f"Cache saved → {cache_file}\n")
    return trials


def get_combined_windowed_dataframe(trials, window_size_sec, step_sec=None, target_fs=64.0):
    """
    Slices raw trials (from session 1 and session 2) into windows, combines them,
    and returns a pandas DataFrame.
    """
    if step_sec is None:
        step_sec = window_size_sec / 2.0

    samples_per_window = int(round(window_size_sec * target_fs))
    step_samples = int(round(step_sec * target_fs))

    window_records = []
    for t in trials:
        eeg_seq = t['eeg']      # [N, n_ch]
        oc_seq = t['ocular']    # [N, 11]
        ph_seq = t['physio']    # [N, 1]
        n_samples = eeg_seq.shape[0]

        for start in range(0, n_samples - samples_per_window + 1, step_samples):
            end = start + samples_per_window
            
            # Slice the continuous raw signals for this window
            eeg_win = eeg_seq[start:end]
            oc_win = oc_seq[start:end]
            ph_win = ph_seq[start:end]
            
            window_records.append({
                'subject': t['subject'],
                'session': t['session'],
                'task': t['task'],
                'experiment': t['experiment'],
                'start_time': start / target_fs,
                'label': t['label'],
                'memory_score': t['memory_score'],
                'eeg': eeg_win,
                'ocular': oc_win,
                'physio': ph_win,
            })

    df = pd.DataFrame(window_records)
    return df


if __name__ == "__main__":
    BASE = r"f:\DATA C DRIVE\BBBD experiments"

    preprocess_experiment(
        exp_dir    = os.path.join(BASE, "experiment2"),
        exp_id     = 2,
        cache_file = os.path.join(BASE, "data_cache", "exp2_raw_trials.pkl"),
    )

    preprocess_experiment(
        exp_dir    = os.path.join(BASE, "experiment3"),
        exp_id     = 3,
        cache_file = os.path.join(BASE, "data_cache", "exp3_raw_trials.pkl"),
    )
