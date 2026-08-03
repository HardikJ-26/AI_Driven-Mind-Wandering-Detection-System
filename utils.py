import os
import gzip
import numpy as np
import pandas as pd
import mne
from scipy.signal import welch

def read_tsv_gz(filepath):
    """Reads a compressed .tsv.gz file and returns a pandas DataFrame."""
    with gzip.open(filepath, 'rt') as f:
        content = f.read()

    from io import StringIO
    lines = content.strip().split('\n')
    if len(lines) == 0:
        return pd.DataFrame()

    # Try parsing first line as floats — if it succeeds, there is no header
    first_parts = lines[0].strip().split('\t')
    has_header = True
    try:
        [float(p) for p in first_parts if p.strip() != '']
        has_header = False
    except ValueError:
        has_header = True

    df = pd.read_csv(StringIO(content), sep='\t', header=0 if has_header else None)
    return df


def read_eeg_bdf(bdf_path, target_fs=64.0):
    """
    Reads a BioSemi BDF file and returns EEG data as a numpy array [channels, samples]
    and the sampling frequency.
    Only keeps the first 64 EEG channels and excludes known non-EEG channels.
    Optionally resamples the raw object to target_fs.
    """
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    ch_names = raw.ch_names

    # Drop known non-EEG channel types (EXG = external electrodes, Status = trigger channel)
    non_eeg_prefixes = ('EXG', 'GSR', 'Erg', 'Resp', 'Temp', 'Status', 'GS', 'ER', 'RE', 'TE', 'ST')
    eeg_ch_names = [ch for ch in ch_names if not any(ch.startswith(p) for p in non_eeg_prefixes)]

    # Limit to first 64
    eeg_ch_names = eeg_ch_names[:64]
    if len(eeg_ch_names) == 0:
        eeg_ch_names = ch_names[:64]

    raw.pick(eeg_ch_names, verbose=False)
    if target_fs is not None and raw.info['sfreq'] != target_fs:
        raw.resample(target_fs, verbose=False)
    
    data, _ = raw[:, :]
    sfreq = raw.info['sfreq']
    return data, sfreq, eeg_ch_names


def compute_band_powers(eeg_data, sfreq,
                        bands=None):
    """
    Computes PSD and extracts average band power for each channel.
    eeg_data: shape (n_channels, n_samples)
    Returns dict of band_name -> array of shape (n_channels,)
    """
    if bands is None:
        bands = {'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30)}

    n_channels, n_samples = eeg_data.shape
    nperseg = min(n_samples, int(2 * sfreq))  # 2-second windows if possible
    if nperseg < 16:
        return {band: np.zeros(n_channels) for band in bands}

    freqs, psd = welch(eeg_data, fs=sfreq, nperseg=nperseg, axis=-1)

    band_powers = {}
    for band_name, (low, high) in bands.items():
        idx = np.where((freqs >= low) & (freqs <= high))[0]
        if len(idx) == 0:
            band_powers[band_name] = np.zeros(n_channels)
        else:
            band_powers[band_name] = np.mean(psd[:, idx], axis=-1)

    return band_powers


def compute_rmssd(rpeaks, start_time, end_time):
    """
    Computes RMSSD (Root Mean Square of Successive Differences) for R-peak timestamps
    that fall within [start_time, end_time].
    rpeaks: 1-D numpy array of timestamps in seconds.
    """
    window_peaks = rpeaks[(rpeaks >= start_time) & (rpeaks <= end_time)]
    if len(window_peaks) < 3:
        return 0.0
    rr_intervals = np.diff(window_peaks)
    diff_rr = np.diff(rr_intervals)
    return float(np.sqrt(np.mean(diff_rr ** 2)))
