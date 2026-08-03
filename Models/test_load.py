import os
import gzip
import pandas as pd
import mne

def main():
    print("MNE Version:", mne.__version__)
    
    # Paths for sub-01, ses-01, task-stim01
    sub_id = "sub-01"
    ses_id = "ses-01"
    task_id = "task-stim01"
    deriv_dir = f"f:\\DATA C DRIVE\\BBBD experiments\\experiment2\\derivatives\\{sub_id}\\{ses_id}"
    
    eeg_path = os.path.join(deriv_dir, "eeg", f"{sub_id}_{ses_id}_{task_id}_desc-eeg.bdf")
    pupil_path = os.path.join(deriv_dir, "eyetrack", f"{sub_id}_{ses_id}_{task_id}_desc-pupil_eyetrack.tsv.gz")
    hr_path = os.path.join(deriv_dir, "beh", f"{sub_id}_{ses_id}_{task_id}_desc-heartrate.tsv.gz")
    
    print("\nChecking file existence:")
    print("EEG path exists:", os.path.exists(eeg_path))
    print("Pupil path exists:", os.path.exists(pupil_path))
    print("Heart rate path exists:", os.path.exists(hr_path))
    
    if os.path.exists(eeg_path):
        print("\nLoading EEG BDF file...")
        raw = mne.io.read_raw_bdf(eeg_path, preload=True, verbose=False)
        ch_names = raw.ch_names
        print("Total channels in BDF:", len(ch_names))
        print("First 10 channel names:", ch_names[:10])
        print("Sampling rate (sfreq):", raw.info['sfreq'])
        data, times = raw[:, :]
        print("EEG shape:", data.shape)
        
    if os.path.exists(pupil_path):
        print("\nLoading Pupil TSV file...")
        with gzip.open(pupil_path, 'rt') as f:
            df = pd.read_csv(f, sep='\t')
            print("Pupil DataFrame shape:", df.shape)
            print("Pupil Columns:", df.columns.tolist())
            print("First 5 rows:\n", df.head())
            
    if os.path.exists(hr_path):
        print("\nLoading Heart Rate TSV file...")
        with gzip.open(hr_path, 'rt') as f:
            df = pd.read_csv(f, sep='\t')
            print("Heart Rate DataFrame shape:", df.shape)
            print("Heart Rate Columns:", df.columns.tolist())
            print("First 5 rows:\n", df.head())

if __name__ == "__main__":
    main()
