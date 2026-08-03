import os
import pandas as pd
import numpy as np
import mne
import warnings

warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = 'C:/Users/asus/Downloads/exp4'
DERIVATIVES_DIR = os.path.join(BASE_DIR, 'derivatives')
LABELS_PATH = os.path.join(BASE_DIR, 'RESULTS', 'master_encoded_labels.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'RESULTS', 'final_extracted_features.csv')

def extract_tsv_features(file_path, column_name=None):
    """Reads a TSV file and returns mean and std of its primary data column."""
    if not os.path.exists(file_path):
        return np.nan, np.nan
    try:
        df = pd.read_csv(file_path, sep='\t', compression='infer')
        # If the file has no header, pandas assigns columns. We just take the last column.
        # TSV files from this dataset typically have values in the last column.
        if df.shape[1] > 0:
            col_data = df.iloc[:, -1]
            # Convert to numeric, coercing errors
            col_data = pd.to_numeric(col_data, errors='coerce').dropna()
            if len(col_data) > 0:
                return col_data.mean(), col_data.std()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return np.nan, np.nan

def extract_eeg_features(file_path):
    """Reads BDF file and extracts Global Field Power (GFP) mean and std."""
    if not os.path.exists(file_path):
        return np.nan, np.nan
    try:
        # Load raw data without preloading fully into memory to save RAM if possible
        raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)
        # Drop EOG and auxiliary channels if they exist, keeping only EEG
        raw.pick_types(eeg=True, eog=False, stim=False, exclude='bads')
        data = raw.get_data() # Shape: (n_channels, n_times)  — units: Volts
        
        # Convert Volts -> microvolts (standard EEG unit)
        data_uv = data * 1e6
        
        # Global Field Power (GFP): std deviation across all sensors at each time point
        gfp = np.std(data_uv, axis=0)
        return np.mean(gfp), np.std(gfp)
    except Exception as e:
        print(f"Error processing EEG {file_path}: {e}")
    return np.nan, np.nan

def process_dataset():
    extracted_features = []
    
    subjects = [d for d in os.listdir(DERIVATIVES_DIR) if d.startswith('sub-')]
    
    print(f"Found {len(subjects)} subjects to process. This may take a while if processing EEG...")
    
    for sub in sorted(subjects):
        sub_dir = os.path.join(DERIVATIVES_DIR, sub)
        sessions = [s for s in os.listdir(sub_dir) if s.startswith('ses-')]
        
        for ses in sorted(sessions):
            # To find which stimuli exist, we can look at the files in the 'beh' or 'eeg' directory
            beh_dir = os.path.join(sub_dir, ses, 'beh')
            eyetrack_dir = os.path.join(sub_dir, ses, 'eyetrack')
            eeg_dir = os.path.join(sub_dir, ses, 'eeg')
            
            # Find unique stimuli for this subject/session
            stimuli = set()
            for d in [beh_dir, eyetrack_dir, eeg_dir]:
                if os.path.exists(d):
                    for f in os.listdir(d):
                        if 'task-stim' in f:
                            # Extract stimulus number, e.g., 'task-stim04' -> '04' -> 4
                            parts = f.split('_')
                            for p in parts:
                                if p.startswith('task-stim'):
                                    stim_no = int(p.replace('task-stim', ''))
                                    stimuli.add(stim_no)
            
            for stim_no in sorted(list(stimuli)):
                print(f"Processing {sub} | {ses} | Stimulus {stim_no}")
                stim_str = f"stim{stim_no:02d}"
                prefix = f"{sub}_{ses}_task-{stim_str}"
                
                # Extract BEH Features
                hr_mean, hr_std = extract_tsv_features(os.path.join(beh_dir, f"{prefix}_desc-heartrate.tsv"))
                br_mean, br_std = extract_tsv_features(os.path.join(beh_dir, f"{prefix}_desc-breathrate.tsv"))
                
                # Extract Eyetrack Features
                pupil_mean, pupil_std = extract_tsv_features(os.path.join(eyetrack_dir, f"{prefix}_desc-pupil_eyetrack.tsv"))
                blink_mean, blink_std = extract_tsv_features(os.path.join(eyetrack_dir, f"{prefix}_desc-blinkrate.tsv"))
                saccade_mean, saccade_std = extract_tsv_features(os.path.join(eyetrack_dir, f"{prefix}_desc-saccaderate.tsv"))
                
                # Extract EEG Features
                eeg_file = os.path.join(eeg_dir, f"{prefix}_desc-eeg.bdf")
                eeg_gfp_mean, eeg_gfp_std = extract_eeg_features(eeg_file)
                
                extracted_features.append({
                    'participant_id': sub,
                    'session': ses,
                    'stimulus_no': stim_no,
                    'hr_mean': hr_mean, 'hr_std': hr_std,
                    'br_mean': br_mean, 'br_std': br_std,
                    'pupil_mean': pupil_mean, 'pupil_std': pupil_std,
                    'blink_rate_mean': blink_mean, 'blink_rate_std': blink_std,
                    'saccade_rate_mean': saccade_mean, 'saccade_rate_std': saccade_std,
                    'eeg_gfp_mean': eeg_gfp_mean, 'eeg_gfp_std': eeg_gfp_std
                })

    df_features = pd.DataFrame(extracted_features)
    
    print("Feature extraction complete. Merging with labels...")
    
    if os.path.exists(LABELS_PATH):
        df_labels = pd.read_csv(LABELS_PATH)
        
        # We need to map attention_label (which is constant per subject) to all rows
        # We need to map learning_label (which depends on stimulus) only to matching stimulus
        
        # Extract subject-level attention labels
        attention_map = df_labels.drop_duplicates('participant_id')[['participant_id', 'attention_label']]
        
        # Extract stimulus-level learning labels
        learning_map = df_labels[['participant_id', 'stimulus_no', 'learning_label']]
        
        # Merge attention
        df_final = df_features.merge(attention_map, on='participant_id', how='left')
        
        # Merge learning
        df_final = df_final.merge(learning_map, on=['participant_id', 'stimulus_no'], how='left')
    else:
        print(f"Warning: Labels file not found at {LABELS_PATH}. Outputting features only.")
        df_final = df_features
        
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully saved final feature dataset to: {OUTPUT_PATH}")
    print(df_final.head())

if __name__ == "__main__":
    process_dataset()
    print("df_final shape:", pd.read_csv(OUTPUT_PATH).shape)
