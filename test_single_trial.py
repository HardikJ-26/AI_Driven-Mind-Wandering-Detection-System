"""
test_single_trial.py
Quick sanity-check: loads one trial from Exp2 sub-01/ses-01/stim01
and prints shapes so we know the full pipeline is working.
"""
import sys, os
sys.path.insert(0, r'f:\DATA C DRIVE\BBBD experiments')
os.chdir(r'f:\DATA C DRIVE\BBBD experiments')

from src.data_preprocessing import extract_features_for_trial

print("Testing single trial: sub-01, ses-01, task-stim01, exp2...")
trial = extract_features_for_trial(
    sub=1, ses=1, task=1,
    exp_dir=r'f:\DATA C DRIVE\BBBD experiments\experiment2',
    exp_id=2
)
if trial:
    eeg    = trial['eeg']
    ocular = trial['ocular']
    physio = trial['physio']
    label  = trial['label']
    mem    = trial['memory_score']
    print("SUCCESS!")
    print(f"  EEG shape    : {eeg.shape}")
    print(f"  Ocular shape : {ocular.shape}")
    print(f"  Physio shape : {physio.shape}")
    print(f"  Label        : {label}  (0=Attentive, 1=Distracted)")
    print(f"  Memory score : {mem:.3f}")
else:
    print("FAILED - trial returned None")
