"""
run_all.py
──────────
Master pipeline script for the BBBD Multimodal Study.

Execution order:
  1. Feature extraction & caching  (src/data_preprocessing.py)
  2. Training on Experiment 2      (train_exp2.py)
  3. Training on Experiment 3      (train_exp3.py)
  4. Cross-experiment evaluation   (evaluate_cross.py)
  5. Attention trend analysis      (analyze_trends.py)

All results are written to:
  f:\\DATA C DRIVE\\BBBD experiments\\results\\
"""

import os
import sys
import time

BASE = r"f:\DATA C DRIVE\BBBD experiments"
CACHE_DIR = os.path.join(BASE, "data_cache")
RESULTS_DIR = os.path.join(BASE, "results")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── helper ──────────────────────────────────────────────────────────────────
def header(msg):
    bar = "=" * 64
    print(f"\n{bar}\n  {msg}\n{bar}\n")

# ── Step 1: Preprocessing ────────────────────────────────────────────────────
header("STEP 1 / 5 — Feature Extraction & Caching")

from src.data_preprocessing import preprocess_experiment

exp2_cache = os.path.join(CACHE_DIR, "exp2_raw_trials.pkl")
exp3_cache = os.path.join(CACHE_DIR, "exp3_raw_trials.pkl")

for cache_file, exp_dir, exp_id in [
    (exp2_cache, os.path.join(BASE, "experiment2"), 2),
    (exp3_cache, os.path.join(BASE, "experiment3"), 3),
]:
    if os.path.exists(cache_file):
        print(f"[REBUILD] Rebuilding cache: {cache_file}")
    preprocess_experiment(
        exp_dir    = exp_dir,
        exp_id     = exp_id,
        cache_file = cache_file,
        target_fs  = 64.0,
    )

# ── Step 2: Train on Experiment 2 ───────────────────────────────────────────
header("STEP 2 / 5 — Training on Experiment 2")
import train_exp2
train_exp2.main()

# ── Step 3: Train on Experiment 3 ───────────────────────────────────────────
header("STEP 3 / 5 — Training on Experiment 3")
import train_exp3
train_exp3.main()

# ── Step 4: Cross-Experiment Evaluation ─────────────────────────────────────
header("STEP 4 / 5 — Cross-Experiment Evaluation")
import evaluate_cross
evaluate_cross.main()

# ── Step 5: Attention Trend & Mind-Wandering Analysis ───────────────────────
header("STEP 5 / 5 — Attention Trend & Mind-Wandering Analysis")
import analyze_trends
analyze_trends.main()

header("ALL STEPS COMPLETED")
print(f"Results are saved in: {RESULTS_DIR}")
