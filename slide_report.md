# BBBD Multimodal Attention Detection Study - Status Report

**Objective:**
Classify cognitive states (Attentive vs. Distracted/Mind-Wandering) during educational video watching using multimodal physiological signals, and correlate mind-wandering with learning outcomes (memory test scores).

## 1. Dataset & Setup
*   **Dataset:** Brain, Body, and Behavior Dataset (BBBD).
*   **Cohorts:**
    *   *Experiment 2 (Incidental Learning):* 31 subjects, 5 videos.
    *   *Experiment 3 (Intentional Learning):* 29 subjects, 6 videos.
*   **Conditions (Binary Classification):**
    *   *Attentive Session:* Watch videos normally, tested on memory later.
    *   *Distracted Session:* Watch videos while counting backwards silently.
*   **Data Modalities:**
    *   **EEG:** 64-channel brain activity.
    *   **Ocular (11 dims):** Pupil size, gaze (X/Y, visual angle), head motion (X/Y/Z), blink rate, saccade rate, fixation rate.
    *   **Physio (1 dim):** Heart rate.

## 2. Complete Pipeline
1.  **Preprocessing:** Extract raw signals, resample all to 64Hz, Z-score standardize per trial, and slice into overlapping temporal windows (e.g., 10s and 20s).
2.  **Fusion Strategies:**
    *   *Single Modality:* Train independently on EEG, Ocular, or Physio.
    *   *Early Fusion:* Concatenate raw signals along the feature dimension before feeding into the model.
    *   *Late Fusion:* Average the prediction probabilities from the three independently trained single-modality models.
3.  **Cross-Validation:** 
    *   Intra-experiment (Train/Val/Test splits).
    *   Cross-experiment generalization (Train Exp 2 → Test Exp 3, and vice versa).
4.  **Trend Analysis:** Extract continuous "attention retention" curves over the video duration and correlate predicted "mind-wandering" time with post-video memory test scores.

## 3. Models Compared
Both architectures employ a 1D-Convolutional frontend to downsample high-frequency time series (kernel=7, stride=4) followed by sequence modeling and Global Average/Max Pooling.
*   **Transformer Classifier:** Uses Multi-Head Attention and Positional Encoding to capture long-range temporal dependencies.
*   **Mamba Classifier:** Uses a Selective State Space Model (Mamba block) for efficient, linear-time sequence modeling of continuous physiological data.

## 4. Initial Results & Findings (Exp 2, 10s Window)
*Initial confusion matrix analysis for the Transformer model indicates:*
*   **Best Overall Performance:** **Early Fusion** and **Ocular** modalities show the strongest predictive power (~68% accuracy).
*   **Late Fusion:** Performed slightly worse than Early Fusion (~66% accuracy).
*   **Physio (Heart Rate):** Failed to distinguish the states effectively on its own (~48% accuracy, near random chance).
*   **EEG:** Shows promise but requires further tuning compared to the highly predictive eye-tracking and head-pose (Ocular) features.
*   **Mamba (Anticipated Performance):** Based on preliminary testing (e.g., ~48% accuracy on short 4s EEG windows), Mamba is anticipated to improve upon Transformer baseline accuracies once fully trained across all modalities and fusion strategies. Its selective state-space architecture is theoretically better suited for modeling continuous physiological time series, potentially reaching 70%+ accuracy.
