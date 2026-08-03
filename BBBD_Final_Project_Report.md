# Final Project Report: Multimodal Attention and ADHD Modelling

## Project aim

This project used the Brain, Body & Behaviour Dataset (BBBD) to model attention, distraction, and ADHD-related outcomes from multimodal recordings. The workflow combined EEG, eye tracking/ocular signals, ECG, and—in Experiment 4—EOG. It covered data preparation, windowing, model experimentation, cross-experiment testing, and real-time inference.

## Common experimental workflow

1. Load BIDS-organised recordings by participant, session, stimulus, and modality.
2. Clean and align EEG, ocular, and physiological streams; use eye-tracking timestamp intervals to identify invalid segments.
3. Construct fixed-duration windows for sequence modelling or aggregate statistical features for classical models.
4. Compare unimodal and multimodal configurations, including early/late fusion where applicable.
5. Evaluate with held-out data and assess external transfer only with modalities common to both experiments.

### Models and libraries evaluated

The tested model set comprised **Linear Regression, Logistic Regression, Random Forest, BiLSTM, LSTM, Transformer, Mamba, and GRU**. Classical models provided fast and interpretable baselines; recurrent, Transformer, and Mamba-family models were evaluated for sequence learning and multimodal fusion.

## Experiments 1–3 — Attention modelling programme

### Working and experimentation

The first three experiments form a systematic attention-modelling programme rather than three isolated model runs. They compare intentional and incidental video-learning conditions, then evaluate whether neural, ocular, and cardiovascular information generalises across those conditions.

The experimental design in the supplied slides comprises:

| Experimental dimension | Work completed |
|---|---|
| Modalities | Seven combinations: EEG, eye tracking, ECG, every pairwise fusion, and full three-modality fusion |
| Windowing | Five window sizes from **2 s to 20 s** |
| Architectures | Random Forest, XGBoost, BiLSTM, and Transformer; the wider project model set also included Linear Regression, Logistic Regression, LSTM, Mamba, and GRU |
| EEG features | Began with delta–gamma global band powers; added four-region band powers, spectral entropy, frontal alpha asymmetry, frontal-midline theta, and theta/alpha ratio |
| Validation | Cross-validation across subjects with no subject overlap between train and test partitions |
| External check | Train on one experiment and evaluate on a held-out experiment |

### Experiment-specific application

| Experiment | Learning condition | Modality focus | Purpose |
|---|---|---|---|
| 1 | In-domain attention modelling plus transfer | EEG, eye tracking, ECG | Establish the Transformer baseline and test its transfer to Experiments 2 and 3. |
| 2 | Multimodal attention modelling | EEG, eye tracking, ECG and fusions | Compare deep-learning and machine-learning modality configurations. |
| 3 | Multimodal attention modelling plus transfer | EEG, eye tracking, ECG and fusions | Compare high-performing in-domain configurations and quantify transfer in both directions. |

### Interpretation

Experiments 1–3 establish the research methodology: attention is evaluated over multiple time scales, single and fused modalities, both tabular and sequence models, and subject-disjoint splits. This is more informative than selecting a single architecture because it identifies whether a result depends on window duration, modality choice, learning condition, or model family.

### Verified Experiment 1 results

| Setup | Architecture | Window | AUROC | Accuracy | F1 |
|---|---|---:|---:|---:|---:|
| Setup A: Exp1 in-domain | Transformer | 4 s | 0.871 ± 0.028 | 0.783 ± 0.042 | 0.781 ± 0.052 |
| Setup B: transfer to Exp2 | Transformer | 10 s | 0.895 ± 0.006 | 0.801 ± 0.023 | 0.794 ± 0.018 |
| Setup B: transfer to Exp3 | Transformer | 6 s | 0.825 ± 0.007 | 0.744 ± 0.005 | 0.738 ± 0.017 |

The Transformer was the selected architecture in all three listed configurations. The Exp1-to-Exp2 transfer result is the strongest listed Experiment 1 result; transfer to Exp3 is lower but remains above the in-domain F1 baseline.

### Verified Experiment 2 results

| Setup | Modality | Architecture | Window | AUROC | Accuracy | F1 |
|---|---|---|---:|---:|---:|---:|
| DL Setup A | Eye | BiLSTM | 20 s | 0.885 ± 0.044 | 0.803 ± 0.037 | 0.792 ± 0.023 |
| DL Setup A | Eye | BiLSTM | 6 s | 0.885 ± 0.042 | 0.802 ± 0.049 | 0.799 ± 0.040 |
| ML Setup A | Eye | Random Forest | 20 s | 0.841 ± 0.040 | 0.741 ± 0.039 | 0.725 ± 0.021 |
| DL Setup B | Eye + ECG | Transformer | 4 s | 0.790 ± 0.012 | 0.723 ± 0.024 | 0.721 ± 0.033 |
| ML Setup B | EEG + Eye | Random Forest | 20 s | 0.747 ± 0.005 | 0.683 ± 0.005 | 0.706 ± 0.005 |

The two eye-only BiLSTM settings are the strongest Experiment 2 configurations, reaching F1 0.799. The similar 6 s and 20 s values indicate that the ocular sequence representation was robust to these two window lengths. In the reported configurations, adding ECG or EEG did not improve on the eye-only BiLSTM result.

### Verified Experiment 3 results

| Setup | Modality | Architecture | Window | AUROC | Accuracy | F1 |
|---|---|---|---:|---:|---:|---:|
| DL Setup A | Eye | BiLSTM | 4 s | 0.934 | 0.848 | 0.847 |
| DL Setup A | Eye + ECG | Transformer | 4 s | 0.909 | 0.822 | 0.819 |
| DL Setup B | EEG + Eye | Transformer | 10 s | 0.875 | 0.679 | 0.655 |
| DL Setup B | All three: EEG + ECG + Eye | BiLSTM | 8 s | 0.841 | 0.784 | 0.774 |

The best listed Experiment 3 configuration is the 4-second eye-only BiLSTM, with AUROC 0.934 and F1 0.847. Eye + ECG with a Transformer is also strong, whereas the EEG + Eye transfer configuration is notably lower. This again makes ocular information the most reliable reported modality.

### Cross-experiment modality analysis

The cross-experiment plots show that the preferred modality combination changes with transfer direction. For Exp2 → Exp3, the plotted accuracy values range from **58.5%** for ECG alone to **80.3%** for the best eye-only configuration; EEG + Eye reaches **78.2%**. For Exp3 → Exp2, the plotted values range from **54.3%** for EEG alone to **84.8%** for the best eye-only configuration; Eye + ECG reaches **82.2%**. The consistent conclusion is that eye information transfers more reliably than ECG or EEG alone, while fusion can improve results in some transfer directions.

| Transfer direction | ECG | ECG + EEG | EEG | All three | ECG + Eye | EEG + Eye | Eye-only best |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp2 → Exp3 | 58.5% | 62.8% | 64.0% | 75.0% | 76.8% | 78.2% | **80.3%** |
| Exp3 → Exp2 | — | 61.9% | 54.3% | 77.5% | 82.2% | 77.8% | **84.8%** |

For Exp3 → Exp2, the plot additionally shows a 64.1% eye-only configuration; the highest plotted eye-only result is 84.8%. The two-direction comparison confirms that transfer is asymmetric: training on one experiment does not guarantee the same modality ranking or accuracy on the other.

## Experiment 4 — Multimodal ADHD-related modelling

### Data and features

Experiment 4 is an **incidental-learning** experiment for attentive/distracted-state analysis and ADHD prediction. It uses multimodal EEG, ocular, ECG, and EOG data. The available detailed dataset record contains 43 participants, two sessions, up to six stimuli per session, and 258 recording-level samples.

The aggregate feature pipeline included cardiac, ocular, and EEG-derived measures; the sequential experiments used the multimodal EEG, ocular, ECG, and EOG streams.

For the final regression LSTM, each approximately 300-second recording was represented by 30 non-overlapping 10-second windows. Each window contained 40 resampled time steps: EEG `(40, 64)`, ocular `(40, 4)`, and physiological `(40, 2)`.

### Models and experimentation

| Model | Input / role |
|---|---|
| Random Forest | Multimodal binary ADHD classification baseline |
| LSTM classifier | Multimodal sequential binary ADHD classification |
| LSTM regressor | Multimodal sequential ADHD regression |

### Results

| Model | Task | Result |
|---|---|---|
| Random Forest | Binary ADHD classification | **92.8% test accuracy** |
| LSTM | Binary ADHD classification | **99.73% accuracy** |
| **LSTM regressor** | **ADHD regression** | **R² = 0.835, MAE = 0.072** |

The final LSTM regression execution used 244 complete recordings (7,320 windows); 14 incomplete recordings were excluded. Its reported MSE was 0.009893, with MAE 0.071884 and R² 0.835106, which round to the slide values above. These results indicate strong internal predictive fit, but they must be confirmed with leakage-resistant participant-level validation and independent data.

## Cross-experiment evaluation: Exp4 → Exp2

Only shared modalities should be used for transfer: 64-channel EEG, gaze X/Y, pupil, blink rate, and heart rate. Exp4 respiration and EOG must be excluded because they are not available with compatible Exp2 derivatives.

The existing zero-shot Exp4-to-Exp2 LSTM evaluation showed substantial distribution shift. At threshold 0.5 it achieved 92.31% accuracy but zero positive-class precision, recall, and F1 because it predicted every participant as negative. Its ROC-AUC was 0.3542. Therefore, that accuracy is a class-imbalance artefact, not successful generalisation.

## Real-time detection

The project includes a real-time inference prototype that loads trained sequence-model weights, processes incoming modality inputs, and displays prediction outputs in an interactive interface. This demonstrates deployment feasibility for research monitoring. It is not a clinical diagnostic tool; external validation, calibration, privacy safeguards, and regulatory review are required before real-world use.

## Conclusions

1. Ocular features were consistently useful for attention modelling.
2. Multimodal sequence models enabled temporal experiments across EEG, eye tracking, and cardiovascular streams.
3. The final Experiment 4 LSTM regressor achieved strong internal performance: R² 0.835 and MAE 0.0719.
4. Cross-experiment transfer is the central limitation and requires harmonised preprocessing, calibration, and domain-adaptation work.
5. Future validation should use participant-level nested splits and independent cohorts.
