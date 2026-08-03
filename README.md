# Multimodal ADHD Prediction using EEG, Eye Tracking & Physiological Signals

A deep learning framework for ADHD prediction using multimodal physiological signals from the **Brain, Body & Behavior Dataset (BBBD)**. This project explores both classical machine learning and deep learning approaches to model neural, ocular, and physiological biomarkers associated with ADHD.

---

## Overview

ADHD diagnosis primarily relies on behavioral assessments and self-report questionnaires. This project investigates whether multimodal biosignals can be leveraged to assist ADHD screening through data-driven models.

The workflow includes signal preprocessing, feature engineering, multimodal sequence modeling, and comparative evaluation across multiple architectures.

---

## Features

- Multimodal physiological data processing
- EEG feature extraction using Global Field Power (GFP)
- Eye-tracking and physiological signal preprocessing
- Missing value handling and feature engineering
- Logistic Regression baseline
- Multimodal LSTM for temporal sequence modeling
- Cross-Attention Transformer for multimodal fusion
- Cross-experiment generalization analysis

---

## Dataset

This project uses **Brain, Body & Behavior Dataset (BBBD)**.

**Modalities used**

- EEG
- Eye Tracking
- Pupil Diameter
- Blink Rate
- Saccade Rate
- ECG / Heart Rate
- Respiration

Ground-truth labels are derived from the **Adult ADHD Self-Report Scale (ASRS)**.

---

## Methodology

The complete pipeline consists of four stages:

```
Raw BBBD Dataset
        │
        ▼
Signal Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ▼
Performance Evaluation
```

Three complementary models were implemented and evaluated:

- **Logistic Regression** – interpretable baseline using engineered statistical features.
- **Multimodal LSTM** – independent temporal encoders for neural, ocular, and physiological signals.
- **Cross-Attention Transformer** – transformer-based multimodal fusion for ADHD severity prediction.

---

## Tech Stack

- Python
- PyTorch
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- MNE
- Matplotlib

---

## Repository Structure

```
.
├── data/
├── notebooks/
├── src/
├── models/
├── results/
├── report/
│   └── Project_Report.pdf
├── requirements.txt
└── README.md
```

---

## Results

The project compares classical machine learning and deep learning approaches for multimodal ADHD prediction while also evaluating model robustness through cross-experiment validation. Detailed experimental methodology, model architectures, ablation studies, and quantitative results are available in the project report.

---

## References

- The Brain, Body, and Behavior Dataset (BBBD): Multimodal Recordings during Educational Videos

---
