# Interpretability Framework for Deep Learning Models in ECG Signal Analysis

AI was used to assist with writing the code.

## Overview

This repository contains the materials for a Bachelor of Engineering thesis in Information and Communication Technology (Health Technology major). The work addresses a critical gap in medical AI: the lack of transparency in deep learning models used for electrocardiogram (ECG) signal analysis.

While deep learning models have demonstrated strong performance in detecting cardiovascular conditions such as atrial fibrillation, their "black box" nature presents a significant barrier to clinical adoption. Healthcare professionals need to understand _why_ a model arrives at a particular diagnosis before they can trust and act on it. This thesis develops an interpretability framework that combines multiple explainability methods to provide clinically meaningful explanations for ECG-based AI decisions.

## Problem Statement

AI systems deployed in clinical settings must meet stringent requirements for transparency and reliability. The EU AI Act classifies medical AI systems as high-risk, requiring them to be sufficiently transparent for healthcare providers to interpret their outputs. Similarly, the Medical Device Regulation (MDR) imposes strict requirements on software used for diagnostic purposes. Current deep learning approaches for ECG analysis, while accurate, often fail to meet these interpretability requirements.

## Research Focus

The thesis investigates and compares multiple explainability techniques for ECG signal analysis, including:

- **Gradient-based methods** — HiResCAM and GradCAM, which highlight input regions most influential to model predictions
- **SHAP-based methods** — GradientSHAP and KernelSHAP, which quantify feature contributions through Shapley value approximations
- **Faithfulness evaluation** — deletion curves and AOPC scores to objectively compare explanation quality across methods

## Dataset

This project uses the [ECG Arrhythmia Database](https://physionet.org/content/ecg-arrhythmia/1.0.0/) — 45,152 clinical 12-lead ECGs (500 Hz, 10 seconds) with SNOMED-CT diagnostic labels covering ~80 arrhythmia conditions.

The framework classifies the top 5 most frequent conditions:

| Abbreviation | Condition           |
| ------------ | ------------------- |
| SB           | Sinus Bradycardia   |
| SR           | Sinus Rhythm        |
| AF           | Atrial Fibrillation |
| ST           | Sinus Tachycardia   |
| TWC          | T-Wave Change       |

## Project Structure

```
ecg-xai-framework/
├── configs/
│   ├── data.yaml               # dataset, sample rate, preprocessing params
│   ├── model.yaml              # model name, target layer, checkpoint paths
│   ├── training.yaml           # training hyperparameters
│   └── xai.yaml                # which XAI methods to run, evaluation settings
├── data/
│   ├── raw/                    # downloaded PhysioNet files
│   ├── processed/              # metadata CSV
│   └── preprocessed/           # cached .npy files
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_xai_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── dataset.py          # ArrhythmiaDataset (PyTorch Dataset)
│   │   └── preprocessing.py    # bandpass filter, normalisation, segmentation
│   ├── models/
│   │   └── cnn.py              # xresnet1d model building and loading
│   ├── training/
│   │   └── trainer.py          # training loop with early stopping
│   ├── explainability/
│   │   ├── captum_methods.py   # GradientSHAP
│   │   ├── gradcam.py          # HiResCAM and GradCAM via signal_grad_cam
│   │   ├── kernel_shap.py      # KernelSHAP (model-agnostic, temporal segmentation)
│   │   └── evaluation.py       # deletion curve / AOPC faithfulness metric
│   └── visualization/
│       └── plots.py            # ECG + attribution overlay, method comparison
├── outputs/
│   ├── models/                 # saved model checkpoints
│   ├── explanations/           # saved attribution arrays (.npy)
│   └── figures/                # generated plots
├── environment.yml
└── requirements.txt
```

## Licence

MIT Licence
