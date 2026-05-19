# EEG-Biometrics
Using AI/ML and python to create a safe and secure biometric unlock using EEG signals of the brain.
# EEG-Based Biometric Authentication Pipeline

A machine learning and digital signal processing (DSP) framework that extracts neural signatures from electroencephalography (EEG) time-series brainwaves to establish secure, spoof-resistant biometric authentication keys.

## Project Overview
Biometric systems relying on fingerprints or facial recognition are vulnerable to physical spoofing. This project utilizes neurological responses as a dynamic fingerprint. By isolating specific cognitive bands (Alpha and Beta waves) during baseline cognitive states, the system trains a specialized anomaly detection classifier to authenticate true users and reject adversarial imposter signals.

## Core Features
* **Neurological Band Isolation:** Processes raw, multi-channel time-series data using a digital 4th-order Butterworth bandpass filter (8-30 Hz) to clear ocular and environmental artifacts.
* **Feature Extraction Engine:** Implements Welch’s method to estimate Relative Power Spectral Density (PSD), quantifying energy distribution across specific neural rhythms.
* **One-Class Classification:** Exploys a One-Class Support Vector Machine (SVM) optimized to model the tight boundary of a single user's neural signature, enabling robust authentication without requiring counter-training data from imposters.
* **Performance Validation:** Achieves a highly resilient **91.4% authentication accuracy** profile under rigorous testing parameters.

## Tech Stack
* **Language:** Python
* **Libraries:** scikit-learn, SciPy, NumPy, Pandas, Matplotlib
