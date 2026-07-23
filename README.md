# EOG-Based Real-Time Robot Control Using 1D-CNN

A real-time machine learning system for classifying Electrooculography (EOG)
signals and translating eye activities into mobile robot control commands.

This project was developed as my final project in Biomedical Engineering
at Telkom University.

---

## Overview

This project develops a single-channel EOG-based Human–Machine Interface (HMI)
for real-time mobile robot control.

EOG signals are acquired using BITalino and classified into four eye-activity classes:

- Left
- Right
- Blink
- Idle / Neutral

The predicted activities are processed through decision logic and mapped into
movement commands for a Pioneer P3DX mobile robot simulated in CoppeliaSim.

The complete system covers:

**EOG Acquisition → Signal Processing → Dataset Preparation → 1D-CNN Training → Model Evaluation → Real-Time Inference → Decision Logic → Robot Control**

---

## System Architecture

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/b6a5113a-25b9-4515-9ffc-6d1c1a280c59" />

The system consists of two main stages:

### Offline — Model Development

1. EOG data acquisition
2. Dataset creation
3. Quality control and preprocessing
4. Dataset splitting
5. 1D-CNN training and evaluation

### Online — Real-Time Operation

1. Load the trained model
2. Acquire real-time EOG signals
3. Apply signal preprocessing
4. Perform model inference
5. Apply blink detection and decision logic
6. Generate robot control commands
7. Send commands to the Pioneer P3DX robot in CoppeliaSim

---

## Dataset

The model-development dataset consisted of:

- **5 subjects**
- **20 trials per subject**
- **100 total trials**
- **25 trials per class**

Each subject performed:

- 5 idle trials
- 5 left-eye-movement trials
- 5 right-eye-movement trials
- 5 blink trials

An additional **5 unseen subjects** were used for cross-subject evaluation
without retraining the model.

---

## Signal Processing

Raw EOG signals were acquired using BITalino at a sampling frequency of
**1000 Hz**.

The preprocessing pipeline included:

1. **4th-order Butterworth low-pass filtering**
   - Cutoff frequency: **15 Hz**

2. **Baseline correction**

3. **Activity segmentation**
   - Segment duration: **2 seconds**

4. **Downsampling**
   - From **1000 Hz to 250 Hz**

5. **Normalization**

The resulting signal segment contained **500 samples** per trial segment
for model input.

---

## Machine Learning

A **One-Dimensional Convolutional Neural Network (1D-CNN)** was used to
classify the temporal patterns of the EOG signals.

### Classification Classes

1. Left
2. Right
3. Blink
4. Idle

### Dataset Split

The 100-trial model-development dataset was divided using a stratified
**80:20 train-test split**:

- Training set: **80 trials**
- Internal test set: **20 trials**

The 80 training trials were further divided internally into:

- Training subset: **64 trials**
- Validation subset: **16 trials**

The validation subset was used to determine the best training epoch.

After epoch selection, the model was retrained using all **80 training trials**
before final evaluation on the held-out **20-trial internal test set**.

---

## Model Training Pipeline

<img width="1448" height="1086" alt="Dataset Split and 1D-CNN Training Pipeline" src="https://github.com/user-attachments/assets/a6c76b66-3981-47fd-9997-67223c346a84" />

The final trained model was exported for real-time inference using
PyTorch model serialization and **TorchScript**.

---

## Model Evaluation

### Internal Test Performance

Evaluation on the held-out 20-trial internal test set achieved:

- **Accuracy: 90.00%**
- **Macro F1-score: 90.28%**

### Cross-Subject Evaluation

The trained model was also evaluated using **100 trials from 5 unseen subjects**
who were not involved in model training, normalization parameter estimation,
or model selection.

Results:

- **Accuracy: 87.00%**
- **Macro F1-score: 86.74%**
- **Subject-level accuracy range: 75–95%**

The blink class was the most challenging class:

- **Blink Recall: 72%**
- **Blink F1-score: 80%**

These results indicate that the model was able to generalize to previously
unseen subjects, although inter-subject EOG variability still affected
classification performance.

---

## Real-Time System

The trained 1D-CNN was integrated with the complete real-time system:

- BITalino EOG acquisition
- Real-time signal preprocessing
- Model inference
- Blink detection
- Decision logic
- Robot command mapping
- CoppeliaSim Remote API
- Pioneer P3DX mobile robot simulation
- Prediction and command logging

### Robot Command Mapping

| EOG Activity | Robot Command |
|---|---|
| Idle | HOLD |
| Left | TURN LEFT |
| Right | TURN RIGHT |
| Single Blink | STOP |
| Double Blink | MOVE FORWARD |

Blink detection was used to differentiate between a single blink and
a double blink after the model classified the signal as a blink.

---

## Real-Time Performance

The integrated system was evaluated using **20 real-time trials**.

Results:

- **Label prediction accuracy: 80%**
- **Robot-command agreement: 75%**
- **Communication and command execution success: 100%**
- **Average inference time: 2.61 ms**

The inference time demonstrates that the trained model is computationally
fast enough for real-time operation within the developed system.

---

## Real-Time Robot Control

<img width="1448" height="1086" alt="Real-Time EOG Classification and Robot Control Interface" src="https://github.com/user-attachments/assets/01a62088-38ca-4447-b9d2-9a151b2ed697" />

The real-time interface displays:

- EOG activity instruction
- Filtered EOG signal
- Model prediction
- Prediction confidence
- Blink count
- Final robot command
- Camera preview for experiment documentation
- Pioneer P3DX robot response in CoppeliaSim

The camera preview is used only for experiment documentation and is not used
as an input to the machine learning model.

---

## Technologies

### Machine Learning & Data

- Python
- PyTorch
- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib

### Biomedical Signal Acquisition

- BITalino
- Electrooculography (EOG)

### Simulation & Integration

- CoppeliaSim
- CoppeliaSim Remote API
- Pioneer P3DX

---

## Project Structure

```text
eog-1dcnn-robot-control/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data_acquisition.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── realtime_inference.py
│   └── robot_control.py
│
├── results/
│   ├── training_history/
│   └── confusion_matrices/
│
├── docs/
│   ├── system_architecture.png
│   ├── training_pipeline.png
│   └── realtime_interface.png
│
└── models/
    └── README.md
