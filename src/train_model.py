#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a Four-Class EOG 1D-CNN
=============================

Reproduces the final five-subject model-development experiment:

- 5 subjects × 20 trials = 100 trials.
- Stratified outer split: 80 training trials and 20 internal-test trials.
- Stratification uses the subject-class combination, so every subject and
  every class are represented in both outer splits.
- The 80 training trials are split again into 64 training and 16 validation
  trials for epoch selection.
- The final model is retrained from scratch on all 80 outer-training trials.
- The 20 outer-test trials are used only for the final internal test.

This internal test measures new trials from the same five subjects; it is not
a subject-independent evaluation. Generalization to unseen subjects is handled
by ``evaluate_realtime.py`` and ``analyze_evaluation.py``.

Default repository layout:
    data/training/
    outputs/training/
    models/eog_5_subjek_80_20/

Example:
    python src/train_model.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# KONFIGURASI DATA
# ============================================================
RAW_FS = 1000
PROCESSED_FS = 250
DOWNSAMPLE_FACTOR = RAW_FS // PROCESSED_FS

LOWPASS_HZ = 15.0
FILTER_ORDER = 4

BASELINE_SAMPLES = 2 * RAW_FS
ACTION_SAMPLES = 2 * RAW_FS
RETURN_SAMPLES = 2 * RAW_FS
REST_SAMPLES = 5 * RAW_FS
TRIAL_SAMPLES = (
    BASELINE_SAMPLES
    + ACTION_SAMPLES
    + RETURN_SAMPLES
    + REST_SAMPLES
)

EXPECTED_SUBJECTS = 5
EXPECTED_TRIALS_PER_SUBJECT = 20
EXPECTED_TRIALS_PER_CLASS_PER_SUBJECT = 5

CLASS_NAMES = ["diam", "kiri", "kanan", "kedip"]
LABEL_TO_INDEX = {
    "diam": 0,
    "kiri": 1,
    "kanan": 2,
    "kedip": 3,
}

REQUIRED_COLUMNS = [
    "waktu_detik",
    "sinyal_a1",
    "status",
    "trial_ke",
]

TEST_SIZE = 0.20
VALIDATION_SIZE_FROM_TRAIN = 0.20

RANDOM_SEED = 42

# Training configuration with early stopping.
MAX_EPOCHS = 250
EARLY_STOPPING_PATIENCE = 40
BATCH_SIZE = 16
LEARNING_RATE = 0.0015
MIN_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
DROPOUT = 0.20


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(RANDOM_SEED)
torch.set_num_threads(2)


# ============================================================
# MODEL 1D-CNN
# ============================================================
class EOG1DCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                1,
                16,
                kernel_size=11,
                padding=5,
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                16,
                32,
                kernel_size=9,
                padding=4,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                32,
                64,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                64,
                96,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, len(CLASS_NAMES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)


# ============================================================
# UTILITAS
# ============================================================
SOS = butter(
    FILTER_ORDER,
    LOWPASS_HZ,
    btype="lowpass",
    fs=RAW_FS,
    output="sos",
)


def natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def normalize_status(value: object) -> str:
    text = str(value).strip().lower()

    mapping = {
        "left": "kiri",
        "right": "kanan",
        "blink": "kedip",
        "rest": "diam",
        "center_start": "tengah_awal",
        "center_return": "tengah_kembali",
        "break": "jeda",
    }

    return mapping.get(text, text)


def causal_lowpass(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)

    if signal.size == 0:
        raise ValueError("Sinyal kosong.")

    initial_state = sosfilt_zi(SOS) * signal[0]
    filtered, _ = sosfilt(
        SOS,
        signal,
        zi=initial_state,
    )
    return filtered


def get_action_label(status_values: pd.Series) -> str:
    labels = [
        value
        for value in pd.unique(status_values)
        if value in CLASS_NAMES
    ]

    if len(labels) != 1:
        raise ValueError(
            "Setiap trial harus memiliki tepat satu kelas aksi. "
            f"Ditemukan: {labels}"
        )

    return labels[0]


def infer_subject_id(
    dataframe: pd.DataFrame,
    csv_path: Path,
) -> str:
    for column in ("subjek_id", "subject_id"):
        if column in dataframe.columns:
            values = (
                dataframe[column]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            if len(values) != 1:
                raise ValueError(
                    f"{csv_path.name}: kolom {column} harus "
                    "berisi satu ID subjek."
                )

            return values[0]

    return csv_path.stem


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# LOAD DAN PREPROCESSING DATA
# ============================================================
def load_dataset(
    data_dir: Path,
    file_glob: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    csv_files = sorted(
        data_dir.glob(file_glob),
        key=natural_key,
    )

    if len(csv_files) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Ditemukan {len(csv_files)} file CSV di {data_dir}. "
            f"Harus tepat {EXPECTED_SUBJECTS} file."
        )

    windows: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[str] = []
    file_names: list[str] = []
    trial_numbers: list[int] = []
    audit_rows: list[dict[str, object]] = []

    detected_subjects: set[str] = set()

    for csv_path in csv_files:
        dataframe = pd.read_csv(csv_path)

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{csv_path.name}: kolom hilang {missing_columns}"
            )

        subject_id = infer_subject_id(
            dataframe,
            csv_path,
        )

        if subject_id in detected_subjects:
            raise ValueError(
                f"ID subjek ganda: {subject_id}"
            )

        detected_subjects.add(subject_id)

        work = dataframe[
            REQUIRED_COLUMNS
        ].copy()

        work["waktu_detik"] = pd.to_numeric(
            work["waktu_detik"],
            errors="coerce",
        )
        work["sinyal_a1"] = pd.to_numeric(
            work["sinyal_a1"],
            errors="coerce",
        )
        work["trial_ke"] = pd.to_numeric(
            work["trial_ke"],
            errors="coerce",
        )
        work["status"] = work["status"].map(
            normalize_status
        )

        if work.isna().any().any():
            raise ValueError(
                f"{csv_path.name}: terdapat data kosong/invalid."
            )

        work["trial_ke"] = (
            work["trial_ke"]
            .astype(np.int64)
        )

        expected_samples = (
            EXPECTED_TRIALS_PER_SUBJECT
            * TRIAL_SAMPLES
        )

        if len(work) != expected_samples:
            raise ValueError(
                f"{csv_path.name}: jumlah sampel {len(work):,}, "
                f"seharusnya {expected_samples:,}."
            )

        raw_signal = work[
            "sinyal_a1"
        ].to_numpy(dtype=np.float64)

        if np.any(
            (raw_signal < 0)
            | (raw_signal > 1023)
        ):
            raise ValueError(
                f"{csv_path.name}: ada nilai di luar rentang ADC."
            )

        work["filtered"] = causal_lowpass(
            raw_signal
        )

        class_counter: Counter[str] = Counter()

        for trial_id, trial_df in work.groupby(
            "trial_ke",
            sort=True,
        ):
            if len(trial_df) != TRIAL_SAMPLES:
                raise ValueError(
                    f"{csv_path.name}, trial {trial_id}: "
                    f"jumlah sampel {len(trial_df)}, "
                    f"seharusnya {TRIAL_SAMPLES}."
                )

            action_label = get_action_label(
                trial_df["status"]
            )
            class_counter[action_label] += 1

            baseline = trial_df.loc[
                trial_df["status"] == "tengah_awal",
                "filtered",
            ].to_numpy(dtype=np.float64)

            action = trial_df.loc[
                trial_df["status"] == action_label,
                "filtered",
            ].to_numpy(dtype=np.float64)

            if len(baseline) != BASELINE_SAMPLES:
                raise ValueError(
                    f"{csv_path.name}, trial {trial_id}: "
                    "baseline tidak berjumlah 2000 sampel."
                )

            if len(action) != ACTION_SAMPLES:
                raise ValueError(
                    f"{csv_path.name}, trial {trial_id}: "
                    "aksi tidak berjumlah 2000 sampel."
                )

            baseline_median = float(
                np.median(baseline)
            )

            corrected = (
                action - baseline_median
            )
            processed = corrected[
                ::DOWNSAMPLE_FACTOR
            ]

            if processed.shape != (500,):
                raise RuntimeError(
                    f"{csv_path.name}, trial {trial_id}: "
                    f"shape {processed.shape}, harus (500,)."
                )

            windows.append(
                processed.astype(np.float32)
            )
            labels.append(
                LABEL_TO_INDEX[action_label]
            )
            subjects.append(subject_id)
            file_names.append(csv_path.name)
            trial_numbers.append(int(trial_id))

        for class_name in CLASS_NAMES:
            if (
                class_counter[class_name]
                != EXPECTED_TRIALS_PER_CLASS_PER_SUBJECT
            ):
                raise ValueError(
                    f"{csv_path.name}: kelas {class_name} "
                    f"berjumlah {class_counter[class_name]}, "
                    f"seharusnya "
                    f"{EXPECTED_TRIALS_PER_CLASS_PER_SUBJECT}."
                )

        audit_rows.append({
            "subject_id": subject_id,
            "file_name": csv_path.name,
            "raw_samples": int(len(work)),
            "trials": int(
                work["trial_ke"].nunique()
            ),
            "diam": int(class_counter["diam"]),
            "kiri": int(class_counter["kiri"]),
            "kanan": int(class_counter["kanan"]),
            "kedip": int(class_counter["kedip"]),
            "clipped_samples": int(
                np.sum(
                    (raw_signal <= 1)
                    | (raw_signal >= 1022)
                )
            ),
        })

        print(
            f"{csv_path.name}: "
            f"subjek={subject_id}, "
            f"{len(work):,} sampel, "
            f"{sum(class_counter.values())} trial"
        )

    X = np.asarray(
        windows,
        dtype=np.float32,
    )
    y = np.asarray(
        labels,
        dtype=np.int64,
    )
    subject_array = np.asarray(
        subjects,
        dtype=str,
    )
    file_array = np.asarray(
        file_names,
        dtype=str,
    )
    trial_array = np.asarray(
        trial_numbers,
        dtype=np.int64,
    )

    if X.shape != (100, 500):
        raise ValueError(
            f"Shape dataset {X.shape}, seharusnya (100, 500)."
        )

    return (
        X,
        y,
        subject_array,
        file_array,
        trial_array,
        pd.DataFrame(audit_rows),
    )


# ============================================================
# SPLIT 80:20
# ============================================================
def make_split(
    y: np.ndarray,
    subjects: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))

    # Kombinasi subjek dan kelas dijadikan strata.
    # Karena setiap subjek mempunyai 5 trial per kelas,
    # hasilnya tepat 4 train + 1 test per kelas per subjek.
    stratification_key = np.asarray([
        f"{subject}__{CLASS_NAMES[label]}"
        for subject, label in zip(subjects, y)
    ])

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratification_key,
    )

    return (
        np.sort(train_indices),
        np.sort(test_indices),
    )


def print_split_summary(
    split_name: str,
    indices: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
) -> None:
    print()
    print(split_name)
    print("-" * 60)
    print("Jumlah trial:", len(indices))

    print("Distribusi kelas:")
    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        count = int(
            np.sum(y[indices] == class_index)
        )
        print(f"  {class_name:6s}: {count}")

    print("Distribusi subjek:")
    for subject in sorted(
        np.unique(subjects)
    ):
        count = int(
            np.sum(subjects[indices] == subject)
        )
        print(f"  {subject:8s}: {count}")


# ============================================================
# NORMALISASI
# ============================================================
def normalize_from_training(
    X: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    mean = float(
        np.mean(X[train_indices])
    )
    std = float(
        np.std(X[train_indices])
    )

    if (
        not np.isfinite(mean)
        or not np.isfinite(std)
        or std <= 0
    ):
        raise RuntimeError(
            "Mean/std data training tidak valid."
        )

    normalized = (
        X - mean
    ) / std

    return (
        normalized[:, np.newaxis, :],
        mean,
        std,
    )


# ============================================================
# TRAINING
# ============================================================
def train_model(
    X_normalized: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[
    EOG1DCNN,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    # Validation hanya diambil dari data training.
    train_labels = y[train_indices]

    inner_train_indices, validation_indices = (
        train_test_split(
            train_indices,
            test_size=VALIDATION_SIZE_FROM_TRAIN,
            random_state=RANDOM_SEED,
            stratify=train_labels,
        )
    )

    X_inner_train = torch.tensor(
        X_normalized[inner_train_indices],
        dtype=torch.float32,
    )
    y_inner_train = torch.tensor(
        y[inner_train_indices],
        dtype=torch.long,
    )

    X_validation = torch.tensor(
        X_normalized[validation_indices],
        dtype=torch.float32,
    )
    y_validation = torch.tensor(
        y[validation_indices],
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(
            X_inner_train,
            y_inner_train,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(
            RANDOM_SEED
        ),
    )

    set_seed(RANDOM_SEED)
    model = EOG1DCNN()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=MIN_LEARNING_RATE,
    )

    loss_function = nn.CrossEntropyLoss()

    best_state = None
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_training_loss = 0.0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_function(
                logits,
                batch_y,
            )
            loss.backward()
            optimizer.step()

            total_training_loss += (
                float(loss.item())
                * len(batch_y)
            )

        mean_training_loss = (
            total_training_loss
            / len(inner_train_indices)
        )

        model.eval()

        with torch.no_grad():
            train_logits = model(
                X_inner_train
            )
            validation_logits = model(
                X_validation
            )

            validation_loss = float(
                loss_function(
                    validation_logits,
                    y_validation,
                ).item()
            )

            train_predictions = (
                train_logits
                .argmax(dim=1)
                .cpu()
                .numpy()
            )
            validation_predictions = (
                validation_logits
                .argmax(dim=1)
                .cpu()
                .numpy()
            )

        train_accuracy = accuracy_score(
            y[inner_train_indices],
            train_predictions,
        )
        validation_accuracy = accuracy_score(
            y[validation_indices],
            validation_predictions,
        )
        validation_macro_f1 = f1_score(
            y[validation_indices],
            validation_predictions,
            average="macro",
            zero_division=0,
        )

        scheduler.step(
            validation_loss
        )

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        history_rows.append({
            "epoch": float(epoch),
            "train_loss": float(
                mean_training_loss
            ),
            "validation_loss": float(
                validation_loss
            ),
            "train_accuracy": float(
                train_accuracy
            ),
            "validation_accuracy": float(
                validation_accuracy
            ),
            "validation_macro_f1": float(
                validation_macro_f1
            ),
            "learning_rate": current_lr,
        })

        if (
            validation_loss
            < best_validation_loss - 1e-5
        ):
            best_validation_loss = (
                validation_loss
            )
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epoch == 1
            or epoch % 10 == 0
        ):
            print(
                f"Epoch {epoch:03d} | "
                f"loss train={mean_training_loss:.4f} | "
                f"loss val={validation_loss:.4f} | "
                f"acc train={train_accuracy * 100:6.2f}% | "
                f"acc val={validation_accuracy * 100:6.2f}% | "
                f"lr={current_lr:.6f}"
            )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                f"Early stopping pada epoch {epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError(
            "Best model tidak tersimpan."
        )

    model.load_state_dict(
        best_state
    )
    model.eval()

    print(
        f"Best epoch: {best_epoch}, "
        f"validation loss: "
        f"{best_validation_loss:.4f}"
    )

    return (
        model,
        pd.DataFrame(history_rows),
        np.sort(inner_train_indices),
        np.sort(validation_indices),
    )



# ============================================================
# RETRAIN MODEL FINAL PADA SELURUH 80 DATA TRAINING
# ============================================================
def retrain_final_model(
    X_normalized: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    selected_epochs: int,
) -> tuple[EOG1DCNN, pd.DataFrame]:
    X_train = torch.tensor(
        X_normalized[train_indices],
        dtype=torch.float32,
    )
    y_train = torch.tensor(
        y[train_indices],
        dtype=torch.long,
    )

    loader = DataLoader(
        TensorDataset(
            X_train,
            y_train,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(
            RANDOM_SEED
        ),
    )

    set_seed(RANDOM_SEED)
    model = EOG1DCNN()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_function = nn.CrossEntropyLoss()

    history_rows: list[dict[str, float]] = []

    for epoch in range(1, selected_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_function(
                logits,
                batch_y,
            )
            loss.backward()
            optimizer.step()

            total_loss += (
                float(loss.item())
                * len(batch_y)
            )

        mean_loss = (
            total_loss
            / len(train_indices)
        )

        model.eval()

        with torch.no_grad():
            predictions = (
                model(X_train)
                .argmax(dim=1)
                .cpu()
                .numpy()
            )

        accuracy = accuracy_score(
            y[train_indices],
            predictions,
        )

        history_rows.append({
            "epoch": float(epoch),
            "training_loss": float(mean_loss),
            "training_accuracy": float(accuracy),
        })

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == selected_epochs
        ):
            print(
                f"Final epoch {epoch:03d}/{selected_epochs} | "
                f"loss={mean_loss:.4f} | "
                f"acc={accuracy * 100:6.2f}%"
            )

    model.eval()

    return (
        model,
        pd.DataFrame(history_rows),
    )


# ============================================================
# EVALUASI
# ============================================================
def predict(
    model: nn.Module,
    X_normalized: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X_tensor = torch.tensor(
        X_normalized[indices],
        dtype=torch.float32,
    )

    model.eval()

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = (
            torch.softmax(logits, dim=1)
            .cpu()
            .numpy()
        )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    return predictions, probabilities


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    csv_path: Path,
    image_path: Path,
) -> None:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(len(CLASS_NAMES))
        ),
    )

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=[
            f"aktual_{name}"
            for name in CLASS_NAMES
        ],
        columns=[
            f"prediksi_{name}"
            for name in CLASS_NAMES
        ],
    )
    matrix_dataframe.to_csv(
        csv_path
    )

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )
    image = axis.imshow(matrix)

    axis.set_xticks(
        range(len(CLASS_NAMES))
    )
    axis.set_xticklabels(
        CLASS_NAMES
    )
    axis.set_yticks(
        range(len(CLASS_NAMES))
    )
    axis.set_yticklabels(
        CLASS_NAMES
    )
    axis.set_xlabel("Prediksi")
    axis.set_ylabel("Aktual")
    axis.set_title(title)

    for row in range(len(CLASS_NAMES)):
        for column in range(
            len(CLASS_NAMES)
        ):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    figure.colorbar(
        image,
        ax=axis,
    )
    figure.tight_layout()
    figure.savefig(
        image_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_history_plot(
    history: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.plot(
        history["epoch"],
        history["train_accuracy"] * 100,
        label="Training accuracy",
    )
    axis.plot(
        history["epoch"],
        history["validation_accuracy"] * 100,
        label="Validation accuracy",
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Akurasi (%)")
    axis.set_title(
        "Riwayat Training Model 5 Subjek"
    )
    axis.grid(
        True,
        alpha=0.30,
    )
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


# ============================================================
# MAIN
# ============================================================
def run(
    data_dir: Path,
    output_dir: Path,
    model_dir: Path,
    file_glob: str,
) -> None:
    start_time = time.time()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("TRAINING 5 SUBJEK — STRATIFIED 80:20")
    print("MODEL DILATIH DARI AWAL, BUKAN FINE-TUNING")
    print("=" * 76)

    (
        X,
        y,
        subjects,
        file_names,
        trial_numbers,
        audit_dataframe,
    ) = load_dataset(
        data_dir=data_dir,
        file_glob=file_glob,
    )

    audit_dataframe.to_csv(
        output_dir
        / "01_dataset_audit.csv",
        index=False,
    )

    train_indices, test_indices = (
        make_split(
            y=y,
            subjects=subjects,
        )
    )

    print_split_summary(
        "DATA TRAINING",
        train_indices,
        y,
        subjects,
    )
    print_split_summary(
        "DATA TESTING",
        test_indices,
        y,
        subjects,
    )

    partition = np.full(
        len(y),
        "train",
        dtype=object,
    )
    partition[test_indices] = "test"

    split_manifest = pd.DataFrame({
        "index": np.arange(len(y)),
        "subject_id": subjects,
        "file_name": file_names,
        "trial_ke": trial_numbers,
        "kelas": [
            CLASS_NAMES[label]
            for label in y
        ],
        "partition": partition,
    })

    split_manifest.to_csv(
        output_dir
        / "02_split_manifest_80_20.csv",
        index=False,
    )

    (
        X_normalized,
        normalization_mean,
        normalization_std,
    ) = normalize_from_training(
        X=X,
        train_indices=train_indices,
    )

    (
        selection_model,
        history,
        inner_train_indices,
        validation_indices,
    ) = train_model(
        X_normalized=X_normalized,
        y=y,
        train_indices=train_indices,
    )

    history.to_csv(
        output_dir
        / "03_epoch_selection_history.csv",
        index=False,
    )

    save_history_plot(
        history=history,
        output_path=(
            output_dir
            / "04_epoch_selection_history.png"
        ),
    )

    selected_epoch = int(
        history.loc[
            history["validation_loss"].idxmin(),
            "epoch",
        ]
    )

    print()
    print(
        f"Epoch terpilih dari validation: "
        f"{selected_epoch}"
    )
    print(
        "Model final dilatih ulang dari awal "
        "menggunakan seluruh 80 data training."
    )
    print()

    model, final_history = retrain_final_model(
        X_normalized=X_normalized,
        y=y,
        train_indices=train_indices,
        selected_epochs=selected_epoch,
    )

    final_history.to_csv(
        output_dir
        / "05_final_training_history_80_data.csv",
        index=False,
    )

    # Training accuracy dihitung pada seluruh 80 data training.
    train_predictions, train_probabilities = (
        predict(
            model=model,
            X_normalized=X_normalized,
            indices=train_indices,
        )
    )

    # Test accuracy dihitung pada 20 data test yang tidak dipakai training.
    test_predictions, test_probabilities = (
        predict(
            model=model,
            X_normalized=X_normalized,
            indices=test_indices,
        )
    )

    train_accuracy = accuracy_score(
        y[train_indices],
        train_predictions,
    )
    train_macro_f1 = f1_score(
        y[train_indices],
        train_predictions,
        average="macro",
        zero_division=0,
    )

    test_accuracy = accuracy_score(
        y[test_indices],
        test_predictions,
    )
    test_macro_f1 = f1_score(
        y[test_indices],
        test_predictions,
        average="macro",
        zero_division=0,
    )

    prediction_rows: list[dict[str, object]] = []

    for partition_name, indices, predictions, probabilities in [
        (
            "train",
            train_indices,
            train_predictions,
            train_probabilities,
        ),
        (
            "test",
            test_indices,
            test_predictions,
            test_probabilities,
        ),
    ]:
        for local_index, dataset_index in enumerate(
            indices
        ):
            row: dict[str, object] = {
                "partition": partition_name,
                "subject_id": subjects[
                    dataset_index
                ],
                "file_name": file_names[
                    dataset_index
                ],
                "trial_ke": int(
                    trial_numbers[dataset_index]
                ),
                "aktual": CLASS_NAMES[
                    y[dataset_index]
                ],
                "prediksi": CLASS_NAMES[
                    predictions[local_index]
                ],
                "benar": bool(
                    y[dataset_index]
                    == predictions[local_index]
                ),
                "confidence": float(
                    np.max(
                        probabilities[local_index]
                    )
                ),
            }

            for class_index, class_name in enumerate(
                CLASS_NAMES
            ):
                row[f"prob_{class_name}"] = float(
                    probabilities[
                        local_index,
                        class_index,
                    ]
                )

            prediction_rows.append(row)

    prediction_dataframe = pd.DataFrame(
        prediction_rows
    )
    prediction_dataframe.to_csv(
        output_dir
        / "06_predictions_train_test.csv",
        index=False,
    )

    save_confusion_matrix(
        y_true=y[train_indices],
        y_pred=train_predictions,
        title=(
            "Confusion Matrix Data Training "
            "(80 Trial)"
        ),
        csv_path=(
            output_dir
            / "07_confusion_matrix_train.csv"
        ),
        image_path=(
            output_dir
            / "08_confusion_matrix_train.png"
        ),
    )

    save_confusion_matrix(
        y_true=y[test_indices],
        y_pred=test_predictions,
        title=(
            "Confusion Matrix Data Testing "
            "(20 Trial)"
        ),
        csv_path=(
            output_dir
            / "09_confusion_matrix_test.csv"
        ),
        image_path=(
            output_dir
            / "10_confusion_matrix_test.png"
        ),
    )

    train_report = classification_report(
        y[train_indices],
        train_predictions,
        labels=list(
            range(len(CLASS_NAMES))
        ),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(
        train_report
    ).transpose().to_csv(
        output_dir
        / "11_classification_report_train.csv"
    )

    test_report = classification_report(
        y[test_indices],
        test_predictions,
        labels=list(
            range(len(CLASS_NAMES))
        ),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(
        test_report
    ).transpose().to_csv(
        output_dir
        / "12_classification_report_test.csv"
    )

    # Hasil per subjek pada data testing.
    test_subject_rows = []

    for subject in sorted(
        np.unique(subjects)
    ):
        subject_mask = (
            subjects[test_indices]
            == subject
        )

        subject_true = y[
            test_indices[subject_mask]
        ]
        subject_pred = test_predictions[
            subject_mask
        ]

        test_subject_rows.append({
            "subject_id": subject,
            "jumlah_test": int(
                len(subject_true)
            ),
            "benar": int(
                np.sum(
                    subject_true == subject_pred
                )
            ),
            "salah": int(
                np.sum(
                    subject_true != subject_pred
                )
            ),
            "test_accuracy": float(
                accuracy_score(
                    subject_true,
                    subject_pred,
                )
            ),
        })

    pd.DataFrame(
        test_subject_rows
    ).to_csv(
        output_dir
        / "13_test_metrics_per_subject.csv",
        index=False,
    )

    state_dict_path = (
        model_dir
        / "eog_5_subjek_80_20_state_dict.pth"
    )
    torchscript_path = (
        model_dir
        / "eog_5_subjek_80_20_torchscript.pt"
    )
    config_path = (
        model_dir
        / "model_config.json"
    )

    torch.save(
        model.state_dict(),
        state_dict_path,
    )

    example_input = torch.zeros(
        (1, 1, 500),
        dtype=torch.float32,
    )

    traced_model = torch.jit.trace(
        model,
        example_input,
    )
    traced_model.save(
        str(torchscript_path)
    )

    parameter_count = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
        )
    )

    elapsed = time.time() - start_time

    config = {
        "model_name": (
            "EOG1DCNN_5_subjects_80_20"
        ),
        "training_type": (
            "train_from_scratch; not fine-tuning; "
            "epoch selected using internal validation, then retrained on all 80 training trials"
        ),
        "split_method": (
            "stratified random 80:20 by "
            "subject-class combination"
        ),
        "important_note": (
            "same five subjects appear in train and test; "
            "this is not subject-independent evaluation"
        ),
        "subjects": sorted(
            np.unique(subjects).tolist()
        ),
        "training_subjects": sorted(
            np.unique(subjects).tolist()
        ),
        "class_order": CLASS_NAMES,
        "total_trials": int(len(X)),
        "train_trials": int(
            len(train_indices)
        ),
        "test_trials": int(
            len(test_indices)
        ),
        "inner_train_trials": int(
            len(inner_train_indices)
        ),
        "validation_trials_for_epoch_selection": int(
            len(validation_indices)
        ),
        "selected_final_epochs": int(selected_epoch),
        "training_accuracy": float(
            train_accuracy
        ),
        "training_macro_f1": float(
            train_macro_f1
        ),
        "test_accuracy": float(
            test_accuracy
        ),
        "test_macro_f1": float(
            test_macro_f1
        ),
        "normalization_mean": float(
            normalization_mean
        ),
        "normalization_std": float(
            normalization_std
        ),
        "raw_sampling_rate_hz": RAW_FS,
        "processed_sampling_rate_hz": (
            PROCESSED_FS
        ),
        "lowpass_hz": LOWPASS_HZ,
        "filter_order": FILTER_ORDER,
        "window_samples": 500,
        "processed_samples_per_window": 500,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "random_seed": RANDOM_SEED,
        "trainable_parameters": (
            parameter_count
        ),
        "state_dict_sha256": sha256_file(
            state_dict_path
        ),
        "torchscript_sha256": sha256_file(
            torchscript_path
        ),
        "processing_time_seconds": float(
            elapsed
        ),
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            config,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    summary = {
        "dataset_total": int(len(X)),
        "data_training": int(
            len(train_indices)
        ),
        "data_testing": int(
            len(test_indices)
        ),
        "training_benar": int(
            np.sum(
                y[train_indices]
                == train_predictions
            )
        ),
        "training_salah": int(
            np.sum(
                y[train_indices]
                != train_predictions
            )
        ),
        "training_accuracy": float(
            train_accuracy
        ),
        "training_macro_f1": float(
            train_macro_f1
        ),
        "testing_benar": int(
            np.sum(
                y[test_indices]
                == test_predictions
            )
        ),
        "testing_salah": int(
            np.sum(
                y[test_indices]
                != test_predictions
            )
        ),
        "testing_accuracy": float(
            test_accuracy
        ),
        "testing_macro_f1": float(
            test_macro_f1
        ),
    }

    with (
        output_dir
        / "14_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 76)
    print("HASIL AKHIR")
    print("=" * 76)
    print(
        f"Dataset total      : {len(X)} trial"
    )
    print(
        f"Data training      : "
        f"{len(train_indices)} trial"
    )
    print(
        f"Data testing       : "
        f"{len(test_indices)} trial"
    )
    print("-" * 76)
    print(
        f"Training benar     : "
        f"{summary['training_benar']}"
    )
    print(
        f"Training salah     : "
        f"{summary['training_salah']}"
    )
    print(
        f"TRAINING ACCURACY  : "
        f"{train_accuracy * 100:.2f}%"
    )
    print(
        f"Training Macro F1  : "
        f"{train_macro_f1 * 100:.2f}%"
    )
    print("-" * 76)
    print(
        f"Testing benar      : "
        f"{summary['testing_benar']}"
    )
    print(
        f"Testing salah      : "
        f"{summary['testing_salah']}"
    )
    print(
        f"TEST ACCURACY      : "
        f"{test_accuracy * 100:.2f}%"
    )
    print(
        f"Test Macro F1      : "
        f"{test_macro_f1 * 100:.2f}%"
    )
    print("-" * 76)
    print("Model              :", torchscript_path)
    print("Output             :", output_dir)
    print("=" * 76)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a 1D-CNN for four-class EOG classification using "
            "a stratified 80:20 split."
        )
    )

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent if script_dir.name == "src" else script_dir

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "training",
        help="Directory containing the five training-subject CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "training",
        help="Directory for training reports and figures.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_root / "models" / "eog_5_subjek_80_20",
        help="Directory for the trained state_dict, TorchScript model, and config.",
    )
    parser.add_argument(
        "--file-glob",
        type=str,
        default="*.csv",
    )

    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()

    run(
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        model_dir=arguments.model_dir,
        file_glob=arguments.file_glob,
    )


if __name__ == "__main__":
    main()
