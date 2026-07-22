#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Real-Time EOG Evaluation on Unseen Subjects
============================================

Runs the trained TorchScript 1D-CNN on new participants using the same
2 s baseline, 2 s action, 2 s return, and 5 s rest protocol used during
data acquisition. Per-trial output contains the actual label, prediction,
confidence, correctness flag, and inference time.

Expected model files:
    models/eog_5_subjek_80_20/eog_5_subjek_80_20_torchscript.pt
    models/eog_5_subjek_80_20/model_config.json

Example:
    python src/evaluate_realtime.py --subject E01 --mac YOUR_BITALINO_MAC

The raw participant recordings are written under outputs/ and should not be
committed to a public repository.
"""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
import argparse
import os
import csv
import json
import random
import re
import threading
import time
import traceback
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, sosfilt, sosfilt_zi
from sklearn.metrics import f1_score

try:
    from bitalino import BITalino
except ImportError as exc:
    raise SystemExit(
        "Library bitalino belum terpasang.\n"
        "Install ke Python yang dipakai menjalankan script:\n"
        "python -m pip install bitalino"
    ) from exc


# =========================================================
# 1. PATH DAN ARGUMEN
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "eog_5_subjek_80_20"
)
MODEL_PATH = (
    MODEL_DIR
    / "eog_5_subjek_80_20_torchscript.pt"
)
CONFIG_PATH = (
    MODEL_DIR
    / "model_config.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "evaluasi_realtime_5_subjek"
)

DEFAULT_MAC_ADDRESS = os.getenv("BITALINO_MAC", "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluasi realtime model EOG 1D-CNN "
            "pada subjek baru."
        )
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="",
        help="Kode subjek evaluasi, contoh E01.",
    )
    parser.add_argument(
        "--mac",
        type=str,
        default=DEFAULT_MAC_ADDRESS,
        help="BITalino MAC address. Can also be set with BITALINO_MAC.",
    )
    return parser.parse_args()


ARGS = parse_args()
MAC_ADDRESS = ARGS.mac.strip()
if not MAC_ADDRESS:
    raise SystemExit(
        "BITalino MAC address is required. Use --mac YOUR_BITALINO_MAC "
        "or set the BITALINO_MAC environment variable."
    )


# =========================================================
# 2. KONFIGURASI PROTOKOL
# =========================================================

RAW_FS = 1000
CHANNELS = [0]  # BITalino A1
ANALOG_START_COLUMN = 5
READ_BLOCK_SAMPLES = 50

SCHEDULE_CLASSES = [
    "kiri",
    "kanan",
    "kedip",
    "diam",
]

REPETITIONS_PER_CLASS = 5
TOTAL_TRIALS = (
    REPETITIONS_PER_CLASS
    * len(SCHEDULE_CLASSES)
)

COUNTDOWN_SEC = 3.0
BASELINE_SEC = 2.0
ACTION_SEC = 2.0
RETURN_SEC = 2.0
REST_SEC = 5.0
TRIAL_SEC = (
    BASELINE_SEC
    + ACTION_SEC
    + RETURN_SEC
    + REST_SEC
)

BASELINE_SAMPLES = int(BASELINE_SEC * RAW_FS)
ACTION_SAMPLES = int(ACTION_SEC * RAW_FS)
EXPECTED_SESSION_SAMPLES = int(
    TOTAL_TRIALS
    * TRIAL_SEC
    * RAW_FS
)

DISPLAY_SEC = 4.0
DISPLAY_SAMPLES = int(DISPLAY_SEC * RAW_FS)


# =========================================================
# 3. LOAD MODEL DAN CONFIG
# =========================================================

def load_model_and_config() -> tuple[
    torch.jit.ScriptModule,
    dict[str, Any],
]:
    if not MODEL_PATH.exists():
        raise SystemExit(
            "Model TorchScript tidak ditemukan:\n"
            f"{MODEL_PATH}"
        )

    if not CONFIG_PATH.exists():
        raise SystemExit(
            "model_config.json tidak ditemukan:\n"
            f"{CONFIG_PATH}"
        )

    config = json.loads(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    required_keys = [
        "class_order",
        "normalization_mean",
        "normalization_std",
        "lowpass_hz",
        "filter_order",
    ]

    missing = [
        key
        for key in required_keys
        if key not in config
    ]

    if missing:
        raise SystemExit(
            "Config model tidak lengkap. "
            f"Key yang hilang: {missing}"
        )

    raw_sampling_rate = int(
        config.get(
            "raw_sampling_rate_hz",
            RAW_FS,
        )
    )
    processed_sampling_rate = int(
        config.get(
            "processed_sampling_rate_hz",
            250,
        )
    )

    if raw_sampling_rate != RAW_FS:
        raise SystemExit(
            "Sampling rate model tidak sesuai.\n"
            f"Model: {raw_sampling_rate} Hz\n"
            f"Script: {RAW_FS} Hz"
        )

    if (
        processed_sampling_rate <= 0
        or raw_sampling_rate
        % processed_sampling_rate
        != 0
    ):
        raise SystemExit(
            "Sampling rate processed pada config tidak valid."
        )

    inferred_downsample = (
        raw_sampling_rate
        // processed_sampling_rate
    )
    downsample_factor = int(
        config.get(
            "downsample_factor",
            inferred_downsample,
        )
    )

    window_samples = int(
        config.get(
            "processed_samples_per_window",
            config.get(
                "window_samples",
                ACTION_SAMPLES
                // downsample_factor,
            ),
        )
    )

    if downsample_factor != inferred_downsample:
        raise SystemExit(
            "Downsample factor tidak sesuai.\n"
            f"Ditemukan: {downsample_factor}\n"
            f"Seharusnya: {inferred_downsample}"
        )

    expected_window = (
        ACTION_SAMPLES
        // downsample_factor
    )

    if window_samples != expected_window:
        raise SystemExit(
            "Ukuran input model tidak sesuai.\n"
            f"Config: {window_samples}\n"
            f"Seharusnya: {expected_window}"
        )

    normalization_std = float(
        config["normalization_std"]
    )

    if (
        not np.isfinite(normalization_std)
        or normalization_std <= 0
    ):
        raise SystemExit(
            "normalization_std tidak valid."
        )

    config["downsample_factor"] = (
        downsample_factor
    )
    config["processed_samples_per_window"] = (
        window_samples
    )
    config["training_subjects"] = (
        config.get(
            "training_subjects",
            config.get(
                "subjects",
                [],
            ),
        )
    )

    model = torch.jit.load(
        str(MODEL_PATH),
        map_location="cpu",
    )
    model.eval()

    with torch.inference_mode():
        test_output = model(
            torch.zeros(
                (
                    1,
                    1,
                    window_samples,
                ),
                dtype=torch.float32,
            )
        )

    class_order = list(
        config["class_order"]
    )

    if tuple(test_output.shape) != (
        1,
        len(class_order),
    ):
        raise SystemExit(
            "Output model tidak sesuai.\n"
            f"Ditemukan: {tuple(test_output.shape)}\n"
            f"Seharusnya: (1, {len(class_order)})"
        )

    return model, config


MODEL, MODEL_CONFIG = (
    load_model_and_config()
)
MODEL_CLASSES = list(
    MODEL_CONFIG["class_order"]
)

if set(MODEL_CLASSES) != set(
    SCHEDULE_CLASSES
):
    raise SystemExit(
        "Daftar kelas model tidak sesuai.\n"
        f"Model: {MODEL_CLASSES}\n"
        f"Protokol: {SCHEDULE_CLASSES}"
    )

MODEL_LOWPASS_HZ = float(
    MODEL_CONFIG["lowpass_hz"]
)
MODEL_FILTER_ORDER = int(
    MODEL_CONFIG["filter_order"]
)
DOWNSAMPLE_FACTOR = int(
    MODEL_CONFIG["downsample_factor"]
)
WINDOW_SAMPLES = int(
    MODEL_CONFIG[
        "processed_samples_per_window"
    ]
)
NORMALIZATION_MEAN = float(
    MODEL_CONFIG["normalization_mean"]
)
NORMALIZATION_STD = float(
    MODEL_CONFIG["normalization_std"]
)

TRAINING_SUBJECTS = {
    str(value).strip().upper()
    for value in MODEL_CONFIG.get(
        "training_subjects",
        [],
    )
}


# =========================================================
# 4. KODE SUBJEK DAN FILE OUTPUT
# =========================================================

def ask_subject_code(
    initial_value: str,
) -> str:
    print("=" * 72)
    print("EVALUASI REALTIME EOG 1D-CNN")
    print("=" * 72)
    print("Contoh kode subjek evaluasi: E01")
    print()

    candidate = (
        initial_value
        .strip()
        .upper()
    )

    while True:
        if not candidate:
            candidate = (
                input(
                    "Masukkan kode subjek evaluasi: "
                )
                .strip()
                .upper()
            )

        if not re.fullmatch(
            r"[A-Z]\d{2}(R\d+)?",
            candidate,
        ):
            print(
                "Format tidak sesuai. "
                "Gunakan contoh E01 atau E01R2."
            )
            candidate = ""
            continue

        if candidate in TRAINING_SUBJECTS:
            print(
                f"{candidate} termasuk subjek training. "
                "Gunakan subjek baru."
            )
            candidate = ""
            continue

        return candidate


SUBJECT_CODE = ask_subject_code(
    ARGS.subject
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_FINAL_PATH = (
    OUTPUT_ROOT
    / f"{SUBJECT_CODE}_raw.csv"
)
RAW_TEMP_PATH = (
    OUTPUT_ROOT
    / f"{SUBJECT_CODE}_raw_sedang_rekam.tmp"
)

RESULT_FINAL_PATH = (
    OUTPUT_ROOT
    / f"{SUBJECT_CODE}_evaluasi_per_trial.csv"
)
RESULT_TEMP_PATH = (
    OUTPUT_ROOT
    / f"{SUBJECT_CODE}_evaluasi_sedang_rekam.tmp"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / f"{SUBJECT_CODE}_ringkasan.csv"
)

existing_paths = [
    RAW_FINAL_PATH,
    RESULT_FINAL_PATH,
    SUMMARY_PATH,
]

existing = [
    path
    for path in existing_paths
    if path.exists()
]

if existing:
    formatted = "\n".join(
        str(path)
        for path in existing
    )
    raise SystemExit(
        "File evaluasi subjek ini sudah ada "
        "dan tidak akan ditimpa:\n"
        f"{formatted}\n\n"
        "Gunakan kode pengulangan, misalnya E01R2."
    )

for temporary_path in [
    RAW_TEMP_PATH,
    RESULT_TEMP_PATH,
]:
    if temporary_path.exists():
        try:
            temporary_path.unlink()
        except OSError as exc:
            raise SystemExit(
                "Tidak dapat menghapus file sementara:\n"
                f"{temporary_path}\n{exc}"
            ) from exc


# =========================================================
# 5. FILTER
# =========================================================

DISPLAY_SOS = butter(
    MODEL_FILTER_ORDER,
    MODEL_LOWPASS_HZ,
    btype="lowpass",
    fs=RAW_FS,
    output="sos",
)

MODEL_SOS = butter(
    MODEL_FILTER_ORDER,
    MODEL_LOWPASS_HZ,
    btype="lowpass",
    fs=RAW_FS,
    output="sos",
)


# =========================================================
# 6. STATE PROGRAM
# =========================================================

lock = threading.RLock()
stop_event = threading.Event()

device = None
reader_thread = None

absolute_sample = 0
last_sequence = None

display_filter_zi = None
model_filter_zi = None

raw_plot_buffer: list[float] = []
filtered_plot_buffer: list[float] = []
time_plot_buffer: list[float] = []

raw_csv_handle = None
raw_csv_writer = None
result_csv_handle = None
result_csv_writer = None
samples_since_flush = 0

trial_buffers: dict[
    int,
    dict[str, list[float]],
] = {}

evaluation_rows: list[
    dict[str, object]
] = []

state = {
    "running": True,
    "connected": False,
    "reader_error": None,

    "countdown_active": False,
    "countdown_end": None,

    "session_active": False,
    "session_finished": False,
    "session_aborted": False,
    "session_saved": False,
    "completed_once": False,

    "session_start_absolute_sample": None,
    "session_samples_written": 0,

    "schedule": [],
    "schedule_end_samples": [],
    "current_segment": None,

    "packet_gap_events": 0,
    "estimated_missing_samples": 0,
    "clipped_a1_samples": 0,

    "last_result": None,
    "final_summary": None,

    "message": "Tekan C untuk menghubungkan BITalino.",
}


# =========================================================
# 7. JADWAL 20 TRIAL
# =========================================================

def build_schedule() -> list[
    dict[str, object]
]:
    schedule: list[
        dict[str, object]
    ] = []

    offset = 0
    trial_id = 0
    secure_random = random.SystemRandom()

    phases = [
        ("baseline", BASELINE_SEC),
        ("action", ACTION_SEC),
        ("return", RETURN_SEC),
        ("rest", REST_SEC),
    ]

    for group_number in range(
        1,
        REPETITIONS_PER_CLASS + 1,
    ):
        labels = SCHEDULE_CLASSES.copy()
        secure_random.shuffle(labels)

        for label in labels:
            trial_id += 1

            for phase, duration_sec in phases:
                duration_samples = int(
                    round(
                        duration_sec
                        * RAW_FS
                    )
                )

                schedule.append({
                    "trial_id": trial_id,
                    "group": group_number,
                    "target_label": label,
                    "phase": phase,
                    "start_sample": offset,
                    "end_sample_exclusive": (
                        offset
                        + duration_samples
                    ),
                })

                offset += duration_samples

    return schedule


def get_segment(
    session_sample: int,
    schedule: list[
        dict[str, object]
    ],
    end_samples: list[int],
) -> dict[str, object] | None:
    if (
        session_sample < 0
        or not schedule
    ):
        return None

    if session_sample >= end_samples[-1]:
        return None

    index = bisect_right(
        end_samples,
        session_sample,
    )
    return schedule[index]


# =========================================================
# 8. FILE OUTPUT
# =========================================================

RAW_COLUMNS = [
    "waktu_detik",
    "sinyal_a1",
    "status",
    "trial_ke",
]

RESULT_COLUMNS = [
    "subject_id",
    "trial_ke",
    "aktual",
    "prediksi",
    "confidence",
    "benar",
    "inference_ms",
]

SUMMARY_COLUMNS = [
    "subject_id",
    "total_trial",
    "benar",
    "salah",
    "accuracy",
    "macro_f1",
    "rata_rata_confidence",
    "rata_rata_inference_ms",
]


def open_temporary_files() -> None:
    global raw_csv_handle
    global raw_csv_writer
    global result_csv_handle
    global result_csv_writer

    raw_csv_handle = RAW_TEMP_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
        buffering=1,
    )
    raw_csv_writer = csv.DictWriter(
        raw_csv_handle,
        fieldnames=RAW_COLUMNS,
    )
    raw_csv_writer.writeheader()

    result_csv_handle = (
        RESULT_TEMP_PATH.open(
            "w",
            newline="",
            encoding="utf-8",
            buffering=1,
        )
    )
    result_csv_writer = csv.DictWriter(
        result_csv_handle,
        fieldnames=RESULT_COLUMNS,
    )
    result_csv_writer.writeheader()


def close_files() -> None:
    global raw_csv_handle
    global raw_csv_writer
    global result_csv_handle
    global result_csv_writer
    global samples_since_flush

    for handle in [
        raw_csv_handle,
        result_csv_handle,
    ]:
        if handle is not None:
            try:
                handle.flush()
            except Exception:
                pass

            try:
                handle.close()
            except Exception:
                pass

    raw_csv_handle = None
    raw_csv_writer = None
    result_csv_handle = None
    result_csv_writer = None
    samples_since_flush = 0


def remove_temporary_files() -> None:
    close_files()

    for path in [
        RAW_TEMP_PATH,
        RESULT_TEMP_PATH,
    ]:
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def calculate_summary() -> dict[
    str,
    object
]:
    actual = [
        str(row["aktual"])
        for row in evaluation_rows
    ]
    predicted = [
        str(row["prediksi"])
        for row in evaluation_rows
    ]

    correct_count = int(
        sum(
            bool(row["benar"])
            for row in evaluation_rows
        )
    )
    total = len(evaluation_rows)
    wrong_count = total - correct_count

    accuracy = (
        correct_count / total
        if total > 0
        else 0.0
    )

    macro_f1 = (
        float(
            f1_score(
                actual,
                predicted,
                labels=MODEL_CLASSES,
                average="macro",
                zero_division=0,
            )
        )
        if total > 0
        else 0.0
    )

    mean_confidence = (
        float(
            np.mean([
                float(row["confidence"])
                for row in evaluation_rows
            ])
        )
        if total > 0
        else 0.0
    )

    mean_inference = (
        float(
            np.mean([
                float(row["inference_ms"])
                for row in evaluation_rows
            ])
        )
        if total > 0
        else 0.0
    )

    return {
        "subject_id": SUBJECT_CODE,
        "total_trial": total,
        "benar": correct_count,
        "salah": wrong_count,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "rata_rata_confidence": (
            mean_confidence
        ),
        "rata_rata_inference_ms": (
            mean_inference
        ),
    }


def save_summary(
    summary: dict[str, object],
) -> None:
    with SUMMARY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_COLUMNS,
        )
        writer.writeheader()
        writer.writerow(summary)


def finalize_session() -> None:
    with lock:
        if state["session_saved"]:
            return

        aborted = bool(
            state["session_aborted"]
        )
        written = int(
            state[
                "session_samples_written"
            ]
        )
        result_count = len(
            evaluation_rows
        )
        state["session_saved"] = True

    close_files()

    complete = bool(
        not aborted
        and written
        == EXPECTED_SESSION_SAMPLES
        and result_count
        == TOTAL_TRIALS
    )

    if not complete:
        for path in [
            RAW_TEMP_PATH,
            RESULT_TEMP_PATH,
        ]:
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

        with lock:
            state["session_aborted"] = True
            state["message"] = (
                "Sesi tidak lengkap. "
                "Data parsial dibuang dan harus diulang."
            )

        print()
        print("SESI TIDAK LENGKAP")
        print("Sampel tersimpan :", written)
        print("Prediksi selesai :", result_count)
        print("Data parsial tidak disimpan.")
        return

    RAW_TEMP_PATH.replace(
        RAW_FINAL_PATH
    )
    RESULT_TEMP_PATH.replace(
        RESULT_FINAL_PATH
    )

    summary = calculate_summary()
    save_summary(summary)

    with lock:
        state["completed_once"] = True
        state["final_summary"] = summary
        state["message"] = (
            "Sesi selesai. "
            "Hasil evaluasi telah disimpan."
        )

    print()
    print("=" * 72)
    print("HASIL EVALUASI REALTIME")
    print("=" * 72)
    print("Subjek       :", SUBJECT_CODE)
    print(
        "Total trial  :",
        summary["total_trial"],
    )
    print(
        "Benar        :",
        summary["benar"],
    )
    print(
        "Salah        :",
        summary["salah"],
    )
    print(
        "Accuracy     :",
        f"{float(summary['accuracy']) * 100:.2f}%",
    )
    print(
        "Macro F1     :",
        f"{float(summary['macro_f1']) * 100:.2f}%",
    )
    print(
        "Mean conf.   :",
        f"{float(summary['rata_rata_confidence']):.3f}",
    )
    print(
        "Mean infer.  :",
        f"{float(summary['rata_rata_inference_ms']):.3f} ms",
    )
    print("Raw          :", RAW_FINAL_PATH)
    print("Per trial    :", RESULT_FINAL_PATH)
    print("Ringkasan    :", SUMMARY_PATH)
    print("=" * 72)


# =========================================================
# 9. PREPROCESSING DAN PREDIKSI
# =========================================================

def predict_trial(
    trial_id: int,
    actual_label: str,
) -> dict[str, object]:
    buffer = trial_buffers.get(
        trial_id
    )

    if buffer is None:
        raise RuntimeError(
            f"Buffer trial {trial_id} tidak ditemukan."
        )

    baseline = np.asarray(
        buffer["baseline"],
        dtype=np.float64,
    )
    action = np.asarray(
        buffer["action"],
        dtype=np.float64,
    )

    if baseline.shape != (
        BASELINE_SAMPLES,
    ):
        raise RuntimeError(
            f"Trial {trial_id}: baseline "
            f"{baseline.shape}, harus "
            f"({BASELINE_SAMPLES},)."
        )

    if action.shape != (
        ACTION_SAMPLES,
    ):
        raise RuntimeError(
            f"Trial {trial_id}: aksi "
            f"{action.shape}, harus "
            f"({ACTION_SAMPLES},)."
        )

    baseline_median = float(
        np.median(baseline)
    )
    corrected = (
        action
        - baseline_median
    )
    processed = corrected[
        ::DOWNSAMPLE_FACTOR
    ]

    if processed.shape != (
        WINDOW_SAMPLES,
    ):
        raise RuntimeError(
            f"Trial {trial_id}: window "
            f"{processed.shape}, harus "
            f"({WINDOW_SAMPLES},)."
        )

    normalized = (
        processed
        - NORMALIZATION_MEAN
    ) / NORMALIZATION_STD

    if not np.isfinite(
        normalized
    ).all():
        raise RuntimeError(
            f"Trial {trial_id}: hasil "
            "normalisasi mengandung NaN/Inf."
        )

    model_input = torch.tensor(
        normalized[
            np.newaxis,
            np.newaxis,
            :,
        ],
        dtype=torch.float32,
    )

    start_time = time.perf_counter()

    with torch.inference_mode():
        logits = MODEL(
            model_input
        )
        probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )
            .cpu()
            .numpy()[0]
        )

    inference_ms = (
        time.perf_counter()
        - start_time
    ) * 1000.0

    prediction_index = int(
        np.argmax(probabilities)
    )
    predicted_label = (
        MODEL_CLASSES[
            prediction_index
        ]
    )
    confidence = float(
        probabilities[
            prediction_index
        ]
    )
    correct = bool(
        predicted_label
        == actual_label
    )

    result = {
        "subject_id": SUBJECT_CODE,
        "trial_ke": trial_id,
        "aktual": actual_label,
        "prediksi": predicted_label,
        "confidence": confidence,
        "benar": correct,
        "inference_ms": (
            inference_ms
        ),
    }

    evaluation_rows.append(
        result
    )

    if result_csv_writer is None:
        raise RuntimeError(
            "File hasil evaluasi belum dibuka."
        )

    result_csv_writer.writerow(
        result
    )

    if result_csv_handle is not None:
        result_csv_handle.flush()

    print(
        f"Trial {trial_id:02d} | "
        f"aktual={actual_label:6s} | "
        f"prediksi={predicted_label:6s} | "
        f"confidence={confidence:.3f} | "
        f"{'BENAR' if correct else 'SALAH'}"
    )

    return result


# =========================================================
# 10. BUFFER TAMPILAN
# =========================================================

def append_plot_data(
    time_value: float,
    raw_value: float,
    filtered_value: float,
) -> None:
    raw_plot_buffer.append(
        float(raw_value)
    )
    filtered_plot_buffer.append(
        float(filtered_value)
    )
    time_plot_buffer.append(
        float(time_value)
    )

    overflow = (
        len(raw_plot_buffer)
        - DISPLAY_SAMPLES
    )

    if overflow > 0:
        del raw_plot_buffer[:overflow]
        del filtered_plot_buffer[:overflow]
        del time_plot_buffer[:overflow]


# =========================================================
# 11. PEMBACAAN BITALINO
# =========================================================

def acquisition_worker() -> None:
    global absolute_sample
    global last_sequence
    global display_filter_zi
    global model_filter_zi
    global samples_since_flush

    try:
        while not stop_event.is_set():
            block = np.asarray(
                device.read(
                    READ_BLOCK_SAMPLES
                )
            )

            if (
                block.ndim != 2
                or block.shape[1]
                <= ANALOG_START_COLUMN
            ):
                raise RuntimeError(
                    "Format data BITalino "
                    f"tidak sesuai: {block.shape}"
                )

            sequence_values = block[
                :,
                0,
            ].astype(int)
            a1_block = block[
                :,
                ANALOG_START_COLUMN,
            ].astype(float)

            if display_filter_zi is None:
                display_filter_zi = (
                    sosfilt_zi(
                        DISPLAY_SOS
                    )
                    * a1_block[0]
                )

            (
                display_filtered_block,
                display_filter_zi,
            ) = sosfilt(
                DISPLAY_SOS,
                a1_block,
                zi=display_filter_zi,
            )

            # Filter model memakai state terpisah dari filter tampilan.
            # State di-reset saat sesi dimulai agar preprocessing sama
            # dengan training yang memfilter satu file dari awal.
            with lock:
                if model_filter_zi is None:
                    model_filter_zi = (
                        sosfilt_zi(
                            MODEL_SOS
                        )
                        * a1_block[0]
                    )

                (
                    model_filtered_block,
                    model_filter_zi,
                ) = sosfilt(
                    MODEL_SOS,
                    a1_block,
                    zi=model_filter_zi,
                )

                for row_index in range(
                    len(block)
                ):
                    sequence = int(
                        sequence_values[
                            row_index
                        ]
                    )

                    if last_sequence is not None:
                        expected = (
                            last_sequence + 1
                        ) % 16

                        if (
                            sequence != expected
                            and state[
                                "session_active"
                            ]
                        ):
                            missing_before = (
                                sequence
                                - expected
                            ) % 16

                            if missing_before == 0:
                                missing_before = 16

                            state[
                                "packet_gap_events"
                            ] += 1
                            state[
                                "estimated_missing_samples"
                            ] += missing_before

                    last_sequence = sequence

                    raw_a1 = int(
                        round(
                            float(
                                a1_block[
                                    row_index
                                ]
                            )
                        )
                    )
                    display_filtered_a1 = float(
                        display_filtered_block[
                            row_index
                        ]
                    )
                    sample_time = (
                        absolute_sample
                        / RAW_FS
                    )

                    append_plot_data(
                        sample_time,
                        raw_a1,
                        display_filtered_a1,
                    )

                    if state["session_active"]:
                        model_filtered_a1 = float(
                            model_filtered_block[
                                row_index
                            ]
                        )

                        session_sample = (
                            absolute_sample
                            - state[
                                "session_start_absolute_sample"
                            ]
                        )

                        segment = get_segment(
                            session_sample,
                            state["schedule"],
                            state[
                                "schedule_end_samples"
                            ],
                        )

                        if segment is None:
                            state[
                                "session_active"
                            ] = False
                            state[
                                "session_finished"
                            ] = True
                            state[
                                "current_segment"
                            ] = None
                            state["message"] = (
                                "Perekaman selesai. "
                                "Menyimpan hasil."
                            )

                        else:
                            phase = str(
                                segment["phase"]
                            )
                            trial_id = int(
                                segment[
                                    "trial_id"
                                ]
                            )
                            actual_label = str(
                                segment[
                                    "target_label"
                                ]
                            )

                            if phase == "baseline":
                                status = (
                                    "tengah_awal"
                                )
                            elif phase == "return":
                                status = (
                                    "tengah_kembali"
                                )
                            elif phase == "rest":
                                status = "jeda"
                            else:
                                status = (
                                    actual_label
                                )

                            if raw_csv_writer is None:
                                raise RuntimeError(
                                    "File raw belum dibuka."
                                )

                            raw_csv_writer.writerow({
                                "waktu_detik": (
                                    f"{session_sample / RAW_FS:.3f}"
                                ),
                                "sinyal_a1": raw_a1,
                                "status": status,
                                "trial_ke": trial_id,
                            })

                            state[
                                "session_samples_written"
                            ] += 1
                            samples_since_flush += 1

                            if (
                                samples_since_flush
                                >= RAW_FS
                            ):
                                if (
                                    raw_csv_handle
                                    is not None
                                ):
                                    raw_csv_handle.flush()

                                samples_since_flush = 0

                            if (
                                raw_a1 <= 1
                                or raw_a1 >= 1022
                            ):
                                state[
                                    "clipped_a1_samples"
                                ] += 1

                            trial_buffer = (
                                trial_buffers.setdefault(
                                    trial_id,
                                    {
                                        "baseline": [],
                                        "action": [],
                                    },
                                )
                            )

                            if phase == "baseline":
                                trial_buffer[
                                    "baseline"
                                ].append(
                                    model_filtered_a1
                                )
                            elif phase == "action":
                                trial_buffer[
                                    "action"
                                ].append(
                                    model_filtered_a1
                                )

                            previous = state[
                                "current_segment"
                            ]

                            changed = bool(
                                previous is None
                                or int(
                                    previous[
                                        "trial_id"
                                    ]
                                )
                                != trial_id
                                or str(
                                    previous["phase"]
                                )
                                != phase
                            )

                            # Prediksi dibuat sekali, tepat saat
                            # fase action selesai dan masuk return.
                            if (
                                changed
                                and previous is not None
                                and str(
                                    previous[
                                        "phase"
                                    ]
                                )
                                == "action"
                                and phase == "return"
                                and int(
                                    previous[
                                        "trial_id"
                                    ]
                                )
                                == trial_id
                            ):
                                result = predict_trial(
                                    trial_id=trial_id,
                                    actual_label=actual_label,
                                )
                                state[
                                    "last_result"
                                ] = result

                            state[
                                "current_segment"
                            ] = segment.copy()

                            if changed:
                                play_cue_sound(
                                    phase
                                )

                    absolute_sample += 1

    except Exception as exc:
        traceback.print_exc()

        with lock:
            state["reader_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            state["connected"] = False

            if state["session_active"]:
                state[
                    "session_active"
                ] = False
                state[
                    "session_finished"
                ] = True
                state[
                    "session_aborted"
                ] = True

            state["message"] = (
                "Pembacaan BITalino gagal."
            )


# =========================================================
# 12. SUARA CUE
# =========================================================

def play_cue_sound(
    phase: str,
) -> None:
    try:
        import winsound

        if phase == "action":
            frequency = 1000
            duration_ms = 140
        elif phase == "rest":
            frequency = 500
            duration_ms = 100
        else:
            frequency = 700
            duration_ms = 80

        threading.Thread(
            target=winsound.Beep,
            args=(
                frequency,
                duration_ms,
            ),
            daemon=True,
        ).start()

    except Exception:
        pass


# =========================================================
# 13. CONNECT DAN DISCONNECT
# =========================================================

def connect_device() -> None:
    global device
    global reader_thread
    global absolute_sample
    global last_sequence
    global display_filter_zi

    with lock:
        if state["connected"]:
            state["message"] = (
                "BITalino sudah terhubung."
            )
            return

    try:
        print(
            f"Menghubungkan ke "
            f"{MAC_ADDRESS} ..."
        )

        stop_event.clear()
        absolute_sample = 0
        last_sequence = None
        display_filter_zi = None

        raw_plot_buffer.clear()
        filtered_plot_buffer.clear()
        time_plot_buffer.clear()

        device = BITalino(
            MAC_ADDRESS
        )
        device.start(
            RAW_FS,
            CHANNELS,
        )

        reader_thread = threading.Thread(
            target=acquisition_worker,
            daemon=True,
        )
        reader_thread.start()

        with lock:
            state["connected"] = True
            state["reader_error"] = None
            state["message"] = (
                "BITalino terhubung. "
                "Tekan S untuk mulai."
            )

        print("BITalino terhubung.")

    except Exception as exc:
        device = None
        traceback.print_exc()

        with lock:
            state["connected"] = False
            state["message"] = (
                f"Koneksi gagal: {exc}"
            )


def disconnect_device() -> None:
    global device
    global reader_thread

    stop_event.set()

    if (
        reader_thread is not None
        and reader_thread.is_alive()
    ):
        reader_thread.join(
            timeout=2.0
        )

    if device is not None:
        try:
            device.stop()
        except Exception:
            pass

        try:
            device.close()
        except Exception:
            pass

    device = None
    reader_thread = None

    with lock:
        state["connected"] = False


# =========================================================
# 14. MULAI DAN BATALKAN SESI
# =========================================================

def request_start_session() -> None:
    with lock:
        if not state["connected"]:
            state["message"] = (
                "BITalino belum terhubung. "
                "Tekan C."
            )
            return

        if state["completed_once"]:
            state["message"] = (
                "Evaluasi sudah selesai. "
                "Tekan Q."
            )
            return

        if (
            state["session_active"]
            or state["countdown_active"]
        ):
            state["message"] = (
                "Sesi sedang berjalan."
            )
            return

        state["session_finished"] = False
        state["session_aborted"] = False
        state["session_saved"] = False
        state["countdown_active"] = True
        state["countdown_end"] = (
            time.monotonic()
            + COUNTDOWN_SEC
        )
        state["message"] = (
            "Bersiap. Sesi dimulai "
            "setelah hitung mundur."
        )


def begin_session_after_countdown() -> None:
    global model_filter_zi

    schedule = build_schedule()

    remove_temporary_files()
    open_temporary_files()

    trial_buffers.clear()
    evaluation_rows.clear()

    # Filter model harus mulai dari awal sesi,
    # sama seperti preprocessing file training.
    with lock:
        model_filter_zi = None

        state["schedule"] = schedule
        state[
            "schedule_end_samples"
        ] = [
            int(
                segment[
                    "end_sample_exclusive"
                ]
            )
            for segment in schedule
        ]

        state[
            "session_start_absolute_sample"
        ] = absolute_sample
        state[
            "session_samples_written"
        ] = 0
        state["packet_gap_events"] = 0
        state[
            "estimated_missing_samples"
        ] = 0
        state[
            "clipped_a1_samples"
        ] = 0

        state[
            "current_segment"
        ] = schedule[0].copy()

        state["last_result"] = None
        state["final_summary"] = None

        state[
            "countdown_active"
        ] = False
        state["session_active"] = True
        state["session_finished"] = False
        state["session_aborted"] = False
        state["session_saved"] = False

        state["message"] = (
            "Sesi berjalan. "
            "Ikuti perintah pada layar."
        )

    play_cue_sound("baseline")

    print()
    print("EVALUASI DIMULAI")
    print("Subjek        :", SUBJECT_CODE)
    print("Total trial   :", TOTAL_TRIALS)
    print(
        "Trial/kelas   :",
        REPETITIONS_PER_CLASS,
    )
    print(
        "Durasi/trial  :",
        TRIAL_SEC,
        "detik",
    )
    print(
        "Total rekaman :",
        TOTAL_TRIALS
        * TRIAL_SEC,
        "detik",
    )


def abort_session() -> None:
    with lock:
        if state["countdown_active"]:
            state[
                "countdown_active"
            ] = False
            state["message"] = (
                "Hitung mundur dibatalkan."
            )
            return

        if not state["session_active"]:
            state["message"] = (
                "Tidak ada sesi "
                "yang sedang direkam."
            )
            return

        state["session_active"] = False
        state["session_finished"] = True
        state["session_aborted"] = True
        state["current_segment"] = None
        state["message"] = (
            "Sesi dibatalkan. "
            "Data parsial akan dibuang."
        )


# =========================================================
# 15. TAMPILAN PERINTAH
# =========================================================

def draw_marker(
    axis,
    display: str,
) -> None:
    y = -0.03

    if display == "kiri":
        axis.scatter(
            0,
            y,
            s=120,
            facecolors="none",
            linewidths=1.5,
        )
        axis.scatter(
            -0.92,
            y,
            s=1100,
            facecolors="none",
            linewidths=3,
        )
        axis.scatter(
            -0.92,
            y,
            s=240,
        )

    elif display == "kanan":
        axis.scatter(
            0,
            y,
            s=120,
            facecolors="none",
            linewidths=1.5,
        )
        axis.scatter(
            0.92,
            y,
            s=1100,
            facecolors="none",
            linewidths=3,
        )
        axis.scatter(
            0.92,
            y,
            s=240,
        )

    elif display == "kedip":
        axis.scatter(
            0,
            y,
            s=1100,
            facecolors="none",
            linewidths=3,
        )
        axis.scatter(
            0,
            y,
            s=600,
            marker="x",
            linewidths=4,
        )

    else:
        axis.scatter(
            0,
            y,
            s=1100,
            facecolors="none",
            linewidths=3,
        )
        axis.scatter(
            0,
            y,
            s=240,
        )


def format_result(
    result: dict[
        str,
        object
    ] | None,
) -> str:
    if result is None:
        return ""

    status = (
        "BENAR"
        if bool(result["benar"])
        else "SALAH"
    )

    return (
        f"Aktual: "
        f"{str(result['aktual']).upper()}   |   "
        f"Prediksi: "
        f"{str(result['prediksi']).upper()}   |   "
        f"Confidence: "
        f"{float(result['confidence']):.3f}   |   "
        f"{status}"
    )


def draw_cue(
    axis,
    segment,
    countdown_remaining=None,
    completed=False,
    aborted=False,
    last_result=None,
    final_summary=None,
) -> None:
    axis.clear()
    axis.set_xlim(
        -1.25,
        1.25,
    )
    axis.set_ylim(
        -1.0,
        1.0,
    )
    axis.set_aspect("equal")
    axis.axis("off")

    if countdown_remaining is not None:
        number = max(
            1,
            int(
                np.ceil(
                    countdown_remaining
                )
            ),
        )

        axis.text(
            0,
            0.22,
            str(number),
            ha="center",
            va="center",
            fontsize=72,
            fontweight="bold",
        )
        axis.text(
            0,
            -0.46,
            (
                "Tatap titik tengah dan "
                "jaga kepala tetap lurus."
            ),
            ha="center",
            va="center",
            fontsize=17,
        )
        return

    if completed:
        axis.text(
            0,
            0.44,
            "EVALUASI SELESAI",
            ha="center",
            va="center",
            fontsize=34,
            fontweight="bold",
        )

        if final_summary is not None:
            axis.text(
                0,
                0.02,
                (
                    f"Benar "
                    f"{int(final_summary['benar'])} "
                    f"dari "
                    f"{int(final_summary['total_trial'])} "
                    f"trial\n"
                    f"Accuracy "
                    f"{float(final_summary['accuracy']) * 100:.2f}% "
                    f"| Macro F1 "
                    f"{float(final_summary['macro_f1']) * 100:.2f}%"
                ),
                ha="center",
                va="center",
                fontsize=19,
                fontweight="bold",
            )

        axis.text(
            0,
            -0.54,
            "Tekan Q untuk keluar.",
            ha="center",
            va="center",
            fontsize=15,
        )
        return

    if aborted:
        axis.text(
            0,
            0.18,
            "SESI DIBATALKAN",
            ha="center",
            va="center",
            fontsize=32,
            fontweight="bold",
        )
        axis.text(
            0,
            -0.28,
            (
                "Data parsial dibuang. "
                "Tekan S untuk mengulang."
            ),
            ha="center",
            va="center",
            fontsize=16,
        )
        return

    if segment is None:
        axis.text(
            0,
            0.30,
            (
                f"EVALUASI EOG — "
                f"{SUBJECT_CODE}"
            ),
            ha="center",
            va="center",
            fontsize=29,
            fontweight="bold",
        )
        axis.text(
            0,
            -0.12,
            (
                "C = hubungkan   |   "
                "S = mulai   |   "
                "A = batalkan   |   "
                "Q = keluar"
            ),
            ha="center",
            va="center",
            fontsize=15,
        )
        axis.text(
            0,
            -0.45,
            (
                f"Model: {MODEL_PATH.name}\n"
                f"Preprocessing: low-pass "
                f"{MODEL_LOWPASS_HZ:g} Hz, "
                f"{RAW_FS}→"
                f"{RAW_FS // DOWNSAMPLE_FACTOR} Hz, "
                f"window {WINDOW_SAMPLES} sampel"
            ),
            ha="center",
            va="center",
            fontsize=12,
        )
        return

    phase = str(
        segment["phase"]
    )
    label = str(
        segment["target_label"]
    )

    if phase == "baseline":
        title = (
            "TATAP TITIK TENGAH"
        )
        instruction = (
            "Jangan gerakkan mata dan "
            "jangan berkedip. "
            "Kepala tetap lurus."
        )
        display = "diam"

    elif phase == "return":
        title = (
            "KEMBALIKAN MATA KE TENGAH"
        )
        instruction = (
            "Tatap titik tengah dan "
            "jaga kepala tetap lurus."
        )
        display = "diam"

    elif phase == "rest":
        title = (
            "ISTIRAHAT 5 DETIK"
        )
        instruction = (
            "Boleh berkedip dan rileks. "
            "Jangan menyentuh elektroda."
        )
        display = "diam"

    elif label == "kiri":
        title = (
            "GERAKKAN MATA KE KIRI SEKARANG"
        )
        instruction = (
            "Gerakkan mata saja, jangan menoleh. "
            "Tahan pandangan ke kiri "
            "sampai perintah berubah."
        )
        display = "kiri"

    elif label == "kanan":
        title = (
            "GERAKKAN MATA KE KANAN SEKARANG"
        )
        instruction = (
            "Gerakkan mata saja, jangan menoleh. "
            "Tahan pandangan ke kanan "
            "sampai perintah berubah."
        )
        display = "kanan"

    elif label == "kedip":
        title = (
            "KEDIP SATU KALI SEKARANG"
        )
        instruction = (
            "Setelah satu kali kedip, "
            "buka mata dan tatap titik tengah."
        )
        display = "kedip"

    else:
        title = (
            "TETAP TATAP TITIK TENGAH"
        )
        instruction = (
            "Jangan gerakkan mata dan "
            "jangan berkedip "
            "sampai perintah berubah."
        )
        display = "diam"

    axis.text(
        0,
        0.76,
        title,
        ha="center",
        va="center",
        fontsize=27,
        fontweight="bold",
    )

    axis.text(
        0,
        -0.45,
        instruction,
        ha="center",
        va="center",
        fontsize=15,
    )

    axis.text(
        0,
        -0.67,
        (
            f"Trial "
            f"{int(segment['trial_id'])} "
            f"dari {TOTAL_TRIALS}"
        ),
        ha="center",
        va="center",
        fontsize=13,
    )

    if (
        phase in {
            "return",
            "rest",
        }
        and last_result is not None
    ):
        axis.text(
            0,
            -0.88,
            format_result(
                last_result
            ),
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

    draw_marker(
        axis,
        display,
    )


# =========================================================
# 16. KONTROL KEYBOARD
# =========================================================

def on_key(event) -> None:
    key = (
        event.key.lower()
        if event.key
        else ""
    )

    if key == "c":
        connect_device()
    elif key == "s":
        request_start_session()
    elif key == "a":
        abort_session()
    elif key == "q":
        with lock:
            active = bool(
                state["session_active"]
            )
            counting = bool(
                state["countdown_active"]
            )

        if active or counting:
            abort_session()

        with lock:
            state["running"] = False


def on_close(_event) -> None:
    with lock:
        active = bool(
            state["session_active"]
        )
        counting = bool(
            state["countdown_active"]
        )

    if active or counting:
        abort_session()

    with lock:
        state["running"] = False


# =========================================================
# 17. MAIN
# =========================================================

def main() -> None:
    print()
    print("=" * 72)
    print("EVALUASI REALTIME EOG 1D-CNN")
    print("=" * 72)
    print("Subjek         :", SUBJECT_CODE)
    print("Model          :", MODEL_PATH)
    print(
        "Subjek training:",
        sorted(
            TRAINING_SUBJECTS
        ),
    )
    print("Channel        : A1")
    print("Sampling raw   :", RAW_FS, "Hz")
    print(
        "Low-pass       :",
        MODEL_LOWPASS_HZ,
        "Hz, orde",
        MODEL_FILTER_ORDER,
    )
    print(
        "Sampling model :",
        RAW_FS
        // DOWNSAMPLE_FACTOR,
        "Hz",
    )
    print(
        "Window model   :",
        WINDOW_SAMPLES,
        "sampel",
    )
    print("Total trial    :", TOTAL_TRIALS)
    print(
        "Repetisi/kelas :",
        REPETITIONS_PER_CLASS,
    )
    print()
    print("Setiap trial:")
    print(
        "  2 detik : tatap titik tengah"
    )
    print(
        "  2 detik : lakukan perintah"
    )
    print(
        "  2 detik : kembali ke tengah"
    )
    print(
        "  5 detik : istirahat"
    )
    print()
    print(
        "Total sesi     :",
        TOTAL_TRIALS
        * TRIAL_SEC
        + COUNTDOWN_SEC,
        "detik",
    )
    print()
    print("Output:")
    print("  Raw          :", RAW_FINAL_PATH)
    print("  Evaluasi     :", RESULT_FINAL_PATH)
    print("  Ringkasan    :", SUMMARY_PATH)
    print()
    print("Kontrol:")
    print(
        "  C = hubungkan BITalino"
    )
    print(
        "  S = mulai seluruh sesi"
    )
    print(
        "  A = batalkan; data parsial dibuang"
    )
    print(
        "  Q = keluar"
    )
    print("=" * 72)

    plt.ion()

    figure = plt.figure(
        figsize=(14, 8.5)
    )
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=[
            1.0,
            1.15,
        ],
        hspace=0.14,
    )

    cue_axis = figure.add_subplot(
        grid[0, 0]
    )
    signal_axis = figure.add_subplot(
        grid[1, 0]
    )

    figure.canvas.manager.set_window_title(
        "Evaluasi Realtime EOG 1D-CNN"
    )

    figure.canvas.mpl_connect(
        "key_press_event",
        on_key,
    )
    figure.canvas.mpl_connect(
        "close_event",
        on_close,
    )

    raw_line, = signal_axis.plot(
        [],
        [],
        linewidth=0.8,
        alpha=0.45,
        label="A1 raw — disimpan",
    )

    filtered_line, = (
        signal_axis.plot(
            [],
            [],
            linewidth=1.5,
            label=(
                f"A1 low-pass "
                f"{MODEL_LOWPASS_HZ:g} Hz"
            ),
        )
    )

    signal_axis.set_title(
        "Sinyal EOG A1 Realtime"
    )
    signal_axis.set_xlabel(
        "Waktu akuisisi (detik)"
    )
    signal_axis.set_ylabel(
        "ADC count"
    )
    signal_axis.set_ylim(
        0,
        1023,
    )
    signal_axis.grid(
        True,
        alpha=0.25,
    )
    signal_axis.legend(
        loc="upper right"
    )

    status_text = signal_axis.text(
        0.01,
        0.97,
        "",
        transform=signal_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    last_cue_key = None

    try:
        while True:
            with lock:
                running = bool(
                    state["running"]
                )
                connected = bool(
                    state["connected"]
                )
                countdown_active = bool(
                    state[
                        "countdown_active"
                    ]
                )
                countdown_end = state[
                    "countdown_end"
                ]
                session_active = bool(
                    state["session_active"]
                )
                session_finished = bool(
                    state[
                        "session_finished"
                    ]
                )
                session_saved = bool(
                    state["session_saved"]
                )
                completed_once = bool(
                    state[
                        "completed_once"
                    ]
                )
                session_aborted = bool(
                    state[
                        "session_aborted"
                    ]
                )
                segment = (
                    state[
                        "current_segment"
                    ].copy()
                    if state[
                        "current_segment"
                    ]
                    else None
                )
                message = str(
                    state["message"]
                )
                gaps = int(
                    state[
                        "packet_gap_events"
                    ]
                )
                missing = int(
                    state[
                        "estimated_missing_samples"
                    ]
                )
                clipping = int(
                    state[
                        "clipped_a1_samples"
                    ]
                )
                reader_error = state[
                    "reader_error"
                ]
                last_result = (
                    state[
                        "last_result"
                    ].copy()
                    if state[
                        "last_result"
                    ]
                    else None
                )
                final_summary = (
                    state[
                        "final_summary"
                    ].copy()
                    if state[
                        "final_summary"
                    ]
                    else None
                )

            if (
                not running
                or not plt.fignum_exists(
                    figure.number
                )
            ):
                break

            countdown_remaining = None

            if countdown_active:
                countdown_remaining = (
                    float(countdown_end)
                    - time.monotonic()
                )

                if countdown_remaining <= 0:
                    begin_session_after_countdown()
                    countdown_remaining = None

            if (
                session_finished
                and not session_saved
            ):
                finalize_session()

                with lock:
                    completed_once = bool(
                        state[
                            "completed_once"
                        ]
                    )
                    session_aborted = bool(
                        state[
                            "session_aborted"
                        ]
                    )
                    message = str(
                        state["message"]
                    )
                    final_summary = (
                        state[
                            "final_summary"
                        ].copy()
                        if state[
                            "final_summary"
                        ]
                        else None
                    )

            if (
                countdown_remaining
                is not None
            ):
                cue_key = (
                    "countdown",
                    int(
                        np.ceil(
                            countdown_remaining
                        )
                    ),
                )
            elif completed_once:
                cue_key = (
                    "completed",
                    int(
                        final_summary[
                            "benar"
                        ]
                    )
                    if final_summary
                    else -1,
                )
            elif (
                session_aborted
                and session_saved
            ):
                cue_key = (
                    "aborted",
                )
            elif segment is not None:
                cue_key = (
                    int(
                        segment[
                            "trial_id"
                        ]
                    ),
                    str(
                        segment["phase"]
                    ),
                    (
                        int(
                            last_result[
                                "trial_ke"
                            ]
                        )
                        if last_result
                        else -1
                    ),
                )
            else:
                cue_key = (
                    "idle",
                    connected,
                )

            if cue_key != last_cue_key:
                draw_cue(
                    cue_axis,
                    segment,
                    countdown_remaining=(
                        countdown_remaining
                    ),
                    completed=completed_once,
                    aborted=(
                        session_aborted
                        and session_saved
                    ),
                    last_result=(
                        last_result
                    ),
                    final_summary=(
                        final_summary
                    ),
                )
                last_cue_key = cue_key

            with lock:
                times = np.asarray(
                    time_plot_buffer,
                    dtype=float,
                )
                raw_values = np.asarray(
                    raw_plot_buffer,
                    dtype=float,
                )
                filtered_values = (
                    np.asarray(
                        filtered_plot_buffer,
                        dtype=float,
                    )
                )

            if len(times) > 0:
                raw_line.set_data(
                    times,
                    raw_values,
                )
                filtered_line.set_data(
                    times,
                    filtered_values,
                )

                right = float(
                    times[-1]
                )
                left = max(
                    0.0,
                    right - DISPLAY_SEC,
                )
                signal_axis.set_xlim(
                    left,
                    max(
                        right,
                        DISPLAY_SEC,
                    ),
                )

                recent = (
                    raw_values[
                        -RAW_FS:
                    ]
                    if len(raw_values)
                    >= RAW_FS
                    else raw_values
                )

                result_text = (
                    format_result(
                        last_result
                    )
                    if last_result
                    else "Belum ada prediksi."
                )

                status_text.set_text(
                    f"{'TERHUBUNG' if connected else 'OFFLINE'} | "
                    f"{'MEREKAM' if session_active else 'SIAP'} | "
                    f"A1 1 detik: "
                    f"min {np.min(recent):.0f}, "
                    f"max {np.max(recent):.0f}, "
                    f"std {np.std(recent):.2f} | "
                    f"gap {gaps}, "
                    f"missing {missing}, "
                    f"clipping {clipping}\n"
                    f"{result_text}\n"
                    f"{message}"
                )
            else:
                signal_axis.set_xlim(
                    0,
                    DISPLAY_SEC,
                )
                status_text.set_text(
                    f"{'TERHUBUNG' if connected else 'OFFLINE'} | "
                    f"{message}"
                )

            if reader_error:
                status_text.set_text(
                    status_text.get_text()
                    + "\nERROR: "
                    + str(reader_error)
                )

            figure.canvas.draw_idle()
            figure.canvas.flush_events()
            plt.pause(0.02)

    except KeyboardInterrupt:
        print(
            "\nProgram dihentikan "
            "dari keyboard."
        )

    finally:
        with lock:
            active = bool(
                state["session_active"]
            )
            counting = bool(
                state["countdown_active"]
            )
            needs_finalize = bool(
                state["session_finished"]
                and not state[
                    "session_saved"
                ]
            )

        if active or counting:
            abort_session()
            needs_finalize = True

        if needs_finalize:
            finalize_session()

        remove_temporary_files()
        disconnect_device()

        plt.ioff()
        plt.close(figure)

        print("Program selesai.")


if __name__ == "__main__":
    main()
