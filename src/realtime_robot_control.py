#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Real-Time EOG Classification and Pioneer P3DX Robot Control
============================================================

Loads the trained TorchScript 1D-CNN, acquires single-channel EOG from
BITalino, applies the same preprocessing used during training, and maps
model outputs plus blink-count logic to commands for a Pioneer P3DX robot
in CoppeliaSim.

Preprocessing:
1. BITalino A1 at 1000 Hz.
2. 4th-order causal Butterworth low-pass filter at 15 Hz.
3. Median baseline from the 2 s baseline phase.
4. Baseline-corrected 2 s activity segment.
5. Downsample from 1000 Hz to 250 Hz.
6. 500-sample model input.
7. Normalize using mean/std stored in model_config.json.

Command mapping:
- idle         -> HOLD
- left         -> TURN LEFT
- right        -> TURN RIGHT
- single blink -> STOP
- double blink -> MOVE FORWARD

Example:
    python src/realtime_robot_control.py --subject E01 --mac YOUR_BITALINO_MAC

Camera preview is optional and is used only for experiment documentation.
"""

from __future__ import annotations

import argparse
import os
import csv
import hashlib
import json
import math
import queue
import random
import re
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, sosfilt, sosfilt_zi

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from bitalino import BITalino
except ImportError as exc:
    raise SystemExit(
        "Library bitalino belum terpasang.\n"
        "Install dengan:\n"
        "python -m pip install bitalino"
    ) from exc

try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
except ImportError as exc:
    raise SystemExit(
        "Library CoppeliaSim ZeroMQ belum terpasang.\n"
        "Install dengan:\n"
        "python -m pip install coppeliasim-zmqremoteapi-client"
    ) from exc


# ============================================================
# 1. KONFIGURASI DASAR
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Script boleh disimpan di MAIN atau MAIN/scripts.
if (SCRIPT_DIR / "models").exists():
    PROJECT_ROOT = SCRIPT_DIR
else:
    PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_DIR = PROJECT_ROOT / "models" / "eog_5_subjek_80_20"
MODEL_PATH = MODEL_DIR / "eog_5_subjek_80_20_torchscript.pt"
CONFIG_PATH = MODEL_DIR / "model_config.json"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "realtime_robot_command_5_subjek"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_MAC_ADDRESS = os.getenv("BITALINO_MAC", "").strip()

RAW_FS = 1000
PROCESSED_FS = 250
CHANNELS = [0]          # BITalino A1
A1_COLUMN = 5           # Posisi kanal A1 pada hasil BITalino
READ_BLOCK = 20

BASELINE_SEC = 2.0
ACTION_SEC = 2.0
RETURN_SEC = 2.0
REST_SEC = 5.0
COUNTDOWN_SEC = 3

BASELINE_SAMPLES = int(BASELINE_SEC * RAW_FS)
ACTION_SAMPLES = int(ACTION_SEC * RAW_FS)
RETURN_SAMPLES = int(RETURN_SEC * RAW_FS)
REST_SAMPLES = int(REST_SEC * RAW_FS)
TRIAL_SAMPLES = (
    BASELINE_SAMPLES
    + ACTION_SAMPLES
    + RETURN_SAMPLES
    + REST_SAMPLES
)

CLASS_ORDER_EXPECTED = ["diam", "kiri", "kanan", "kedip"]
WINDOW_SAMPLES_EXPECTED = int(ACTION_SEC * PROCESSED_FS)

REPETITIONS_PER_COMMAND = 4
TOTAL_COMMAND_TYPES = 5
TOTAL_TRIALS = REPETITIONS_PER_COMMAND * TOTAL_COMMAND_TYPES
RANDOM_SEED = 20260715

# Threshold keputusan ditetapkan sebelum pengujian final.
DIRECTION_CONFIDENCE_MIN = 0.55
STOP_KEDIP_SUPPORT_MIN = 0.35
FORWARD_KEDIP_SUPPORT_MIN = 0.55

# Blink detector pada sinyal aktivitas setelah koreksi baseline.
MIN_BLINK_THRESHOLD_ADC = 70.0
MAX_BLINK_THRESHOLD_ADC = 180.0
BLINK_SIGMA_MULTIPLIER = 8.0
BLINK_ENVELOPE_SMOOTH_SEC = 0.050
BLINK_MERGE_GAP_SEC = 0.20
BLINK_MIN_EVENT_SEC = 0.030
BLINK_MAX_EVENT_SEC = 0.800

# Tampilan dibuat kecil dan ringan agar nyaman berdampingan dengan CoppeliaSim.
DISPLAY_SEC = 4.0
DISPLAY_SAMPLES = int(DISPLAY_SEC * RAW_FS)
DISPLAY_MAX_POINTS = 500
DISPLAY_UPDATE_SEC = 0.12
DISPLAY_SMOOTH_SAMPLES = 21       # Khusus tampilan, tidak masuk model.
DISPLAY_MIN_HALF_SPAN_ADC = 40.0
DISPLAY_MAX_HALF_SPAN_ADC = 260.0
DISPLAY_SPAN_SHRINK = 0.97
DISPLAY_SPAN_GROW = 0.35

WINDOW_WIDTH_PX = 720
WINDOW_HEIGHT_PX = 520
WINDOW_DPI = 100

CAMERA_WIDTH = 180
CAMERA_HEIGHT = 135
CAMERA_FPS = 3.0

# CoppeliaSim Pioneer P3DX.
ROBOT_PATH = "/PioneerP3DX"
LEFT_MOTOR_PATH = "/PioneerP3DX/leftMotor"
RIGHT_MOTOR_PATH = "/PioneerP3DX/rightMotor"

FORWARD_SPEED = 1.20
FORWARD_DURATION_SEC = 3.0
TURN_INNER_SPEED = 0.45
TURN_OUTER_SPEED = 1.20
TURN_DURATION_SEC = 1.50

START_SIMULATION_FROM_PYTHON = True
STOP_SIMULATION_ON_EXIT = True


# ============================================================
# 2. ARGUMEN PROGRAM
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Uji realtime command robot menggunakan model EOG 1D-CNN "
            "hasil training lima subjek."
        )
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="",
        help="Kode subjek uji, misalnya E01.",
    )
    parser.add_argument(
        "--mac",
        type=str,
        default=DEFAULT_MAC_ADDRESS,
        help="BITalino MAC address. Can also be set with BITALINO_MAC.",
    )
    parser.add_argument(
        "--allow-training-subject",
        action="store_true",
        help=(
            "Izinkan kode subjek yang juga terdapat pada data training. "
            "Tanpa opsi ini, program menolak subjek training."
        ),
    )
    parser.add_argument(
        "--no-raw-log",
        action="store_true",
        help="Tidak menyimpan log raw per sampel.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Indeks kamera OpenCV. Default 0.",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Jalankan tanpa preview kamera.",
    )
    return parser.parse_args()


ARGS = parse_args()
MAC_ADDRESS = ARGS.mac.strip()
if not MAC_ADDRESS:
    raise SystemExit(
        "BITalino MAC address is required. Use --mac YOUR_BITALINO_MAC "
        "or set the BITALINO_MAC environment variable."
    )


def ask_subject_code(initial: str) -> str:
    value = initial.strip().upper()

    while not value:
        value = input("Masukkan kode subjek uji, contoh E01: ").strip().upper()

    if not re.fullmatch(r"[A-Z]\d{2}(R\d+)?", value):
        raise SystemExit(
            "Format kode subjek tidak sesuai. Gunakan contoh E01, E02, atau E01R2."
        )

    return value


SUBJECT_ID = ask_subject_code(ARGS.subject)


# ============================================================
# 3. UTILITAS VALIDASI MODEL
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_and_config() -> tuple[torch.jit.ScriptModule, dict[str, Any]]:
    if not MODEL_PATH.exists():
        raise SystemExit(
            "Model TorchScript tidak ditemukan:\n"
            f"{MODEL_PATH}\n\n"
            "Jalankan training 5 subjek terlebih dahulu."
        )

    if not CONFIG_PATH.exists():
        raise SystemExit(
            "model_config.json tidak ditemukan:\n"
            f"{CONFIG_PATH}"
        )

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    required_keys = [
        "class_order",
        "normalization_mean",
        "normalization_std",
        "raw_sampling_rate_hz",
        "processed_sampling_rate_hz",
        "lowpass_hz",
        "filter_order",
    ]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise SystemExit(f"Config model tidak lengkap. Key hilang: {missing}")

    class_order = list(config["class_order"])
    if class_order != CLASS_ORDER_EXPECTED:
        raise SystemExit(
            "Urutan kelas model tidak sesuai.\n"
            f"Model     : {class_order}\n"
            f"Seharusnya: {CLASS_ORDER_EXPECTED}"
        )

    raw_fs = int(config["raw_sampling_rate_hz"])
    processed_fs = int(config["processed_sampling_rate_hz"])
    if raw_fs != RAW_FS or processed_fs != PROCESSED_FS:
        raise SystemExit(
            "Sampling rate model tidak sesuai.\n"
            f"Model: raw={raw_fs}, processed={processed_fs}\n"
            f"Script: raw={RAW_FS}, processed={PROCESSED_FS}"
        )

    if raw_fs % processed_fs != 0:
        raise SystemExit("Raw sampling rate tidak habis dibagi processed sampling rate.")

    downsample_factor = int(
        config.get("downsample_factor", raw_fs // processed_fs)
    )
    if downsample_factor != RAW_FS // PROCESSED_FS:
        raise SystemExit(
            f"Downsample factor tidak sesuai: {downsample_factor}."
        )

    window_samples = int(
        config.get(
            "processed_samples_per_window",
            config.get("window_samples", WINDOW_SAMPLES_EXPECTED),
        )
    )
    if window_samples != WINDOW_SAMPLES_EXPECTED:
        raise SystemExit(
            "Ukuran input model tidak sesuai.\n"
            f"Model: {window_samples}\n"
            f"Seharusnya: {WINDOW_SAMPLES_EXPECTED}"
        )

    if not np.isclose(float(config["lowpass_hz"]), 15.0, atol=1e-9):
        raise SystemExit("Model tidak menggunakan low-pass 15 Hz.")

    if int(config["filter_order"]) != 4:
        raise SystemExit("Model tidak menggunakan filter orde 4.")

    normalization_mean = float(config["normalization_mean"])
    normalization_std = float(config["normalization_std"])
    if not np.isfinite(normalization_mean):
        raise SystemExit("normalization_mean tidak valid.")
    if not np.isfinite(normalization_std) or normalization_std <= 0:
        raise SystemExit("normalization_std tidak valid.")

    expected_hash = str(config.get("torchscript_sha256", "")).strip().lower()
    actual_hash = sha256_file(MODEL_PATH)
    if expected_hash and expected_hash != actual_hash:
        raise SystemExit(
            "Hash model tidak sesuai dengan model_config.json.\n"
            "Kemungkinan file model dan config berasal dari training yang berbeda."
        )

    config["downsample_factor"] = downsample_factor
    config["processed_samples_per_window"] = window_samples
    config["training_subjects"] = list(
        config.get("training_subjects", config.get("subjects", []))
    )
    config["actual_torchscript_sha256"] = actual_hash

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model = torch.jit.load(str(MODEL_PATH), map_location="cpu")
    model.eval()

    with torch.inference_mode():
        output = model(
            torch.zeros(
                (1, 1, window_samples),
                dtype=torch.float32,
            )
        )

    if tuple(output.shape) != (1, len(class_order)):
        raise SystemExit(
            "Output model tidak sesuai.\n"
            f"Ditemukan: {tuple(output.shape)}\n"
            f"Seharusnya: (1, {len(class_order)})"
        )

    return model, config


MODEL, MODEL_CONFIG = load_model_and_config()
CLASS_ORDER = list(MODEL_CONFIG["class_order"])
NORMALIZATION_MEAN = float(MODEL_CONFIG["normalization_mean"])
NORMALIZATION_STD = float(MODEL_CONFIG["normalization_std"])
DOWNSAMPLE_FACTOR = int(MODEL_CONFIG["downsample_factor"])
MODEL_LOWPASS_HZ = float(MODEL_CONFIG["lowpass_hz"])
MODEL_FILTER_ORDER = int(MODEL_CONFIG["filter_order"])
WINDOW_SAMPLES = int(MODEL_CONFIG["processed_samples_per_window"])
TRAINING_SUBJECTS = {
    str(value).strip().upper()
    for value in MODEL_CONFIG.get("training_subjects", [])
}

if SUBJECT_ID in TRAINING_SUBJECTS and not ARGS.allow_training_subject:
    raise SystemExit(
        f"Subjek {SUBJECT_ID} terdapat pada data training.\n"
        "Gunakan subjek baru untuk pengujian final.\n"
        "Untuk uji teknis saja, tambahkan --allow-training-subject."
    )


# ============================================================
# 4. FILTER CAUSAL KONTINU
# ============================================================

class ContinuousLowpassFilter:
    def __init__(self) -> None:
        self.sos = butter(
            MODEL_FILTER_ORDER,
            MODEL_LOWPASS_HZ,
            btype="lowpass",
            fs=RAW_FS,
            output="sos",
        )
        self.zi: np.ndarray | None = None

    def process(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return values

        if self.zi is None:
            self.zi = sosfilt_zi(self.sos) * values[0]

        filtered, self.zi = sosfilt(
            self.sos,
            values,
            zi=self.zi,
        )
        return filtered


# ============================================================
# 5. PROTOKOL COMMAND
# ============================================================

@dataclass(frozen=True)
class CommandCue:
    cue_id: str
    title: str
    instruction: str
    expected_model_label: str
    expected_command: str


COMMAND_CUES = [
    CommandCue(
        cue_id="diam",
        title="TETAP DIAM",
        instruction=(
            "Tatap titik tengah. Jangan gerakkan mata dan jangan berkedip."
        ),
        expected_model_label="diam",
        expected_command="HOLD",
    ),
    CommandCue(
        cue_id="kiri",
        title="GERAKKAN MATA KE KIRI",
        instruction=(
            "Gerakkan mata saja ke kiri tanpa menoleh. Tahan sampai instruksi berubah."
        ),
        expected_model_label="kiri",
        expected_command="BELOK_KIRI",
    ),
    CommandCue(
        cue_id="kanan",
        title="GERAKKAN MATA KE KANAN",
        instruction=(
            "Gerakkan mata saja ke kanan tanpa menoleh. Tahan sampai instruksi berubah."
        ),
        expected_model_label="kanan",
        expected_command="BELOK_KANAN",
    ),
    CommandCue(
        cue_id="kedip_satu",
        title="KEDIP SATU KALI",
        instruction=(
            "Lakukan satu kedipan yang jelas, kemudian buka mata dan tetap diam."
        ),
        expected_model_label="kedip",
        expected_command="STOP",
    ),
    CommandCue(
        cue_id="kedip_ganda",
        title="KEDIP DUA KALI",
        instruction=(
            "Lakukan dua kedipan terpisah dan jelas dalam waktu dua detik."
        ),
        expected_model_label="kedip",
        expected_command="MAJU",
    ),
]


def build_schedule() -> list[CommandCue]:
    rng = random.Random(RANDOM_SEED)
    schedule: list[CommandCue] = []

    for _ in range(REPETITIONS_PER_COMMAND):
        block = COMMAND_CUES.copy()
        rng.shuffle(block)
        schedule.extend(block)

    if len(schedule) != TOTAL_TRIALS:
        raise RuntimeError("Jumlah trial schedule tidak sesuai.")

    return schedule


# ============================================================
# 6. CONTROLLER ROBOT COPPELIASIM
# ============================================================

class CoppeliaRobotController:
    def __init__(self) -> None:
        self.client = None
        self.sim = None
        self.robot = None
        self.left_motor = None
        self.right_motor = None

        self.command_queue: queue.Queue[tuple[int, str]] = queue.Queue()
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.thread: threading.Thread | None = None

        self.lock = threading.RLock()
        self.connected = False
        self.connection_error: str | None = None
        self.active_command: str | None = None
        self.active_request_id: int | None = None
        self.motion_deadline: float | None = None
        self.last_started_id: int | None = None
        self.last_completed_id: int | None = None
        self.last_applied_command = "HOLD"
        self.status = "BELUM TERHUBUNG"
        self.message = "CoppeliaSim belum terhubung."
        self.request_counter = 0

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "connected": self.connected,
                "status": self.status,
                "message": self.message,
                "active_command": self.active_command,
                "last_applied_command": self.last_applied_command,
                "last_started_id": self.last_started_id,
                "last_completed_id": self.last_completed_id,
            }

    def _set_state(self, status: str, message: str) -> None:
        with self.lock:
            self.status = status
            self.message = message

    def start(self, timeout_sec: float = 12.0) -> None:
        if self.thread is not None and self.thread.is_alive():
            return

        self.stop_event.clear()
        self.ready_event.clear()
        self.connection_error = None
        self.thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="CoppeliaRobotController",
        )
        self.thread.start()

        if not self.ready_event.wait(timeout=timeout_sec):
            raise TimeoutError(
                f"CoppeliaSim tidak merespons dalam {timeout_sec:.0f} detik."
            )

        if self.connection_error:
            raise RuntimeError(self.connection_error)

        if not self.connected:
            raise RuntimeError("CoppeliaSim tidak berhasil terhubung.")

    def _next_request_id(self) -> int:
        with self.lock:
            self.request_counter += 1
            return self.request_counter

    def submit(self, command: str) -> int | None:
        command = str(command).upper()

        if command == "HOLD":
            with self.lock:
                self.last_applied_command = "HOLD"
                self.status = "IDLE"
                self.message = "HOLD: robot tetap diam."
            return None

        if not self.connected:
            return None

        request_id = self._next_request_id()

        # STOP harus menghapus command tertunda.
        if command == "STOP":
            while True:
                try:
                    self.command_queue.get_nowait()
                except queue.Empty:
                    break

        self.command_queue.put((request_id, command))
        return request_id

    def was_completed(self, request_id: int | None) -> bool:
        if request_id is None:
            return True
        with self.lock:
            return self.last_completed_id == request_id

    def _set_velocities(self, left: float, right: float) -> None:
        self.sim.setJointTargetVelocity(self.left_motor, float(left))
        self.sim.setJointTargetVelocity(self.right_motor, float(right))

    def _stop_motion(self) -> None:
        if self.sim is not None and self.left_motor is not None:
            self._set_velocities(0.0, 0.0)

        with self.lock:
            self.active_command = None
            self.active_request_id = None
            self.motion_deadline = None

    def _complete_active(self, message: str) -> None:
        with self.lock:
            completed_id = self.active_request_id
            completed_command = self.active_command

        self._stop_motion()

        with self.lock:
            self.last_completed_id = completed_id
            if completed_command is not None:
                self.last_applied_command = completed_command
            self.status = "IDLE"
            self.message = message

    def _apply(self, request_id: int, command: str) -> None:
        command = command.upper()

        # Command baru boleh memotong gerakan sebelumnya.
        if self.active_command is not None:
            self._stop_motion()

        with self.lock:
            self.active_request_id = request_id
            self.active_command = command
            self.last_started_id = request_id
            self.last_applied_command = command

        now = time.monotonic()

        if command == "STOP":
            self._set_velocities(0.0, 0.0)
            with self.lock:
                self.active_command = None
                self.active_request_id = None
                self.motion_deadline = None
                self.last_completed_id = request_id
                self.status = "STOP"
                self.message = "STOP diterapkan. Robot berhenti."
            return

        if command == "MAJU":
            self._set_velocities(FORWARD_SPEED, FORWARD_SPEED)
            with self.lock:
                self.motion_deadline = now + FORWARD_DURATION_SEC
                self.status = "MAJU"
                self.message = f"MAJU selama {FORWARD_DURATION_SEC:.1f} detik."
            return

        if command == "BELOK_KIRI":
            self._set_velocities(TURN_INNER_SPEED, TURN_OUTER_SPEED)
            with self.lock:
                self.motion_deadline = now + TURN_DURATION_SEC
                self.status = "BELOK_KIRI"
                self.message = f"BELOK KIRI selama {TURN_DURATION_SEC:.1f} detik."
            return

        if command == "BELOK_KANAN":
            self._set_velocities(TURN_OUTER_SPEED, TURN_INNER_SPEED)
            with self.lock:
                self.motion_deadline = now + TURN_DURATION_SEC
                self.status = "BELOK_KANAN"
                self.message = f"BELOK KANAN selama {TURN_DURATION_SEC:.1f} detik."
            return

        self._stop_motion()
        with self.lock:
            self.last_completed_id = request_id
            self.status = "ERROR"
            self.message = f"Command tidak dikenal: {command}"

    def _worker(self) -> None:
        try:
            self._set_state("MENGHUBUNGKAN", "Menghubungkan ZeroMQ Remote API...")

            self.client = RemoteAPIClient()
            self.sim = self.client.require("sim")
            self.robot = self.sim.getObject(ROBOT_PATH)
            self.left_motor = self.sim.getObject(LEFT_MOTOR_PATH)
            self.right_motor = self.sim.getObject(RIGHT_MOTOR_PATH)

            if START_SIMULATION_FROM_PYTHON:
                simulation_state = self.sim.getSimulationState()
                stopped_state = getattr(self.sim, "simulation_stopped", 0)
                if simulation_state == stopped_state:
                    self.sim.startSimulation()
                    time.sleep(0.5)

            self._set_velocities(0.0, 0.0)

            with self.lock:
                self.connected = True
                self.status = "IDLE"
                self.message = "Robot siap menerima command."
            self.ready_event.set()

            while not self.stop_event.is_set():
                latest_request: tuple[int, str] | None = None

                while True:
                    try:
                        request = self.command_queue.get_nowait()
                    except queue.Empty:
                        break

                    # STOP selalu menang. Selain STOP, gunakan command terbaru.
                    if request[1] == "STOP":
                        latest_request = request
                        while True:
                            try:
                                self.command_queue.get_nowait()
                            except queue.Empty:
                                break
                        break

                    latest_request = request

                if latest_request is not None:
                    self._apply(*latest_request)

                with self.lock:
                    deadline = self.motion_deadline
                    active = self.active_command

                if active is not None and deadline is not None:
                    if time.monotonic() >= deadline:
                        self._complete_active(f"{active} selesai. Robot kembali HOLD.")

                time.sleep(0.01)

        except Exception as exc:
            self.connection_error = f"{type(exc).__name__}: {exc}"
            with self.lock:
                self.connected = False
                self.status = "ERROR"
                self.message = self.connection_error
            self.ready_event.set()

        finally:
            try:
                self._stop_motion()
            except Exception:
                pass

            if self.sim is not None and STOP_SIMULATION_ON_EXIT:
                try:
                    self.sim.stopSimulation()
                except Exception:
                    pass

            with self.lock:
                self.connected = False

    def shutdown(self) -> None:
        try:
            self.submit("STOP")
            time.sleep(0.1)
        except Exception:
            pass

        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.thread = None


# ============================================================
# 7. PREVIEW KAMERA DAN TAMPILAN REALTIME
# ============================================================

class CameraPreview:
    """Preview kamera ringan untuk dokumentasi, bukan input model."""

    def __init__(
        self,
        camera_index: int = 0,
        enabled: bool = True,
    ) -> None:
        self.camera_index = int(camera_index)
        self.enabled = bool(enabled)
        self.capture = None
        self.thread = None
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.status = "KAMERA BELUM DIMULAI"
        self.error = None

    def start(self) -> bool:
        if not self.enabled:
            self.status = "KAMERA DINONAKTIFKAN"
            return False

        if cv2 is None:
            self.status = "OPENCV BELUM TERPASANG"
            self.error = (
                "Install kamera dengan: "
                "python -m pip install opencv-python"
            )
            return False

        if self.thread is not None and self.thread.is_alive():
            return True

        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="CameraPreviewWorker",
        )
        self.thread.start()
        return True

    def _worker(self) -> None:
        try:
            self.capture = cv2.VideoCapture(
                self.camera_index,
                cv2.CAP_DSHOW if sys.platform.startswith("win") else 0,
            )

            if not self.capture.isOpened():
                raise RuntimeError(
                    f"Kamera indeks {self.camera_index} tidak dapat dibuka."
                )

            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

            self.status = "KAMERA AKTIF"
            minimum_interval = 1.0 / CAMERA_FPS

            while not self.stop_event.is_set():
                started = time.monotonic()
                ok, frame = self.capture.read()

                if not ok or frame is None:
                    self.status = "FRAME KAMERA GAGAL"
                    time.sleep(0.20)
                    continue

                frame = cv2.resize(
                    frame,
                    (CAMERA_WIDTH, CAMERA_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                with self.frame_lock:
                    self.latest_frame = frame

                elapsed = time.monotonic() - started
                time.sleep(max(0.0, minimum_interval - elapsed))

        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = "KAMERA TIDAK TERSEDIA"

        finally:
            if self.capture is not None:
                try:
                    self.capture.release()
                except Exception:
                    pass
            self.capture = None

    def get_frame(self) -> np.ndarray | None:
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def shutdown(self) -> None:
        self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None

        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = None


class RealtimeDisplay:
    """
    Interface compact:
    - Instruksi tetap di bagian atas.
    - Grafik sinyal filtered di kiri bawah.
    - Kamera kecil dan status robot di kanan.

    Grafik memakai sinyal low-pass 15 Hz, kemudian hanya untuk tampilan:
    - dikurangi median berjalan,
    - diberi smoothing ringan,
    - memakai skala Y stabil.

    Proses tersebut tidak mengubah data yang masuk ke model 1D-CNN.
    """

    def __init__(self, camera_preview: CameraPreview) -> None:
        plt.ion()
        self.camera_preview = camera_preview

        self.figure = plt.figure(
            figsize=(
                WINDOW_WIDTH_PX / WINDOW_DPI,
                WINDOW_HEIGHT_PX / WINDOW_DPI,
            ),
            dpi=WINDOW_DPI,
        )

        grid = self.figure.add_gridspec(
            3,
            2,
            height_ratios=[0.62, 1.0, 0.52],
            width_ratios=[1.55, 0.72],
            left=0.07,
            right=0.97,
            top=0.95,
            bottom=0.10,
            hspace=0.20,
            wspace=0.18,
        )

        self.cue_axis = self.figure.add_subplot(grid[0, :])
        self.signal_axis = self.figure.add_subplot(grid[1:, 0])
        self.camera_axis = self.figure.add_subplot(grid[1, 1])
        self.status_axis = self.figure.add_subplot(grid[2, 1])

        self.cue_axis.axis("off")
        self.status_axis.axis("off")
        self.camera_axis.set_xticks([])
        self.camera_axis.set_yticks([])

        # Instruksi tetap di atas seperti antarmuka akuisisi/evaluasi.
        self.title_text = self.cue_axis.text(
            0.5,
            0.78,
            "",
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            transform=self.cue_axis.transAxes,
        )
        self.instruction_text = self.cue_axis.text(
            0.5,
            0.44,
            "",
            ha="center",
            va="center",
            fontsize=10.5,
            wrap=True,
            transform=self.cue_axis.transAxes,
        )
        self.progress_text = self.cue_axis.text(
            0.5,
            0.12,
            "",
            ha="center",
            va="center",
            fontsize=9.5,
            transform=self.cue_axis.transAxes,
        )

        # Grafik filtered untuk tampilan.
        self.signal_axis.set_title(
            "Sinyal EOG filtered 15 Hz",
            fontsize=10,
            pad=5,
        )
        self.signal_axis.set_xlabel(
            "Waktu relatif (detik)",
            fontsize=8,
        )
        self.signal_axis.set_ylabel(
            "ADC relatif",
            fontsize=8,
        )
        self.signal_axis.tick_params(labelsize=7)
        self.signal_axis.grid(True, alpha=0.25)
        self.signal_line, = self.signal_axis.plot(
            [],
            [],
            linewidth=1.25,
        )
        self.signal_axis.axhline(0.0, linewidth=0.7, alpha=0.55)
        self.signal_axis.set_xlim(-DISPLAY_SEC, 0.0)
        self.signal_axis.set_ylim(
            -DISPLAY_MIN_HALF_SPAN_ADC,
            DISPLAY_MIN_HALF_SPAN_ADC,
        )

        placeholder = np.zeros(
            (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
            dtype=np.uint8,
        )
        self.camera_artist = self.camera_axis.imshow(
            placeholder,
            aspect="equal",
        )
        self.camera_axis.set_title(
            "Preview kamera",
            fontsize=9,
            pad=4,
        )
        self.camera_status_text = self.camera_axis.text(
            0.5,
            0.5,
            self.camera_preview.status,
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
            transform=self.camera_axis.transAxes,
        )

        self.result_text = self.status_axis.text(
            0.5,
            0.70,
            "Belum ada prediksi",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            wrap=True,
            transform=self.status_axis.transAxes,
        )
        self.robot_text = self.status_axis.text(
            0.5,
            0.24,
            "Robot belum terhubung",
            ha="center",
            va="center",
            fontsize=8.2,
            wrap=True,
            transform=self.status_axis.transAxes,
        )

        self.figure.canvas.manager.set_window_title(
            "Realtime EOG + Pioneer P3DX"
        )
        self._configure_window_size()

        self.filtered_buffer: deque[float] = deque(
            maxlen=DISPLAY_SAMPLES
        )
        self.last_plot_update = 0.0
        self.last_camera_update = 0.0
        self.display_half_span = DISPLAY_MIN_HALF_SPAN_ADC

    def _configure_window_size(self) -> None:
        """Paksa ukuran jendela kecil jika backend mendukung Tk."""
        try:
            manager = self.figure.canvas.manager
            window = getattr(manager, "window", None)
            if window is not None and hasattr(window, "wm_geometry"):
                window.wm_geometry(
                    f"{WINDOW_WIDTH_PX}x{WINDOW_HEIGHT_PX}+8+40"
                )
        except Exception:
            pass

    def append_signal(self, values: np.ndarray) -> None:
        self.filtered_buffer.extend(
            np.asarray(values, dtype=float).tolist()
        )

    @staticmethod
    def _smooth_for_display(values: np.ndarray) -> np.ndarray:
        if values.size < DISPLAY_SMOOTH_SAMPLES:
            return values

        kernel = np.ones(
            DISPLAY_SMOOTH_SAMPLES,
            dtype=np.float64,
        ) / DISPLAY_SMOOTH_SAMPLES

        return np.convolve(values, kernel, mode="same")

    def _update_camera_if_due(self, force: bool = False) -> None:
        now = time.monotonic()
        if (
            not force
            and now - self.last_camera_update < 1.0 / CAMERA_FPS
        ):
            return

        self.last_camera_update = now
        frame = self.camera_preview.get_frame()

        if frame is not None:
            self.camera_artist.set_data(frame)
            self.camera_status_text.set_text("")
        else:
            self.camera_status_text.set_text(
                self.camera_preview.status
            )

    def update_signal_if_due(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_plot_update < DISPLAY_UPDATE_SEC:
            self._update_camera_if_due(force=False)
            return

        self.last_plot_update = now
        values = np.asarray(self.filtered_buffer, dtype=np.float64)

        if values.size > 0:
            # Centering ini hanya untuk visualisasi. Input model tetap
            # memakai sinyal filtered asli dan baseline per trial.
            center = float(np.median(values))
            centered = values - center
            centered = self._smooth_for_display(centered)

            if centered.size > DISPLAY_MAX_POINTS:
                indices = np.linspace(
                    0,
                    centered.size - 1,
                    DISPLAY_MAX_POINTS,
                    dtype=int,
                )
                plot_values = centered[indices]
                plot_time = (
                    np.arange(centered.size, dtype=float)[indices]
                    - (centered.size - 1)
                ) / RAW_FS
            else:
                plot_values = centered
                plot_time = (
                    np.arange(centered.size, dtype=float)
                    - (centered.size - 1)
                ) / RAW_FS

            self.signal_line.set_data(plot_time, plot_values)
            self.signal_axis.set_xlim(-DISPLAY_SEC, 0.0)

            low = float(np.percentile(centered, 1.0))
            high = float(np.percentile(centered, 99.0))
            target_half = max(
                DISPLAY_MIN_HALF_SPAN_ADC,
                abs(low) * 1.15,
                abs(high) * 1.15,
            )
            target_half = min(
                DISPLAY_MAX_HALF_SPAN_ADC,
                target_half,
            )

            if target_half > self.display_half_span:
                factor = DISPLAY_SPAN_GROW
            else:
                factor = 1.0 - DISPLAY_SPAN_SHRINK

            self.display_half_span += factor * (
                target_half - self.display_half_span
            )
            self.display_half_span = float(
                np.clip(
                    self.display_half_span,
                    DISPLAY_MIN_HALF_SPAN_ADC,
                    DISPLAY_MAX_HALF_SPAN_ADC,
                )
            )

            self.signal_axis.set_ylim(
                -self.display_half_span,
                self.display_half_span,
            )

        self._update_camera_if_due(force=force)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)

    def update_text(
        self,
        title: str,
        instruction: str = "",
        progress: str = "",
        result: str = "",
        robot: str = "",
    ) -> None:
        self.title_text.set_text(title)
        self.instruction_text.set_text(instruction)
        self.progress_text.set_text(progress)

        if result:
            self.result_text.set_text(result)

        if robot:
            self.robot_text.set_text(robot)

        self._update_camera_if_due(force=True)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.01)

    def close(self) -> None:
        plt.ioff()
        plt.close(self.figure)


# ============================================================
# 8. RAW LOGGER
# ============================================================

class RawSessionLogger:
    COLUMNS = [
        "sample_index",
        "waktu_detik",
        "sinyal_a1_raw",
        "sinyal_a1_filtered",
        "phase",
        "trial_ke",
        "cue_id",
        "expected_model_label",
        "expected_command",
    ]

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.handle = None
        self.writer = None
        self.sample_index = 0
        self.rows_since_flush = 0

        if path is not None:
            self.handle = path.open("w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.handle, fieldnames=self.COLUMNS)
            self.writer.writeheader()

    def write_block(
        self,
        raw: np.ndarray,
        filtered: np.ndarray,
        phase: str,
        trial_number: int,
        cue: CommandCue,
    ) -> None:
        if self.writer is None:
            self.sample_index += len(raw)
            return

        rows = []
        for raw_value, filtered_value in zip(raw, filtered):
            rows.append({
                "sample_index": self.sample_index,
                "waktu_detik": self.sample_index / RAW_FS,
                "sinyal_a1_raw": float(raw_value),
                "sinyal_a1_filtered": float(filtered_value),
                "phase": phase,
                "trial_ke": trial_number,
                "cue_id": cue.cue_id,
                "expected_model_label": cue.expected_model_label,
                "expected_command": cue.expected_command,
            })
            self.sample_index += 1

        self.writer.writerows(rows)
        self.rows_since_flush += len(rows)
        if self.rows_since_flush >= 1000:
            self.handle.flush()
            self.rows_since_flush = 0

    def close(self) -> None:
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()
            self.handle = None
            self.writer = None


# ============================================================
# 9. AKUISISI, PREPROCESSING, DAN INFERENCE
# ============================================================

def capture_exact_samples(
    device: BITalino,
    filter_state: ContinuousLowpassFilter,
    display: RealtimeDisplay,
    logger: RawSessionLogger,
    duration_sec: float,
    phase: str,
    trial_number: int,
    cue: CommandCue,
) -> tuple[np.ndarray, np.ndarray]:
    target_samples = int(round(duration_sec * RAW_FS))
    raw_parts: list[np.ndarray] = []
    filtered_parts: list[np.ndarray] = []
    collected = 0

    while collected < target_samples:
        if not plt.fignum_exists(display.figure.number):
            raise KeyboardInterrupt("Jendela GUI ditutup.")

        remaining = target_samples - collected
        block_size = min(READ_BLOCK, remaining)
        block = np.asarray(device.read(block_size))

        if block.ndim != 2 or block.shape[1] <= A1_COLUMN:
            raise RuntimeError(
                "Format data BITalino tidak sesuai. "
                f"Shape ditemukan: {block.shape}"
            )

        raw = block[:, A1_COLUMN].astype(np.float64)
        if not np.isfinite(raw).all():
            raise RuntimeError("Data BITalino mengandung NaN/Inf.")

        filtered = filter_state.process(raw)

        raw_parts.append(raw)
        filtered_parts.append(filtered)
        logger.write_block(
            raw,
            filtered,
            phase,
            trial_number,
            cue,
        )

        display.append_signal(filtered)
        display.update_signal_if_due()
        collected += len(raw)

    return np.concatenate(raw_parts), np.concatenate(filtered_parts)


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(1e-9, 1.4826 * mad)


def preprocess_and_predict(
    baseline_filtered: np.ndarray,
    action_filtered: np.ndarray,
) -> dict[str, Any]:
    if baseline_filtered.shape != (BASELINE_SAMPLES,):
        raise RuntimeError(
            f"Baseline {baseline_filtered.shape}, harus ({BASELINE_SAMPLES},)."
        )
    if action_filtered.shape != (ACTION_SAMPLES,):
        raise RuntimeError(
            f"Action {action_filtered.shape}, harus ({ACTION_SAMPLES},)."
        )

    baseline_median = float(np.median(baseline_filtered))
    corrected = action_filtered - baseline_median
    processed = corrected[::DOWNSAMPLE_FACTOR]

    if processed.shape != (WINDOW_SAMPLES,):
        raise RuntimeError(
            f"Input processed {processed.shape}, harus ({WINDOW_SAMPLES},)."
        )

    normalized = (processed - NORMALIZATION_MEAN) / NORMALIZATION_STD
    if not np.isfinite(normalized).all():
        raise RuntimeError("Hasil normalisasi mengandung NaN/Inf.")

    tensor = torch.tensor(
        normalized[np.newaxis, np.newaxis, :],
        dtype=torch.float32,
    )

    inference_start = time.perf_counter()
    with torch.inference_mode():
        logits = MODEL(tensor)
        probabilities = (
            torch.softmax(logits, dim=1)
            .cpu()
            .numpy()[0]
        )
    inference_ms = (time.perf_counter() - inference_start) * 1000.0

    prediction_index = int(np.argmax(probabilities))
    prediction = CLASS_ORDER[prediction_index]
    confidence = float(probabilities[prediction_index])

    sorted_probabilities = np.sort(probabilities)
    margin = float(sorted_probabilities[-1] - sorted_probabilities[-2])

    return {
        "prediction": prediction,
        "confidence": confidence,
        "margin": margin,
        "probabilities": probabilities,
        "inference_ms": inference_ms,
        "baseline_median": baseline_median,
        "baseline_std": float(np.std(baseline_filtered)),
        "baseline_p2p": float(np.ptp(baseline_filtered)),
        "baseline_sigma": robust_sigma(baseline_filtered),
        "corrected": corrected,
        "processed": processed,
        "normalized": normalized,
    }


# ============================================================
# 10. BLINK DETECTOR DAN KEPUTUSAN COMMAND
# ============================================================

def make_envelope(values: np.ndarray) -> np.ndarray:
    smooth_samples = max(1, int(BLINK_ENVELOPE_SMOOTH_SEC * RAW_FS))
    kernel = np.ones(smooth_samples, dtype=np.float64) / smooth_samples
    return np.convolve(np.abs(values), kernel, mode="same")


def merge_segments(
    segments: list[tuple[int, int]],
    max_gap_samples: int,
) -> list[tuple[int, int]]:
    if not segments:
        return []

    merged = [list(segments[0])]
    for start, end in segments[1:]:
        gap = start - merged[-1][1] - 1
        if gap <= max_gap_samples:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [(int(start), int(end)) for start, end in merged]


def detect_blinks(
    corrected: np.ndarray,
    baseline_sigma: float,
) -> tuple[int, list[dict[str, Any]], float]:
    threshold = float(
        np.clip(
            max(
                MIN_BLINK_THRESHOLD_ADC,
                BLINK_SIGMA_MULTIPLIER * baseline_sigma,
            ),
            MIN_BLINK_THRESHOLD_ADC,
            MAX_BLINK_THRESHOLD_ADC,
        )
    )

    envelope = make_envelope(corrected)
    mask = envelope >= threshold
    changes = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1

    segments = merge_segments(
        list(zip(starts.tolist(), ends.tolist())),
        int(BLINK_MERGE_GAP_SEC * RAW_FS),
    )

    min_samples = int(BLINK_MIN_EVENT_SEC * RAW_FS)
    max_samples = int(BLINK_MAX_EVENT_SEC * RAW_FS)

    events: list[dict[str, Any]] = []
    for start, end in segments:
        duration = end - start + 1
        if min_samples <= duration <= max_samples:
            events.append({
                "start_sample": start,
                "end_sample": end,
                "duration_ms": duration / RAW_FS * 1000.0,
                "peak_adc": float(np.max(envelope[start:end + 1])),
            })

    return len(events), events, threshold


def decide_command(
    prediction: str,
    confidence: float,
    probabilities: np.ndarray,
    blink_count: int,
    clipped_samples: int,
) -> tuple[str, str, str]:
    if clipped_samples > 0:
        return (
            "HOLD",
            "REJECTED_CLIPPING",
            "Command ditahan karena terdapat sampel clipping.",
        )

    probability_by_class = {
        class_name: float(probabilities[index])
        for index, class_name in enumerate(CLASS_ORDER)
    }
    p_kedip = probability_by_class["kedip"]

    if blink_count >= 3:
        return (
            "HOLD",
            "REJECTED_TOO_MANY_BLINKS",
            "Terdeteksi lebih dari dua event kedip.",
        )

    if blink_count == 2:
        if p_kedip >= FORWARD_KEDIP_SUPPORT_MIN:
            return (
                "MAJU",
                "ACCEPTED_DOUBLE_BLINK",
                "Dua kedip terdeteksi dan probabilitas kedip memenuhi threshold.",
            )
        return (
            "HOLD",
            "REJECTED_DOUBLE_BLINK_LOW_SUPPORT",
            "Dua kedip terdeteksi tetapi dukungan CNN belum cukup.",
        )

    if blink_count == 1:
        if p_kedip >= STOP_KEDIP_SUPPORT_MIN:
            return (
                "STOP",
                "ACCEPTED_SINGLE_BLINK",
                "Satu kedip terdeteksi dan diterjemahkan menjadi STOP.",
            )
        return (
            "HOLD",
            "REJECTED_SINGLE_BLINK_LOW_SUPPORT",
            "Satu event terdeteksi tetapi dukungan CNN terlalu rendah.",
        )

    if prediction == "diam":
        return (
            "HOLD",
            "ACCEPTED_DIAM",
            "Prediksi diam, robot tetap HOLD.",
        )

    if prediction == "kiri" and confidence >= DIRECTION_CONFIDENCE_MIN:
        return (
            "BELOK_KIRI",
            "ACCEPTED_LEFT",
            "Prediksi kiri memenuhi threshold confidence.",
        )

    if prediction == "kanan" and confidence >= DIRECTION_CONFIDENCE_MIN:
        return (
            "BELOK_KANAN",
            "ACCEPTED_RIGHT",
            "Prediksi kanan memenuhi threshold confidence.",
        )

    if prediction == "kedip":
        return (
            "HOLD",
            "REJECTED_KEDIP_WITHOUT_EVENT",
            "CNN memprediksi kedip tetapi blink counter tidak menemukan event valid.",
        )

    return (
        "HOLD",
        "REJECTED_LOW_CONFIDENCE",
        "Confidence arah belum mencapai threshold.",
    )


# ============================================================
# 11. PENYIMPANAN HASIL
# ============================================================

def write_trial_results(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    model_correct = sum(int(row["model_label_correct"]) for row in rows)
    command_correct = sum(int(row["command_correct"]) for row in rows)
    robot_completed = sum(int(row["robot_execution_completed"]) for row in rows)

    per_command: dict[str, dict[str, int | float]] = {}
    for command_name in ["HOLD", "BELOK_KIRI", "BELOK_KANAN", "STOP", "MAJU"]:
        selected = [row for row in rows if row["expected_command"] == command_name]
        count = len(selected)
        correct = sum(int(row["command_correct"]) for row in selected)
        per_command[command_name] = {
            "trials": count,
            "correct": correct,
            "accuracy": (correct / count) if count else 0.0,
        }

    return {
        "subject_id": SUBJECT_ID,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": "20 trial; 4 repetitions x 5 robot commands",
        "total_trials": total,
        "model_label_correct": model_correct,
        "model_label_accuracy": (model_correct / total) if total else 0.0,
        "command_correct": command_correct,
        "command_accuracy": (command_correct / total) if total else 0.0,
        "robot_execution_completed": robot_completed,
        "robot_execution_completion_rate": (
            robot_completed / total if total else 0.0
        ),
        "per_expected_command": per_command,
        "model_path": str(MODEL_PATH),
        "model_sha256": MODEL_CONFIG["actual_torchscript_sha256"],
        "training_subjects": sorted(TRAINING_SUBJECTS),
        "normalization_mean": NORMALIZATION_MEAN,
        "normalization_std": NORMALIZATION_STD,
        "lowpass_hz": MODEL_LOWPASS_HZ,
        "filter_order": MODEL_FILTER_ORDER,
        "raw_sampling_rate_hz": RAW_FS,
        "processed_sampling_rate_hz": PROCESSED_FS,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "window_samples": WINDOW_SAMPLES,
        "direction_confidence_min": DIRECTION_CONFIDENCE_MIN,
        "stop_kedip_support_min": STOP_KEDIP_SUPPORT_MIN,
        "forward_kedip_support_min": FORWARD_KEDIP_SUPPORT_MIN,
    }


# ============================================================
# 12. PROGRAM UTAMA
# ============================================================

def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_ROOT / f"{SUBJECT_ID}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=False)

    trial_log_path = session_dir / "robot_command_trials.csv"
    summary_path = session_dir / "summary.json"
    raw_log_path = None if ARGS.no_raw_log else session_dir / "raw_session.csv"

    print("=" * 76)
    print("UJI REALTIME EOG 1D-CNN + COMMAND ROBOT")
    print("=" * 76)
    print("Subjek               :", SUBJECT_ID)
    print("Model                :", MODEL_PATH)
    print("Subjek training      :", sorted(TRAINING_SUBJECTS))
    print("Normalisasi mean     :", f"{NORMALIZATION_MEAN:.6f}")
    print("Normalisasi std      :", f"{NORMALIZATION_STD:.6f}")
    print("Preprocessing        :", "LPF 15 Hz orde 4 causal, median baseline, 1000->250 Hz")
    print("Input model          :", f"1 x {WINDOW_SAMPLES}")
    print("Total trial command  :", TOTAL_TRIALS)
    print("Kamera               :", "nonaktif" if ARGS.no_camera else f"index {ARGS.camera_index}")
    print("Output               :", session_dir)
    print()

    schedule = build_schedule()
    camera_preview = CameraPreview(
        camera_index=ARGS.camera_index,
        enabled=not ARGS.no_camera,
    )
    camera_preview.start()
    display = RealtimeDisplay(camera_preview)
    filter_state = ContinuousLowpassFilter()
    raw_logger = RawSessionLogger(raw_log_path)
    robot = CoppeliaRobotController()
    device = None
    trial_rows: list[dict[str, Any]] = []

    try:
        display.update_text(
            "MENGHUBUNGKAN SISTEM",
            "Menghubungkan CoppeliaSim dan BITalino.",
        )

        robot.start()
        robot_snapshot = robot.snapshot()
        print("CoppeliaSim          :", robot_snapshot["message"])

        device = BITalino(MAC_ADDRESS)
        device.start(RAW_FS, CHANNELS)
        print("BITalino             : terhubung", MAC_ADDRESS)

        display.update_text(
            "SISTEM SIAP",
            "Tatap titik tengah dan jaga posisi kepala tetap lurus.",
            robot=robot.snapshot()["message"],
        )

        for number in range(COUNTDOWN_SEC, 0, -1):
            display.update_text(
                str(number),
                "Sesi uji command segera dimulai.",
                robot=robot.snapshot()["message"],
            )
            time.sleep(1.0)

        for trial_number, cue in enumerate(schedule, start=1):
            progress = f"Trial {trial_number}/{TOTAL_TRIALS}"

            # ---------------- BASELINE 2 DETIK ----------------
            display.update_text(
                "TATAP TITIK TENGAH",
                "Jangan gerakkan mata dan jangan berkedip.",
                progress=progress + " | Baseline 2 detik",
                robot=robot.snapshot()["message"],
            )
            baseline_raw, baseline_filtered = capture_exact_samples(
                device,
                filter_state,
                display,
                raw_logger,
                BASELINE_SEC,
                "baseline",
                trial_number,
                cue,
            )

            # ---------------- ACTION 2 DETIK ----------------
            display.update_text(
                cue.title,
                cue.instruction,
                progress=progress + " | Aksi 2 detik",
                robot=robot.snapshot()["message"],
            )
            action_raw, action_filtered = capture_exact_samples(
                device,
                filter_state,
                display,
                raw_logger,
                ACTION_SEC,
                "action",
                trial_number,
                cue,
            )

            prediction_result = preprocess_and_predict(
                baseline_filtered,
                action_filtered,
            )

            blink_count, blink_events, blink_threshold = detect_blinks(
                prediction_result["corrected"],
                prediction_result["baseline_sigma"],
            )

            clipped_samples = int(
                np.sum(
                    (baseline_raw <= 1)
                    | (baseline_raw >= 1022)
                )
                + np.sum(
                    (action_raw <= 1)
                    | (action_raw >= 1022)
                )
            )

            final_command, decision_status, decision_reason = decide_command(
                prediction_result["prediction"],
                prediction_result["confidence"],
                prediction_result["probabilities"],
                blink_count,
                clipped_samples,
            )

            request_id = robot.submit(final_command)
            model_label_correct = (
                prediction_result["prediction"] == cue.expected_model_label
            )
            command_correct = final_command == cue.expected_command

            probabilities = prediction_result["probabilities"]
            result_text = (
                f"Prediksi: {prediction_result['prediction'].upper()} "
                f"({prediction_result['confidence'] * 100:.1f}%) | "
                f"Blink: {blink_count} | Command: {final_command}"
            )

            print(
                f"Trial {trial_number:02d} | "
                f"cue={cue.cue_id:12s} | "
                f"label={prediction_result['prediction']:6s} | "
                f"conf={prediction_result['confidence']:.3f} | "
                f"blink={blink_count} | "
                f"command={final_command:12s} | "
                f"{'BENAR' if command_correct else 'SALAH'}"
            )

            # ---------------- RETURN 2 DETIK ----------------
            display.update_text(
                "KEMBALI KE TITIK TENGAH",
                "Kembalikan pandangan ke titik tengah.",
                progress=progress + " | Return 2 detik",
                result=result_text,
                robot=robot.snapshot()["message"],
            )
            capture_exact_samples(
                device,
                filter_state,
                display,
                raw_logger,
                RETURN_SEC,
                "return",
                trial_number,
                cue,
            )

            # ---------------- REST 5 DETIK ----------------
            display.update_text(
                "JEDA",
                "Istirahat dan tetap lihat titik tengah.",
                progress=progress + " | Jeda 5 detik",
                result=result_text,
                robot=robot.snapshot()["message"],
            )
            capture_exact_samples(
                device,
                filter_state,
                display,
                raw_logger,
                REST_SEC,
                "rest",
                trial_number,
                cue,
            )

            robot_snapshot = robot.snapshot()
            robot_execution_completed = robot.was_completed(request_id)

            row: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "subject_id": SUBJECT_ID,
                "trial_ke": trial_number,
                "cue_id": cue.cue_id,
                "expected_model_label": cue.expected_model_label,
                "predicted_model_label": prediction_result["prediction"],
                "model_label_correct": bool(model_label_correct),
                "confidence": float(prediction_result["confidence"]),
                "margin": float(prediction_result["margin"]),
                "prob_diam": float(probabilities[0]),
                "prob_kiri": float(probabilities[1]),
                "prob_kanan": float(probabilities[2]),
                "prob_kedip": float(probabilities[3]),
                "blink_count": int(blink_count),
                "blink_events": json.dumps(blink_events, ensure_ascii=False),
                "blink_threshold_adc": float(blink_threshold),
                "expected_command": cue.expected_command,
                "final_command": final_command,
                "command_correct": bool(command_correct),
                "decision_status": decision_status,
                "decision_reason": decision_reason,
                "queued_request_id": request_id if request_id is not None else "",
                "robot_execution_completed": bool(robot_execution_completed),
                "robot_last_applied_command": robot_snapshot["last_applied_command"],
                "robot_status_end_trial": robot_snapshot["status"],
                "inference_ms": float(prediction_result["inference_ms"]),
                "baseline_median_adc": float(prediction_result["baseline_median"]),
                "baseline_std_adc": float(prediction_result["baseline_std"]),
                "baseline_p2p_adc": float(prediction_result["baseline_p2p"]),
                "baseline_robust_sigma_adc": float(prediction_result["baseline_sigma"]),
                "clipped_samples_baseline_action": clipped_samples,
                "normalization_mean": NORMALIZATION_MEAN,
                "normalization_std": NORMALIZATION_STD,
            }
            trial_rows.append(row)
            write_trial_results(trial_log_path, trial_rows)

        summary = build_summary(trial_rows)
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        display.update_signal_if_due(force=True)
        display.update_text(
            "SESI SELESAI",
            (
                f"Akurasi command: {summary['command_correct']}/"
                f"{summary['total_trials']} "
                f"({summary['command_accuracy'] * 100:.2f}%)"
            ),
            result=f"Hasil tersimpan di:\n{session_dir}",
            robot=robot.snapshot()["message"],
        )

        print()
        print("=" * 76)
        print("HASIL AKHIR")
        print("=" * 76)
        print(
            "Akurasi label model : "
            f"{summary['model_label_correct']}/{summary['total_trials']} "
            f"= {summary['model_label_accuracy'] * 100:.2f}%"
        )
        print(
            "Akurasi command     : "
            f"{summary['command_correct']}/{summary['total_trials']} "
            f"= {summary['command_accuracy'] * 100:.2f}%"
        )
        print("Log trial            :", trial_log_path)
        print("Ringkasan             :", summary_path)
        if raw_log_path is not None:
            print("Log raw               :", raw_log_path)
        print()
        print("Tutup jendela grafik untuk keluar.")

        while plt.fignum_exists(display.figure.number):
            plt.pause(0.1)

    except KeyboardInterrupt:
        print("\nSesi dihentikan oleh pengguna.")

    except Exception:
        print("\nTERJADI ERROR")
        traceback.print_exc()
        raise

    finally:
        try:
            write_trial_results(trial_log_path, trial_rows)
            if trial_rows:
                summary_path.write_text(
                    json.dumps(
                        build_summary(trial_rows),
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
        except Exception:
            traceback.print_exc()

        raw_logger.close()

        if device is not None:
            try:
                device.stop()
            except Exception:
                pass
            try:
                device.close()
            except Exception:
                pass

        robot.shutdown()
        camera_preview.shutdown()
        display.close()


if __name__ == "__main__":
    main()
