"""
BITalino EOG Data Acquisition
=============================

Records single-channel EOG data from BITalino A1 at 1000 Hz using a
20-trial protocol (left, right, blink, idle). Each trial contains 2 s
baseline, 2 s action, 2 s return-to-center, and 5 s rest.

The CSV stores raw A1 ADC values. A 15 Hz low-pass filtered signal is used
only for the live display. Participant recordings are written under
``data/training`` and should not be committed to the public repository.

Set the BITalino address with the ``BITALINO_MAC`` environment variable or
enter it when the script starts.
"""

from pathlib import Path
from bisect import bisect_right
import csv
import random
import re
import os
import threading
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

try:
    from bitalino import BITalino
except ImportError as exc:
    raise SystemExit(
        "Library bitalino belum terpasang.\n"
        "Install ke Python yang dipakai menjalankan script:\n"
        "python -m pip install bitalino"
    ) from exc


# =========================================================
# 1. KONFIGURASI FINAL
# =========================================================

# Read the BITalino address from an environment variable instead of
# publishing a device identifier in source control.
MAC_ADDRESS = os.getenv("BITALINO_MAC", "").strip()
if not MAC_ADDRESS:
    MAC_ADDRESS = input("Enter BITalino MAC address: ").strip()
if not MAC_ADDRESS:
    raise SystemExit("BITalino MAC address is required.")

RAW_FS = 1000
CHANNELS = [0]  # BITalino A1
ANALOG_START_COLUMN = 5
READ_BLOCK_SAMPLES = 50

CLASSES = ["kiri", "kanan", "kedip", "diam"]

# Lima kelompok internal.
# Setiap kelompok berisi satu kiri, satu kanan, satu kedip,
# dan satu diam dengan urutan yang diacak.
REPETITIONS_PER_CLASS = 5
TOTAL_TRIALS = REPETITIONS_PER_CLASS * len(CLASSES)

COUNTDOWN_SEC = 3.0
BASELINE_SEC = 2.0
ACTION_SEC = 2.0
RETURN_SEC = 2.0
REST_SEC = 5.0
TRIAL_SEC = BASELINE_SEC + ACTION_SEC + RETURN_SEC + REST_SEC

# Filter hanya untuk tampilan realtime.
# CSV tetap menyimpan RAW ADC A1 pada 1000 Hz.
DISPLAY_LOWPASS_HZ = 15.0
DISPLAY_FILTER_ORDER = 4
DISPLAY_SEC = 4.0
DISPLAY_SAMPLES = int(DISPLAY_SEC * RAW_FS)

# Repository-relative output path. Raw participant recordings should not
# be committed to the public repository.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "training"


# =========================================================
# 2. KODE SUBJEK DAN NAMA FILE
# =========================================================

def ask_subject_code():
    print("=" * 68)
    print("AKUISISI DATA TRAINING EOG")
    print("=" * 68)
    print("Contoh kode subjek : S01")
    print("Jika mengulang     : S01R2")
    print()

    while True:
        code = input("Masukkan kode subjek: ").strip().upper()

        if not re.fullmatch(r"S\d{2}(R\d+)?", code):
            print(
                "Format tidak sesuai. Gunakan S01, S02, ..., "
                "atau S01R2 untuk pengulangan."
            )
            continue

        return code


SUBJECT_CODE = ask_subject_code()

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

FINAL_CSV_PATH = OUTPUT_ROOT / f"{SUBJECT_CODE}.csv"
TEMP_CSV_PATH = OUTPUT_ROOT / f"{SUBJECT_CODE}_sedang_rekam.tmp"

if FINAL_CSV_PATH.exists():
    raise SystemExit(
        f"\nFile sudah ada dan tidak akan ditimpa:\n{FINAL_CSV_PATH}\n\n"
        "Gunakan kode pengulangan, misalnya S01R2."
    )

# Membersihkan file sementara dari program yang sebelumnya tidak selesai.
if TEMP_CSV_PATH.exists():
    try:
        TEMP_CSV_PATH.unlink()
    except OSError as exc:
        raise SystemExit(
            f"Tidak dapat menghapus file sementara:\n{TEMP_CSV_PATH}\n{exc}"
        ) from exc


# =========================================================
# 3. STATE PROGRAM
# =========================================================

lock = threading.RLock()
stop_event = threading.Event()

device = None
reader_thread = None

absolute_sample = 0
last_sequence = None

raw_plot_buffer = []
filtered_plot_buffer = []
time_plot_buffer = []

display_sos = butter(
    DISPLAY_FILTER_ORDER,
    DISPLAY_LOWPASS_HZ,
    btype="lowpass",
    fs=RAW_FS,
    output="sos",
)
display_filter_zi = None

csv_handle = None
csv_writer = None
samples_since_flush = 0

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

    "message": "Tekan C untuk menghubungkan BITalino.",
}


# =========================================================
# 4. JADWAL 20 TRIAL
# =========================================================

def build_schedule():
    schedule = []
    offset = 0
    trial_id = 0
    secure_random = random.SystemRandom()

    phases = [
        ("baseline", BASELINE_SEC),
        ("action", ACTION_SEC),
        ("return", RETURN_SEC),
        ("rest", REST_SEC),
    ]

    for group_number in range(1, REPETITIONS_PER_CLASS + 1):
        labels = CLASSES.copy()
        secure_random.shuffle(labels)

        for label in labels:
            trial_id += 1

            for phase, duration_sec in phases:
                duration_samples = int(round(duration_sec * RAW_FS))

                schedule.append({
                    "trial_id": trial_id,
                    "group": group_number,
                    "target_label": label,
                    "phase": phase,
                    "start_sample": offset,
                    "end_sample_exclusive": offset + duration_samples,
                })

                offset += duration_samples

    return schedule


def get_segment(session_sample, schedule, end_samples):
    if session_sample < 0 or not schedule:
        return None

    if session_sample >= end_samples[-1]:
        return None

    index = bisect_right(end_samples, session_sample)
    return schedule[index]


# =========================================================
# 5. SATU-SATUNYA FILE OUTPUT
# =========================================================

CSV_COLUMNS = [
    "waktu_detik",
    "sinyal_a1",
    "status",
    "trial_ke",
]


def open_temporary_csv():
    global csv_handle, csv_writer

    csv_handle = TEMP_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
        buffering=1,
    )

    csv_writer = csv.DictWriter(
        csv_handle,
        fieldnames=CSV_COLUMNS,
    )
    csv_writer.writeheader()


def close_csv():
    global csv_handle, csv_writer, samples_since_flush

    if csv_handle is not None:
        try:
            csv_handle.flush()
        except Exception:
            pass

        try:
            csv_handle.close()
        except Exception:
            pass

    csv_handle = None
    csv_writer = None
    samples_since_flush = 0


def remove_temporary_csv():
    close_csv()

    if TEMP_CSV_PATH.exists():
        try:
            TEMP_CSV_PATH.unlink()
        except OSError:
            pass


def finalize_session():
    with lock:
        if state["session_saved"]:
            return

        aborted = state["session_aborted"]
        written = state["session_samples_written"]
        state["session_saved"] = True

    close_csv()

    if aborted:
        if TEMP_CSV_PATH.exists():
            try:
                TEMP_CSV_PATH.unlink()
            except OSError:
                pass

        with lock:
            state["message"] = (
                "Sesi dibatalkan. Data parsial dibuang. "
                "Tekan S untuk mengulang."
            )

        print("\nSESI DIBATALKAN")
        print("Data parsial tidak disimpan.")
        return

    if written != int(TOTAL_TRIALS * TRIAL_SEC * RAW_FS):
        if TEMP_CSV_PATH.exists():
            try:
                TEMP_CSV_PATH.unlink()
            except OSError:
                pass

        with lock:
            state["session_aborted"] = True
            state["message"] = (
                "Jumlah sampel tidak lengkap. "
                "Data dibuang dan sesi harus diulang."
            )

        print("\nDATA TIDAK LENGKAP")
        print("Jumlah sampel :", written)
        print("Data tidak disimpan.")
        return

    TEMP_CSV_PATH.replace(FINAL_CSV_PATH)

    with lock:
        state["completed_once"] = True
        state["message"] = (
            f"Sesi selesai. Data tersimpan sebagai {FINAL_CSV_PATH.name}"
        )

    print("\nSESI SELESAI")
    print("Subjek :", SUBJECT_CODE)
    print("Trial  :", TOTAL_TRIALS)
    print("Sampel :", written)
    print("File   :", FINAL_CSV_PATH)


# =========================================================
# 6. BUFFER TAMPILAN
# =========================================================

def append_plot_data(time_value, raw_value, filtered_value):
    raw_plot_buffer.append(float(raw_value))
    filtered_plot_buffer.append(float(filtered_value))
    time_plot_buffer.append(float(time_value))

    overflow = len(raw_plot_buffer) - DISPLAY_SAMPLES

    if overflow > 0:
        del raw_plot_buffer[:overflow]
        del filtered_plot_buffer[:overflow]
        del time_plot_buffer[:overflow]


# =========================================================
# 7. PEMBACAAN BITALINO
# =========================================================

def acquisition_worker():
    global absolute_sample
    global last_sequence
    global display_filter_zi
    global samples_since_flush

    try:
        while not stop_event.is_set():
            block = device.read(READ_BLOCK_SAMPLES)

            if block.ndim != 2 or block.shape[1] <= ANALOG_START_COLUMN:
                raise RuntimeError(
                    f"Format data BITalino tidak sesuai: {block.shape}"
                )

            sequence_values = block[:, 0].astype(int)
            a1_block = block[:, ANALOG_START_COLUMN].astype(float)

            if display_filter_zi is None:
                display_filter_zi = (
                    sosfilt_zi(display_sos) * a1_block[0]
                )

            filtered_block, display_filter_zi = sosfilt(
                display_sos,
                a1_block,
                zi=display_filter_zi,
            )

            with lock:
                for row_index in range(len(block)):
                    sequence = int(sequence_values[row_index])

                    if last_sequence is not None:
                        expected = (last_sequence + 1) % 16

                        if sequence != expected and state["session_active"]:
                            missing_before = (sequence - expected) % 16

                            if missing_before == 0:
                                missing_before = 16

                            state["packet_gap_events"] += 1
                            state["estimated_missing_samples"] += missing_before

                    last_sequence = sequence

                    raw_a1 = int(round(float(a1_block[row_index])))
                    filtered_a1 = float(filtered_block[row_index])
                    sample_time = absolute_sample / RAW_FS

                    append_plot_data(
                        sample_time,
                        raw_a1,
                        filtered_a1,
                    )

                    if state["session_active"]:
                        session_sample = (
                            absolute_sample
                            - state["session_start_absolute_sample"]
                        )

                        segment = get_segment(
                            session_sample,
                            state["schedule"],
                            state["schedule_end_samples"],
                        )

                        if segment is None:
                            state["session_active"] = False
                            state["session_finished"] = True
                            state["current_segment"] = None
                            state["message"] = "Perekaman selesai. Menyimpan CSV."

                        else:
                            phase = segment["phase"]

                            if phase == "baseline":
                                status = "tengah_awal"
                            elif phase == "return":
                                status = "tengah_kembali"
                            elif phase == "rest":
                                status = "jeda"
                            else:
                                status = segment["target_label"]

                            csv_writer.writerow({
                                "waktu_detik": f"{session_sample / RAW_FS:.3f}",
                                "sinyal_a1": raw_a1,
                                "status": status,
                                "trial_ke": segment["trial_id"],
                            })

                            state["session_samples_written"] += 1
                            samples_since_flush += 1

                            if samples_since_flush >= RAW_FS:
                                csv_handle.flush()
                                samples_since_flush = 0

                            if raw_a1 <= 1 or raw_a1 >= 1022:
                                state["clipped_a1_samples"] += 1

                            previous = state["current_segment"]
                            state["current_segment"] = segment.copy()

                            changed = (
                                previous is None
                                or previous["trial_id"] != segment["trial_id"]
                                or previous["phase"] != segment["phase"]
                            )

                            if changed:
                                play_cue_sound(segment["phase"])

                    absolute_sample += 1

    except Exception as exc:
        traceback.print_exc()

        with lock:
            state["reader_error"] = f"{type(exc).__name__}: {exc}"
            state["connected"] = False

            if state["session_active"]:
                state["session_active"] = False
                state["session_finished"] = True
                state["session_aborted"] = True

            state["message"] = "Pembacaan BITalino gagal."


# =========================================================
# 8. SUARA CUE
# =========================================================

def play_cue_sound(phase):
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
            args=(frequency, duration_ms),
            daemon=True,
        ).start()

    except Exception:
        pass


# =========================================================
# 9. CONNECT DAN DISCONNECT
# =========================================================

def connect_device():
    global device
    global reader_thread
    global absolute_sample
    global last_sequence
    global display_filter_zi

    with lock:
        if state["connected"]:
            state["message"] = "BITalino sudah terhubung."
            return

    try:
        print(f"Menghubungkan ke {MAC_ADDRESS} ...")

        stop_event.clear()
        absolute_sample = 0
        last_sequence = None
        display_filter_zi = None

        raw_plot_buffer.clear()
        filtered_plot_buffer.clear()
        time_plot_buffer.clear()

        device = BITalino(MAC_ADDRESS)
        device.start(RAW_FS, CHANNELS)

        reader_thread = threading.Thread(
            target=acquisition_worker,
            daemon=True,
        )
        reader_thread.start()

        with lock:
            state["connected"] = True
            state["reader_error"] = None
            state["message"] = "BITalino terhubung. Tekan S untuk mulai."

        print("BITalino terhubung.")

    except Exception as exc:
        device = None
        traceback.print_exc()

        with lock:
            state["connected"] = False
            state["message"] = f"Koneksi gagal: {exc}"


def disconnect_device():
    global device, reader_thread

    stop_event.set()

    if reader_thread is not None and reader_thread.is_alive():
        reader_thread.join(timeout=2.0)

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
# 10. MULAI DAN BATALKAN SESI
# =========================================================

def request_start_session():
    with lock:
        if not state["connected"]:
            state["message"] = "BITalino belum terhubung. Tekan C."
            return

        if state["completed_once"]:
            state["message"] = (
                "Data subjek ini sudah selesai dan tersimpan. Tekan Q."
            )
            return

        if state["session_active"] or state["countdown_active"]:
            state["message"] = "Sesi sedang berjalan."
            return

        state["session_finished"] = False
        state["session_aborted"] = False
        state["session_saved"] = False
        state["countdown_active"] = True
        state["countdown_end"] = time.monotonic() + COUNTDOWN_SEC
        state["message"] = "Bersiap. Sesi dimulai setelah hitung mundur."


def begin_session_after_countdown():
    schedule = build_schedule()

    remove_temporary_csv()
    open_temporary_csv()

    with lock:
        state["schedule"] = schedule
        state["schedule_end_samples"] = [
            segment["end_sample_exclusive"]
            for segment in schedule
        ]

        state["session_start_absolute_sample"] = absolute_sample
        state["session_samples_written"] = 0
        state["packet_gap_events"] = 0
        state["estimated_missing_samples"] = 0
        state["clipped_a1_samples"] = 0

        state["current_segment"] = schedule[0].copy()

        state["countdown_active"] = False
        state["session_active"] = True
        state["session_finished"] = False
        state["session_aborted"] = False
        state["session_saved"] = False

        state["message"] = "Sesi berjalan. Ikuti perintah pada layar."

    play_cue_sound("baseline")

    print("\nPEREKAMAN DIMULAI")
    print("Subjek        :", SUBJECT_CODE)
    print("Total trial   :", TOTAL_TRIALS)
    print("Trial/kelas   :", REPETITIONS_PER_CLASS)
    print("Durasi/trial  :", TRIAL_SEC, "detik")
    print("Total rekaman :", TOTAL_TRIALS * TRIAL_SEC, "detik")


def abort_session():
    with lock:
        if state["countdown_active"]:
            state["countdown_active"] = False
            state["message"] = "Hitung mundur dibatalkan."
            return

        if not state["session_active"]:
            state["message"] = "Tidak ada sesi yang sedang direkam."
            return

        state["session_active"] = False
        state["session_finished"] = True
        state["session_aborted"] = True
        state["current_segment"] = None
        state["message"] = "Sesi dibatalkan. Data parsial akan dibuang."


# =========================================================
# 11. TAMPILAN PERINTAH
# =========================================================

def draw_marker(axis, display):
    y = -0.03

    if display == "kiri":
        axis.scatter(0, y, s=120, facecolors="none", linewidths=1.5)
        axis.scatter(-0.92, y, s=1100, facecolors="none", linewidths=3)
        axis.scatter(-0.92, y, s=240)

    elif display == "kanan":
        axis.scatter(0, y, s=120, facecolors="none", linewidths=1.5)
        axis.scatter(0.92, y, s=1100, facecolors="none", linewidths=3)
        axis.scatter(0.92, y, s=240)

    elif display == "kedip":
        axis.scatter(0, y, s=1100, facecolors="none", linewidths=3)
        axis.scatter(0, y, s=600, marker="x", linewidths=4)

    else:
        axis.scatter(0, y, s=1100, facecolors="none", linewidths=3)
        axis.scatter(0, y, s=240)


def draw_cue(
    axis,
    segment,
    countdown_remaining=None,
    completed=False,
    aborted=False,
):
    axis.clear()
    axis.set_xlim(-1.25, 1.25)
    axis.set_ylim(-1.0, 1.0)
    axis.set_aspect("equal")
    axis.axis("off")

    if countdown_remaining is not None:
        number = max(1, int(np.ceil(countdown_remaining)))

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
            "Tatap titik tengah dan jaga kepala tetap lurus.",
            ha="center",
            va="center",
            fontsize=17,
        )
        return

    if completed:
        axis.text(
            0,
            0.18,
            "SESI SELESAI",
            ha="center",
            va="center",
            fontsize=34,
            fontweight="bold",
        )
        axis.text(
            0,
            -0.25,
            f"Data tersimpan: {FINAL_CSV_PATH.name}",
            ha="center",
            va="center",
            fontsize=17,
        )
        axis.text(
            0,
            -0.52,
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
            "Data parsial dibuang. Tekan S untuk mengulang.",
            ha="center",
            va="center",
            fontsize=16,
        )
        return

    if segment is None:
        axis.text(
            0,
            0.20,
            f"AKUISISI EOG — {SUBJECT_CODE}",
            ha="center",
            va="center",
            fontsize=29,
            fontweight="bold",
        )
        axis.text(
            0,
            -0.20,
            "C = hubungkan   |   S = mulai   |   "
            "A = batalkan   |   Q = keluar",
            ha="center",
            va="center",
            fontsize=15,
        )
        return

    phase = segment["phase"]
    label = segment["target_label"]

    if phase == "baseline":
        title = "TATAP TITIK TENGAH"
        instruction = (
            "Jangan gerakkan mata dan jangan berkedip. "
            "Kepala tetap lurus."
        )
        display = "diam"

    elif phase == "return":
        title = "KEMBALIKAN MATA KE TENGAH"
        instruction = (
            "Tatap titik tengah dan jaga kepala tetap lurus."
        )
        display = "diam"

    elif phase == "rest":
        title = "ISTIRAHAT 5 DETIK"
        instruction = (
            "Boleh berkedip dan rileks. "
            "Jangan menyentuh elektroda."
        )
        display = "diam"

    elif label == "kiri":
        title = "GERAKKAN MATA KE KIRI SEKARANG"
        instruction = (
            "Gerakkan mata saja, jangan menoleh. "
            "Tahan pandangan ke kiri sampai perintah berubah."
        )
        display = "kiri"

    elif label == "kanan":
        title = "GERAKKAN MATA KE KANAN SEKARANG"
        instruction = (
            "Gerakkan mata saja, jangan menoleh. "
            "Tahan pandangan ke kanan sampai perintah berubah."
        )
        display = "kanan"

    elif label == "kedip":
        title = "KEDIP SATU KALI SEKARANG"
        instruction = (
            "Setelah satu kali kedip, buka mata "
            "dan tatap titik tengah."
        )
        display = "kedip"

    else:
        title = "TETAP TATAP TITIK TENGAH"
        instruction = (
            "Jangan gerakkan mata dan jangan berkedip "
            "sampai perintah berubah."
        )
        display = "diam"

    axis.text(
        0,
        0.76,
        title,
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
    )

    axis.text(
        0,
        -0.53,
        instruction,
        ha="center",
        va="center",
        fontsize=15,
    )

    axis.text(
        0,
        -0.76,
        f"Perintah {segment['trial_id']} dari {TOTAL_TRIALS}",
        ha="center",
        va="center",
        fontsize=13,
    )

    draw_marker(axis, display)


# =========================================================
# 12. KONTROL KEYBOARD
# =========================================================

def on_key(event):
    key = event.key.lower() if event.key else ""

    if key == "c":
        connect_device()

    elif key == "s":
        request_start_session()

    elif key == "a":
        abort_session()

    elif key == "q":
        with lock:
            active = state["session_active"]
            counting = state["countdown_active"]

        if active or counting:
            abort_session()

        with lock:
            state["running"] = False


def on_close(_event):
    with lock:
        active = state["session_active"]
        counting = state["countdown_active"]

    if active or counting:
        abort_session()

    with lock:
        state["running"] = False


# =========================================================
# 13. MAIN
# =========================================================

def main():
    print()
    print("=" * 68)
    print("PROTOKOL")
    print("=" * 68)
    print("Subjek         :", SUBJECT_CODE)
    print("Output         :", FINAL_CSV_PATH)
    print("Channel        : A1 RAW ADC")
    print("Sampling       :", RAW_FS, "Hz")
    print("Total trial    :", TOTAL_TRIALS)
    print("Repetisi/kelas :", REPETITIONS_PER_CLASS)
    print()
    print("Setiap trial:")
    print("  2 detik  : tatap titik tengah")
    print("  2 detik  : lakukan perintah")
    print("  2 detik  : kembali ke tengah")
    print("  5 detik  : istirahat")
    print()
    print("Total sesi     :", TOTAL_TRIALS * TRIAL_SEC + COUNTDOWN_SEC, "detik")
    print()
    print("Kontrol:")
    print("  C = hubungkan BITalino")
    print("  S = mulai seluruh sesi")
    print("  A = batalkan; data parsial dibuang")
    print("  Q = keluar")
    print("=" * 68)

    plt.ion()

    figure = plt.figure(figsize=(14, 8.5))
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.15],
        hspace=0.14,
    )

    cue_axis = figure.add_subplot(grid[0, 0])
    signal_axis = figure.add_subplot(grid[1, 0])

    figure.canvas.mpl_connect("key_press_event", on_key)
    figure.canvas.mpl_connect("close_event", on_close)

    raw_line, = signal_axis.plot(
        [],
        [],
        linewidth=0.8,
        alpha=0.45,
        label="A1 raw — data yang disimpan",
    )

    filtered_line, = signal_axis.plot(
        [],
        [],
        linewidth=1.5,
        label="A1 low-pass 15 Hz — hanya tampilan",
    )

    signal_axis.set_title("Sinyal EOG A1 Realtime")
    signal_axis.set_xlabel("Waktu akuisisi (detik)")
    signal_axis.set_ylabel("ADC count")
    signal_axis.set_ylim(0, 1023)
    signal_axis.grid(True, alpha=0.25)
    signal_axis.legend(loc="upper right")

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
                running = state["running"]
                connected = state["connected"]
                countdown_active = state["countdown_active"]
                countdown_end = state["countdown_end"]
                session_active = state["session_active"]
                session_finished = state["session_finished"]
                session_saved = state["session_saved"]
                completed_once = state["completed_once"]
                session_aborted = state["session_aborted"]
                segment = (
                    state["current_segment"].copy()
                    if state["current_segment"]
                    else None
                )
                message = state["message"]
                gaps = state["packet_gap_events"]
                missing = state["estimated_missing_samples"]
                clipping = state["clipped_a1_samples"]
                reader_error = state["reader_error"]

            if not running or not plt.fignum_exists(figure.number):
                break

            countdown_remaining = None

            if countdown_active:
                countdown_remaining = countdown_end - time.monotonic()

                if countdown_remaining <= 0:
                    begin_session_after_countdown()
                    countdown_remaining = None

            if session_finished and not session_saved:
                finalize_session()

                with lock:
                    completed_once = state["completed_once"]
                    session_aborted = state["session_aborted"]
                    message = state["message"]

            if countdown_remaining is not None:
                cue_key = (
                    "countdown",
                    int(np.ceil(countdown_remaining)),
                )
            elif completed_once:
                cue_key = ("completed",)
            elif session_aborted and session_saved:
                cue_key = ("aborted",)
            elif segment is not None:
                cue_key = (
                    segment["trial_id"],
                    segment["phase"],
                )
            else:
                cue_key = ("idle", connected)

            if cue_key != last_cue_key:
                draw_cue(
                    cue_axis,
                    segment,
                    countdown_remaining=countdown_remaining,
                    completed=completed_once,
                    aborted=(session_aborted and session_saved),
                )
                last_cue_key = cue_key

            with lock:
                times = np.asarray(time_plot_buffer, dtype=float)
                raw_values = np.asarray(raw_plot_buffer, dtype=float)
                filtered_values = np.asarray(
                    filtered_plot_buffer,
                    dtype=float,
                )

            if len(times) > 0:
                raw_line.set_data(times, raw_values)
                filtered_line.set_data(times, filtered_values)

                right = float(times[-1])
                left = max(0.0, right - DISPLAY_SEC)
                signal_axis.set_xlim(left, max(right, DISPLAY_SEC))

                recent = (
                    raw_values[-RAW_FS:]
                    if len(raw_values) >= RAW_FS
                    else raw_values
                )

                status_text.set_text(
                    f"{'TERHUBUNG' if connected else 'OFFLINE'} | "
                    f"{'MEREKAM' if session_active else 'SIAP'} | "
                    f"A1 1 detik: min {np.min(recent):.0f}, "
                    f"max {np.max(recent):.0f}, "
                    f"std {np.std(recent):.2f} | "
                    f"gap {gaps}, missing {missing}, clipping {clipping}\n"
                    f"{message}"
                )
            else:
                signal_axis.set_xlim(0, DISPLAY_SEC)
                status_text.set_text(
                    f"{'TERHUBUNG' if connected else 'OFFLINE'} | {message}"
                )

            if reader_error:
                status_text.set_text(
                    status_text.get_text()
                    + "\nERROR: "
                    + reader_error
                )

            figure.canvas.draw_idle()
            figure.canvas.flush_events()
            plt.pause(0.02)

    except KeyboardInterrupt:
        print("\nProgram dihentikan dari keyboard.")

    finally:
        with lock:
            active = state["session_active"]
            counting = state["countdown_active"]
            needs_finalize = (
                state["session_finished"]
                and not state["session_saved"]
            )

        if active or counting:
            abort_session()
            needs_finalize = True

        if needs_finalize:
            finalize_session()

        remove_temporary_csv()
        disconnect_device()

        plt.ioff()
        plt.close(figure)

        print("Program selesai.")


if __name__ == "__main__":
    main()
