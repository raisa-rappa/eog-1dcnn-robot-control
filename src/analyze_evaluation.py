#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aggregate Evaluation Results from Five Unseen Subjects.

This script reads five per-trial evaluation CSV files, validates the expected
20 trials per subject and five trials per class, aggregates all 100 trials,
and produces subject-level, class-level, and overall metrics plus figures.

It performs analysis only; it does not retrain or modify the model.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


CLASS_ORDER = ["diam", "kiri", "kanan", "kedip"]
EXPECTED_SUBJECTS = 5
EXPECTED_TRIALS_PER_SUBJECT = 20
EXPECTED_TRIALS_PER_CLASS = 5
EXPECTED_TOTAL_TRIALS = EXPECTED_SUBJECTS * EXPECTED_TRIALS_PER_SUBJECT

REQUIRED_COLUMNS = [
    "subject_id",
    "trial_ke",
    "aktual",
    "prediksi",
    "confidence",
    "benar",
    "inference_ms",
]


def natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def normalize_label(value: object) -> str:
    text = str(value).strip().lower()
    mapping = {
        "left": "kiri",
        "right": "kanan",
        "blink": "kedip",
        "rest": "diam",
        "no_command": "diam",
        "hold": "diam",
    }
    return mapping.get(text, text)


def parse_boolean(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"true", "1", "ya", "yes", "benar"}:
        return True
    if text in {"false", "0", "tidak", "no", "salah"}:
        return False

    raise ValueError(f"Nilai kolom benar tidak dapat dibaca: {value!r}")


def load_evaluation_files(
    input_dir: Path,
    file_glob: str,
) -> tuple[pd.DataFrame, list[Path]]:
    files = sorted(input_dir.glob(file_glob), key=natural_key)

    if len(files) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Ditemukan {len(files)} file evaluasi di:\n{input_dir}\n"
            f"Seharusnya tepat {EXPECTED_SUBJECTS} file dengan pola "
            f"{file_glob!r}."
        )

    frames: list[pd.DataFrame] = []
    subject_ids: set[str] = set()

    for path in files:
        dataframe = pd.read_csv(path)

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in dataframe.columns
        ]
        if missing:
            raise ValueError(
                f"{path.name}: kolom berikut tidak ditemukan: {missing}"
            )

        work = dataframe[REQUIRED_COLUMNS].copy()

        if len(work) != EXPECTED_TRIALS_PER_SUBJECT:
            raise ValueError(
                f"{path.name}: jumlah trial {len(work)}, seharusnya "
                f"{EXPECTED_TRIALS_PER_SUBJECT}."
            )

        work["subject_id"] = (
            work["subject_id"].astype(str).str.strip().str.upper()
        )
        unique_subjects = work["subject_id"].dropna().unique().tolist()

        if len(unique_subjects) != 1:
            raise ValueError(
                f"{path.name}: harus berisi tepat satu subject_id. "
                f"Ditemukan: {unique_subjects}"
            )

        subject_id = unique_subjects[0]
        if subject_id in subject_ids:
            raise ValueError(f"subject_id ganda ditemukan: {subject_id}")
        subject_ids.add(subject_id)

        for column in ["trial_ke", "confidence", "inference_ms"]:
            work[column] = pd.to_numeric(work[column], errors="coerce")

        if work[["trial_ke", "confidence", "inference_ms"]].isna().any().any():
            raise ValueError(
                f"{path.name}: terdapat angka kosong atau tidak valid."
            )

        trial_fraction = np.modf(work["trial_ke"].to_numpy(dtype=float))[0]
        if not np.allclose(trial_fraction, 0.0):
            raise ValueError(f"{path.name}: trial_ke harus bilangan bulat.")

        work["trial_ke"] = work["trial_ke"].astype(int)
        expected_trials = list(range(1, EXPECTED_TRIALS_PER_SUBJECT + 1))
        actual_trials = sorted(work["trial_ke"].tolist())

        if actual_trials != expected_trials:
            raise ValueError(
                f"{path.name}: trial_ke harus lengkap dari 1 sampai "
                f"{EXPECTED_TRIALS_PER_SUBJECT}."
            )

        work["aktual"] = work["aktual"].map(normalize_label)
        work["prediksi"] = work["prediksi"].map(normalize_label)

        invalid_actual = sorted(set(work["aktual"]) - set(CLASS_ORDER))
        invalid_prediction = sorted(set(work["prediksi"]) - set(CLASS_ORDER))

        if invalid_actual:
            raise ValueError(
                f"{path.name}: label aktual tidak dikenal: {invalid_actual}"
            )
        if invalid_prediction:
            raise ValueError(
                f"{path.name}: label prediksi tidak dikenal: "
                f"{invalid_prediction}"
            )

        class_counts = work["aktual"].value_counts().to_dict()
        for class_name in CLASS_ORDER:
            count = int(class_counts.get(class_name, 0))
            if count != EXPECTED_TRIALS_PER_CLASS:
                raise ValueError(
                    f"{path.name}: kelas aktual {class_name} berjumlah "
                    f"{count}, seharusnya {EXPECTED_TRIALS_PER_CLASS}."
                )

        if not np.isfinite(work["confidence"].to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name}: confidence mengandung NaN/Inf.")
        if not np.isfinite(work["inference_ms"].to_numpy(dtype=float)).all():
            raise ValueError(f"{path.name}: inference_ms mengandung NaN/Inf.")
        if ((work["confidence"] < 0) | (work["confidence"] > 1)).any():
            raise ValueError(f"{path.name}: confidence harus berada pada 0–1.")
        if (work["inference_ms"] < 0).any():
            raise ValueError(
                f"{path.name}: inference_ms tidak boleh negatif."
            )

        work["benar_file"] = work["benar"].map(parse_boolean)
        work["benar"] = work["aktual"] == work["prediksi"]
        mismatch = int(np.sum(work["benar_file"] != work["benar"]))

        if mismatch > 0:
            raise ValueError(
                f"{path.name}: terdapat {mismatch} nilai benar yang tidak "
                "sesuai dengan aktual dan prediksi."
            )

        work["sumber_file"] = path.name
        frames.append(work)
        print(f"{subject_id}: {len(work)} trial valid")

    combined = pd.concat(frames, ignore_index=True)

    if len(combined) != EXPECTED_TOTAL_TRIALS:
        raise ValueError(
            f"Total trial {len(combined)}, seharusnya "
            f"{EXPECTED_TOTAL_TRIALS}."
        )

    return combined, files


def load_optional_qc(
    input_dir: Path,
    subject_ids: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for subject_id in subject_ids:
        path = input_dir / f"{subject_id}_ringkasan.csv"
        if not path.exists():
            continue

        dataframe = pd.read_csv(path)
        if dataframe.empty:
            continue

        row = dataframe.iloc[0].to_dict()
        row["subject_id"] = subject_id
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    qc = pd.DataFrame(rows)
    wanted_columns = [
        "subject_id",
        "packet_gap_events",
        "estimated_missing_samples",
        "clipped_a1_samples",
    ]
    existing = [column for column in wanted_columns if column in qc.columns]
    return qc[existing].copy()


def calculate_subject_metrics(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for subject_id, subject_df in combined.groupby("subject_id", sort=True):
        actual = subject_df["aktual"].to_numpy()
        predicted = subject_df["prediksi"].to_numpy()
        correct = int(np.sum(actual == predicted))
        total = int(len(subject_df))

        rows.append({
            "subject_id": subject_id,
            "total_trial": total,
            "benar": correct,
            "salah": total - correct,
            "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(
                f1_score(
                    actual,
                    predicted,
                    labels=CLASS_ORDER,
                    average="macro",
                    zero_division=0,
                )
            ),
            "rata_rata_confidence": float(subject_df["confidence"].mean()),
            "rata_rata_inference_ms": float(
                subject_df["inference_ms"].mean()
            ),
        })

    return pd.DataFrame(rows)


def calculate_class_metrics(combined: pd.DataFrame) -> pd.DataFrame:
    report = classification_report(
        combined["aktual"],
        combined["prediksi"],
        labels=CLASS_ORDER,
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for class_name in CLASS_ORDER:
        values = report[class_name]
        rows.append({
            "kelas": class_name,
            "precision": float(values["precision"]),
            "recall": float(values["recall"]),
            "f1_score": float(values["f1-score"]),
            "support": int(values["support"]),
        })

    return pd.DataFrame(rows)


def calculate_overall_summary(combined: pd.DataFrame) -> dict[str, object]:
    actual = combined["aktual"].to_numpy()
    predicted = combined["prediksi"].to_numpy()
    correct = int(np.sum(actual == predicted))
    total = int(len(combined))

    return {
        "total_subjek": int(combined["subject_id"].nunique()),
        "total_trial": total,
        "benar": correct,
        "salah": total - correct,
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "rata_rata_confidence": float(combined["confidence"].mean()),
        "rata_rata_inference_ms": float(combined["inference_ms"].mean()),
    }


def save_confusion_matrix(
    combined: pd.DataFrame,
    csv_path: Path,
    image_path: Path,
) -> None:
    matrix = confusion_matrix(
        combined["aktual"],
        combined["prediksi"],
        labels=CLASS_ORDER,
    )

    dataframe = pd.DataFrame(
        matrix,
        index=[f"aktual_{name}" for name in CLASS_ORDER],
        columns=[f"prediksi_{name}" for name in CLASS_ORDER],
    )
    dataframe.to_csv(csv_path)

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix)
    axis.set_xticks(range(len(CLASS_ORDER)))
    axis.set_xticklabels(CLASS_ORDER)
    axis.set_yticks(range(len(CLASS_ORDER)))
    axis.set_yticklabels(CLASS_ORDER)
    axis.set_xlabel("Prediksi")
    axis.set_ylabel("Aktual")
    axis.set_title("Confusion Matrix Evaluasi 5 Subjek Baru")

    for row in range(len(CLASS_ORDER)):
        for column in range(len(CLASS_ORDER)):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(image_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_subject_accuracy_chart(
    subject_metrics: pd.DataFrame,
    overall_accuracy: float,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    accuracy_percent = subject_metrics["accuracy"] * 100
    bars = axis.bar(subject_metrics["subject_id"], accuracy_percent)

    axis.axhline(
        overall_accuracy * 100,
        linestyle="--",
        linewidth=1.2,
        label=f"Akurasi keseluruhan {overall_accuracy * 100:.2f}%",
    )
    axis.set_ylim(0, 105)
    axis.set_xlabel("Subjek Evaluasi")
    axis.set_ylabel("Akurasi (%)")
    axis.set_title("Akurasi Evaluasi per Subjek")
    axis.grid(True, axis="y", alpha=0.30, linestyle="--")

    for bar, value in zip(bars, accuracy_percent):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_correct_wrong_chart(
    subject_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    subject_ids = subject_metrics["subject_id"].tolist()
    correct = subject_metrics["benar"].to_numpy(dtype=int)
    wrong = subject_metrics["salah"].to_numpy(dtype=int)

    axis.bar(subject_ids, correct, label="Benar", hatch="///")
    axis.bar(subject_ids, wrong, bottom=correct, label="Salah", hatch="...")
    axis.set_ylim(0, EXPECTED_TRIALS_PER_SUBJECT + 2)
    axis.set_xlabel("Subjek Evaluasi")
    axis.set_ylabel("Jumlah Trial")
    axis.set_title("Jumlah Prediksi Benar dan Salah")
    axis.grid(True, axis="y", alpha=0.30, linestyle="--")
    axis.legend()

    for index in range(len(subject_ids)):
        if correct[index] > 0:
            axis.text(
                index,
                correct[index] / 2,
                str(correct[index]),
                ha="center",
                va="center",
            )
        if wrong[index] > 0:
            axis.text(
                index,
                correct[index] + wrong[index] / 2,
                str(wrong[index]),
                ha="center",
                va="center",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_analysis(
    input_dir: Path,
    output_dir: Path,
    file_glob: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("ANALISIS EVALUASI 5 SUBJEK")
    print("=" * 64)

    combined, input_files = load_evaluation_files(
        input_dir=input_dir,
        file_glob=file_glob,
    )

    combined = combined.sort_values(
        by=["subject_id", "trial_ke"]
    ).reset_index(drop=True)

    subject_metrics = calculate_subject_metrics(combined)
    class_metrics = calculate_class_metrics(combined)
    overall = calculate_overall_summary(combined)

    qc = load_optional_qc(
        input_dir=input_dir,
        subject_ids=subject_metrics["subject_id"].tolist(),
    )
    if not qc.empty:
        subject_metrics = subject_metrics.merge(
            qc,
            on="subject_id",
            how="left",
        )

    combined.to_csv(
        output_dir / "01_gabungan_evaluasi_100_trial.csv",
        index=False,
    )
    subject_metrics.to_csv(
        output_dir / "02_hasil_per_subjek.csv",
        index=False,
    )
    class_metrics.to_csv(
        output_dir / "03_hasil_per_kelas.csv",
        index=False,
    )
    pd.DataFrame([overall]).to_csv(
        output_dir / "04_hasil_keseluruhan.csv",
        index=False,
    )

    with (output_dir / "05_hasil_keseluruhan.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(overall, handle, indent=2, ensure_ascii=False)

    save_confusion_matrix(
        combined=combined,
        csv_path=output_dir / "06_confusion_matrix.csv",
        image_path=output_dir / "07_confusion_matrix.png",
    )
    save_subject_accuracy_chart(
        subject_metrics=subject_metrics,
        overall_accuracy=float(overall["accuracy"]),
        output_path=output_dir / "08_akurasi_per_subjek.png",
    )
    save_correct_wrong_chart(
        subject_metrics=subject_metrics,
        output_path=output_dir / "09_benar_salah_per_subjek.png",
    )

    pd.DataFrame({
        "file_input": [path.name for path in input_files]
    }).to_csv(
        output_dir / "10_daftar_file_input.csv",
        index=False,
    )

    print()
    print("HASIL PER SUBJEK")
    print("-" * 64)

    for row in subject_metrics.itertuples(index=False):
        print(
            f"{row.subject_id}: {row.benar} benar, "
            f"{row.salah} salah, akurasi {row.accuracy * 100:.2f}%"
        )

    print("-" * 64)
    print(f"Total      : {overall['total_trial']} trial")
    print(f"Benar      : {overall['benar']}")
    print(f"Salah      : {overall['salah']}")
    print(f"Accuracy   : {float(overall['accuracy']) * 100:.2f}%")
    print(f"Macro F1   : {float(overall['macro_f1']) * 100:.2f}%")
    print(f"Confidence : {float(overall['rata_rata_confidence']):.3f}")
    print(f"Inference  : {float(overall['rata_rata_inference_ms']):.3f} ms")
    print("-" * 64)
    print("Output     :", output_dir)
    print("=" * 64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gabungkan dan analisis hasil evaluasi EOG dari 5 subjek baru."
        )
    )

    script_dir = Path(__file__).resolve().parent

    if (script_dir / "outputs" / "evaluasi_realtime_5_subjek").exists():
        default_input = script_dir / "outputs" / "evaluasi_realtime_5_subjek"
        default_output = script_dir / "outputs" / "analisis_evaluasi_5_subjek"
    else:
        project_root = script_dir.parent
        default_input = project_root / "outputs" / "evaluasi_realtime_5_subjek"
        default_output = project_root / "outputs" / "analisis_evaluasi_5_subjek"

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="Folder yang berisi 5 file *_evaluasi_per_trial.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Folder output hasil analisis.",
    )
    parser.add_argument(
        "--file-glob",
        type=str,
        default="*_evaluasi_per_trial.csv",
        help="Pola nama file evaluasi.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_glob=args.file_glob,
    )


if __name__ == "__main__":
    main()
