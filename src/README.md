# Source Code

The source directory contains the end-to-end implementation used in the EOG
robot-control final project.

| File | Purpose |
| --- | --- |
| `data_acquisition.py` | Acquire 20 EOG trials per participant using BITalino. |
| `train_model.py` | Preprocess the five-subject dataset, train the PyTorch 1D-CNN, evaluate the internal test set, and export TorchScript. |
| `evaluate_realtime.py` | Evaluate the trained model in real time on unseen participants. |
| `analyze_evaluation.py` | Aggregate the five unseen-subject evaluation sessions and calculate overall/per-class metrics. |
| `realtime_robot_control.py` | Run real-time EOG inference, blink decision logic, and Pioneer P3DX control in CoppeliaSim. |

## Typical workflow

```text
Data acquisition
    ↓
Model training and internal testing
    ↓
Real-time evaluation on unseen subjects
    ↓
Aggregate evaluation analysis
    ↓
Real-time robot-control test
```

## Environment

Python 3.10+ is recommended. Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

For BITalino scripts, provide the device address using `--mac` where supported
or the `BITALINO_MAC` environment variable. Raw participant data is intentionally
excluded from the public repository.
