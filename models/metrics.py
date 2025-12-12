import matplotlib
matplotlib.use("Agg")  # headless backend

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import math
import threading
import os, uuid
"""
Author: Ben
"""

# Base directories.
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "models" / "train"
METRICS_DIR = BASE_DIR / "static" / "metrics"

# History and confusion JSON files.
HISTORY_PATH = TRAIN_DIR / "history.json"
CONFUSION_PATH = TRAIN_DIR / "confusion_matrix.json"

_plot_lock = threading.Lock()


def _save_atomic(fig, out_path: Path) -> None:
    tmp = out_path.with_suffix(out_path.suffix + f'.{uuid.uuid4().hex}{out_path.suffix}')
    fig.savefig(tmp, dpi=120, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp, out_path)

def _ensure_metrics_dir() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Placefolder PNGs if no data is found.
def _placeholder_plot(metric_name: str) -> str:
    """Simple 'No data available' PNG."""
    _ensure_metrics_dir()
    out_path = METRICS_DIR / f"{metric_name}.png"

    with _plot_lock:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=16)
        ax.axis("off")
        fig.tight_layout()
        _save_atomic(fig, out_path)
    return str(out_path)


def _load_history() -> dict | None:
    if not HISTORY_PATH.exists():
        return None
    with HISTORY_PATH.open("r") as f:
        hist = json.load(f)
    return hist if isinstance(hist, dict) else None


def _load_confusion() -> np.ndarray | None:
    if not CONFUSION_PATH.exists():
        return None
    with CONFUSION_PATH.open("r") as f:
        cm_list = json.load(f)
    return np.array(cm_list, dtype=float)

# This definition generates a PNG for the given metric and return its file path.
# accuracy, loss, auc, precision, and recall use "history.json"
# confusion uses "confusion_matrix.json"
def plot_metric(metric_name: str) -> str:

    metric_name = metric_name.lower()

    if metric_name == "confusion":
        return _plot_confusion()

    history = _load_history()
    if history is None:
        return _placeholder_plot(metric_name)

    # Maps names to Keras keys from supervised_model.py
    aliases = {
        "accuracy": ["accuracy"],
        "loss": ["loss"],
        "auc": ["AUC", "auc", "roc_auc"],
        "precision": ["Precision", "precision"],
        "recall": ["Recall", "recall"],
    }

    candidates = aliases.get(metric_name, [metric_name])
    key = next((k for k in candidates if k in history), None)
    if key is None:
        return _placeholder_plot(metric_name)

    # Tries to find validation version.
    val_candidates = [
        f"val_{key}",
        f"val_{key.lower()}",
        f"val_{key.upper()}",
        f"val_{key.capitalize()}",
    ]
    val_key = next((vk for vk in val_candidates if vk in history), None)

    train_values = history[key]
    epochs = range(1, len(train_values) + 1)

    _ensure_metrics_dir()
    out_path = METRICS_DIR / f"{metric_name}.png"

    with _plot_lock:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, train_values, label="Training")

        if val_key is not None:
            val_values = history[val_key]
            ax.plot(epochs, val_values, label="Validation")

        ax.set_title(f"{metric_name.capitalize()} over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        _save_atomic(fig, out_path)

    print("WROTE", out_path, "bytes:", out_path.stat().st_size)
    return str(out_path)

# Plots confusion matrix from JSON. Falls back to placeholder.
def _plot_confusion() -> str:
    print("PLOTTING CONFUSION")
    cm = _load_confusion()
    metric_name = "confusion"

    if cm is None:
        return _placeholder_plot(metric_name)

    _ensure_metrics_dir()
    out_path = METRICS_DIR / f"{metric_name}.png"

    with _plot_lock:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest")
        fig.colorbar(im, ax=ax)
        ax.set_title("Confusion Matrix")
        tick_marks = np.arange(cm.shape[0])
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        fig.tight_layout()
        _save_atomic(fig, out_path)
        return str(out_path)

if (__name__ == "__main__"):
    plot_metric("loss")
    plot_metric("accuracy")
    plot_metric("auc")
    plot_metric("confusion")