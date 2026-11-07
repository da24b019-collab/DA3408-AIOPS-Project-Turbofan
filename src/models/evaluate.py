"""
Evaluation helpers: metrics, threshold choice and plots.

Kept separate from train.py so they can be unit-tested on tiny examples
and reused later by the API and the retraining step.
"""

import matplotlib

matplotlib.use("Agg")  # draw plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(y_true, prob, target_recall: float) -> float:
    """
    Pick the decision threshold for "at risk".

    A missed failure costs far more than an extra inspection, so we require
    recall >= target_recall (catch at least that share of real failures).
    Among all thresholds meeting that rule, we take the HIGHEST one, because
    a higher threshold raises fewer false alarms (better precision).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    # precision_recall_curve returns one more recall value than thresholds;
    # recall[i] is the recall when using thresholds[i].
    meets_target = recall[:-1] >= target_recall
    if not meets_target.any():
        # Target impossible: fall back to the lowest threshold (flag everything risky)
        return float(thresholds.min())
    return float(thresholds[meets_target].max())


def classification_metrics(y_true, prob, threshold: float) -> dict:
    """All the metrics we report. Accuracy is kept only for comparison."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, prob),
        "roc_auc": roc_auc_score(y_true, prob),
        "accuracy": accuracy_score(y_true, pred),
    }


def plot_pr_curve(y_true, prob, threshold: float, path, title: str) -> None:
    """Precision-recall curve with the chosen threshold marked."""
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    idx = int(np.argmin(np.abs(thresholds - threshold)))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="steelblue")
    ax.scatter(
        [recall[idx]],
        [precision[idx]],
        color="red",
        zorder=3,
        label=f"chosen threshold = {threshold:.3f}",
    )
    ax.set_xlabel("Recall (share of real failures caught)")
    ax.set_ylabel("Precision (share of alarms that were real)")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(y_true, prob, threshold: float, path, title: str) -> None:
    """Counts of correct and wrong predictions at the chosen threshold."""
    pred = (np.asarray(prob) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, pred, display_labels=["safe", "at risk"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
