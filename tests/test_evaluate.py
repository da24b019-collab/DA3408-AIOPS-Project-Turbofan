"""
Unit tests for the evaluation helpers (threshold choice and metrics).
Small hand-made examples, so the expected answers can be checked by eye.
"""

import pytest

from src.models.evaluate import choose_threshold, classification_metrics

# 4 safe engines (0) and 4 at-risk engines (1), with model probabilities
Y_TRUE = [0, 0, 0, 0, 1, 1, 1, 1]
PROB = [0.10, 0.20, 0.35, 0.60, 0.40, 0.70, 0.80, 0.90]


def test_threshold_meets_target_recall():
    t = choose_threshold(Y_TRUE, PROB, target_recall=1.0)
    # To catch all 4 failures, the threshold must be at or below 0.40 (the lowest at-risk prob)
    assert t <= 0.40
    assert classification_metrics(Y_TRUE, PROB, t)["recall"] == 1.0


def test_threshold_is_as_high_as_possible():
    # Catching 3 of 4 failures (recall 0.75) allows a higher threshold: 0.70
    t = choose_threshold(Y_TRUE, PROB, target_recall=0.75)
    assert t == pytest.approx(0.70)
    m = classification_metrics(Y_TRUE, PROB, t)
    assert m["recall"] == 0.75
    assert m["precision"] == 1.0  # no false alarms at 0.70


def test_metrics_by_hand():
    m = classification_metrics(Y_TRUE, PROB, threshold=0.5)
    # Predicted at risk: 0.60 (wrong), 0.70, 0.80, 0.90 (right). Missed: 0.40.
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["accuracy"] == pytest.approx(6 / 8)


def test_always_safe_model_has_zero_recall():
    m = classification_metrics(Y_TRUE, [0.0] * 8, threshold=0.5)
    assert m["recall"] == 0.0
    assert m["accuracy"] == 0.5  # accuracy alone hides that it catches nothing
