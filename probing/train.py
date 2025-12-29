"""
Linear probe training with train/test split and selectivity checks.

This module provides the core probe training functionality:
- Logistic regression for classification
- 80/20 train/test split for accuracy evaluation
- Selectivity checks (random baseline comparison)
"""

import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from easyprobe.datamodels import ProbeResult, ProbeTask, ProbeType


def train_single_probe(task: ProbeTask) -> ProbeResult:
    """
    Train a single linear probe and evaluate.

    This function is designed to be called in parallel via ProcessPoolExecutor.
    It receives all necessary data in the ProbeTask and returns a ProbeResult.

    The probe training process:
    1. Create logistic regression model
    2. Train on 80% of data, evaluate on 20%
    3. Optionally compute random baseline (shuffle labels, retrain)
    4. Calculate selectivity (accuracy - random_baseline)

    Args:
        task: ProbeTask containing activations, labels, and settings

    Returns:
        ProbeResult with accuracy, random baseline, selectivity, and timing info
    """
    # Capture timing and process info
    start_time = time.time()
    pid = os.getpid()

    # Choose model based on probe type
    if task.probe_type == ProbeType.CLASSIFICATION:
        # Logistic regression for classification
        # C = 1/regularization (sklearn uses inverse regularization)
        model = LogisticRegression(
            penalty="l2",
            C=1.0 / task.regularization,
            max_iter=1000,
            solver="lbfgs",
        )
    else:
        raise ValueError(f"Unsupported probe type: {task.probe_type}")

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        task.activations,
        task.labels,
        test_size=0.2,
        stratify=task.labels,
        random_state=42,
    )

    # Train and evaluate
    model.fit(X_train, y_train)
    accuracy = float(model.score(X_test, y_test))
    accuracy_std = None  # No std with single split

    # Store weights from the trained model (before selectivity check may overwrite)
    weights = model.coef_.copy()
    bias = model.intercept_.copy()

    # Selectivity check (random baseline)
    random_baseline = None
    random_baseline_std = None
    selectivity = None

    if task.include_selectivity:
        random_scores = []
        for _ in range(task.random_trials):
            # Shuffle labels and retrain
            shuffled_labels = np.random.permutation(task.labels)
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                task.activations,
                shuffled_labels,
                test_size=0.2,
                stratify=shuffled_labels,
                random_state=42,
            )
            model.fit(X_train_r, y_train_r)
            random_scores.append(model.score(X_test_r, y_test_r))

        random_baseline = float(np.mean(random_scores))
        random_baseline_std = float(np.std(random_scores))

        selectivity = accuracy - random_baseline

    end_time = time.time()
    training_duration_s = end_time - start_time

    return ProbeResult(
        layer=task.layer,
        component=task.component,
        position=task.position,
        accuracy=accuracy,
        accuracy_std=accuracy_std,
        random_baseline=random_baseline,
        random_baseline_std=random_baseline_std,
        selectivity=selectivity,
        probe_type=task.probe_type,
        n_samples=len(task.labels),
        pid=pid,
        start_time=start_time,
        end_time=end_time,
        training_duration_s=training_duration_s,
        weights=weights,
        bias=bias,
    )
