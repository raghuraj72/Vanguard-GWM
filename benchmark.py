"""
benchmark.py - Synthetic Pipeline Execution Benchmark for Vanguard-GWM

IMPORTANT:
This benchmark evaluates Vanguard-GWM execution on synthetic data.
The current neural model uses randomly initialized weights, so these
results must NOT be interpreted as validated cybersecurity detection
performance or real-world accuracy.
"""

import time

import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier

from src.engine import VanguardEngine


SEED = 42
N_SAMPLES = 500
N_FEATURES = 10
ATTACK_RATIO = 0.15


def create_synthetic_dataset():
    """Create a reproducible synthetic binary classification dataset."""

    rng = np.random.default_rng(SEED)

    features = rng.normal(
        0,
        1,
        size=(N_SAMPLES, N_FEATURES),
    )

    labels = np.zeros(N_SAMPLES, dtype=np.int64)

    attack_count = int(N_SAMPLES * ATTACK_RATIO)

    attack_indices = rng.choice(
        N_SAMPLES,
        size=attack_count,
        replace=False,
    )

    labels[attack_indices] = 1

    # Synthetic attack signature.
    features[attack_indices] += rng.normal(
        1.8,
        0.4,
        size=(attack_count, N_FEATURES),
    )

    return features, labels


def compute_metrics(y_true, y_pred):
    """Calculate standard binary classification metrics."""

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "False Positive Rate": fpr,
    }


def run_classical_baselines(
    x_train,
    x_test,
    y_train,
    y_test,
):
    """Train and evaluate classical ML baselines."""

    models = {
        "Logistic Regression": LogisticRegression(
            random_state=SEED,
            max_iter=500,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=SEED,
        ),
        "Static MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=SEED,
        ),
    }

    results = []

    for name, model in models.items():

        start_time = time.perf_counter()

        model.fit(
            x_train,
            y_train,
        )

        predictions = model.predict(x_test)

        latency_ms = (
            (time.perf_counter() - start_time)
            * 1000
        )

        metrics = compute_metrics(
            y_test,
            predictions,
        )

        results.append(
            {
                "Model": name,
                **metrics,
                "Evaluation Time (ms)": latency_ms,
            }
        )

    return results


def run_vanguard_pipeline(
    x_test,
    y_test,
):
    """
    Evaluate the current Vanguard-GWM inference pipeline.

    NOTE:
    The current neural architecture is untrained. Therefore this evaluates
    pipeline execution and synthetic inference behavior, not validated
    cybersecurity detection accuracy.
    """

    engine = VanguardEngine(
        in_node_features=5,
        in_edge_features=5,
        latent_dim=64,
        horizon_k=5,
        risk_threshold=0.60,
    )

    predictions = []
    inference_times = []

    for sample in x_test:

        # Convert the synthetic feature vector into
        # a small graph snapshot.
        node_features = torch.tensor(
            sample[:5],
            dtype=torch.float32,
        ).repeat(4, 1)

        edge_index = torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 0],
            ],
            dtype=torch.long,
        )

        edge_attr = torch.tensor(
            sample[:5],
            dtype=torch.float32,
        ).repeat(4, 1)

        start_time = time.perf_counter()

        result = engine.process_graph_snapshot(
            node_features,
            edge_index,
            edge_attr,
            target_ip="192.168.1.105",
        )

        latency_ms = (
            (time.perf_counter() - start_time)
            * 1000
        )

        inference_times.append(latency_ms)

        predictions.append(
            int(result["peak_risk"] >= 0.60)
        )

    metrics = compute_metrics(
        y_test,
        np.array(predictions),
    )

    return {
        "Model": "Vanguard-GWM (Untrained Pipeline)",
        **metrics,
        "Evaluation Time (ms)": np.mean(
            inference_times
        ),
    }


def main():
    print("=" * 70)
    print("VANGUARD-GWM SYNTHETIC PIPELINE EXECUTION BENCHMARK")
    print("=" * 70)

    print(
        "\nWARNING: Vanguard-GWM is currently evaluated "
        "with randomly initialized neural weights."
    )

    print(
        "These results are NOT validated cybersecurity "
        "detection performance.\n"
    )

    x, y = create_synthetic_dataset()

    split = int(0.70 * len(x))

    x_train = x[:split]
    x_test = x[split:]

    y_train = y[:split]
    y_test = y[split:]

    results = run_classical_baselines(
        x_train,
        x_test,
        y_train,
        y_test,
    )

    vanguard_result = run_vanguard_pipeline(
        x_test,
        y_test,
    )

    results.append(vanguard_result)

    df_results = pd.DataFrame(results)

    display_df = df_results.copy()

    for column in [
        "Precision",
        "Recall",
        "F1-Score",
        "False Positive Rate",
    ]:
        display_df[column] = display_df[column].map(
            lambda value: f"{value:.3f}"
        )

    display_df["Evaluation Time (ms)"] = (
        display_df["Evaluation Time (ms)"]
        .map(lambda value: f"{value:.2f}")
    )

    print("\n" + display_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("Benchmark completed.")
    print(
        "Use these results as synthetic pipeline evidence only."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()