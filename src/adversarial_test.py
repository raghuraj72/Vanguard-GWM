"""
adversarial_test.py - Graph Adversarial Perturbation & Stress-Testing Suite

Evaluates Vanguard-GWM sensitivity to controlled graph perturbations:

1. Edge-dropping perturbations
2. Node/edge feature noise injection

The test reports changes in the model's risk score and latent
representation relative to a clean baseline.

Important:
    This is a robustness/stress test, not a validated security benchmark.
    Synthetic perturbations do not establish real-world attack resilience.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from src.config import settings
from src.world_model import (
    GATv2SpatialEncoder,
    MultiTaskDecoder,
)


logger = logging.getLogger("Vanguard-Adversarial")


# ============================================================
# TEST CONFIGURATION
# ============================================================

NUM_NODES = 12
NUM_EDGES = 30
NODE_FEATURES = 5
EDGE_FEATURES = 5

EDGE_DROP_RATES = (0.10, 0.25, 0.50)
FEATURE_NOISE_LEVELS = (0.10, 0.50, 1.00, 2.00)

RANDOM_SEED = 42


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _validate_probability(
    value: float,
    field_name: str,
) -> float:
    """Validate a probability/rate in the inclusive range [0, 1]."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a number."
        )

    value = float(value)

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0.0 and 1.0."
        )

    return value


def _validate_non_negative_float(
    value: float,
    field_name: str,
) -> float:
    """Validate a non-negative numeric value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a number."
        )

    value = float(value)

    if value < 0.0:
        raise ValueError(
            f"{field_name} cannot be negative."
        )

    return value


# ============================================================
# GRAPH GENERATION
# ============================================================

def _build_clean_graph(
    num_nodes: int = NUM_NODES,
    num_edges: int = NUM_EDGES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a deterministic synthetic graph for robustness testing.

    Returns:
        x:
            Node features with shape [num_nodes, 5].

        edge_index:
            Directed graph connectivity with shape [2, num_edges].

        edge_attr:
            Edge features with shape [num_edges, 5].
    """

    if num_nodes <= 0:
        raise ValueError("num_nodes must be greater than zero.")

    max_edges = num_nodes * (num_nodes - 1)

    if num_edges <= 0:
        raise ValueError("num_edges must be greater than zero.")

    if num_edges > max_edges:
        raise ValueError(
            f"num_edges cannot exceed {max_edges} for "
            f"{num_nodes} nodes."
        )

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    x = torch.randn(
        num_nodes,
        NODE_FEATURES,
        generator=generator,
    )

    # Build all possible directed non-self-loop edges.
    candidates = [
        (src, dst)
        for src in range(num_nodes)
        for dst in range(num_nodes)
        if src != dst
    ]

    permutation = torch.randperm(
        len(candidates),
        generator=generator,
    )[:num_edges]

    selected_edges = [
        candidates[index]
        for index in permutation.tolist()
    ]

    src = torch.tensor(
        [edge[0] for edge in selected_edges],
        dtype=torch.long,
    )

    dst = torch.tensor(
        [edge[1] for edge in selected_edges],
        dtype=torch.long,
    )

    edge_index = torch.stack(
        [src, dst],
        dim=0,
    )

    edge_attr = torch.randn(
        num_edges,
        EDGE_FEATURES,
        generator=generator,
    )

    return x, edge_index, edge_attr


# ============================================================
# MODEL EVALUATION
# ============================================================

def _evaluate_graph(
    encoder: GATv2SpatialEncoder,
    decoder: MultiTaskDecoder,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """
    Run the encoder and decoder on a graph.

    Returns:
        risk_score:
            Model-derived risk score in the range [0, 1].

        graph_latent:
            Graph-level latent representation.
    """

    with torch.no_grad():
        _, graph_latent = encoder(
            x,
            edge_index,
            edge_attr,
        )

        predictions = decoder(
            graph_latent,
        )

        risk_score = float(
            predictions["risk_score"].squeeze().item()
        )

    return risk_score, graph_latent


# ============================================================
# MAIN STRESS TEST
# ============================================================

def run_adversarial_stress_test() -> dict[str, Any]:
    """
    Run the Vanguard-GWM synthetic robustness stress test.

    Returns:
        Dictionary containing the clean baseline and all perturbation
        results.

    Important:
        Results indicate sensitivity to synthetic perturbations only.
        They should not be interpreted as proof of real-world
        adversarial robustness.
    """

    print("=" * 70)
    print(
        "🛡️ VANGUARD-GWM ADVERSARIAL ROBUSTNESS "
        "& PERTURBATION STRESS TEST"
    )
    print("=" * 70)

    latent_dim = settings.model_latent_dim

    # --------------------------------------------------------
    # Model initialization
    # --------------------------------------------------------

    encoder = GATv2SpatialEncoder(
        in_node_features=NODE_FEATURES,
        in_edge_features=EDGE_FEATURES,
        latent_dim=latent_dim,
    )

    decoder = MultiTaskDecoder(
        latent_dim=latent_dim,
    )

    encoder.eval()
    decoder.eval()

    # --------------------------------------------------------
    # Clean baseline graph
    # --------------------------------------------------------

    x, edge_index, edge_attr = _build_clean_graph()

    clean_risk, clean_latent = _evaluate_graph(
        encoder,
        decoder,
        x,
        edge_index,
        edge_attr,
    )

    logger.info(
        "Baseline clean graph risk score: %.4f",
        clean_risk,
    )

    print(
        f"\nBaseline Clean Graph Risk Score: "
        f"{clean_risk:.4f}"
    )

    # --------------------------------------------------------
    # Result container
    # --------------------------------------------------------

    results: dict[str, Any] = {
        "baseline": {
            "risk_score": clean_risk,
            "num_nodes": NUM_NODES,
            "num_edges": NUM_EDGES,
        },
        "edge_drop": [],
        "feature_noise": [],
    }

    # ========================================================
    # 1. EDGE-DROPPING PERTURBATION
    # ========================================================

    print(
        "\n--- 1. Edge-Dropping Perturbation Test ---"
    )

    for rate in EDGE_DROP_RATES:
        rate = _validate_probability(
            rate,
            "edge_drop_rate",
        )

        generator = torch.Generator()
        generator.manual_seed(
            RANDOM_SEED + int(rate * 1000)
        )

        keep_mask = (
            torch.rand(
                NUM_EDGES,
                generator=generator,
            )
            > rate
        )

        # Always preserve at least one edge.
        if not bool(keep_mask.any()):
            keep_mask[0] = True

        perturbed_edge_index = edge_index[
            :,
            keep_mask,
        ]

        perturbed_edge_attr = edge_attr[
            keep_mask
        ]

        perturbed_risk, perturbed_latent = _evaluate_graph(
            encoder,
            decoder,
            x,
            perturbed_edge_index,
            perturbed_edge_attr,
        )

        risk_drift = abs(
            clean_risk - perturbed_risk
        )

        latent_drift = torch.linalg.vector_norm(
            clean_latent - perturbed_latent
        ).item()

        actual_drop_rate = (
            1.0
            - (
                perturbed_edge_index.size(1)
                / NUM_EDGES
            )
        )

        result = {
            "requested_drop_rate": rate,
            "actual_drop_rate": actual_drop_rate,
            "remaining_edges": int(
                perturbed_edge_index.size(1)
            ),
            "risk_score": perturbed_risk,
            "risk_drift": risk_drift,
            "latent_drift": latent_drift,
        }

        results["edge_drop"].append(result)

        print(
            f"Requested Drop: {rate * 100:4.0f}% | "
            f"Actual Drop: {actual_drop_rate * 100:4.1f}% | "
            f"Risk: {perturbed_risk:.4f} | "
            f"Risk Drift: {risk_drift:.4f} | "
            f"Latent Drift: {latent_drift:.4f}"
        )

    # ========================================================
    # 2. FEATURE NOISE INJECTION
    # ========================================================

    print(
        "\n--- 2. Feature Noise Injection Test ---"
    )

    for sigma in FEATURE_NOISE_LEVELS:
        sigma = _validate_non_negative_float(
            sigma,
            "noise_sigma",
        )

        generator = torch.Generator()
        generator.manual_seed(
            RANDOM_SEED + int(sigma * 100)
        )

        node_noise = torch.randn(
            x.shape,
            generator=generator,
        )

        edge_noise = torch.randn(
            edge_attr.shape,
            generator=generator,
        )

        noisy_x = (
            x
            + node_noise * sigma
        )

        noisy_edge_attr = (
            edge_attr
            + edge_noise * sigma
        )

        perturbed_risk, perturbed_latent = _evaluate_graph(
            encoder,
            decoder,
            noisy_x,
            edge_index,
            noisy_edge_attr,
        )

        risk_drift = abs(
            clean_risk - perturbed_risk
        )

        latent_drift = torch.linalg.vector_norm(
            clean_latent - perturbed_latent
        ).item()

        result = {
            "noise_sigma": sigma,
            "risk_score": perturbed_risk,
            "risk_drift": risk_drift,
            "latent_drift": latent_drift,
        }

        results["feature_noise"].append(result)

        print(
            f"Noise Sigma (σ): {sigma:3.1f} | "
            f"Risk: {perturbed_risk:.4f} | "
            f"Risk Drift: {risk_drift:.4f} | "
            f"Latent Drift: {latent_drift:.4f}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    all_risk_drifts = [
        item["risk_drift"]
        for item in results["edge_drop"]
    ]

    all_risk_drifts.extend(
        item["risk_drift"]
        for item in results["feature_noise"]
    )

    results["summary"] = {
        "max_risk_drift": (
            max(all_risk_drifts)
            if all_risk_drifts
            else 0.0
        ),
        "tests_completed": (
            len(results["edge_drop"])
            + len(results["feature_noise"])
        ),
    }

    print("=" * 70)
    print(
        "✅ Synthetic robustness stress testing complete."
    )
    print(
        "ℹ️ Risk/latent drift was measured; "
        "attention-weight stability was not directly measured."
    )
    print(
        "⚠️ Results are experimental stress-test signals, "
        "not validated real-world robustness metrics."
    )
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_adversarial_stress_test()