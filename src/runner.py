"""
runner.py - Real-Time Graph Telemetry Streamer & Predictive Engine Runner

Runs the Vanguard-GWM telemetry -> graph -> forecasting pipeline over a
sequence of graph snapshots.

Important:
    This runner performs inference and policy evaluation only.
    It does NOT automatically execute SDN quarantine or honeynet actions.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.config import settings
from src.data_pipeline import StreamingTelemetryParser
from src.engine import VanguardEngine


logger = logging.getLogger("Vanguard-Runner")


def _validate_positive_int(
    value: int,
    field_name: str,
) -> int:
    """Validate a positive integer parameter."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{field_name} must be greater than zero."
        )

    return value


def _validate_non_negative_float(
    value: float,
    field_name: str,
) -> float:
    """Validate a non-negative numeric parameter."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a number."
        )

    if value < 0:
        raise ValueError(
            f"{field_name} cannot be negative."
        )

    return float(value)


def run_pipeline(
    iterations: int = 5,
    delta_t_sec: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Run the Vanguard-GWM streaming inference pipeline.

    Args:
        iterations:
            Number of telemetry graph windows to process.

        delta_t_sec:
            Delay between snapshots in seconds.

    Returns:
        A list containing the inference result and processing
        latency for each processed snapshot.

    Raises:
        ValueError:
            If iterations or delta_t_sec is invalid.

    Note:
        The runner performs prediction and policy evaluation only.
        It does not automatically trigger network enforcement.
    """

    iterations = _validate_positive_int(
        iterations,
        "iterations",
    )

    delta_t_sec = _validate_non_negative_float(
        delta_t_sec,
        "delta_t_sec",
    )

    logger.info(
        "Initializing Vanguard-GWM Streaming Telemetry Pipeline..."
    )

    parser = StreamingTelemetryParser()

    engine = VanguardEngine(
        in_node_features=5,
        in_edge_features=5,
        latent_dim=settings.model_latent_dim,
        horizon_k=settings.forecast_horizon,
        risk_threshold=settings.defensive_threshold,
    )

    logger.info(
        "Starting streaming loop "
        "(%d windows, interval: %.2fs)...",
        iterations,
        delta_t_sec,
    )

    results: list[dict[str, Any]] = []

    for t_idx in range(iterations):
        logger.info(
            "--- Ingesting Snapshot Window t=%d ---",
            t_idx,
        )

        # ----------------------------------------------------
        # 1. Generate telemetry graph snapshot
        # ----------------------------------------------------

        graph_data = parser.synthesize_mock_window(
            num_nodes=12,
            num_edges=28,
        )

        # ----------------------------------------------------
        # 2. Extract graph tensors
        # ----------------------------------------------------

        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr

        target_ip = "192.168.1.105"

        # ----------------------------------------------------
        # 3. Run Vanguard-GWM inference
        # ----------------------------------------------------

        start_time = time.perf_counter()

        result = engine.process_graph_snapshot(
            x,
            edge_index,
            edge_attr,
            target_ip,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        # ----------------------------------------------------
        # 4. Collect normalized result
        # ----------------------------------------------------

        snapshot_result = {
            "snapshot_index": t_idx,
            "target_ip": result["target_ip"],
            "current_stage": result["current_stage"],
            "peak_risk": result["peak_risk"],
            "risk_trajectory": result["risk_trajectory"],
            "defensive_threshold": result.get(
                "defensive_threshold",
                settings.defensive_threshold,
            ),
            "threshold_crossed": result.get(
                "threshold_crossed",
                False,
            ),
            "sdn_quarantine_enforced": result.get(
                "sdn_quarantine_enforced",
                False,
            ),
            "latency_ms": latency_ms,
        }

        results.append(snapshot_result)

        # ----------------------------------------------------
        # 5. Operational logging
        # ----------------------------------------------------

        logger.info(
            "Target Host: %s",
            result["target_ip"],
        )

        logger.info(
            "Forecasted ATT&CK Phase: %s",
            result["current_stage"],
        )

        logger.info(
            "Peak Risk Score: %.1f",
            result["peak_risk"] * 100.0,
        )

        logger.info(
            "Horizon Risk Trajectory: %s",
            result["risk_trajectory"],
        )

        threshold_crossed = result.get(
            "threshold_crossed",
            False,
        )

        if threshold_crossed:
            logger.info(
                "Defensive Decision: THRESHOLD CROSSED "
                "(human/policy approval required)"
            )
        else:
            logger.info(
                "Defensive Decision: BELOW THRESHOLD"
            )

        logger.info(
            "Automatic SDN Enforcement: DISABLED"
        )

        logger.info(
            "Processing Latency: %.2f ms",
            latency_ms,
        )

        # ----------------------------------------------------
        # 6. Wait after every processed snapshot
        # ----------------------------------------------------

        if delta_t_sec > 0:
            time.sleep(delta_t_sec)

    # --------------------------------------------------------
    # 7. Pipeline completion
    # --------------------------------------------------------

    logger.info(
        "Streaming pipeline completed successfully: "
        "%d/%d windows.",
        len(results),
        iterations,
    )

    return results


if __name__ == "__main__":
    run_pipeline(
        iterations=5,
        delta_t_sec=1.5,
    )