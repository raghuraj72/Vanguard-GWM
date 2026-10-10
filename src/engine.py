"""
engine.py - Core Orchestration Engine for Vanguard-GWM

Connects:
    Streaming telemetry
        -> Graph World Model
        -> Temporal rollout
        -> Risk / MITRE-stage prediction

Important:
    This engine performs inference and policy evaluation only.
    It does NOT automatically push SDN quarantine rules.

    Any defensive enforcement should require an explicit approval
    workflow outside the inference engine.
"""

import logging
from typing import Any, Dict, List

import torch

from src.config import settings
from src.world_model import (
    GATv2SpatialEncoder,
    TemporalDynamicsTransformer,
    MultiTaskDecoder,
)
from src.sdn_enforcer import SDNEnforcer


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger("Vanguard-Engine")

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
    )


# ---------------------------------------------------------------------
# Vanguard Engine
# ---------------------------------------------------------------------


class VanguardEngine:
    """
    Core Vanguard-GWM inference engine.

    Responsibilities:
        1. Encode a graph snapshot.
        2. Maintain a short temporal history.
        3. Forecast future graph states.
        4. Decode risk and attack-stage predictions.
        5. Evaluate the configured defensive decision threshold.

    Important:
        The SDN enforcement component is retained for architectural
        compatibility and future explicit approval workflows.

        This engine NEVER calls the enforcement layer automatically.
    """

    def __init__(
        self,
        in_node_features: int = 5,
        in_edge_features: int = 5,
        latent_dim: int | None = None,
        horizon_k: int | None = None,
        risk_threshold: float | None = None,
    ):
        # -------------------------------------------------------------
        # Configuration
        # -------------------------------------------------------------

        self.latent_dim = (
            settings.model_latent_dim
            if latent_dim is None
            else latent_dim
        )

        self.horizon_k = (
            settings.forecast_horizon
            if horizon_k is None
            else horizon_k
        )

        self.risk_threshold = (
            settings.defensive_threshold
            if risk_threshold is None
            else risk_threshold
        )

        # -------------------------------------------------------------
        # Validate configuration
        # -------------------------------------------------------------

        if self.latent_dim <= 0:
            raise ValueError(
                f"latent_dim must be greater than 0, got {self.latent_dim}"
            )

        if self.horizon_k <= 0:
            raise ValueError(
                f"horizon_k must be greater than 0, got {self.horizon_k}"
            )

        if not 0.0 <= self.risk_threshold <= 1.0:
            raise ValueError(
                "risk_threshold must be between 0.0 and 1.0, "
                f"got {self.risk_threshold}"
            )

        # -------------------------------------------------------------
        # Neural Network Modules
        # -------------------------------------------------------------

        self.spatial_encoder = GATv2SpatialEncoder(
            in_node_features=in_node_features,
            in_edge_features=in_edge_features,
            latent_dim=self.latent_dim,
        )

        self.dynamics_transformer = TemporalDynamicsTransformer(
            latent_dim=self.latent_dim,
            nhead=4,
            num_layers=2,
        )

        self.decoder = MultiTaskDecoder(
            latent_dim=self.latent_dim,
            num_mitre_stages=5,
        )

        # -------------------------------------------------------------
        # SDN Enforcement Component
        # -------------------------------------------------------------
        #
        # Retained for compatibility with the existing architecture
        # and future explicit approval workflows.
        #
        # IMPORTANT:
        # process_graph_snapshot() NEVER calls this component.
        #

        self.sdn_enforcer = SDNEnforcer()

        # -------------------------------------------------------------
        # Temporal History
        # -------------------------------------------------------------

        self.history_buffer: List[torch.Tensor] = []

        # Keep history bounded so memory usage remains predictable.
        self.max_history_snapshots = 8

        logger.info(
            "VanguardEngine initialized | "
            "latent_dim=%s | horizon=%s | threshold=%.2f",
            self.latent_dim,
            self.horizon_k,
            self.risk_threshold,
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def process_graph_snapshot(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        target_ip: str,
    ) -> Dict[str, Any]:
        """
        Process one graph snapshot and forecast future threat state.

        Parameters
        ----------
        x:
            Node feature tensor.

        edge_index:
            Graph connectivity tensor.

        edge_attr:
            Edge feature tensor.

        target_ip:
            Identifier of the target asset.

        Returns
        -------
        Dict[str, Any]
            Forecast and policy-evaluation results.

        Notes
        -----
        This method performs model inference and policy evaluation only.

        It does NOT:
            - modify firewall rules
            - isolate hosts
            - push SDN rules
            - execute external network actions
        """

        # -------------------------------------------------------------
        # Put inference modules into evaluation mode
        # -------------------------------------------------------------

        self.spatial_encoder.eval()
        self.dynamics_transformer.eval()
        self.decoder.eval()

        with torch.no_grad():

            # =========================================================
            # 1. Spatial Perception
            # =========================================================
            # G_t -> z_t

            z_nodes, z_graph = self.spatial_encoder(
                x,
                edge_index,
                edge_attr,
            )

            # z_graph normally has shape:
            # [1, latent_dim]

            if z_graph.dim() == 1:
                z_graph = z_graph.unsqueeze(0)

            if z_graph.dim() != 2:
                raise ValueError(
                    "Expected z_graph to have shape "
                    "[batch, latent_dim], "
                    f"got {tuple(z_graph.shape)}"
                )

            # =========================================================
            # 2. Update Temporal History
            # =========================================================

            self.history_buffer.append(
                z_graph.detach()
            )

            if len(self.history_buffer) > self.max_history_snapshots:
                self.history_buffer.pop(0)

            # =========================================================
            # 3. Temporal Dynamics Rollout
            # =========================================================

            context_seq = torch.cat(
                self.history_buffer,
                dim=0,
            ).unsqueeze(0)

            rollout_latents = (
                self.dynamics_transformer.predict_rollout(
                    context_seq,
                    k_steps=self.horizon_k,
                )
            )

            # Expected:
            # [batch, horizon, latent_dim]

            if rollout_latents.dim() != 3:
                raise ValueError(
                    "Expected rollout_latents to have shape "
                    "[batch, horizon, latent_dim], "
                    f"got {tuple(rollout_latents.shape)}"
                )

            if rollout_latents.shape[1] < self.horizon_k:
                raise ValueError(
                    "Temporal model returned fewer rollout steps "
                    f"than requested: expected {self.horizon_k}, "
                    f"got {rollout_latents.shape[1]}"
                )

            # =========================================================
            # 4. Multi-Task Forecast Decoding
            # =========================================================

            risk_scores: List[float] = []
            phase_predictions: List[int] = []

            for k_idx in range(self.horizon_k):

                z_hat = rollout_latents[
                    :,
                    k_idx,
                    :,
                ]

                preds = self.decoder(z_hat)

                # -----------------------------------------------------
                # Risk
                # -----------------------------------------------------

                if "risk_score" not in preds:
                    raise KeyError(
                        "Decoder output is missing 'risk_score'."
                    )

                risk_value = preds["risk_score"].item()

                # Defensive range validation.
                #
                # This does NOT reinterpret the value as a calibrated
                # probability.

                risk_value = max(
                    0.0,
                    min(
                        1.0,
                        float(risk_value),
                    ),
                )

                # -----------------------------------------------------
                # MITRE-stage prediction
                # -----------------------------------------------------

                if "mitre_stage_logits" not in preds:
                    raise KeyError(
                        "Decoder output is missing "
                        "'mitre_stage_logits'."
                    )

                stage_logits = preds[
                    "mitre_stage_logits"
                ]

                stage_idx = int(
                    torch.argmax(
                        stage_logits,
                        dim=-1,
                    ).item()
                )

                risk_scores.append(
                    risk_value
                )

                phase_predictions.append(
                    stage_idx
                )

            # =========================================================
            # 5. Aggregate Forecast
            # =========================================================

            if not risk_scores:
                raise RuntimeError(
                    "Model returned an empty risk trajectory."
                )

            peak_risk = max(
                risk_scores
            )

            # Current MultiTaskDecoder stage mapping.
            #
            # "Benign Baseline" is a model output label.
            # It is NOT proof that the environment is actually benign.

            stages = [
                "Benign Baseline",
                "Reconnaissance",
                "Initial Access",
                "Lateral Movement",
                "Exfiltration",
            ]

            final_stage_idx = (
                phase_predictions[-1]
            )

            if 0 <= final_stage_idx < len(stages):
                current_stage = stages[
                    final_stage_idx
                ]
            else:
                current_stage = "Unknown"

            # =========================================================
            # 6. Defensive Policy Evaluation
            # =========================================================
            #
            # IMPORTANT:
            #
            # Threshold crossing only produces a policy signal.
            #
            # It does NOT:
            #     - isolate the host
            #     - push an SDN rule
            #     - modify the network
            #
            # Explicit enforcement must happen through a separate
            # approval workflow.

            threshold_crossed = (
                peak_risk >= self.risk_threshold
            )

            if threshold_crossed:
                logger.warning(
                    "Defensive decision threshold crossed for %s "
                    "(peak model risk: %.2f%%, threshold: %.2f%%). "
                    "No automatic enforcement was performed.",
                    target_ip,
                    peak_risk * 100.0,
                    self.risk_threshold * 100.0,
                )

            # =========================================================
            # 7. Structured Result
            # =========================================================

            return {
                # Target
                "target_ip": target_ip,

                # Forecast interpretation
                "current_stage": current_stage,
                "phase_predictions": phase_predictions,

                # Risk
                "peak_risk": round(
                    peak_risk,
                    4,
                ),
                "risk_trajectory": [
                    round(
                        risk,
                        4,
                    )
                    for risk in risk_scores
                ],

                # Policy evaluation
                "defensive_threshold": round(
                    self.risk_threshold,
                    4,
                ),
                "threshold_crossed": (
                    threshold_crossed
                ),

                # Enforcement status
                #
                # Kept for dashboard/test compatibility.
                # Always False because this engine does not execute
                # enforcement.

                "sdn_quarantine_enforced": False,
            }

    # -----------------------------------------------------------------
    # Utility Methods
    # -----------------------------------------------------------------

    def reset_history(self) -> None:
        """
        Clear the temporal graph history.

        Useful when:
            - starting a new telemetry stream
            - switching datasets
            - resetting a dashboard session
            - beginning a new incident analysis
        """

        self.history_buffer.clear()

        logger.info(
            "VanguardEngine temporal history reset."
        )

    def get_history_length(self) -> int:
        """
        Return the number of graph snapshots currently stored.
        """

        return len(
            self.history_buffer
        )

    def is_threshold_crossed(
        self,
        peak_risk: float,
    ) -> bool:
        """
        Evaluate whether a model risk score crosses the configured
        defensive decision threshold.

        This method performs policy evaluation only.
        """

        return (
            float(peak_risk)
            >= self.risk_threshold
        )