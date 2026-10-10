"""
world_model.py - Generative Graph World Model Core Neural Architectures

Includes:
    - GATv2SpatialEncoder
    - TemporalDynamicsTransformer
    - MultiTaskDecoder

The models in this module are intended for forecasting and
decision-support workflows. They do not perform network enforcement.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2SpatialEncoder(nn.Module):
    """
    Graph-attention-inspired spatial encoder.

    Converts node and edge telemetry features into:
        1. Node-level latent representations.
        2. A graph-level pooled latent representation.

    Expected inputs:
        x:
            [num_nodes, in_node_features]

        edge_index:
            [2, num_edges]

        edge_attr:
            [num_edges, in_edge_features]
    """

    def __init__(
        self,
        in_node_features: int = 5,
        in_edge_features: int = 5,
        latent_dim: int = 64,
    ):
        super().__init__()

        self._validate_positive_int(
            in_node_features,
            "in_node_features",
        )
        self._validate_positive_int(
            in_edge_features,
            "in_edge_features",
        )
        self._validate_positive_int(
            latent_dim,
            "latent_dim",
        )

        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.latent_dim = latent_dim

        # Project node and edge telemetry into a shared latent space.
        self.node_proj = nn.Linear(
            in_node_features,
            latent_dim,
        )

        self.edge_proj = nn.Linear(
            in_edge_features,
            latent_dim,
        )

        # Attention score uses:
        # source node + destination node + edge representation.
        self.attn_fc = nn.Linear(
            latent_dim * 3,
            1,
        )

        self.leaky_relu = nn.LeakyReLU(
            negative_slope=0.2,
        )

        self.out_fc = nn.Linear(
            latent_dim,
            latent_dim,
        )

    @staticmethod
    def _validate_positive_int(
        value: int,
        name: str,
    ) -> None:
        if not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer, "
                f"got {type(value).__name__}."
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be greater than zero, "
                f"got {value}."
            )

    def _validate_inputs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")

        if not isinstance(edge_index, torch.Tensor):
            raise TypeError(
                "edge_index must be a torch.Tensor."
            )

        if not isinstance(edge_attr, torch.Tensor):
            raise TypeError(
                "edge_attr must be a torch.Tensor."
            )

        if x.dim() != 2:
            raise ValueError(
                "x must have shape "
                "[num_nodes, in_node_features]."
            )

        if x.size(1) != self.in_node_features:
            raise ValueError(
                "Invalid node feature dimension: "
                f"expected {self.in_node_features}, "
                f"got {x.size(1)}."
            )

        if edge_index.dim() != 2:
            raise ValueError(
                "edge_index must have shape [2, num_edges]."
            )

        if edge_index.size(0) != 2:
            raise ValueError(
                "edge_index must have first dimension 2, "
                f"got {edge_index.size(0)}."
            )

        if edge_attr.dim() != 2:
            raise ValueError(
                "edge_attr must have shape "
                "[num_edges, in_edge_features]."
            )

        if edge_attr.size(1) != self.in_edge_features:
            raise ValueError(
                "Invalid edge feature dimension: "
                f"expected {self.in_edge_features}, "
                f"got {edge_attr.size(1)}."
            )

        num_edges = edge_index.size(1)

        if edge_attr.size(0) != num_edges:
            raise ValueError(
                "edge_index and edge_attr must contain "
                "the same number of edges: "
                f"{num_edges} != {edge_attr.size(0)}."
            )

        if x.size(0) == 0:
            raise ValueError(
                "Graph must contain at least one node."
            )

        if edge_index.numel() > 0:
            min_index = int(edge_index.min().item())
            max_index = int(edge_index.max().item())

            if min_index < 0:
                raise ValueError(
                    "edge_index contains a negative node index."
                )

            if max_index >= x.size(0):
                raise ValueError(
                    "edge_index contains a node index outside "
                    f"the valid range [0, {x.size(0) - 1}]."
                )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode one graph snapshot.

        Returns:
            out_nodes:
                [num_nodes, latent_dim]

            z_graph:
                [1, latent_dim]
        """

        self._validate_inputs(
            x,
            edge_index,
            edge_attr,
        )

        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # Node projection.
        h = self.node_proj(x)

        # Handle graphs with no edges.
        #
        # This keeps inference stable for isolated-node graphs and
        # avoids invalid attention/message-passing operations.
        if num_edges == 0:
            out_nodes = F.elu(
                self.out_fc(h) + h
            )

            z_graph = torch.mean(
                out_nodes,
                dim=0,
                keepdim=True,
            )

            return out_nodes, z_graph

        # Edge projection.
        e = self.edge_proj(edge_attr)

        src = edge_index[0].long()
        dst = edge_index[1].long()

        h_src = h[src]
        h_dst = h[dst]

        # Attention input:
        # [source_node, destination_node, edge_features]
        score_input = torch.cat(
            [
                h_src,
                h_dst,
                e,
            ],
            dim=-1,
        )

        scores = self.leaky_relu(
            self.attn_fc(score_input)
        ).squeeze(-1)

        # Attention normalization across the graph's edges.
        alpha = torch.softmax(
            scores,
            dim=0,
        ).unsqueeze(-1)

        # Message passing aggregation.
        msg = alpha * (
            h_dst + e
        )

        out_nodes = torch.zeros(
            num_nodes,
            self.latent_dim,
            dtype=h.dtype,
            device=x.device,
        )

        out_nodes.index_add_(
            0,
            src,
            msg,
        )

        # Residual connection from original node representation.
        out_nodes = F.elu(
            self.out_fc(out_nodes) + h
        )

        # Graph-level representation.
        z_graph = torch.mean(
            out_nodes,
            dim=0,
            keepdim=True,
        )

        return out_nodes, z_graph


class TemporalDynamicsTransformer(nn.Module):
    """
    Transformer-based temporal dynamics model.

    Given a sequence of graph-level latent representations, the model
    autoregressively predicts future latent states.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self._validate_positive_int(
            latent_dim,
            "latent_dim",
        )
        self._validate_positive_int(
            nhead,
            "nhead",
        )
        self._validate_positive_int(
            num_layers,
            "num_layers",
        )

        if latent_dim % nhead != 0:
            raise ValueError(
                "latent_dim must be divisible by nhead: "
                f"{latent_dim} % {nhead} != 0."
            )

        self.latent_dim = latent_dim
        self.nhead = nhead
        self.num_layers = num_layers

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=latent_dim * 2,
            dropout=0.1,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.transition_head = nn.Linear(
            latent_dim,
            latent_dim,
        )

    @staticmethod
    def _validate_positive_int(
        value: int,
        name: str,
    ) -> None:
        if not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer, "
                f"got {type(value).__name__}."
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be greater than zero, "
                f"got {value}."
            )

    def _validate_context(
        self,
        context_seq: torch.Tensor,
    ) -> None:
        if not isinstance(
            context_seq,
            torch.Tensor,
        ):
            raise TypeError(
                "context_seq must be a torch.Tensor."
            )

        if context_seq.dim() != 3:
            raise ValueError(
                "context_seq must have shape "
                "[batch_size, sequence_length, latent_dim]."
            )

        if context_seq.size(0) <= 0:
            raise ValueError(
                "context_seq must contain at least one batch."
            )

        if context_seq.size(1) <= 0:
            raise ValueError(
                "context_seq must contain at least one "
                "temporal observation."
            )

        if context_seq.size(2) != self.latent_dim:
            raise ValueError(
                "Invalid context latent dimension: "
                f"expected {self.latent_dim}, "
                f"got {context_seq.size(2)}."
            )

    def predict_rollout(
        self,
        context_seq: torch.Tensor,
        k_steps: int = 5,
    ) -> torch.Tensor:
        """
        Autoregressively roll out k future latent states.

        Input:
            context_seq:
                [batch_size, sequence_length, latent_dim]

        Output:
            [batch_size, k_steps, latent_dim]
        """

        self._validate_context(
            context_seq
        )

        if not isinstance(k_steps, int):
            raise TypeError(
                "k_steps must be an integer."
            )

        if k_steps < 0:
            raise ValueError(
                f"k_steps must be >= 0, got {k_steps}."
            )

        if k_steps == 0:
            return context_seq.new_empty(
                (
                    context_seq.size(0),
                    0,
                    self.latent_dim,
                )
            )

        curr_seq = context_seq.clone()

        rollouts = []

        for _ in range(k_steps):
            out = self.transformer(
                curr_seq
            )

            next_z = self.transition_head(
                out[:, -1:, :]
            )

            rollouts.append(
                next_z
            )

            curr_seq = torch.cat(
                [
                    curr_seq,
                    next_z,
                ],
                dim=1,
            )

        return torch.cat(
            rollouts,
            dim=1,
        )


class MultiTaskDecoder(nn.Module):
    """
    Multi-task forecasting decoder.

    Produces:
        - A bounded risk score in [0, 1].
        - MITRE/attack-stage logits.

    The risk output is a model score and should not be interpreted
    as a calibrated probability without appropriate validation.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        num_mitre_stages: int = 5,
    ):
        super().__init__()

        self._validate_positive_int(
            latent_dim,
            "latent_dim",
        )
        self._validate_positive_int(
            num_mitre_stages,
            "num_mitre_stages",
        )

        self.latent_dim = latent_dim
        self.num_mitre_stages = num_mitre_stages

        self.risk_head = nn.Sequential(
            nn.Linear(
                latent_dim,
                32,
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                1,
            ),
            nn.Sigmoid(),
        )

        self.mitre_head = nn.Sequential(
            nn.Linear(
                latent_dim,
                32,
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                num_mitre_stages,
            ),
        )

    @staticmethod
    def _validate_positive_int(
        value: int,
        name: str,
    ) -> None:
        if not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer, "
                f"got {type(value).__name__}."
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be greater than zero, "
                f"got {value}."
            )

    def forward(
        self,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent representations.

        Input:
            z:
                [..., latent_dim]

        Returns:
            {
                "risk_score": [..., 1],
                "mitre_stage_logits": [..., num_mitre_stages]
            }
        """

        if not isinstance(z, torch.Tensor):
            raise TypeError(
                "z must be a torch.Tensor."
            )

        if z.dim() < 2:
            raise ValueError(
                "z must have at least 2 dimensions, "
                "with the final dimension representing latent features."
            )

        if z.size(-1) != self.latent_dim:
            raise ValueError(
                "Invalid latent dimension: "
                f"expected {self.latent_dim}, "
                f"got {z.size(-1)}."
            )

        return {
            "risk_score": self.risk_head(z),
            "mitre_stage_logits": self.mitre_head(z),
        }