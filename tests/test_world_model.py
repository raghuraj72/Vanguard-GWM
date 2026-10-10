import torch

from src.world_model import (
    GATv2SpatialEncoder,
    TemporalDynamicsTransformer,
    MultiTaskDecoder,
)


# -------------------------------------------------------------------
# GATv2SpatialEncoder
# -------------------------------------------------------------------

def test_gatv2_encoder_initializes():
    model = GATv2SpatialEncoder()

    assert model.latent_dim == 64


def test_gatv2_encoder_output_shapes():
    model = GATv2SpatialEncoder(
        in_node_features=5,
        in_edge_features=5,
        latent_dim=64,
    )

    num_nodes = 10
    num_edges = 24

    x = torch.randn(num_nodes, 5)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 5)

    node_latent, graph_latent = model(
        x,
        edge_index,
        edge_attr,
    )

    assert node_latent.shape == (num_nodes, 64)
    assert graph_latent.shape == (1, 64)


def test_gatv2_encoder_custom_latent_dimension():
    latent_dim = 32

    model = GATv2SpatialEncoder(
        in_node_features=5,
        in_edge_features=5,
        latent_dim=latent_dim,
    )

    x = torch.randn(8, 5)
    edge_index = torch.randint(0, 8, (2, 12))
    edge_attr = torch.randn(12, 5)

    node_latent, graph_latent = model(
        x,
        edge_index,
        edge_attr,
    )

    assert node_latent.shape == (8, latent_dim)
    assert graph_latent.shape == (1, latent_dim)


def test_gatv2_encoder_outputs_finite_values():
    model = GATv2SpatialEncoder()

    x = torch.randn(10, 5)
    edge_index = torch.randint(0, 10, (2, 20))
    edge_attr = torch.randn(20, 5)

    node_latent, graph_latent = model(
        x,
        edge_index,
        edge_attr,
    )

    assert torch.isfinite(node_latent).all()
    assert torch.isfinite(graph_latent).all()


def test_gatv2_encoder_supports_empty_edges():
    model = GATv2SpatialEncoder()

    x = torch.randn(5, 5)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 5))

    node_latent, graph_latent = model(
        x,
        edge_index,
        edge_attr,
    )

    assert node_latent.shape == (5, 64)
    assert graph_latent.shape == (1, 64)
    assert torch.isfinite(node_latent).all()


# -------------------------------------------------------------------
# TemporalDynamicsTransformer
# -------------------------------------------------------------------

def test_temporal_transformer_initializes():
    model = TemporalDynamicsTransformer()

    assert model.latent_dim == 64


def test_temporal_transformer_rollout_shape():
    model = TemporalDynamicsTransformer(
        latent_dim=64,
        nhead=4,
        num_layers=2,
    )

    context_seq = torch.randn(1, 5, 64)

    rollout = model.predict_rollout(
        context_seq,
        k_steps=5,
    )

    assert rollout.shape == (1, 5, 64)


def test_temporal_transformer_custom_rollout_length():
    model = TemporalDynamicsTransformer()

    context_seq = torch.randn(1, 3, 64)

    rollout = model.predict_rollout(
        context_seq,
        k_steps=8,
    )

    assert rollout.shape == (1, 8, 64)


def test_temporal_transformer_outputs_finite_values():
    model = TemporalDynamicsTransformer()

    context_seq = torch.randn(1, 4, 64)

    rollout = model.predict_rollout(
        context_seq,
        k_steps=5,
    )

    assert torch.isfinite(rollout).all()


# -------------------------------------------------------------------
# MultiTaskDecoder
# -------------------------------------------------------------------

def test_decoder_initializes():
    model = MultiTaskDecoder(
        latent_dim=64,
        num_mitre_stages=5,
    )

    assert model is not None


def test_decoder_output_keys():
    model = MultiTaskDecoder()

    z = torch.randn(1, 64)

    output = model(z)

    assert "risk_score" in output
    assert "mitre_stage_logits" in output


def test_decoder_output_shapes():
    model = MultiTaskDecoder(
        latent_dim=64,
        num_mitre_stages=5,
    )

    z = torch.randn(1, 64)

    output = model(z)

    assert output["risk_score"].shape == (1, 1)
    assert output["mitre_stage_logits"].shape == (1, 5)


def test_decoder_risk_score_is_between_zero_and_one():
    model = MultiTaskDecoder()

    z = torch.randn(10, 64)

    output = model(z)

    risk_score = output["risk_score"]

    assert torch.all(risk_score >= 0)
    assert torch.all(risk_score <= 1)


def test_decoder_supports_custom_mitre_stage_count():
    model = MultiTaskDecoder(
        latent_dim=64,
        num_mitre_stages=8,
    )

    z = torch.randn(2, 64)

    output = model(z)

    assert output["risk_score"].shape == (2, 1)
    assert output["mitre_stage_logits"].shape == (2, 8)


def test_decoder_outputs_finite_values():
    model = MultiTaskDecoder()

    z = torch.randn(4, 64)

    output = model(z)

    assert torch.isfinite(output["risk_score"]).all()
    assert torch.isfinite(output["mitre_stage_logits"]).all()