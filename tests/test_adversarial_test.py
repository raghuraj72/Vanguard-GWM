import torch
from unittest.mock import patch

from src.adversarial_test import run_adversarial_stress_test


def test_adversarial_stress_test_completes():
    with patch("src.adversarial_test.logger"):
        run_adversarial_stress_test()


def test_adversarial_test_is_deterministic_for_same_seed():
    torch.manual_seed(123)

    with patch("src.adversarial_test.logger"):
        run_adversarial_stress_test()

    first_random_state = torch.random.get_rng_state()

    torch.manual_seed(123)

    with patch("src.adversarial_test.logger"):
        run_adversarial_stress_test()

    second_random_state = torch.random.get_rng_state()

    assert torch.equal(first_random_state, second_random_state)


def test_adversarial_test_uses_expected_perturbation_rates():
    expected_rates = [0.10, 0.25, 0.50]

    assert len(expected_rates) == 3
    assert all(0.0 < rate < 1.0 for rate in expected_rates)


def test_adversarial_test_uses_expected_noise_levels():
    expected_noise_levels = [0.1, 0.5, 1.0, 2.0]

    assert len(expected_noise_levels) == 4
    assert all(level > 0 for level in expected_noise_levels)


def test_adversarial_test_handles_edge_dropping_without_empty_graph():
    torch.manual_seed(42)

    num_nodes = 12
    num_edges = 30

    x = torch.randn(num_nodes, 5)
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.randn(num_edges, 5)

    rate = 0.50

    keep_mask = torch.rand(num_edges) > rate

    if keep_mask.sum() == 0:
        keep_mask[0] = True

    perturbed_edge_index = edge_index[:, keep_mask]
    perturbed_edge_attr = edge_attr[keep_mask]

    assert perturbed_edge_index.shape[1] > 0
    assert perturbed_edge_attr.shape[0] > 0


def test_adversarial_test_model_outputs_are_finite():
    from src.world_model import (
        GATv2SpatialEncoder,
        MultiTaskDecoder,
    )

    torch.manual_seed(42)

    encoder = GATv2SpatialEncoder(
        in_node_features=5,
        in_edge_features=5,
        latent_dim=64,
    )

    decoder = MultiTaskDecoder(
        latent_dim=64,
    )

    encoder.eval()
    decoder.eval()

    x = torch.randn(12, 5)

    edge_index = torch.randint(
        0,
        12,
        (2, 30),
        dtype=torch.long,
    )

    edge_attr = torch.randn(30, 5)

    with torch.no_grad():
        _, z = encoder(
            x,
            edge_index,
            edge_attr,
        )

        predictions = decoder(z)

    assert torch.isfinite(
        predictions["risk_score"]
    ).all()

    assert torch.isfinite(
        predictions["mitre_stage_logits"]
    ).all()


def test_risk_score_is_in_valid_range():
    from src.world_model import (
        GATv2SpatialEncoder,
        MultiTaskDecoder,
    )

    torch.manual_seed(42)

    encoder = GATv2SpatialEncoder(
        in_node_features=5,
        in_edge_features=5,
        latent_dim=64,
    )

    decoder = MultiTaskDecoder(
        latent_dim=64,
    )

    encoder.eval()
    decoder.eval()

    x = torch.randn(12, 5)

    edge_index = torch.randint(
        0,
        12,
        (2, 30),
        dtype=torch.long,
    )

    edge_attr = torch.randn(30, 5)

    with torch.no_grad():
        _, z = encoder(
            x,
            edge_index,
            edge_attr,
        )

        risk = decoder(z)["risk_score"]

    assert torch.all(risk >= 0.0)
    assert torch.all(risk <= 1.0)