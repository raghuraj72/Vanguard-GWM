import torch

from src.engine import VanguardEngine


def create_test_graph(num_nodes=10, num_edges=20):
    x = torch.randn(num_nodes, 5)

    edge_index = torch.randint(
        0,
        num_nodes,
        (2, num_edges),
        dtype=torch.long,
    )

    edge_attr = torch.randn(num_edges, 5)

    return x, edge_index, edge_attr


def test_engine_initializes():
    engine = VanguardEngine()

    assert engine.latent_dim == 64
    assert engine.horizon_k == 5
    assert engine.risk_threshold == 0.60


def test_engine_custom_configuration():
    engine = VanguardEngine(
        in_node_features=5,
        in_edge_features=5,
        latent_dim=32,
        horizon_k=3,
        risk_threshold=0.75,
    )

    assert engine.latent_dim == 32
    assert engine.horizon_k == 3
    assert engine.risk_threshold == 0.75


def test_engine_has_required_components():
    engine = VanguardEngine()

    assert engine.spatial_encoder is not None
    assert engine.dynamics_transformer is not None
    assert engine.decoder is not None
    assert engine.sdn_enforcer is not None


def test_engine_history_starts_empty():
    engine = VanguardEngine()

    assert engine.history_buffer == []


def test_process_graph_snapshot_returns_dictionary():
    engine = VanguardEngine()

    x, edge_index, edge_attr = create_test_graph()

    result = engine.process_graph_snapshot(
        x,
        edge_index,
        edge_attr,
        target_ip="192.168.1.10",
    )

    assert isinstance(result, dict)


def test_process_graph_snapshot_contains_required_keys():
    engine = VanguardEngine()

    x, edge_index, edge_attr = create_test_graph()

    result = engine.process_graph_snapshot(
        x,
        edge_index,
        edge_attr,
        target_ip="192.168.1.10",
    )

    expected_keys = {
        "target_ip",
        "current_stage",
        "peak_risk",
        "risk_trajectory",
        "sdn_quarantine_enforced",
    }

    assert expected_keys.issubset(result.keys())


def test_target_ip_is_preserved():
    engine = VanguardEngine()

    x, edge_index, edge_attr = create_test_graph()

    target_ip = "10.0.0.42"

    result = engine.process_graph_snapshot(
        x,
        edge_index,
        edge_attr,
        target_ip=target_ip,
    )

    assert result["target_ip"] == target_ip


def test_risk_trajectory_matches_horizon():
    horizon = 7

    engine = VanguardEngine(
        horizon_k=horizon,
    )

    x, edge_index, edge_attr = create_test_graph()

    result = engine.process_graph_snapshot(
        x,
        edge_index,
        edge_attr,
        target_ip="192.168.1.20",
    )

    assert len(result["risk_trajectory"]) == horizon


def test_risk_values_are_valid():
    engine = VanguardEngine()

    x, edge_index, edge_attr = create_test_graph()

    result = engine.process_graph_snapshot(
        x,
        edge_index,
        edge_attr,
        target_ip="192.168.1.30",
    )

    for risk in result["risk_trajectory"]:
        assert 0.0 <= risk <= 1.0

    assert 0.0 <= result["peak_risk"] <= 1.0


def test_current_stage_is_valid():
    engine = VanguardEngine()

    x, edge_index, edge_attr = create_test_graph()

    result = engine.process_graph_snapshot(
        x,
        edge_index,
        edge_attr,
        target_ip="192.168.1.40",
    )

    valid_stages = {
        "Benign Baseline",
        "Reconnaissance",
        "Initial Access",
        "Lateral Movement",
        "Exfiltration",
    }

    assert result["current_stage"] in valid_stages


def test_history_buffer_updates():
    engine = VanguardEngine()

    x, edge_index, edge_attr = create_test_graph()

    assert len(engine.history_buffer) == 0

    engine.process_graph_snapshot(
        x,
        edge_index,
        edge_attr,
        target_ip="192.168.1.50",
    )

    assert len(engine.history_buffer) == 1


def test_history_buffer_accumulates_snapshots():
    engine = VanguardEngine()

    x, edge_index, edge_attr = create_test_graph()

    for i in range(3):
        engine.process_graph_snapshot(
            x,
            edge_index,
            edge_attr,
            target_ip=f"192.168.1.{50 + i}",
        )

    assert len(engine.history_buffer) == 3


def test_history_buffer_is_limited_to_eight():
    engine = VanguardEngine()

    x, edge_index, edge_attr = create_test_graph()

    for i in range(12):
        engine.process_graph_snapshot(
            x,
            edge_index,
            edge_attr,
            target_ip=f"192.168.1.{i + 1}",
        )

    assert len(engine.history_buffer) == 8


def test_sdn_result_is_boolean():
    engine = VanguardEngine()

    x, edge_index, edge_attr = create_test_graph()

    result = engine.process_graph_snapshot(
        x,
        edge_index,
        edge_attr,
        target_ip="192.168.1.60",
    )

    assert isinstance(
        result["sdn_quarantine_enforced"],
        bool,
    )