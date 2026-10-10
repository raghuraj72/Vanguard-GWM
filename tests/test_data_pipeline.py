import torch
from torch_geometric.data import Data

from src.data_pipeline import StreamingTelemetryParser


def test_parser_initializes_with_default_window():
    parser = StreamingTelemetryParser()

    assert parser.window_size_sec == 2.0


def test_parser_accepts_custom_window_size():
    parser = StreamingTelemetryParser(window_size_sec=5.0)

    assert parser.window_size_sec == 5.0


def test_synthesize_mock_window_returns_graph():
    parser = StreamingTelemetryParser()

    graph = parser.synthesize_mock_window()

    assert isinstance(graph, Data)


def test_graph_has_correct_node_features():
    parser = StreamingTelemetryParser()

    graph = parser.synthesize_mock_window(
        num_nodes=10,
        num_edges=24
    )

    assert graph.x.shape == (10, 5)


def test_graph_has_correct_edges():
    parser = StreamingTelemetryParser()

    graph = parser.synthesize_mock_window(
        num_nodes=10,
        num_edges=24
    )

    assert graph.edge_index.shape == (2, 24)


def test_graph_has_correct_edge_attributes():
    parser = StreamingTelemetryParser()

    graph = parser.synthesize_mock_window(
        num_nodes=10,
        num_edges=24
    )

    assert graph.edge_attr.shape == (24, 5)


def test_graph_uses_float32_features():
    parser = StreamingTelemetryParser()

    graph = parser.synthesize_mock_window()

    assert graph.x.dtype == torch.float32
    assert graph.edge_attr.dtype == torch.float32


def test_graph_contains_valid_node_indices():
    parser = StreamingTelemetryParser()

    num_nodes = 20
    graph = parser.synthesize_mock_window(
        num_nodes=num_nodes,
        num_edges=50
    )

    assert torch.all(graph.edge_index >= 0)
    assert torch.all(graph.edge_index < num_nodes)


def test_custom_graph_size():
    parser = StreamingTelemetryParser()

    graph = parser.synthesize_mock_window(
        num_nodes=50,
        num_edges=100
    )

    assert graph.x.shape == (50, 5)
    assert graph.edge_index.shape == (2, 100)
    assert graph.edge_attr.shape == (100, 5)