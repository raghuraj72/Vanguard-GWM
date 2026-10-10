from unittest.mock import patch, Mock

from src.runner import run_pipeline


@patch("src.runner.time.sleep")
@patch("src.runner.VanguardEngine")
@patch("src.runner.StreamingTelemetryParser")
def test_run_pipeline_completes(mock_parser_class, mock_engine_class, mock_sleep):
    mock_parser = Mock()
    mock_engine = Mock()

    mock_parser_class.return_value = mock_parser
    mock_engine_class.return_value = mock_engine

    mock_graph = Mock()
    mock_graph.x = Mock()
    mock_graph.edge_index = Mock()
    mock_graph.edge_attr = Mock()

    mock_parser.synthesize_mock_window.return_value = mock_graph

    mock_engine.process_graph_snapshot.return_value = {
        "target_ip": "192.168.1.105",
        "current_stage": "Reconnaissance",
        "peak_risk": 0.35,
        "risk_trajectory": [0.20, 0.25, 0.30, 0.35, 0.32],
        "sdn_quarantine_enforced": False,
    }

    run_pipeline(iterations=2, delta_t_sec=0)

    assert mock_parser_class.called
    assert mock_engine_class.called
    assert mock_engine.process_graph_snapshot.call_count == 2


@patch("src.runner.time.sleep")
@patch("src.runner.VanguardEngine")
@patch("src.runner.StreamingTelemetryParser")
def test_run_pipeline_uses_correct_graph_configuration(
    mock_parser_class,
    mock_engine_class,
    mock_sleep,
):
    mock_parser = Mock()
    mock_engine = Mock()

    mock_parser_class.return_value = mock_parser
    mock_engine_class.return_value = mock_engine

    mock_graph = Mock()
    mock_graph.x = Mock()
    mock_graph.edge_index = Mock()
    mock_graph.edge_attr = Mock()

    mock_parser.synthesize_mock_window.return_value = mock_graph

    mock_engine.process_graph_snapshot.return_value = {
        "target_ip": "192.168.1.105",
        "current_stage": "Reconnaissance",
        "peak_risk": 0.25,
        "risk_trajectory": [0.2] * 5,
        "sdn_quarantine_enforced": False,
    }

    run_pipeline(iterations=1, delta_t_sec=0)

    mock_parser.synthesize_mock_window.assert_called_once_with(
        num_nodes=12,
        num_edges=28,
    )


@patch("src.runner.time.sleep")
@patch("src.runner.VanguardEngine")
@patch("src.runner.StreamingTelemetryParser")
def test_run_pipeline_passes_graph_to_engine(
    mock_parser_class,
    mock_engine_class,
    mock_sleep,
):
    mock_parser = Mock()
    mock_engine = Mock()

    mock_parser_class.return_value = mock_parser
    mock_engine_class.return_value = mock_engine

    mock_x = Mock()
    mock_edge_index = Mock()
    mock_edge_attr = Mock()

    mock_graph = Mock()
    mock_graph.x = mock_x
    mock_graph.edge_index = mock_edge_index
    mock_graph.edge_attr = mock_edge_attr

    mock_parser.synthesize_mock_window.return_value = mock_graph

    mock_engine.process_graph_snapshot.return_value = {
        "target_ip": "192.168.1.105",
        "current_stage": "Initial Access",
        "peak_risk": 0.70,
        "risk_trajectory": [0.5, 0.6, 0.7, 0.65, 0.68],
        "sdn_quarantine_enforced": True,
    }

    run_pipeline(iterations=1, delta_t_sec=0)

    mock_engine.process_graph_snapshot.assert_called_once_with(
        mock_x,
        mock_edge_index,
        mock_edge_attr,
        "192.168.1.105",
    )


@patch("src.runner.time.sleep")
@patch("src.runner.VanguardEngine")
@patch("src.runner.StreamingTelemetryParser")
def test_run_pipeline_processes_exact_number_of_iterations(
    mock_parser_class,
    mock_engine_class,
    mock_sleep,
):
    mock_parser = Mock()
    mock_engine = Mock()

    mock_parser_class.return_value = mock_parser
    mock_engine_class.return_value = mock_engine

    mock_graph = Mock()
    mock_graph.x = Mock()
    mock_graph.edge_index = Mock()
    mock_graph.edge_attr = Mock()

    mock_parser.synthesize_mock_window.return_value = mock_graph

    mock_engine.process_graph_snapshot.return_value = {
        "target_ip": "192.168.1.105",
        "current_stage": "Benign Baseline",
        "peak_risk": 0.10,
        "risk_trajectory": [0.1] * 5,
        "sdn_quarantine_enforced": False,
    }

    iterations = 4

    run_pipeline(
        iterations=iterations,
        delta_t_sec=0,
    )

    assert (
        mock_parser.synthesize_mock_window.call_count
        == iterations
    )

    assert (
        mock_engine.process_graph_snapshot.call_count
        == iterations
    )


@patch("src.runner.time.sleep")
@patch("src.runner.VanguardEngine")
@patch("src.runner.StreamingTelemetryParser")
def test_run_pipeline_sleeps_between_iterations(
    mock_parser_class,
    mock_engine_class,
    mock_sleep,
):
    mock_parser = Mock()
    mock_engine = Mock()

    mock_parser_class.return_value = mock_parser
    mock_engine_class.return_value = mock_engine

    mock_graph = Mock()
    mock_graph.x = Mock()
    mock_graph.edge_index = Mock()
    mock_graph.edge_attr = Mock()

    mock_parser.synthesize_mock_window.return_value = mock_graph

    mock_engine.process_graph_snapshot.return_value = {
        "target_ip": "192.168.1.105",
        "current_stage": "Reconnaissance",
        "peak_risk": 0.30,
        "risk_trajectory": [0.2] * 5,
        "sdn_quarantine_enforced": False,
    }

    run_pipeline(
        iterations=3,
        delta_t_sec=1.5,
    )

    assert mock_sleep.call_count == 3

    mock_sleep.assert_called_with(1.5)