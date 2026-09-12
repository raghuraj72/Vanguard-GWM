"""Step 3.3/3.4 — Vanguard-GWM API contract/integration tests.

Static tests run in lightweight environments.
Runtime HTTP tests run when the project's ML dependencies
(notably torch-geometric) are installed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------
# tests/test_api_contract.py
#        ↓ parents[1]
# Vanguard-GWM/
#
# This makes the root-level api.py importable when pytest is executed.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API = ROOT / "api.py"
APP = ROOT / "app.py"


# ---------------------------------------------------------------------------
# Expected API contract
# ---------------------------------------------------------------------------

EXPECTED_ROUTES = {
    ("GET", "/health"),
    ("GET", "/status"),
    ("GET", "/ready"),
    ("GET", "/sdn/status"),
    ("POST", "/telemetry"),
    ("POST", "/forecast"),
    ("POST", "/risk"),
    ("POST", "/analyze"),
    ("POST", "/policy/simulate"),
    ("POST", "/enforcement/request"),
}


POLICIES = {
    "No Mitigation",
    "Host Isolation",
    "Block Suspicious Traffic",
    "Restrict Lateral Movement",
}


# ---------------------------------------------------------------------------
# Static analysis helpers
# ---------------------------------------------------------------------------

def _tree(path: Path) -> ast.Module:
    """Parse a Python source file into an AST."""
    assert path.exists(), f"Required file does not exist: {path}"

    return ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )


def _routes(tree: ast.Module):
    """Extract FastAPI route decorators from the AST."""
    routes = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue

            if not isinstance(dec.func, ast.Attribute):
                continue

            if not isinstance(dec.func.value, ast.Name):
                continue

            if dec.func.value.id != "app":
                continue

            method = dec.func.attr.upper()

            if method not in {
                "GET",
                "POST",
                "PUT",
                "PATCH",
                "DELETE",
            }:
                continue

            if not dec.args:
                continue

            if isinstance(dec.args[0], ast.Constant):
                routes.add((method, dec.args[0].value))

    return routes


def _model_fields(tree: ast.Module):
    """Extract Pydantic request-model field names."""
    fields = {}

    expected_models = {
        "TelemetryRequest",
        "ForecastRequest",
        "RiskRequest",
        "PolicySimulationRequest",
        "EnforcementRequest",
    }

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        if node.name not in expected_models:
            continue

        names = set()

        for stmt in node.body:
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
            ):
                names.add(stmt.target.id)

        fields[node.name] = names

    return fields


# ---------------------------------------------------------------------------
# Static contract tests
# ---------------------------------------------------------------------------

def test_api_syntax_and_route_contract():
    """Verify that api.py exposes the expected API routes."""
    tree = _tree(API)

    actual_routes = _routes(tree)

    assert actual_routes == EXPECTED_ROUTES, (
        f"API route contract mismatch.\n"
        f"Missing: {EXPECTED_ROUTES - actual_routes}\n"
        f"Unexpected: {actual_routes - EXPECTED_ROUTES}"
    )


def test_request_models_expose_expected_fields():
    """Verify required request-model fields."""
    fields = _model_fields(_tree(API))

    assert "TelemetryRequest" in fields
    assert fields["TelemetryRequest"] >= {
        "rows",
        "window_index",
        "mode",
    }

    assert "ForecastRequest" in fields
    assert fields["ForecastRequest"] >= {
        "target_ip",
        "horizon_k",
    }

    assert "RiskRequest" in fields
    assert fields["RiskRequest"] >= {
        "peak_risk",
        "threshold",
    }

    assert "PolicySimulationRequest" in fields
    assert fields["PolicySimulationRequest"] >= {
        "policy",
        "risk_trajectory",
        "threshold",
    }

    assert "EnforcementRequest" in fields
    assert fields["EnforcementRequest"] >= {
        "target_ip",
        "policy",
        "current_max_risk",
        "projected_peak_risk",
        "approved",
    }


def test_policy_contract_and_live_enforcement_boundary():
    """Verify policy names and the explicit live-enforcement boundary."""
    source = API.read_text(encoding="utf-8")

    for policy in POLICIES:
        assert policy in source

    assert 'LIVE_POLICIES = {"Host Isolation"}' in source
    assert '"automatic_enforcement": False' in source
    assert '"simulation_only": True' in source


def test_enforcement_gate_is_explicit_and_ordered():
    """Verify that enforcement safety checks occur before SDN action."""
    source = API.read_text(encoding="utf-8")

    marker = "def request_enforcement"

    assert marker in source, (
        "Could not find request_enforcement endpoint in api.py"
    )

    gate = source[source.index(marker):]

    # api.py uses `payload` for the Pydantic request body.
    #
    # Safety gate order must remain:
    #
    # 1. No Mitigation
    # 2. Policy must be live
    # 3. Current risk must be above threshold
    # 4. Projected residual risk must be below threshold
    # 5. Explicit operator approval
    # 6. SDN controller must be connected
    # 7. Only then may push_quarantine_rule() execute
    checks = [
        'payload.policy == "No Mitigation"',
        "payload.policy not in LIVE_POLICIES",
        "payload.current_max_risk < threshold",
        "payload.projected_peak_risk >= threshold",
        "if not payload.approved",
        'controller_status != "Connected"',
        "push_quarantine_rule",
    ]

    positions = []

    for check in checks:
        assert check in gate, (
            f"Expected enforcement safety check missing: {check}"
        )

        positions.append(gate.index(check))

    assert positions == sorted(positions), (
        "Enforcement safety checks are not in the expected order."
    )


def test_dashboard_uses_backend_for_step_3_contracts():
    """Verify that Streamlit uses the FastAPI backend."""
    source = APP.read_text(encoding="utf-8")

    for endpoint in (
        "/forecast",
        "/policy/simulate",
        "/enforcement/request",
        "/health",
        "/sdn/status",
    ):
        assert endpoint in source

    assert "VANGUARD_API_URL" in source
    assert "FastAPI backend inference" in source


# ---------------------------------------------------------------------------
# Runtime HTTP tests
# ---------------------------------------------------------------------------

def _runtime_api():
    """Load the real FastAPI application for HTTP integration tests.

    torch-geometric is required because api.py imports the Vanguard
    inference stack.
    """
    pytest.importorskip(
        "torch_geometric",
        reason=(
            "Runtime API tests require torch-geometric. "
            "Static contract tests can run without it."
        ),
    )

    from fastapi.testclient import TestClient

    # ROOT has already been inserted into sys.path above.
    import api

    return api, TestClient(api.app)


def test_http_health_and_status():
    """Verify /health and /status response contracts."""
    api, client = _runtime_api()

    health = client.get("/health")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    status = client.get("/status")

    assert status.status_code == 200

    body = status.json()

    assert body["enforcement"]["automatic_enforcement"] is False
    assert "Host Isolation" in body["enforcement"]["live_policies"]


def test_http_risk_contract_and_boundary():
    """Verify risk threshold behavior immediately below and at 0.60."""
    _, client = _runtime_api()

    below = client.post(
        "/risk",
        json={
            "peak_risk": 0.5999,
        },
    )

    assert below.status_code == 200
    assert below.json()["threshold_crossed"] is False

    at = client.post(
        "/risk",
        json={
            "peak_risk": 0.60,
        },
    )

    assert at.status_code == 200
    assert at.json()["threshold_crossed"] is True


def test_http_policy_simulation_contract():
    """Verify Host Isolation counterfactual simulation."""
    _, client = _runtime_api()

    response = client.post(
        "/policy/simulate",
        json={
            "policy": "Host Isolation",
            "risk_trajectory": [
                0.475,
                0.50,
                0.52,
            ],
            "threshold": 0.60,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["policy"] == "Host Isolation"
    assert body["simulation_only"] is True
    assert body["projected_peak_risk"] < 0.60


def test_http_enforcement_blocks_unsafe_requests():
    """Verify the enforcement safety gate rejects unsafe actions."""
    _, client = _runtime_api()

    cases = [
        (
            {
                "target_ip": "192.168.1.105",
                "policy": "No Mitigation",
                "current_max_risk": 0.80,
                "projected_peak_risk": 0.20,
                "approved": True,
            },
            409,
        ),
        (
            {
                "target_ip": "192.168.1.105",
                "policy": "Block Suspicious Traffic",
                "current_max_risk": 0.80,
                "projected_peak_risk": 0.20,
                "approved": True,
            },
            409,
        ),
        (
            {
                "target_ip": "192.168.1.105",
                "policy": "Host Isolation",
                "current_max_risk": 0.50,
                "projected_peak_risk": 0.20,
                "approved": True,
            },
            409,
        ),
        (
            {
                "target_ip": "192.168.1.105",
                "policy": "Host Isolation",
                "current_max_risk": 0.80,
                "projected_peak_risk": 0.60,
                "approved": True,
            },
            409,
        ),
        (
            {
                "target_ip": "192.168.1.105",
                "policy": "Host Isolation",
                "current_max_risk": 0.80,
                "projected_peak_risk": 0.20,
                "approved": False,
            },
            403,
        ),
    ]

    for payload, expected_status in cases:
        response = client.post(
            "/enforcement/request",
            json=payload,
        )

        assert response.status_code == expected_status, response.text


def test_http_forecast_contract_with_stubbed_inference(monkeypatch):
    """Verify /forecast HTTP contract without running neural inference."""
    api, client = _runtime_api()

    fake_result = {
        "target_ip": "192.168.1.105",
        "current_stage": "Lateral Movement",
        "phase_predictions": [
            "Lateral Movement"
        ] * 5,
        "peak_risk": 0.475,
        "risk_trajectory": [
            0.40,
            0.42,
            0.45,
            0.47,
            0.475,
        ],
        "defensive_threshold": 0.60,
        "threshold_crossed": False,
        "sdn_quarantine_enforced": False,
        "latency_ms": 1.23,
        "horizon_k": 5,
        "graph": {
            "nodes": 12,
            "edges": 28,
            "telemetry_rows": 0,
        },
    }

    monkeypatch.setattr(
        api,
        "_forecast_from_request",
        lambda request: fake_result,
    )

    response = client.post(
        "/forecast",
        json={
            "rows": [],
            "window_index": 0,
            "target_ip": "192.168.1.105",
            "horizon_k": 5,
            "mode": "synthetic",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["target_ip"] == "192.168.1.105"
    assert body["horizon_k"] == 5
    assert body["graph"]["nodes"] == 12
    assert body["graph"]["edges"] == 28