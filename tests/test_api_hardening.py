from pathlib import Path
import sys

import pytest


# ------------------------------------------------------------
# Make project root importable when pytest is launched
# from the tests directory.
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _runtime_api():
    """
    Import the runtime API only when the project's ML
    dependencies are available.
    """

    pytest.importorskip(
        "torch_geometric",
        reason=(
            "Runtime API tests require torch-geometric."
        ),
    )

    from fastapi.testclient import TestClient
    import api

    return api, TestClient(api.app)


# ============================================================
# Request tracing
# ============================================================

def test_health_returns_request_id():
    api, client = _runtime_api()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "X-Request-ID" in response.headers

    request_id = response.headers["X-Request-ID"]

    assert request_id
    assert response.json()["request_id"] == request_id


def test_custom_request_id_is_preserved():
    api, client = _runtime_api()

    custom_id = "step-3-4-test-001"

    response = client.get(
        "/health",
        headers={
            "X-Request-ID": custom_id,
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_id
    assert response.json()["request_id"] == custom_id


# ============================================================
# Readiness
# ============================================================

def test_ready_endpoint():
    api, client = _runtime_api()

    response = client.get("/ready")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] in {
        "ready",
        "not_ready",
    }

    assert "dependencies" in body

    assert "torch" in body["dependencies"]
    assert "data_pipeline" in body["dependencies"]
    assert "engine_class" in body["dependencies"]
    assert "sdn_enforcer" in body["dependencies"]


# ============================================================
# Input validation
# ============================================================

def test_invalid_enforcement_ip_is_rejected():
    api, client = _runtime_api()

    response = client.post(
        "/enforcement/request",
        json={
            "target_ip": "not-an-ip",
            "policy": "Host Isolation",
            "current_max_risk": 0.9,
            "projected_peak_risk": 0.2,
            "approved": True,
        },
    )

    assert response.status_code == 422

    body = response.json()

    assert body["error"] == "validation_error"
    assert "fields" in body


def test_unknown_request_fields_are_rejected():
    api, client = _runtime_api()

    response = client.post(
        "/risk",
        json={
            "peak_risk": 0.8,
            "unexpected_field": "should-fail",
        },
    )

    assert response.status_code == 422

    body = response.json()

    assert body["error"] == "validation_error"


def test_oversized_risk_trajectory_is_rejected():
    api, client = _runtime_api()

    response = client.post(
        "/policy/simulate",
        json={
            "policy": "Host Isolation",
            "risk_trajectory": [0.5] * 129,
        },
    )

    assert response.status_code == 422

    body = response.json()

    assert body["error"] == "validation_error"


# ============================================================
# CORS
# ============================================================

def test_cors_headers_are_present_for_dashboard_origin():
    api, client = _runtime_api()

    response = client.options(
        "/health",
        headers={
            "Origin": "http://127.0.0.1:8501",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code in {
        200,
        204,
    }

    assert (
        response.headers.get(
            "access-control-allow-origin"
        )
        == "http://127.0.0.1:8501"
    )


# ============================================================
# Safety gates
# ============================================================

def test_enforcement_requires_operator_approval():
    api, client = _runtime_api()

    response = client.post(
        "/enforcement/request",
        json={
            "target_ip": "192.168.1.105",
            "policy": "Host Isolation",
            "current_max_risk": 0.9,
            "projected_peak_risk": 0.2,
            "approved": False,
        },
    )

    assert response.status_code == 403


def test_non_live_policy_cannot_enforce():
    api, client = _runtime_api()

    response = client.post(
        "/enforcement/request",
        json={
            "target_ip": "192.168.1.105",
            "policy": "Block Suspicious Traffic",
            "current_max_risk": 0.9,
            "projected_peak_risk": 0.2,
            "approved": True,
        },
    )

    assert response.status_code == 409


def test_low_current_risk_cannot_enforce():
    api, client = _runtime_api()

    response = client.post(
        "/enforcement/request",
        json={
            "target_ip": "192.168.1.105",
            "policy": "Host Isolation",
            "current_max_risk": 0.3,
            "projected_peak_risk": 0.1,
            "approved": True,
        },
    )

    assert response.status_code == 409


def test_high_projected_risk_cannot_enforce():
    api, client = _runtime_api()

    response = client.post(
        "/enforcement/request",
        json={
            "target_ip": "192.168.1.105",
            "policy": "Host Isolation",
            "current_max_risk": 0.9,
            "projected_peak_risk": 0.8,
            "approved": True,
        },
    )

    assert response.status_code == 409