"""Step 3.5 — Vanguard-GWM end-to-end API smoke tests."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Runtime API helper
# ---------------------------------------------------------------------------

def _runtime_api():
    """Load the real FastAPI application."""
    pytest.importorskip(
        "torch_geometric",
        reason="E2E smoke tests require torch-geometric.",
    )

    from fastapi.testclient import TestClient

    import api

    return api, TestClient(api.app)


# ---------------------------------------------------------------------------
# End-to-end SOC smoke test
# ---------------------------------------------------------------------------

def test_e2e_soc_smoke_flow(monkeypatch):
    """Exercise the complete SOC decision path without real SDN enforcement."""

    api, client = _runtime_api()

    request_id = "step-3-5-e2e-001"

    # ------------------------------------------------------------------
    # 1. Health
    # ------------------------------------------------------------------

    response = client.get(
        "/health",
        headers={"X-Request-ID": request_id},
    )

    assert response.status_code == 200

    health = response.json()

    assert health["status"] == "ok"
    assert health["request_id"] == request_id
    assert response.headers["X-Request-ID"] == request_id

    # ------------------------------------------------------------------
    # 2. Readiness
    # ------------------------------------------------------------------

    response = client.get(
        "/ready",
        headers={"X-Request-ID": request_id},
    )

    assert response.status_code == 200

    ready = response.json()

    assert ready["status"] in {"ready", "not_ready"}
    assert "dependencies" in ready

    dependencies = ready["dependencies"]

    assert "torch" in dependencies
    assert "data_pipeline" in dependencies
    assert "engine_class" in dependencies
    assert "sdn_enforcer" in dependencies

    # ------------------------------------------------------------------
    # 3. Telemetry ingestion
    #
    # IMPORTANT:
    # /telemetry deliberately rejects synthetic telemetry.
    # Use CSV mode to exercise the accepted ingestion path.
    # ------------------------------------------------------------------

    telemetry_payload = {
        "rows": [
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "src_ip": "192.168.1.100",
                "dst_ip": "192.168.1.105",
                "src_port": 49152,
                "dst_port": 443,
                "protocol": "TCP",
                "bytes": 1200,
                "packets": 12,
            }
        ],
        "window_index": 0,
        "mode": "csv",
    }

    response = client.post(
        "/telemetry",
        json=telemetry_payload,
        headers={"X-Request-ID": request_id},
    )

    assert response.status_code == 200, response.text

    telemetry = response.json()

    assert telemetry["request_id"] == request_id

    # ------------------------------------------------------------------
    # 4. Forecast
    #
    # Stub inference so this smoke test verifies API integration rather
    # than neural-network quality.
    #
    # The forecast endpoint retains its existing synthetic-compatible
    # request contract.
    # ------------------------------------------------------------------

    fake_forecast = {
        "target_ip": "192.168.1.105",
        "current_stage": "Lateral Movement",
        "phase_predictions": [
            "Lateral Movement",
            "Lateral Movement",
            "Lateral Movement",
            "Lateral Movement",
            "Lateral Movement",
        ],
        "peak_risk": 0.75,
        "risk_trajectory": [
            0.50,
            0.58,
            0.64,
            0.70,
            0.75,
        ],
        "defensive_threshold": 0.60,
        "threshold_crossed": True,
        "sdn_quarantine_enforced": False,
        "latency_ms": 1.0,
        "horizon_k": 5,
        "graph": {
            "nodes": 12,
            "edges": 28,
            "telemetry_rows": 1,
        },
    }

    monkeypatch.setattr(
        api,
        "_forecast_from_request",
        lambda request: fake_forecast,
    )

    response = client.post(
        "/forecast",
        json={
            "rows": telemetry_payload["rows"],
            "window_index": 0,
            "target_ip": "192.168.1.105",
            "horizon_k": 5,
            "mode": "synthetic",
        },
        headers={"X-Request-ID": request_id},
    )

    assert response.status_code == 200, response.text

    forecast = response.json()

    assert forecast["target_ip"] == "192.168.1.105"
    assert forecast["peak_risk"] == 0.75
    assert forecast["threshold_crossed"] is True
    assert forecast["sdn_quarantine_enforced"] is False
    assert forecast["request_id"] == request_id

    # ------------------------------------------------------------------
    # 5. Risk assessment
    # ------------------------------------------------------------------

    response = client.post(
        "/risk",
        json={
            "peak_risk": forecast["peak_risk"],
            "threshold": 0.60,
        },
        headers={"X-Request-ID": request_id},
    )

    assert response.status_code == 200, response.text

    risk = response.json()

    assert risk["threshold_crossed"] is True
    assert risk["peak_risk"] == 0.75
    assert risk["threshold"] == 0.60
    assert risk["request_id"] == request_id

    # ------------------------------------------------------------------
    # 6. Counterfactual policy simulation
    # ------------------------------------------------------------------

    response = client.post(
        "/policy/simulate",
        json={
            "policy": "Host Isolation",
            "risk_trajectory": forecast["risk_trajectory"],
            "threshold": 0.60,
        },
        headers={"X-Request-ID": request_id},
    )

    assert response.status_code == 200, response.text

    simulation = response.json()

    assert simulation["policy"] == "Host Isolation"
    assert simulation["simulation_only"] is True
    assert simulation["projected_peak_risk"] < 0.60
    assert simulation["request_id"] == request_id

    # ------------------------------------------------------------------
    # 7. SDN controller status
    # ------------------------------------------------------------------

    response = client.get(
        "/sdn/status",
        headers={"X-Request-ID": request_id},
    )

    assert response.status_code == 200, response.text

    sdn_status = response.json()

    assert "status" in sdn_status
    assert sdn_status["request_id"] == request_id

    # ------------------------------------------------------------------
    # 8. Verify unsafe policy cannot be enforced
    # ------------------------------------------------------------------

    response = client.post(
        "/enforcement/request",
        json={
            "target_ip": "192.168.1.105",
            "policy": "Block Suspicious Traffic",
            "current_max_risk": 0.75,
            "projected_peak_risk": 0.20,
            "approved": True,
        },
        headers={"X-Request-ID": request_id},
    )

    assert response.status_code == 409

    # ------------------------------------------------------------------
    # 9. Verify operator approval is mandatory
    # ------------------------------------------------------------------

    response = client.post(
        "/enforcement/request",
        json={
            "target_ip": "192.168.1.105",
            "policy": "Host Isolation",
            "current_max_risk": 0.75,
            "projected_peak_risk": 0.20,
            "approved": False,
        },
        headers={"X-Request-ID": request_id},
    )

    assert response.status_code == 403

    # ------------------------------------------------------------------
    # 10. Approved Host Isolation reaches the SDN boundary.
    #
    # Mock both controller connectivity and enforcement so this test
    # can NEVER modify a real network.
    # ------------------------------------------------------------------

    monkeypatch.setattr(
        api.SDNEnforcer,
        "check_controller_status",
        lambda self: "Connected",
    )

    monkeypatch.setattr(
        api.SDNEnforcer,
        "push_quarantine_rule",
        lambda self, target_ip: True,
    )

    response = client.post(
        "/enforcement/request",
        json={
            "target_ip": "192.168.1.105",
            "policy": "Host Isolation",
            "current_max_risk": 0.75,
            "projected_peak_risk": 0.20,
            "approved": True,
        },
        headers={"X-Request-ID": request_id},
    )

    assert response.status_code == 200, response.text

    enforcement = response.json()

    assert enforcement["request_id"] == request_id
    assert enforcement["policy"] == "Host Isolation"
    assert enforcement["target_ip"] == "192.168.1.105"