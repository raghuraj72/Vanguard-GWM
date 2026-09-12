from __future__ import annotations

import ipaddress
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, Literal

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import settings
from src.data_pipeline import StreamingTelemetryParser
from src.engine import VanguardEngine
from src.sdn_enforcer import SDNEnforcer


# ============================================================
# Operational configuration
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

MAX_TELEMETRY_ROWS = int(
    os.getenv("VANGUARD_MAX_TELEMETRY_ROWS", "5000")
)

MAX_RISK_POINTS = int(
    os.getenv("VANGUARD_MAX_RISK_POINTS", "128")
)

REQUEST_TIMEOUT_SEC = float(
    os.getenv("VANGUARD_REQUEST_TIMEOUT_SEC", "30")
)

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "VANGUARD_CORS_ORIGINS",
        "http://127.0.0.1:8501,http://localhost:8501",
    ).split(",")
    if origin.strip()
]


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("vanguard.api")


# ============================================================
# Application state
# ============================================================

parser = StreamingTelemetryParser(
    window_size_sec=2.0
)

engine_cache: dict[int, VanguardEngine] = {}

engine_lock = Lock()
inference_lock = Lock()


# ============================================================
# Policy configuration
# ============================================================

POLICY_REDUCTION = {
    "No Mitigation": 0.00,
    "Host Isolation": 0.72,
    "Block Suspicious Traffic": 0.55,
    "Restrict Lateral Movement": 0.48,
}

# Keep this exact declaration for the API contract.
LIVE_POLICIES = {"Host Isolation"}


# ============================================================
# Pydantic request models
# ============================================================

class StrictModel(BaseModel):
    """
    Request model with strict unknown-field rejection.
    """

    model_config = ConfigDict(
        extra="forbid"
    )


class TelemetryRequest(StrictModel):
    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=MAX_TELEMETRY_ROWS,
    )

    window_index: int = Field(
        default=0,
        ge=0,
    )

    mode: Literal[
        "csv",
        "synthetic",
    ] = "csv"


class ForecastRequest(StrictModel):
    """
    Forecast request.

    rows/mode/window_index are retained for compatibility
    with the existing dashboard/API contract.
    """

    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=MAX_TELEMETRY_ROWS,
    )

    window_index: int = Field(
        default=0,
        ge=0,
    )

    target_ip: str = Field(
        min_length=1,
        max_length=64,
    )

    horizon_k: int = Field(
        default=settings.forecast_horizon,
        ge=1,
        le=8,
    )

    mode: Literal[
        "csv",
        "synthetic",
    ] = "synthetic"

    @field_validator("target_ip")
    @classmethod
    def validate_target_ip(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        try:
            ipaddress.ip_address(value)
        except ValueError as exc:
            raise ValueError(
                "target_ip must be a valid IP address"
            ) from exc

        return value


class RiskRequest(StrictModel):
    peak_risk: float = Field(
        ge=0.0,
        le=1.0,
    )

    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class PolicySimulationRequest(StrictModel):
    policy: Literal[
        "No Mitigation",
        "Host Isolation",
        "Block Suspicious Traffic",
        "Restrict Lateral Movement",
    ]

    risk_trajectory: list[float] = Field(
        min_length=1,
        max_length=MAX_RISK_POINTS,
    )

    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class EnforcementRequest(StrictModel):
    target_ip: str = Field(
        min_length=1,
        max_length=64,
    )

    policy: Literal[
        "No Mitigation",
        "Host Isolation",
        "Block Suspicious Traffic",
        "Restrict Lateral Movement",
    ]

    current_max_risk: float = Field(
        ge=0.0,
        le=1.0,
    )

    projected_peak_risk: float = Field(
        ge=0.0,
        le=1.0,
    )

    approved: bool = False

    @field_validator("target_ip")
    @classmethod
    def validate_target_ip(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        try:
            ipaddress.ip_address(value)
        except ValueError as exc:
            raise ValueError(
                "target_ip must be a valid IP address"
            ) from exc

        return value


# ============================================================
# FastAPI application
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting %s",
        settings.app_name,
    )

    logger.info(
        "API configuration: "
        "max_telemetry_rows=%s "
        "max_risk_points=%s "
        "request_timeout_sec=%s",
        MAX_TELEMETRY_ROWS,
        MAX_RISK_POINTS,
        REQUEST_TIMEOUT_SEC,
    )

    yield

    logger.info(
        "Shutting down %s",
        settings.app_name,
    )


app = FastAPI(
    title="Vanguard-GWM API",
    version="0.1.0",
    description=(
        "Threat forecasting, risk analysis, policy simulation, "
        "and gated SDN enforcement API."
    ),
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=["*"],
)


# ============================================================
# Request tracing middleware
# ============================================================

@app.middleware("http")
async def request_tracing_middleware(
    request: Request,
    call_next,
):
    request_id = (
        request.headers.get("X-Request-ID")
        or str(uuid.uuid4())
    )

    request.state.request_id = request_id

    started = time.perf_counter()

    try:
        response = await call_next(request)

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "%s %s -> %s | %.2f ms | request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )

        return response

    except Exception:
        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        logger.exception(
            "Unhandled request failure | "
            "%s %s | %.2f ms | request_id=%s",
            request.method,
            request.url.path,
            elapsed_ms,
            request_id,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": (
                    "An internal server error occurred."
                ),
                "request_id": request_id,
            },
            headers={
                "X-Request-ID": request_id,
            },
        )


# ============================================================
# Validation exception handler
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    request_id = getattr(
        request.state,
        "request_id",
        str(uuid.uuid4()),
    )

    fields = []

    for error in exc.errors():
        location = ".".join(
            str(part)
            for part in error.get("loc", [])
        )

        fields.append(
            {
                "field": location,
                "message": error.get(
                    "msg",
                    "Invalid value",
                ),
                "type": error.get(
                    "type",
                    "validation_error",
                ),
            }
        )

    logger.warning(
        "Request validation failed | "
        "path=%s | request_id=%s",
        request.url.path,
        request_id,
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Request validation failed.",
            "request_id": request_id,
            "fields": fields,
        },
        headers={
            "X-Request-ID": request_id,
        },
    )


# ============================================================
# Engine management
# ============================================================

def _get_engine(
    horizon_k: int,
) -> VanguardEngine:
    with engine_lock:
        if horizon_k not in engine_cache:
            engine_cache[horizon_k] = VanguardEngine(
                in_node_features=5,
                in_edge_features=5,
                latent_dim=settings.model_latent_dim,
                horizon_k=horizon_k,
                risk_threshold=settings.defensive_threshold,
            )

        return engine_cache[horizon_k]


# ============================================================
# Graph conversion
# ============================================================

def _graph_from_rows(
    rows: list[dict[str, Any]],
    window_index: int = 0,
):
    """
    Convert API telemetry rows into a Vanguard graph.

    Pipeline:

        API rows
          ↓
        pandas DataFrame
          ↓
        normalize_dataframe()
          ↓
        dataframe_to_graph()
    """

    if len(rows) > MAX_TELEMETRY_ROWS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Too many telemetry rows. "
                f"Maximum allowed is "
                f"{MAX_TELEMETRY_ROWS}."
            ),
        )

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="Telemetry rows cannot be empty.",
        )

    try:
        # ----------------------------------------------------
        # Convert request rows to DataFrame.
        # ----------------------------------------------------

        dataframe = pd.DataFrame(rows)

        # ----------------------------------------------------
        # Normalize using the actual data pipeline API.
        #
        # StreamingTelemetryParser does NOT expose
        # normalize_rows().
        # ----------------------------------------------------

        normalized = parser.normalize_dataframe(
            dataframe
        )

        # ----------------------------------------------------
        # Build the temporal graph.
        # ----------------------------------------------------

        graph = parser.dataframe_to_graph(
            normalized,
            window_index=window_index,
        )

        return graph

    except HTTPException:
        raise

    except (
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        logger.warning(
            "Telemetry conversion failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=422,
            detail=(
                "Telemetry data could not be "
                "converted into a graph."
            ),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected telemetry graph "
            "conversion failure."
        )

        raise HTTPException(
            status_code=422,
            detail=(
                "Telemetry graph conversion failed."
            ),
        ) from exc


def _graph_for_forecast(
    target_ip: str,
    rows: list[dict[str, Any]] | None = None,
    window_index: int = 0,
):
    """
    Use supplied telemetry when present.

    Otherwise use the existing synthetic forecast
    behavior.
    """

    _ = target_ip

    if rows:
        return _graph_from_rows(
            rows,
            window_index=window_index,
        )

    return parser.synthesize_mock_window(
        num_nodes=12,
        num_edges=28,
    )


# ============================================================
# Forecast helpers
# ============================================================

def _prime_history(
    engine: VanguardEngine,
    graph,
) -> None:
    engine.process_graph_snapshot(graph)


def _forecast_from_request(
    request: ForecastRequest,
):
    engine = _get_engine(
        request.horizon_k
    )

    graph = _graph_for_forecast(
        request.target_ip,
        request.rows,
        request.window_index,
    )

    with inference_lock:
        engine.reset_history()

        _prime_history(
            engine,
            graph,
        )

        result = engine.process_graph_snapshot(
            graph
        )

    return result


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health(
    request: Request,
):
    return {
        "status": "ok",
        "service": settings.app_name,
        "request_id": request.state.request_id,
    }


# ============================================================
# Readiness
# ============================================================

@app.get("/ready")
def ready(
    request: Request,
):
    dependencies = {
        "torch": torch is not None,
        "data_pipeline": parser is not None,
        "engine_class": VanguardEngine is not None,
        "sdn_enforcer": SDNEnforcer is not None,
    }

    ready_state = all(
        dependencies.values()
    )

    return {
        "status": (
            "ready"
            if ready_state
            else "not_ready"
        ),
        "service": settings.app_name,
        "dependencies": dependencies,
        "request_id": request.state.request_id,
    }


# ============================================================
# Status
# ============================================================

@app.get("/status")
def status(
    request: Request,
):
    default_engine = _get_engine(
        settings.forecast_horizon
    )

    return {
        "status": "ok",
        "service": settings.app_name,
        "model": {
            "latent_dim": settings.model_latent_dim,
            "forecast_horizon": settings.forecast_horizon,
            "defensive_threshold": settings.defensive_threshold,
        },
        "engine": {
            "history_length": (
                default_engine.get_history_length()
            ),
        },
        "enforcement": {
            "automatic_enforcement": False,
            "live_policies": sorted(
                LIVE_POLICIES
            ),
        },
        "limits": {
            "max_telemetry_rows": MAX_TELEMETRY_ROWS,
            "max_risk_points": MAX_RISK_POINTS,
            "request_timeout_sec": REQUEST_TIMEOUT_SEC,
        },
        "request_id": request.state.request_id,
    }


# ============================================================
# SDN status
# ============================================================

@app.get("/sdn/status")
def sdn_status(
    request: Request,
):
    enforcer = SDNEnforcer()

    try:
        controller_status = (
            enforcer.check_controller_status()
        )

        return {
            "status": controller_status,
            "simulation_mode": (
                enforcer.simulation_mode
            ),
            "request_id": request.state.request_id,
        }

    finally:
        enforcer.close()


# ============================================================
# Telemetry
# ============================================================

@app.post("/telemetry")
def telemetry(
    payload: TelemetryRequest,
    request: Request,
):
    if payload.mode == "synthetic":
        raise HTTPException(
            status_code=400,
            detail=(
                "Synthetic telemetry is not accepted "
                "by the telemetry ingestion endpoint."
            ),
        )

    if not payload.rows:
        raise HTTPException(
            status_code=400,
            detail="Telemetry rows cannot be empty.",
        )

    graph = _graph_from_rows(
        payload.rows,
        window_index=payload.window_index,
    )

    return {
        "status": "accepted",
        "window_index": payload.window_index,
        "rows": len(payload.rows),
        "nodes": int(
            graph.x.shape[0]
        ),
        "edges": int(
            graph.edge_index.shape[1]
        ),
        "request_id": request.state.request_id,
    }


# ============================================================
# Forecast
# ============================================================

@app.post("/forecast")
def forecast(
    payload: ForecastRequest,
    request: Request,
):
    started = time.perf_counter()

    result = _forecast_from_request(
        payload
    )

    elapsed_ms = (
        time.perf_counter() - started
    ) * 1000.0

    if elapsed_ms > REQUEST_TIMEOUT_SEC * 1000.0:
        logger.warning(
            "Forecast exceeded latency budget | "
            "%.2f ms > %.2f ms | request_id=%s",
            elapsed_ms,
            REQUEST_TIMEOUT_SEC * 1000.0,
            request.state.request_id,
        )

    # Preserve the established API response contract
    # while exposing request tracing and measured latency.
    if isinstance(result, dict):
        response = dict(result)
    else:
        response = {
            "result": result,
        }

    response.setdefault(
        "target_ip",
        payload.target_ip,
    )

    response.setdefault(
        "horizon_k",
        payload.horizon_k,
    )

    response.setdefault(
        "latency_ms",
        round(
            elapsed_ms,
            3,
        ),
    )

    response["request_id"] = (
        request.state.request_id
    )

    return response


# ============================================================
# Risk
# ============================================================

@app.post("/risk")
def risk(
    payload: RiskRequest,
    request: Request,
):
    threshold = (
        settings.defensive_threshold
        if payload.threshold is None
        else payload.threshold
    )

    crossed = (
        payload.peak_risk >= threshold
    )

    return {
        "peak_risk": payload.peak_risk,
        "threshold": threshold,
        "threshold_crossed": crossed,
        "request_id": request.state.request_id,
    }


# ============================================================
# Combined analysis
# ============================================================

@app.post("/analyze")
def analyze(
    payload: ForecastRequest,
    request: Request,
):
    started = time.perf_counter()

    result = _forecast_from_request(
        payload
    )

    elapsed_ms = (
        time.perf_counter() - started
    ) * 1000.0

    threshold = (
        settings.defensive_threshold
    )

    peak_risk = float(
        result.get(
            "peak_risk",
            result.get(
                "max_risk",
                0.0,
            ),
        )
    )

    threshold_crossed = (
        peak_risk >= threshold
    )

    return {
        "target_ip": payload.target_ip,
        "horizon_k": payload.horizon_k,
        "peak_risk": peak_risk,
        "threshold": threshold,
        "threshold_crossed": threshold_crossed,
        "forecast": result,
        "latency_ms": round(
            elapsed_ms,
            3,
        ),
        "request_id": request.state.request_id,
    }


# ============================================================
# Policy simulation
# ============================================================

@app.post("/policy/simulate")
def policy_simulate(
    payload: PolicySimulationRequest,
    request: Request,
):
    threshold = (
        settings.defensive_threshold
        if payload.threshold is None
        else payload.threshold
    )

    reduction = POLICY_REDUCTION[
        payload.policy
    ]

    projected_trajectory = [
        max(
            0.02,
            min(
                1.0,
                risk_value
                * (1.0 - reduction),
            ),
        )
        for risk_value in payload.risk_trajectory
    ]

    projected_peak = max(
        projected_trajectory
    )

    return {
        "policy": payload.policy,
        "reduction_factor": reduction,
        "input_peak_risk": max(
            payload.risk_trajectory
        ),
        "projected_trajectory": (
            projected_trajectory
        ),
        "projected_peak_risk": (
            projected_peak
        ),
        "threshold": threshold,
        "threshold_crossed": (
            projected_peak >= threshold
        ),
        "simulation_only": True,
        "request_id": request.state.request_id,
    }


# ============================================================
# Explicit SDN enforcement
# ============================================================

@app.post("/enforcement/request")
def request_enforcement(
    payload: EnforcementRequest,
    request: Request,
):
    """
    Explicitly request SDN enforcement.

    Safety gates:

    1. No Mitigation -> reject
    2. Non-live policy -> reject
    3. Current risk must meet threshold
    4. Projected risk must fall below threshold
    5. Explicit operator approval required
    6. SDN controller must be connected
    7. Only then submit the SDN rule
    """

    threshold = (
        settings.defensive_threshold
    )

    # --------------------------------------------------------
    # Gate 1
    # --------------------------------------------------------

    if payload.policy == "No Mitigation":
        raise HTTPException(
            status_code=409,
            detail=(
                "No Mitigation cannot trigger "
                "network enforcement."
            ),
        )

    # --------------------------------------------------------
    # Gate 2
    # --------------------------------------------------------

    if payload.policy not in LIVE_POLICIES:
        raise HTTPException(
            status_code=409,
            detail=(
                "This policy is simulation-only "
                "and cannot trigger live enforcement."
            ),
        )

    # --------------------------------------------------------
    # Gate 3
    # --------------------------------------------------------

    if payload.current_max_risk < threshold:
        raise HTTPException(
            status_code=409,
            detail=(
                "Current risk is below the defensive "
                "enforcement threshold."
            ),
        )

    # --------------------------------------------------------
    # Gate 4
    # --------------------------------------------------------

    if payload.projected_peak_risk >= threshold:
        raise HTTPException(
            status_code=409,
            detail=(
                "Projected residual risk remains at "
                "or above the defensive threshold."
            ),
        )

    # --------------------------------------------------------
    # Gate 5
    # --------------------------------------------------------

    if not payload.approved:
        raise HTTPException(
            status_code=403,
            detail=(
                "Explicit operator approval is required "
                "before live enforcement."
            ),
        )

    # --------------------------------------------------------
    # Gate 6
    # --------------------------------------------------------

    enforcer = SDNEnforcer()

    try:
        controller_status = (
            enforcer.check_controller_status()
        )

        if controller_status != "Connected":
            raise HTTPException(
                status_code=503,
                detail=(
                    "SDN controller is unavailable. "
                    "No enforcement action was performed."
                ),
            )

        # ----------------------------------------------------
        # Gate 7
        # ----------------------------------------------------

        success = (
            enforcer.push_quarantine_rule(
                target_ip=payload.target_ip
            )
        )

        if not success:
            raise HTTPException(
                status_code=502,
                detail=(
                    "SDN enforcement request failed."
                ),
            )

        logger.warning(
            "LIVE SDN ENFORCEMENT ACCEPTED | "
            "target_ip=%s | policy=%s | "
            "current_risk=%.4f | "
            "projected_risk=%.4f | "
            "request_id=%s",
            payload.target_ip,
            payload.policy,
            payload.current_max_risk,
            payload.projected_peak_risk,
            request.state.request_id,
        )

        return {
            "status": "enforced",
            "target_ip": payload.target_ip,
            "policy": payload.policy,
            "current_max_risk": (
                payload.current_max_risk
            ),
            "projected_peak_risk": (
                payload.projected_peak_risk
            ),
            "threshold": threshold,
            "controller_status": (
                controller_status
            ),
            "automatic_enforcement": False,
            "operator_approved": True,
            "request_id": (
                request.state.request_id
            ),
        }

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning(
            "Invalid enforcement request: %s | "
            "request_id=%s",
            exc,
            request.state.request_id,
        )

        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid enforcement parameters."
            ),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected enforcement failure | "
            "request_id=%s",
            request.state.request_id,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "SDN enforcement could not be completed."
            ),
        ) from exc

    finally:
        enforcer.close()


# ============================================================
# Application entry point
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )