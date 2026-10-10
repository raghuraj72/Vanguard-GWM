"""
Vanguard-GWM application configuration.

Centralizes environment variables and application settings so that
the dashboard, API, model engine, and future services use the same
configuration source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

# Load variables from .env into the process environment.
# Existing environment variables take precedence.
load_dotenv()


# ============================================================
# ENVIRONMENT VARIABLE HELPERS
# ============================================================

def _get_bool(
    name: str,
    default: bool = False,
) -> bool:
    """Read a boolean environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_int(
    name: str,
    default: int,
) -> int:
    """Read an integer environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name!r} must be an integer. "
            f"Received: {value!r}"
        ) from exc


def _get_float(
    name: str,
    default: float,
) -> float:
    """Read a floating-point environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name!r} must be a number. "
            f"Received: {value!r}"
        ) from exc


# ============================================================
# SETTINGS
# ============================================================

@dataclass(frozen=True)
class Settings:
    """
    Application-wide Vanguard-GWM configuration.

    The object is immutable so configuration cannot accidentally
    change during application runtime.
    """

    # --------------------------------------------------------
    # Application
    # --------------------------------------------------------

    app_name: str
    app_env: str
    debug: bool

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    database_url: str

    # --------------------------------------------------------
    # Vanguard-GWM Model
    # --------------------------------------------------------

    model_latent_dim: int
    forecast_horizon: int
    defensive_threshold: float

    # --------------------------------------------------------
    # FastAPI
    # --------------------------------------------------------

    api_host: str
    api_port: int

    # --------------------------------------------------------
    # Streamlit
    # --------------------------------------------------------

    streamlit_port: int


# ============================================================
# SETTINGS LOADER
# ============================================================

def load_settings() -> Settings:
    """
    Load and validate Vanguard-GWM configuration
    from environment variables.
    """

    loaded_settings = Settings(
        # ====================================================
        # APPLICATION
        # ====================================================

        app_name=os.getenv(
            "APP_NAME",
            "Vanguard-GWM",
        ),

        app_env=os.getenv(
            "APP_ENV",
            "development",
        ),

        debug=_get_bool(
            "DEBUG",
            False,
        ),

        # ====================================================
        # DATABASE
        # ====================================================

        database_url=os.getenv(
            "DATABASE_URL",
            "",
        ),

        # ====================================================
        # VANGUARD-GWM MODEL
        # ====================================================

        model_latent_dim=_get_int(
            "MODEL_LATENT_DIM",
            64,
        ),

        forecast_horizon=_get_int(
            "FORECAST_HORIZON",
            5,
        ),

        defensive_threshold=_get_float(
            "DEFENSIVE_THRESHOLD",
            0.60,
        ),

        # ====================================================
        # FASTAPI
        # ====================================================

        api_host=os.getenv(
            "API_HOST",
            "0.0.0.0",
        ),

        api_port=_get_int(
            "API_PORT",
            8000,
        ),

        # ====================================================
        # STREAMLIT
        # ====================================================

        streamlit_port=_get_int(
            "STREAMLIT_PORT",
            8501,
        ),
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    if loaded_settings.model_latent_dim <= 0:
        raise ValueError(
            "MODEL_LATENT_DIM must be greater than 0."
        )

    if loaded_settings.forecast_horizon <= 0:
        raise ValueError(
            "FORECAST_HORIZON must be greater than 0."
        )

    if not (
        0.0
        <= loaded_settings.defensive_threshold
        <= 1.0
    ):
        raise ValueError(
            "DEFENSIVE_THRESHOLD must be between "
            "0.0 and 1.0."
        )

    if not (
        1
        <= loaded_settings.api_port
        <= 65535
    ):
        raise ValueError(
            "API_PORT must be between 1 and 65535."
        )

    if not (
        1
        <= loaded_settings.streamlit_port
        <= 65535
    ):
        raise ValueError(
            "STREAMLIT_PORT must be between 1 and 65535."
        )

    return loaded_settings


# ============================================================
# GLOBAL SETTINGS INSTANCE
# ============================================================

settings = load_settings()