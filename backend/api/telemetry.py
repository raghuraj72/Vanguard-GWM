from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field


router = APIRouter()


class TelemetryEvent(BaseModel):
    """
    Single network telemetry event accepted by Vanguard-GWM.
    """

    timestamp: datetime = Field(
        ...,
        description="UTC timestamp of the telemetry event",
    )

    source_ip: str = Field(
        ...,
        min_length=1,
        description="Source IP address",
    )

    destination_ip: str = Field(
        ...,
        min_length=1,
        description="Destination IP address",
    )

    protocol: str = Field(
        ...,
        min_length=1,
        description="Network protocol",
    )

    packet_size: float = Field(
        default=0.0,
        ge=0.0,
        description="Packet size in bytes",
    )

    mitre_stage: Optional[str] = Field(
        default=None,
        description="MITRE ATT&CK stage if available",
    )

    anomaly_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Telemetry anomaly score between 0 and 1",
    )


@router.post(
    "/telemetry",
    summary="Ingest network telemetry",
)
async def ingest_telemetry(event: TelemetryEvent):
    """
    Accept a single telemetry event.

    At this stage the event is validated and returned.
    Persistence and Vanguard-GWM inference will be
    connected in later steps.
    """

    return {
        "status": "accepted",
        "message": "Telemetry event received successfully",
        "event": event.model_dump(mode="json"),
    }