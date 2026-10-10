import html
import json
import math
import os
import time

import requests

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import settings
from src.data_pipeline import StreamingTelemetryParser


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Vanguard-GWM | Threat Forecasting SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

WINDOW_SIZE_SEC = 2.0
DEFENSIVE_THRESHOLD = settings.defensive_threshold

MAX_TOPOLOGY_NODES = 18
MAX_TOPOLOGY_EDGES = 35

MAX_HISTORY_SNAPSHOTS = 7


# ============================================================
# CUSTOM PROFESSIONAL THEME
# ============================================================

st.markdown(
    """
    <style>

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        h1 {
            letter-spacing: -0.5px;
        }

        h2,
        h3 {
            letter-spacing: -0.2px;
        }

        div[data-testid="stMetric"] {
            background-color: #0f172a !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
            padding: 14px 18px !important;
            min-height: 105px;
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] label p {
            color: #94a3b8 !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
        }

        div[data-testid="stMetric"]
        div[data-testid="stMetricValue"],
        div[data-testid="stMetric"]
        div[data-testid="stMetricValue"] div {
            color: #38bdf8 !important;
            font-weight: 700 !important;
            font-size: 1.55rem !important;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid #1e293b;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 10px;
        }

        .footer-note {
            color: #64748b;
            font-size: 0.78rem;
            text-align: center;
            padding-top: 8px;
        }

        .telemetry-badge {
            padding: 8px 12px;
            border-radius: 8px;
            border: 1px solid #334155;
            background: #0f172a;
            margin-bottom: 10px;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def append_action_audit(event, target, policy, outcome, details=""):
    """Record operator-visible defensive decisions for the current session."""
    if "action_audit" not in st.session_state:
        st.session_state.action_audit = []

    st.session_state.action_audit.append(
        {
            "timestamp": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
            "event": str(event),
            "target": str(target),
            "policy": str(policy),
            "outcome": str(outcome),
            "details": str(details),
        }
    )
    st.session_state.action_audit = st.session_state.action_audit[-50:]


def policy_is_live_enforceable(policy_name):
    """Return True only for policies with a concrete SDN implementation."""
    return policy_name == "Host Isolation"


def display_dataframe(df: pd.DataFrame):
    try:
        st.dataframe(
            df,
            hide_index=True,
            width="stretch",
        )
    except TypeError:
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
        )


def display_chart(chart):
    try:
        st.altair_chart(
            chart,
            width="stretch",
        )
    except TypeError:
        st.altair_chart(
            chart,
            use_container_width=True,
        )


def display_button(label):
    try:
        return st.button(
            label,
            width="stretch",
        )
    except TypeError:
        return st.button(
            label,
            use_container_width=True,
        )


def clamp(
    value,
    minimum=0.0,
    maximum=1.0,
):
    try:
        value = float(value)
    except Exception:
        return minimum

    if not math.isfinite(value):
        return minimum

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def risk_label(risk):
    risk = clamp(risk)

    if risk >= DEFENSIVE_THRESHOLD:
        return "HIGH"

    if risk >= 0.30:
        return "MODERATE"

    return "LOW"


def risk_icon(risk):
    risk = clamp(risk)

    if risk >= DEFENSIVE_THRESHOLD:
        return "🔴"

    if risk >= 0.30:
        return "🟠"

    return "🟢"


def severity_from_anomaly(score):
    score = clamp(score)

    if score >= 0.80:
        return "Critical"

    if score >= 0.60:
        return "High"

    if score >= 0.30:
        return "Medium"

    return "Low"


def safe_role(ip):
    """
    Visualization-only asset classification.

    This does NOT claim actual asset identity.
    """

    ip = str(ip)

    if ip.startswith("192.168."):
        return "PRIVATE HOST"

    if ip.startswith("10."):
        return "INTERNAL HOST"

    if ip.startswith("172.16."):
        return "PRIVATE HOST"

    return "NETWORK HOST"


def protocol_from_port(port):
    """
    Human-readable visualization helper.
    """

    try:
        port = int(float(port))
    except Exception:
        return "NETWORK"

    known_ports = {
        22: "SSH",
        23: "TELNET",
        53: "DNS",
        80: "HTTP",
        88: "KERBEROS",
        139: "NETBIOS",
        443: "HTTPS",
        445: "SMB",
        3389: "RDP",
        5985: "WINRM",
        5986: "WINRM",
    }

    return known_ports.get(
        port,
        f"PORT {port}",
    )


def format_timestamp(value):
    if value is None:
        return "N/A"

    try:
        return pd.to_datetime(
            float(value),
            unit="s",
            utc=True,
        ).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    except Exception:
        return str(value)


# ============================================================
# CSV NORMALIZATION
# ============================================================

@st.cache_data(show_spinner=False)
def normalize_uploaded_dataframe(df):
    parser = StreamingTelemetryParser(
        window_size_sec=WINDOW_SIZE_SEC
    )

    return parser.normalize_dataframe(
        df.copy()
    )


# ============================================================
# REAL GRAPH BUILDER
# ============================================================

@st.cache_data(show_spinner=False)
def build_real_graph_from_normalized(
    normalized_df,
    window_index,
):
    parser = StreamingTelemetryParser(
        window_size_sec=WINDOW_SIZE_SEC
    )

    return parser.dataframe_to_graph(
        normalized_df.copy(),
        window_index=int(window_index),
    )


# ============================================================
# TEMPORAL WINDOW CALCULATION
# ============================================================

def calculate_max_window_index(
    normalized_df,
):

    if normalized_df is None:
        return 20

    if normalized_df.empty:
        return 0

    if "timestamp" not in normalized_df.columns:
        return 20

    timestamps = pd.to_numeric(
        normalized_df["timestamp"],
        errors="coerce",
    ).dropna()

    if timestamps.empty:
        return 0

    duration = max(
        0.0,
        float(
            timestamps.max()
            - timestamps.min()
        ),
    )

    return max(
        0,
        int(
            math.floor(
                duration
                / WINDOW_SIZE_SEC
            )
        ),
    )


# ============================================================
# TEMPORAL WINDOW INFORMATION
# ============================================================

def get_window_bounds(
    normalized_df,
    window_index,
):

    if (
        normalized_df is None
        or normalized_df.empty
        or "timestamp" not in normalized_df.columns
    ):
        return None, None

    timestamps = pd.to_numeric(
        normalized_df["timestamp"],
        errors="coerce",
    ).dropna()

    if timestamps.empty:
        return None, None

    timestamp_min = float(
        timestamps.min()
    )

    window_start = (
        timestamp_min
        + int(window_index)
        * WINDOW_SIZE_SEC
    )

    window_end = (
        window_start
        + WINDOW_SIZE_SEC
    )

    return (
        window_start,
        window_end,
    )


# ============================================================
# EXTRACT SELECTED WINDOW TELEMETRY
# ============================================================

def get_selected_window_df(
    normalized_df,
    window_index,
):

    if normalized_df is None:
        return pd.DataFrame()

    if normalized_df.empty:
        return normalized_df.copy()

    if "timestamp" not in normalized_df.columns:
        return normalized_df.copy()

    window_start, window_end = (
        get_window_bounds(
            normalized_df,
            window_index,
        )
    )

    if (
        window_start is None
        or window_end is None
    ):
        return pd.DataFrame()

    timestamps = pd.to_numeric(
        normalized_df["timestamp"],
        errors="coerce",
    )

    mask = (
        (timestamps >= window_start)
        &
        (timestamps < window_end)
    )

    return normalized_df[
        mask
    ].copy()


# ============================================================
# AVAILABLE TELEMETRY WINDOWS
# ============================================================

def get_available_window_indices(
    normalized_df,
):

    if (
        normalized_df is None
        or normalized_df.empty
        or "timestamp" not in normalized_df.columns
    ):
        return []

    timestamps = pd.to_numeric(
        normalized_df["timestamp"],
        errors="coerce",
    ).dropna()

    if timestamps.empty:
        return []

    timestamp_min = float(
        timestamps.min()
    )

    window_indices = (
        (
            (
                timestamps
                - timestamp_min
            )
            // WINDOW_SIZE_SEC
        )
        .astype(int)
        .tolist()
    )

    return sorted(
        set(window_indices)
    )


# ============================================================
# BACKEND API CLIENT
# ============================================================

API_BASE_URL = os.getenv(
    "VANGUARD_API_URL",
    f"http://127.0.0.1:{settings.api_port}",
).rstrip("/")


def api_request(
    method,
    endpoint,
    payload=None,
    timeout=120,
):
    """Call the Vanguard-GWM FastAPI backend."""
    url = f"{API_BASE_URL}{endpoint}"

    try:
        response = requests.request(
            method=method,
            url=url,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Vanguard API is unreachable at {API_BASE_URL}. "
            f"Start api.py first. ({type(exc).__name__})"
        ) from exc

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise RuntimeError(
            f"Vanguard API {endpoint} returned HTTP {response.status_code}: {detail}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Vanguard API {endpoint} returned invalid JSON."
        ) from exc


def dataframe_to_api_rows(df):
    """Convert normalized telemetry to JSON-safe API request rows."""
    if df is None or df.empty:
        return []

    safe_df = df.replace([np.inf, -np.inf], np.nan)
    return json.loads(
        safe_df.to_json(
            orient="records",
            date_format="iso",
        )
    )


def run_vanguard_inference(
    window_index,
    horizon,
    target,
    normalized_df=None,
    raw_df=None,
):
    """Run telemetry ingestion and model inference through FastAPI."""
    mode = "csv" if normalized_df is not None else "synthetic"
    rows = dataframe_to_api_rows(
        raw_df if raw_df is not None else normalized_df
    )

    result = api_request(
        "POST",
        "/forecast",
        {
            "rows": rows,
            "window_index": int(window_index),
            "target_ip": str(target),
            "horizon_k": int(horizon),
            "mode": mode,
        },
        timeout=180,
    )

    graph = result.get("graph", {})
    graph_metadata = {
        "mode": "REAL CSV" if mode == "csv" else "SYNTHETIC",
        "nodes": int(graph.get("nodes", 0)),
        "edges": int(graph.get("edges", 0)),
        "node_ips": [],
        "window_start": graph.get("window_start"),
        "window_end": graph.get("window_end"),
        "telemetry_rows": int(graph.get("telemetry_rows", 0)),
        "max_anomaly": graph.get("max_anomaly"),
        "mean_anomaly": graph.get("mean_anomaly"),
        "mitre_stage": str(graph.get("mitre_stage", "Synthetic")),
    }

    return (
        result,
        float(result.get("latency_ms", 0.0)),
        graph_metadata,
    )


# ============================================================
# DYNAMIC REAL TOPOLOGY

# ============================================================

def build_real_topology(
    window_df,
    target_ip,
):

    if (
        window_df is None
        or window_df.empty
    ):
        return {}, []

    required_columns = {
        "src_ip",
        "dst_ip",
    }

    if not required_columns.issubset(
        set(window_df.columns)
    ):
        return {}, []

    grouped_rows = []

    # ========================================================
    # PRESERVE ACTUAL TELEMETRY
    # ========================================================

    for _, row in window_df.iterrows():

        source = str(
            row.get(
                "src_ip",
                "",
            )
        ).strip()

        destination = str(
            row.get(
                "dst_ip",
                "",
            )
        ).strip()

        if (
            not source
            or not destination
        ):
            continue

        anomaly = clamp(
            row.get(
                "anomaly_score",
                0.0,
            )
        )

        try:

            bytes_value = float(
                row.get(
                    "bytes",
                    0.0,
                )
            )

        except Exception:

            bytes_value = 0.0

        try:

            dst_port = float(
                row.get(
                    "dst_port",
                    0.0,
                )
            )

        except Exception:

            dst_port = 0.0

        raw_protocol = str(
            row.get(
                "protocol",
                "NETWORK",
            )
        )

        if dst_port > 0:

            protocol = (
                f"{raw_protocol} / "
                f"{int(dst_port)}"
            )

        else:

            protocol = raw_protocol

        stage = str(
            row.get(
                "mitre_stage",
                "Unknown",
            )
        )

        grouped_rows.append(
            {
                "source": source,

                "destination":
                    destination,

                "protocol":
                    protocol,

                "label":
                    stage,

                "anomaly":
                    anomaly,

                "severity":
                    severity_from_anomaly(
                        anomaly
                    ),

                "bytes":
                    bytes_value,

                "stage":
                    stage,
            }
        )

    # ========================================================
    # HIGHEST-RISK FLOWS FIRST
    # ========================================================

    grouped_rows.sort(
        key=lambda item: (
            item["anomaly"],

            1
            if (
                item["source"]
                == target_ip
                or item["destination"]
                == target_ip
            )
            else 0,
        ),
        reverse=True,
    )

    topology_edges = (
        grouped_rows[
            :MAX_TOPOLOGY_EDGES
        ]
    )

    # ========================================================
    # DISCOVER NODES
    # ========================================================

    node_priority = []

    if target_ip:
        node_priority.append(
            target_ip
        )

    for edge in topology_edges:

        for ip in [
            edge["source"],
            edge["destination"],
        ]:

            if ip not in node_priority:

                node_priority.append(
                    ip
                )

    nodes = {}

    positions = []

    x_positions = [
        120,
        330,
        540,
        750,
    ]

    y_positions = [
        105,
        185,
        265,
        345,
    ]

    for y in y_positions:

        for x in x_positions:

            positions.append(
                (x, y)
            )

    for index, ip in enumerate(
        node_priority[
            :MAX_TOPOLOGY_NODES
        ]
    ):

        x, y = positions[
            index % len(positions)
        ]

        nodes[ip] = {
            "x": x,
            "y": y,
            "role": safe_role(ip),
        }

    return (
        nodes,
        topology_edges,
    )


# ============================================================
# DEMO TOPOLOGY
# ============================================================

def get_demo_topology():

    topology_data = [

        {
            "source":
                "192.168.1.105",

            "destination":
                "10.0.0.5",

            "protocol":
                "TCP / 445",

            "label":
                "SMB",

            "anomaly":
                0.94,

            "severity":
                "Critical",
        },

        {
            "source":
                "192.168.1.105",

            "destination":
                "192.168.1.110",

            "protocol":
                "TCP / 3389",

            "label":
                "RDP",

            "anomaly":
                0.88,

            "severity":
                "Critical",
        },

        {
            "source":
                "10.0.0.5",

            "destination":
                "10.0.0.22",

            "protocol":
                "TCP / 88",

            "label":
                "Kerberos",

            "anomaly":
                0.12,

            "severity":
                "Low",
        },

        {
            "source":
                "172.16.0.4",

            "destination":
                "192.168.1.105",

            "protocol":
                "TCP / 443",

            "label":
                "HTTPS",

            "anomaly":
                0.04,

            "severity":
                "Low",
        },

        {
            "source":
                "192.168.1.200",

            "destination":
                "8.8.8.8",

            "protocol":
                "UDP / 53",

            "label":
                "DNS",

            "anomaly":
                0.01,

            "severity":
                "Low",
        },

    ]

    nodes = {

        "192.168.1.105": {
            "x": 150,
            "y": 190,
            "role": "WEB SERVER",
        },

        "10.0.0.5": {
            "x": 420,
            "y": 90,
            "role": "DOMAIN CONTROLLER",
        },

        "192.168.1.110": {
            "x": 420,
            "y": 290,
            "role": "DB SERVER",
        },

        "10.0.0.22": {
            "x": 690,
            "y": 90,
            "role": "DEV WORKSTATION",
        },

        "172.16.0.4": {
            "x": 690,
            "y": 290,
            "role": "EXTERNAL HOST",
        },

        "192.168.1.200": {
            "x": 150,
            "y": 380,
            "role": "DNS CLIENT",
        },

        "8.8.8.8": {
            "x": 690,
            "y": 400,
            "role": "DNS SERVER",
        },

    }

    return (
        nodes,
        topology_data,
    )


# ============================================================
# SVG TOPOLOGY GENERATOR
# ============================================================

def build_topology_svg(
    topology_nodes,
    topology_edges,
    selected_target,
):

    svg_parts = []

    svg_parts.append(
        """
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<style>

html,
body {

    margin: 0;
    padding: 0;

    width: 100%;
    height: 100%;

    background: #020617;

    overflow: hidden;

    font-family:
        Arial,
        Helvetica,
        sans-serif;
}

.graph-container {

    width: 100%;
    height: 100%;

    background: #020617;
}

</style>

</head>

<body>

<div class="graph-container">

<svg
    viewBox="0 0 840 470"
    width="100%"
    height="470"
    preserveAspectRatio="xMidYMid meet"
    xmlns="http://www.w3.org/2000/svg"
>

<defs>

<marker
    id="arrow-normal"
    markerWidth="10"
    markerHeight="10"
    refX="9"
    refY="3"
    orient="auto"
    markerUnits="strokeWidth"
>
    <path
        d="M0,0 L0,6 L9,3 z"
        fill="#64748b"
    />
</marker>

<marker
    id="arrow-danger"
    markerWidth="10"
    markerHeight="10"
    refX="9"
    refY="3"
    orient="auto"
    markerUnits="strokeWidth"
>
    <path
        d="M0,0 L0,6 L9,3 z"
        fill="#ef4444"
    />
</marker>

<marker
    id="arrow-medium"
    markerWidth="10"
    markerHeight="10"
    refX="9"
    refY="3"
    orient="auto"
    markerUnits="strokeWidth"
>
    <path
        d="M0,0 L0,6 L9,3 z"
        fill="#f59e0b"
    />
</marker>

<filter
    id="target-glow"
    x="-100%"
    y="-100%"
    width="300%"
    height="300%"
>
    <feGaussianBlur
        stdDeviation="5"
        result="blur"
    />

    <feMerge>

        <feMergeNode in="blur"/>

        <feMergeNode in="SourceGraphic"/>

    </feMerge>

</filter>

</defs>

<rect
    x="0"
    y="0"
    width="840"
    height="470"
    rx="12"
    fill="#020617"
/>

<text
    x="25"
    y="30"
    fill="#e2e8f0"
    font-size="14"
    font-weight="700"
>
    Vanguard-GWM Network Communication Graph
</text>

<text
    x="25"
    y="50"
    fill="#64748b"
    font-size="10"
>
    Active inspection target:
"""
    )

    svg_parts.append(
        html.escape(
            str(selected_target)
        )
    )

    svg_parts.append(
        """
</text>
"""
    )

    # ========================================================
    # EDGES
    # ========================================================

    for edge in topology_edges:

        source = str(
            edge["source"]
        )

        destination = str(
            edge["destination"]
        )

        source_node = (
            topology_nodes.get(
                source
            )
        )

        destination_node = (
            topology_nodes.get(
                destination
            )
        )

        if (
            source_node is None
            or destination_node is None
        ):
            continue

        x1 = source_node["x"]
        y1 = source_node["y"]

        x2 = destination_node["x"]
        y2 = destination_node["y"]

        anomaly = clamp(
            edge.get(
                "anomaly",
                0.0,
            )
        )

        if anomaly >= 0.80:

            stroke = "#ef4444"
            marker = (
                "url(#arrow-danger)"
            )
            stroke_width = 3.5

        elif anomaly >= 0.30:

            stroke = "#f59e0b"
            marker = (
                "url(#arrow-medium)"
            )
            stroke_width = 2.5

        else:

            stroke = "#475569"
            marker = (
                "url(#arrow-normal)"
            )
            stroke_width = 1.8

        touches_target = (
            source == selected_target
            or destination
            == selected_target
        )

        if touches_target:
            stroke_width += 1

        mid_x = (
            x1 + x2
        ) / 2

        mid_y = (
            y1 + y2
        ) / 2

        flow_label = str(
            edge.get(
                "label",
                "FLOW",
            )
        )

        protocol = str(
            edge.get(
                "protocol",
                "NETWORK",
            )
        )

        svg_parts.append(
            f"""
<g>

<title>
{html.escape(source)}
→
{html.escape(destination)}
|
{html.escape(protocol)}
|
Telemetry anomaly {anomaly:.2f}
</title>

<line
    x1="{x1}"
    y1="{y1}"
    x2="{x2}"
    y2="{y2}"
    stroke="{stroke}"
    stroke-width="{stroke_width}"
    marker-end="{marker}"
    opacity="0.9"
/>

<rect
    x="{mid_x - 36}"
    y="{mid_y - 10}"
    width="72"
    height="18"
    rx="5"
    fill="#0f172a"
    stroke="#1e293b"
/>

<text
    x="{mid_x}"
    y="{mid_y + 3}"
    text-anchor="middle"
    fill="#94a3b8"
    font-size="8"
>
{html.escape(flow_label[:15])}
</text>

</g>
"""
        )

    # ========================================================
    # NODES
    # ========================================================

    for ip, node in topology_nodes.items():

        x = node["x"]
        y = node["y"]

        role = node["role"]

        is_target = (
            ip == selected_target
        )

        if is_target:

            radius = 34
            fill = "#172554"
            stroke = "#38bdf8"
            stroke_width = 4

            svg_parts.append(
                f"""
<circle
    cx="{x}"
    cy="{y}"
    r="43"
    fill="none"
    stroke="#38bdf8"
    stroke-width="2"
    stroke-dasharray="6 4"
    opacity="0.65"
    filter="url(#target-glow)"
/>
"""
            )

        else:

            radius = 29
            fill = "#0f172a"
            stroke = "#475569"
            stroke_width = 2

        last_octet = str(
            ip
        ).split(".")[-1]

        svg_parts.append(
            f"""
<g>

<title>
{html.escape(ip)}
|
{html.escape(role)}
|
{"ACTIVE TARGET" if is_target else "MONITORED NODE"}
</title>

<circle
    cx="{x}"
    cy="{y}"
    r="{radius}"
    fill="{fill}"
    stroke="{stroke}"
    stroke-width="{stroke_width}"
/>

<text
    x="{x}"
    y="{y - 4}"
    text-anchor="middle"
    fill="#e2e8f0"
    font-size="10"
    font-weight="700"
>
{html.escape(last_octet)}
</text>

<text
    x="{x}"
    y="{y + 11}"
    text-anchor="middle"
    fill="#94a3b8"
    font-size="8"
>
{html.escape(role)}
</text>

</g>
"""
        )

        if is_target:

            svg_parts.append(
                f"""
<text
    x="{x}"
    y="{y + 54}"
    text-anchor="middle"
    fill="#38bdf8"
    font-size="9"
    font-weight="700"
>
ACTIVE TARGET
</text>
"""
            )

    # ========================================================
    # LEGEND
    # ========================================================

    svg_parts.append(
        """
<g>

<circle
    cx="35"
    cy="438"
    r="7"
    fill="#172554"
    stroke="#38bdf8"
    stroke-width="2"
/>

<text
    x="50"
    y="442"
    fill="#94a3b8"
    font-size="10"
>
Active Target
</text>

<line
    x1="170"
    y1="438"
    x2="205"
    y2="438"
    stroke="#ef4444"
    stroke-width="3"
/>

<text
    x="215"
    y="442"
    fill="#94a3b8"
    font-size="10"
>
High Anomaly
</text>

<line
    x1="350"
    y1="438"
    x2="385"
    y2="438"
    stroke="#f59e0b"
    stroke-width="2.5"
/>

<text
    x="395"
    y="442"
    fill="#94a3b8"
    font-size="10"
>
Medium Anomaly
</text>

<line
    x1="565"
    y1="438"
    x2="600"
    y2="438"
    stroke="#475569"
    stroke-width="2"
/>

<text
    x="610"
    y="442"
    fill="#94a3b8"
    font-size="10"
>
Normal Flow
</text>

</g>

</svg>

</div>

</body>

</html>
"""
    )

    return "".join(
        svg_parts
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ Vanguard-GWM: Generative Graph World Model"
)

st.caption(
    "AI-Based Network Attack Forecasting from "
    "Network Telemetry & Graph World Models"
)

st.info(
    "Proactive SOC decision-support dashboard for forecasting "
    "multi-stage network attack risk and evaluating defensive "
    "policies against the predicted trajectory."
)


# ============================================================
# SIDEBAR — INGESTION
# ============================================================

st.sidebar.header(
    "📁 Ingestion & Simulation Controls"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CIC-IDS-2018 / PCAP / CSV Telemetry",
    type=[
        "csv",
        "pcap",
    ],
)


# ============================================================
# FILE INGESTION
# ============================================================

uploaded_df = None
normalized_df = None

telemetry_source = "Synthetic"
telemetry_mode = "Synthetic Demo"


if uploaded_file is not None:

    file_name = (
        uploaded_file.name.lower()
    )

    # ========================================================
    # CSV
    # ========================================================

    if file_name.endswith(".csv"):

        try:

            uploaded_file.seek(0)

            uploaded_df = pd.read_csv(
                uploaded_file
            )

            normalized_df = (
                normalize_uploaded_dataframe(
                    uploaded_df
                )
            )

            if normalized_df.empty:

                raise ValueError(
                    "CSV contains no valid telemetry rows."
                )

            telemetry_source = (
                "CSV Loaded"
            )

            telemetry_mode = (
                "REAL CSV TELEMETRY"
            )

            st.sidebar.success(
                f"Loaded: {uploaded_file.name}"
            )

            st.sidebar.caption(
                f"{len(uploaded_df):,} rows × "
                f"{len(uploaded_df.columns):,} columns"
            )

            st.sidebar.caption(
                f"Normalized telemetry: "
                f"{len(normalized_df):,} rows"
            )

        except Exception as exc:

            st.sidebar.error(
                "Unable to process CSV telemetry."
            )

            st.sidebar.caption(
                f"{type(exc).__name__}: {exc}"
            )

            uploaded_df = None
            normalized_df = None

    # ========================================================
    # PCAP
    # ========================================================

    elif file_name.endswith(
        ".pcap"
    ):

        telemetry_source = (
            "PCAP Pending"
        )

        telemetry_mode = (
            "PCAP — PARSER PENDING"
        )

        st.sidebar.success(
            f"Loaded PCAP: {uploaded_file.name}"
        )

        st.sidebar.warning(
            "PCAP parser integration is not enabled yet. "
            "Inference will remain in synthetic mode."
        )

else:

    st.sidebar.caption(
        "Synthetic Demo Mode: no external telemetry loaded."
    )


# ============================================================
# TEMPORAL RANGE
# ============================================================

if normalized_df is not None:

    max_window_index = (
        calculate_max_window_index(
            normalized_df
        )
    )

    available_windows = (
        get_available_window_indices(
            normalized_df
        )
    )

else:

    max_window_index = 20
    available_windows = []


# ============================================================
# FORECAST HORIZON
# ============================================================

horizon_k = st.sidebar.slider(
    "Forecast Horizon (K-steps ahead)",
    min_value=1,
    max_value=8,
    value=5,
    step=1,
)


# ============================================================
# TEMPORAL SLICE
# ============================================================

default_window = min(
    8,
    max_window_index,
)

selected_window = st.sidebar.slider(
    "Temporal Slice Index (t)",
    min_value=0,
    max_value=max_window_index,
    value=default_window,
    step=1,
)


# ============================================================
# WINDOW AVAILABILITY INFO
# ============================================================

if normalized_df is not None:

    if available_windows:

        if selected_window in available_windows:

            st.sidebar.success(
                f"Window t={selected_window} "
                f"contains telemetry."
            )

        else:

            st.sidebar.warning(
                f"Window t={selected_window} "
                f"contains no direct telemetry rows."
            )

        st.sidebar.caption(
            f"Telemetry-bearing windows: "
            f"{len(available_windows)}"
        )


# ============================================================
# TARGET SELECTION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🎯 Active Inspection Target"
)


if normalized_df is not None:

    discovered_ips = sorted(
        set(
            normalized_df[
                "src_ip"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )
        |
        set(
            normalized_df[
                "dst_ip"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )
    )

    if discovered_ips:

        target_ip = (
            st.sidebar.selectbox(
                "Target Host Node",
                discovered_ips,
            )
        )

        st.sidebar.success(
            f"Real graph target: {target_ip}"
        )

    else:

        target_ip = "UNKNOWN"

        st.sidebar.error(
            "No valid IP addresses found."
        )

else:

    target_node = (
        st.sidebar.selectbox(
            "Target Host Node",
            [
                "192.168.1.105 (Web Server)",
                "192.168.1.110 (DB Server)",
                "10.0.0.5 (Domain Controller)",
                "10.0.0.22 (Dev Workstation)",
            ],
        )
    )

    target_ip = (
        target_node.split()[0]
    )

    st.sidebar.caption(
        f"Demo target: {target_ip}"
    )


# ============================================================
# SELECTED REAL GRAPH
# ============================================================

real_graph = None

if normalized_df is not None:

    try:

        real_graph = (
            build_real_graph_from_normalized(
                normalized_df,
                selected_window,
            )
        )

    except Exception as exc:

        st.error(
            "Unable to construct the selected telemetry graph."
        )

        st.exception(exc)

        st.stop()


# ============================================================
# RUN VANGUARD INFERENCE
# ============================================================

try:

    (
        result,
        inference_latency_ms,
        graph_metadata,
    ) = run_vanguard_inference(
        window_index=selected_window,
        horizon=horizon_k,
        target=target_ip,
        normalized_df=normalized_df,
        raw_df=uploaded_df,
    )

    predicted_risk = [
        clamp(value)
        for value in result[
            "risk_trajectory"
        ]
    ]

    current_max_risk = clamp(
        result[
            "peak_risk"
        ]
    )

    current_stage = str(
        result[
            "current_stage"
        ]
    )

except Exception as exc:

    st.error(
        "Vanguard-GWM inference failed."
    )

    st.warning(
        "Check the telemetry schema, model dependencies, "
        "and src/engine.py."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# REAL TELEMETRY METADATA
# ============================================================

real_current_stage = None
real_max_anomaly = None
real_mean_anomaly = None

if real_graph is not None:

    real_current_stage = str(
        getattr(
            real_graph,
            "mitre_stage",
            "Unknown",
        )
    )

    real_max_anomaly = clamp(
        getattr(
            real_graph,
            "max_anomaly_score",
            0.0,
        )
    )

    real_mean_anomaly = clamp(
        getattr(
            real_graph,
            "mean_anomaly_score",
            0.0,
        )
    )


# ============================================================
# SELECTED WINDOW DATA
# ============================================================

selected_window_df = pd.DataFrame()

if normalized_df is not None:

    selected_window_df = (
        get_selected_window_df(
            normalized_df,
            selected_window,
        )
    )


# ============================================================
# FORECAST AXIS
# ============================================================

future_steps = [
    f"t+{k * 2}s"
    for k in range(
        1,
        horizon_k + 1,
    )
]


# ============================================================
# TOP KPI ROW
# ============================================================

col1, col2, col3, col4 = (
    st.columns(4)
)


with col1:

    st.metric(
        "Predicted Peak Risk Score",
        f"{current_max_risk * 100:.1f} / 100",
        delta="Model-derived score",
        delta_color="off",
    )
    st.caption(
        "Model risk score on a 0–100 scale; not a calibrated probability of attack."
    )


with col2:

    st.metric(
        "Model Threat Stage",
        current_stage,
    )


with col3:

    st.metric(
        "Forecast Horizon",
        f"{horizon_k} steps",
        delta="2 seconds / step",
    )


with col4:

    st.metric(
        "Inference Latency",
        f"{inference_latency_ms:.2f} ms",
        delta="Measured locally",
    )


# ============================================================
# REAL TELEMETRY STATUS
# ============================================================

if normalized_df is not None:

    st.success(
        f"🟢 REAL TELEMETRY ACTIVE — "
        f"{len(normalized_df):,} normalized telemetry rows "
        f"connected to the Vanguard-GWM graph inference path."
    )

    st.caption(
        f"Target: {target_ip} · "
        f"Temporal slice: t={selected_window} · "
        f"Direct rows in slice: "
        f"{len(selected_window_df):,}"
    )

    st.warning(
        "Real telemetry is connected to inference. "
        "Forecast quality still depends on trained model "
        "weights, validation, and calibration."
    )

else:

    st.info(
        "🔵 SYNTHETIC DEMO MODE — "
        "No external telemetry is currently connected. "
        "Upload a CSV to run the graph pipeline on real telemetry."
    )


# ============================================================
# CURRENT RISK STATUS
# ============================================================

st.markdown("")


if current_max_risk >= (
    DEFENSIVE_THRESHOLD
):

    st.error(
        f"{risk_icon(current_max_risk)} HIGH THREAT — "
        f"Forecasted peak model risk is "
        f"{current_max_risk * 100:.1f}%, "
        f"above the 60% defensive decision threshold. "
        f"Target: {target_ip}"
    )

elif current_max_risk >= 0.30:

    st.warning(
        f"{risk_icon(current_max_risk)} MODERATE THREAT — "
        f"Forecasted peak model risk is "
        f"{current_max_risk * 100:.1f}% "
        f"for target {target_ip}."
    )

else:

    st.success(
        f"{risk_icon(current_max_risk)} LOW THREAT — "
        f"Forecasted peak model risk is "
        f"{current_max_risk * 100:.1f}% "
        f"for target {target_ip}."
    )


st.markdown("---")


# ============================================================
# FORECAST + THREAT ASSESSMENT
# ============================================================

col_left, col_right = st.columns(
    [3, 2]
)


# ============================================================
# FORECAST CURVE
# ============================================================

with col_left:

    st.subheader(
        "📈 Autoregressive K-Step Risk Forecast"
    )

    st.caption(
        "Future risk trajectory generated by the "
        "current Vanguard-GWM inference engine."
    )

    forecast_df = pd.DataFrame(
        {
            "Step": list(
                range(
                    1,
                    horizon_k + 1,
                )
            ),

            "Forecast":
                predicted_risk,
        }
    )

    forecast_chart = (
        alt.Chart(
            forecast_df
        )
        .mark_line(
            point=True,
            strokeWidth=3,
        )
        .encode(

            x=alt.X(
                "Step:Q",
                title="Forecast Step (Δt = 2s)",
                axis=alt.Axis(
                    values=list(
                        range(
                            1,
                            horizon_k + 1,
                        )
                    ),
                    labelExpr=(
                        "'t+' + "
                        "(datum.value * 2) + "
                        "'s'"
                    ),
                ),
            ),

            y=alt.Y(
                "Forecast:Q",
                title="Model Risk Score (0–100)",
                scale=alt.Scale(
                    domain=[0, 1]
                ),
                axis=alt.Axis(
                    format="%"
                ),
            ),

            tooltip=[

                alt.Tooltip(
                    "Step:Q",
                    title="Forecast Step",
                ),

                alt.Tooltip(
                    "Forecast:Q",
                    title="Risk Score (0–100)",
                    format=".0%",
                ),

            ],
        )
        .properties(
            height=320
        )
    )

    threshold_df = pd.DataFrame(
        {
            "Threshold": [
                DEFENSIVE_THRESHOLD
            ]
        }
    )

    threshold_line = (
        alt.Chart(
            threshold_df
        )
        .mark_rule(
            strokeDash=[
                6,
                4,
            ]
        )
        .encode(
            y=alt.Y(
                "Threshold:Q"
            ),
            tooltip=[
                alt.Tooltip(
                    "Threshold:Q",
                    title="Decision Threshold",
                    format=".0%",
                )
            ],
        )
    )

    display_chart(
        forecast_chart
        + threshold_line
    )

    st.caption(
        "Risk scores are model outputs and are not calibrated "
        "attack probabilities. The dashed line represents the "
        "60% defensive decision threshold."
    )


# ============================================================
# ACTIVE THREAT ASSESSMENT
# ============================================================

with col_right:

    st.subheader(
        "🚨 Active Threat Assessment"
    )

    if current_max_risk >= (
        DEFENSIVE_THRESHOLD
    ):

        st.error(
            f"HIGH THREAT\n\n"
            f"Forecast risk is above the 60% "
            f"decision threshold for {target_ip}."
        )

    elif current_max_risk >= 0.30:

        st.warning(
            f"MODERATE THREAT\n\n"
            f"Elevated model risk detected for "
            f"{target_ip}."
        )

    else:

        st.success(
            f"NOMINAL\n\n"
            f"Forecast risk remains below the "
            f"defensive threshold for {target_ip}."
        )

    threat_col1, threat_col2 = (
        st.columns(2)
    )

    with threat_col1:

        st.metric(
            "Target Host",
            target_ip,
        )

    with threat_col2:

        st.metric(
            "Threshold",
            "60%",
        )

    with st.container(
        border=True
    ):

        st.markdown(
            "### Assessment Details"
        )

        detail_col1, detail_col2 = (
            st.columns(2)
        )

        with detail_col1:

            st.caption(
                "MODEL THREAT STAGE"
            )

            st.write(
                f"**{current_stage}**"
            )

        with detail_col2:

            st.caption(
                "FORECAST WINDOW"
            )

            st.write(
                f"**{horizon_k * 2} seconds**"
            )

        st.caption(
            "TELEMETRY THREAT STAGE"
        )

        st.write(
            f"**{real_current_stage or 'Synthetic'}**"
        )

        st.caption(
            "TEMPORAL SLICE"
        )

        st.write(
            f"**t = {selected_window}**"
        )

        st.caption(
            "TELEMETRY MODE"
        )

        st.write(
            f"**{telemetry_mode}**"
        )


# ============================================================
# REAL TELEMETRY GRAPH SUMMARY
# ============================================================

if real_graph is not None:

    st.markdown("---")

    st.subheader(
        "🧬 Real Telemetry Graph Summary"
    )

    graph_col1, graph_col2, graph_col3, graph_col4 = (
        st.columns(4)
    )

    with graph_col1:

        st.metric(
            "Graph Nodes",
            graph_metadata[
                "nodes"
            ],
        )

    with graph_col2:

        st.metric(
            "Graph Edges",
            graph_metadata[
                "edges"
            ],
        )

    with graph_col3:

        st.metric(
            "Telemetry Rows",
            graph_metadata[
                "telemetry_rows"
            ],
        )

    with graph_col4:

        st.metric(
            "Max Anomaly",
            f"{real_max_anomaly * 100:.1f}%",
        )

    st.caption(
        f"Graph window: "
        f"{format_timestamp(graph_metadata['window_start'])}"
        f" → "
        f"{format_timestamp(graph_metadata['window_end'])}"
    )

    st.caption(
        f"Mean telemetry anomaly: "
        f"{real_mean_anomaly * 100:.1f}%"
    )


# ============================================================
# DEFENSIVE POLICY SANDBOX
# ============================================================

st.markdown("---")

st.subheader(
    "🧪 Defensive Policy Sandbox"
)

st.caption(
    "Evaluate counterfactual defensive policies against "
    "the predicted attack trajectory. These projected values "
    "are deterministic policy simulations, not new ML predictions."
)


# ============================================================
# POLICY DEFINITIONS
# ============================================================

POLICY_CONFIG = {

    "No Mitigation": {

        "reduction": 0.00,

        "description": (
            "Continue observing the predicted trajectory "
            "without applying a defensive intervention."
        ),

        "action": "MONITOR",
    },

    "Host Isolation": {

        "reduction": 0.72,

        "description": (
            "Simulate isolating the target host from "
            "the monitored network graph."
        ),

        "action": "ISOLATE HOST",
    },

    "Block Suspicious Traffic": {

        "reduction": 0.55,

        "description": (
            "Simulate blocking suspicious flows associated "
            "with the selected target."
        ),

        "action": "BLOCK TRAFFIC",
    },

    "Restrict Lateral Movement": {

        "reduction": 0.48,

        "description": (
            "Simulate restricting east-west communication "
            "from the target host."
        ),

        "action":
            "RESTRICT LATERAL MOVEMENT",
    },

}


# ============================================================
# POLICY SELECTION
# ============================================================

sandbox_col1, sandbox_col2 = (
    st.columns(
        [1, 1]
    )
)


with sandbox_col1:

    selected_policy = st.selectbox(
        "Select Defensive Policy",
        list(
            POLICY_CONFIG.keys()
        ),
        index=1,
    )

    policy = POLICY_CONFIG[
        selected_policy
    ]

    with st.container(
        border=True
    ):

        st.markdown(
            "### Selected Policy"
        )

        st.write(
            f"**{policy['action']}**"
        )

        st.caption(
            policy["description"]
        )

        st.metric(
            "Simulated Reduction",
            f"{policy['reduction'] * 100:.0f}%",
        )


# ============================================================
# COUNTERFACTUAL RISK — API BACKEND
# ============================================================

try:
    policy_simulation = api_request(
        "POST",
        "/policy/simulate",
        {
            "policy": selected_policy,
            "risk_trajectory": predicted_risk,
            "threshold": DEFENSIVE_THRESHOLD,
        },
    )
except RuntimeError as exc:
    st.error(f"Backend policy simulation unavailable: {exc}")
    st.stop()

counterfactual_risk = [
    clamp(value)
    for value in policy_simulation.get(
    "projected_risk_trajectory",
    policy_simulation.get("projected_trajectory", [])
):

baseline_peak = max(predicted_risk)
counterfactual_peak = clamp(
    policy_simulation["projected_peak_risk"]
)

risk_reduction_absolute = max(
    0.0,
    baseline_peak - counterfactual_peak,
)

if baseline_peak > 0:
    risk_reduction_percent = (
        risk_reduction_absolute
        / baseline_peak
        * 100.0
    )
else:
    risk_reduction_percent = 0.0

# ============================================================
# POLICY STATUS
# ============================================================

with sandbox_col2:

    if counterfactual_peak >= (
        DEFENSIVE_THRESHOLD
    ):

        st.error(
            f"⚠️ RESIDUAL RISK HIGH\n\n"
            f"Projected peak remains "
            f"{counterfactual_peak * 100:.1f}%."
        )

    elif counterfactual_peak >= 0.30:

        st.warning(
            f"🟡 RESIDUAL RISK MODERATE\n\n"
            f"Projected peak is "
            f"{counterfactual_peak * 100:.1f}%."
        )

    else:

        st.success(
            f"✅ RISK BELOW THRESHOLD\n\n"
            f"Projected peak is "
            f"{counterfactual_peak * 100:.1f}%."
        )

    metric_col1, metric_col2 = (
        st.columns(2)
    )

    with metric_col1:

        st.metric(
            "Baseline Peak",
            f"{baseline_peak * 100:.1f}%",
        )

    with metric_col2:

        st.metric(
            "Projected Peak",
            f"{counterfactual_peak * 100:.1f}%",

            delta=(
                f"{risk_reduction_percent:.1f}% "
                "relative reduction"
            ),

            delta_color="inverse",
        )


# ============================================================
# POLICY FORECAST
# ============================================================

sandbox_df = pd.DataFrame(
    {
        "Step": list(
            range(
                1,
                horizon_k + 1,
            )
        ),

        "Baseline Risk":
            predicted_risk,

        "Counterfactual Risk":
            counterfactual_risk,
    }
)


st.markdown(
    "#### 📉 Projected Impact of Defensive Policy"
)

sandbox_long_df = sandbox_df.melt(
    id_vars=[
        "Step"
    ],

    value_vars=[
        "Baseline Risk",
        "Counterfactual Risk",
    ],

    var_name="Scenario",

    value_name="Risk Score",
)


sandbox_chart = (
    alt.Chart(
        sandbox_long_df
    )
    .mark_line(
        point=True,
        strokeWidth=3,
    )
    .encode(

        x=alt.X(
            "Step:Q",

            title="Forecast Step (Δt = 2s)",

            axis=alt.Axis(

                values=list(
                    range(
                        1,
                        horizon_k + 1,
                    )
                ),

                labelExpr=(
                    "'t+' + "
                    "(datum.value * 2) + "
                    "'s'"
                ),
            ),
        ),

        y=alt.Y(
            "Risk Score:Q",

            title="Risk Score",

            scale=alt.Scale(
                domain=[
                    0,
                    1,
                ]
            ),

            axis=alt.Axis(
                format="%"
            ),
        ),

        color=alt.Color(
            "Scenario:N",

            scale=alt.Scale(

                domain=[
                    "Baseline Risk",
                    "Counterfactual Risk",
                ],

                range=[
                    "#ef4444",
                    "#10b981",
                ],
            ),

            legend=alt.Legend(
                title="Scenario"
            ),
        ),

        tooltip=[

            alt.Tooltip(
                "Step:Q",
                title="Forecast Step",
            ),

            alt.Tooltip(
                "Scenario:N",
                title="Scenario",
            ),

            alt.Tooltip(
                "Risk Score:Q",
                title="Risk Score",
                format=".2%",
            ),

        ],
    )
    .properties(
        height=300
    )
)


sandbox_threshold = (
    alt.Chart(
        threshold_df
    )
    .mark_rule(
        strokeDash=[
            6,
            4,
        ]
    )
    .encode(
        y=alt.Y(
            "Threshold:Q"
        )
    )
)


display_chart(
    sandbox_chart
    + sandbox_threshold
)


# ============================================================
# DEFENSIVE DECISION
# ============================================================

st.markdown(
    "#### 🎯 Defensive Decision"
)

decision_col1, decision_col2, decision_col3 = (
    st.columns(3)
)


with decision_col1:

    st.metric(
        "Risk Reduction",
        (
            f"{risk_reduction_absolute * 100:.1f} "
            "percentage points"
        ),
    )


with decision_col2:

    st.metric(
        "Relative Reduction",
        f"{risk_reduction_percent:.1f}%",
    )


with decision_col3:

    if counterfactual_peak < (
        DEFENSIVE_THRESHOLD
    ):

        st.success(
            "PASS"
        )

        st.caption(
            "Residual risk < 60%"
        )

    else:

        st.error(
            "FAIL"
        )

        st.caption(
            "Residual risk ≥ 60%"
        )


# ============================================================
# ACTION CONTROL
# ============================================================

st.markdown(
    "#### ⚡ Action Control"
)

action_col1, action_col2 = (
    st.columns(2)
)


with action_col1:

    simulate_clicked = (
        display_button(
            "🧪 Simulate Policy Impact"
        )
    )


with action_col2:

    enforce_clicked = (
        display_button(
            "🛡️ Request SDN Enforcement"
        )
    )


if simulate_clicked:

    st.success(
        f"Counterfactual simulation completed for "
        f"{target_ip} using '{selected_policy}'."
    )

    st.info(
        f"Baseline peak risk: "
        f"{baseline_peak * 100:.1f}% → "
        f"Projected peak risk: "
        f"{counterfactual_peak * 100:.1f}% "
        f"({risk_reduction_percent:.1f}% relative reduction)."
    )


# ============================================================
# SDN ENFORCEMENT GATE
# ============================================================

if enforce_clicked:

    live_supported = policy_is_live_enforceable(selected_policy)
    approval_key = "live_sdn_approval"
    approved = bool(st.session_state.get(approval_key, False))

    if selected_policy == "No Mitigation":
        outcome = "BLOCKED"
        details = "No mitigation policy cannot issue a live SDN action."
        append_action_audit("ENFORCEMENT_REQUEST", target_ip, selected_policy, outcome, details)
        st.warning(details)

    elif not live_supported:
        outcome = "SIMULATION_ONLY"
        details = (
            f"'{selected_policy}' has no action-specific live SDN mapping yet. "
            "The policy remains available for counterfactual simulation only."
        )
        append_action_audit("ENFORCEMENT_REQUEST", target_ip, selected_policy, outcome, details)
        st.info(details)

    elif current_max_risk < DEFENSIVE_THRESHOLD:
        outcome = "BLOCKED"
        details = (
            f"Current model risk is {current_max_risk * 100:.1f}%, below the "
            f"{DEFENSIVE_THRESHOLD * 100:.0f}% defensive threshold."
        )
        append_action_audit("ENFORCEMENT_REQUEST", target_ip, selected_policy, outcome, details)
        st.warning(details)

    elif counterfactual_peak >= DEFENSIVE_THRESHOLD:
        outcome = "BLOCKED"
        details = (
            f"Projected residual risk remains {counterfactual_peak * 100:.1f}%. "
            "The selected policy did not reduce risk below the decision boundary."
        )
        append_action_audit("ENFORCEMENT_REQUEST", target_ip, selected_policy, outcome, details)
        st.warning(details)

    elif not approved:
        outcome = "AWAITING_APPROVAL"
        details = "Explicit operator approval is required before a live SDN request."
        append_action_audit("ENFORCEMENT_REQUEST", target_ip, selected_policy, outcome, details)
        st.warning(details)
        st.caption("Check the explicit approval control below, then request enforcement again.")

    else:
        try:
            enforcement_response = api_request(
                "POST",
                "/enforcement/request",
                {
                    "target_ip": str(target_ip),
                    "policy": selected_policy,
                    "current_max_risk": current_max_risk,
                    "projected_peak_risk": counterfactual_peak,
                    "approved": approved,
                },
            )

            controller_status = enforcement_response.get(
                "controller_status",
                "Unknown",
            )

            if enforcement_response.get("accepted"):
                outcome = "ACCEPTED"
                details = (
                    f"Host isolation rule accepted; projected peak risk "
                    f"{counterfactual_peak * 100:.1f}%."
                )
                append_action_audit(
                    "ENFORCEMENT_REQUEST",
                    target_ip,
                    selected_policy,
                    outcome,
                    details,
                )
                st.success(
                    f"Live SDN host-isolation action accepted for {target_ip}."
                )
                st.caption(details)
            else:
                outcome = "UNAVAILABLE" if enforcement_response.get("status") == "unavailable" else "REJECTED"
                details = (
                    f"SDN controller status: {controller_status}. "
                    "No live rule was confirmed."
                    if outcome == "UNAVAILABLE"
                    else "The SDN controller rejected the enforcement request."
                )
                append_action_audit(
                    "ENFORCEMENT_REQUEST",
                    target_ip,
                    selected_policy,
                    outcome,
                    details,
                )
                st.warning(details)

        except RuntimeError as exc:
            outcome = "ERROR"
            details = str(exc)
            append_action_audit(
                "ENFORCEMENT_REQUEST",
                target_ip,
                selected_policy,
                outcome,
                details,
            )
            st.warning("SDN enforcement request failed safely; no live action was confirmed.")
            st.caption(details)


# The approval control is deliberately separate from the request button.
st.markdown("#### 🔐 Enforcement Gate")

if policy_is_live_enforceable(selected_policy):
    st.session_state["live_sdn_approval"] = st.checkbox(
        "I explicitly approve a live SDN action for the selected target.",
        value=bool(st.session_state.get("live_sdn_approval", False)),
        help=(
            "This approval is required before the dashboard can invoke the "
            "currently implemented Host Isolation SDN action."
        ),
    )
    st.caption(
        "Live mapping: Host Isolation → SDN host quarantine. "
        "Other policies are simulation-only until action-specific controller mappings are implemented."
    )
else:
    st.info(
        f"'{selected_policy}' is currently simulation-only. "
        "No live SDN rule can be issued for this policy."
    )


# ============================================================
# OPERATOR ACTION AUDIT TRAIL
# ============================================================

with st.expander("📋 Operator Action Audit Trail", expanded=False):
    audit_rows = st.session_state.get("action_audit", [])

    if audit_rows:
        audit_df = pd.DataFrame(audit_rows[::-1])
        display_dataframe(audit_df)
    else:
        st.caption("No defensive actions have been requested in this session.")


st.caption(
    "⚠️ Policy impact values are deterministic "
    "counterfactual simulations for decision support. "
    "They are not calibrated probabilities and are not "
    "outputs of a separately trained intervention model."
)


# ============================================================
# UPLOADED TELEMETRY PREVIEW
# ============================================================

if uploaded_df is not None:

    st.markdown("---")

    st.subheader(
        "📥 Uploaded Telemetry Preview"
    )

    st.caption(
        "The uploaded CSV is normalized and transformed "
        "into temporal graph snapshots."
    )

    preview_df = uploaded_df.head(
        100
    )

    display_dataframe(
        preview_df
    )


# ============================================================
# SELECTED WINDOW TELEMETRY
# ============================================================

if normalized_df is not None:

    st.markdown("---")

    st.subheader(
        "⏱️ Selected Temporal Window"
    )

    window_start, window_end = (
        get_window_bounds(
            normalized_df,
            selected_window,
        )
    )

    if (
        window_start is not None
        and window_end is not None
    ):

        st.caption(
            f"{format_timestamp(window_start)} "
            f"→ "
            f"{format_timestamp(window_end)}"
        )

    if not selected_window_df.empty:

        st.success(
            f"{len(selected_window_df):,} direct telemetry "
            f"records in this temporal slice."
        )

        display_dataframe(
            selected_window_df.head(100)
        )

    else:

        st.warning(
            "No telemetry events fall inside this exact "
            "2-second temporal slice."
        )

        st.caption(
            "Move the temporal slice slider to a telemetry-bearing "
            "window to inspect actual flow records."
        )


# ============================================================
# NETWORK ASSET INVENTORY
# ============================================================

if normalized_df is not None:

    st.markdown("---")

    st.subheader(
        "🖥️ Discovered Network Assets"
    )

    discovered_ips = sorted(
        set(
            normalized_df[
                "src_ip"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )
        |
        set(
            normalized_df[
                "dst_ip"
            ]
            .dropna()
            .astype(str)
            .tolist()
        )
    )

    asset_df = pd.DataFrame(
        {
            "IP Address":
                discovered_ips,

            "Graph Role": [
                safe_role(ip)
                for ip in discovered_ips
            ],

            "Active Target": [
                "Yes"
                if ip == target_ip
                else "No"
                for ip in discovered_ips
            ],
        }
    )

    display_dataframe(
        asset_df
    )


# ============================================================
# MODEL CONFIGURATION + TELEMETRY FEATURES
# ============================================================

st.markdown("---")

col_exp1, col_exp2 = (
    st.columns(
        [1, 1]
    )
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

with col_exp1:

    st.subheader(
        "🧠 Vanguard-GWM Inference Configuration"
    )

    st.caption(
        "Architecture parameters used by the current "
        "inference request. The current dashboard uses "
        "randomly initialized model weights; displayed risk "
        "scores are not validated cybersecurity predictions."
    )

    model_config = pd.DataFrame(
        {
            "Component": [

                "Node features",
                "Edge features",
                "Latent dimension",
                "Transformer layers",
                "Forecast horizon",
                "History buffer",
                "Decision threshold",
                "Model weights",
                "Inference status",
                "Validation status",

            ],

            "Value": [

                "5",
                "5",
                "64",
                "2",
                f"{horizon_k} steps",
                f"{MAX_HISTORY_SNAPSHOTS} snapshots",
                "60%",
                "Randomly initialized / untrained",
                "Pipeline functional",
                "Predictive performance not validated",

            ],
        }
    )

    display_dataframe(
        model_config
    )

    if normalized_df is not None:

        st.success(
            "REAL CSV GRAPH CONNECTED"
        )

        st.caption(
            "Telemetry → normalization → temporal window → "
            "graph construction → graph encoder → "
            "temporal forecasting."
        )

    else:

        st.info(
            "Synthetic graph generator is active."
        )


# ============================================================
# TELEMETRY FEATURE OVERVIEW
# ============================================================

with col_exp2:

    st.subheader(
        "🔍 Telemetry Feature Overview"
    )

    st.caption(
        "Node-level features consumed by the graph encoder."
    )

    feature_df = pd.DataFrame(
        {
            "Telemetry Feature": [

                "In-Degree",
                "Out-Degree",
                "Average Bytes",
                "Port Entropy",
                "Role ID",

            ],

            "Representation": [

                "Node feature",
                "Node feature",
                "Node feature",
                "Node feature",
                "Node feature",

            ],
        }
    )

    display_dataframe(
        feature_df
    )

    st.caption(
        "CSV-derived graph edges carry protocol ID, "
        "log-scaled bytes, duration, packets, and "
        "destination-port features."
    )


# ============================================================
# NETWORK GRAPH
# ============================================================

st.markdown("---")

st.subheader(
    "🌐 Monitored Network Graph ($G_t$)"
)


# Keep the graph source explicit so the visualization is never mistaken
# for an observed production topology when the dashboard is in demo mode.
if normalized_df is not None:
    st.success(
        "🟢 REAL CSV TELEMETRY GRAPH"
    )
    st.caption(
        "This topology is constructed from normalized CSV telemetry "
        "for the selected temporal window."
    )
else:
    st.info(
        "🔵 SYNTHETIC DEMO GRAPH"
    )
    st.caption(
        "This topology is deterministically generated for demonstration. "
        "It does not represent an observed network. Upload CSV telemetry "
        "to build the graph from real flow records."
    )


if normalized_df is not None:

    topology_nodes, topology_data = (
        build_real_topology(
            selected_window_df,
            target_ip,
        )
    )

    # ========================================================
    # EMPTY EXACT WINDOW
    # ========================================================

    if not topology_nodes:

        graph_ips = [
            str(ip)
            for ip in getattr(
                real_graph,
                "node_ips",
                [],
            )
        ]

        topology_nodes = {}

        positions = [

            (120, 105),
            (330, 105),
            (540, 105),
            (750, 105),

            (120, 230),
            (330, 230),
            (540, 230),
            (750, 230),

            (120, 355),
            (330, 355),
            (540, 355),
            (750, 355),

        ]

        ordered_ips = [
            target_ip
        ]

        ordered_ips.extend(
            [
                ip
                for ip in graph_ips
                if ip != target_ip
            ]
        )

        for index, ip in enumerate(
            ordered_ips[
                :MAX_TOPOLOGY_NODES
            ]
        ):

            x, y = positions[
                index
                % len(positions)
            ]

            topology_nodes[ip] = {
                "x": x,
                "y": y,
                "role": safe_role(ip),
            }

        topology_data = []

        topology_caption = (
            "The selected 2-second window contains no direct "
            "telemetry flow records. Node inventory is shown "
            "from the selected graph snapshot."
        )

    else:

        topology_caption = (
            "Selected-window topology from real CSV telemetry. "
            "Flow colors are based on the observed CSV anomaly_score. "
            "The selected target is highlighted in blue."
        )

else:

    topology_nodes, topology_data = (
        get_demo_topology()
    )

    topology_caption = (
        "Synthetic monitored-flow topology generated by the dashboard. "
        "These nodes and flows are demonstration data, not observed "
        "network traffic."
    )


st.caption(
    topology_caption
)


# ============================================================
# BUILD + RENDER SVG
# ============================================================

topology_svg = build_topology_svg(
    topology_nodes,
    topology_data,
    target_ip,
)


with st.container(
    border=True
):

    st.html(topology_svg)


# ============================================================
# TOPOLOGY TABLE
# ============================================================

if normalized_df is not None:
    st.markdown(
        "#### 🔎 Flow-Level Telemetry — Real CSV"
    )
else:
    st.markdown(
        "#### 🔎 Flow-Level Telemetry — Synthetic Demo"
    )


if topology_data:

    topology_table = pd.DataFrame(
        {

            "Source IP": [
                item["source"]
                for item in topology_data
            ],

            "Destination IP": [
                item["destination"]
                for item in topology_data
            ],

            "Protocol": [
                item["protocol"]
                for item in topology_data
            ],

            "MITRE / Flow Type": [
                item["label"]
                for item in topology_data
            ],

            "Anomaly Score": [
                item["anomaly"]
                for item in topology_data
            ],

            "Severity": [
                item["severity"]
                for item in topology_data
            ],

            "Target Flow": [

                "Yes"
                if (
                    item["source"]
                    == target_ip
                    or item["destination"]
                    == target_ip
                )

                else "No"

                for item in topology_data
            ],

        }
    )

    display_dataframe(
        topology_table
    )

else:

    st.info(
        "No flow-level telemetry records are available "
        "for the selected temporal window."
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

st.markdown("---")

st.subheader(
    "🟢 System Status"
)


status_col1, status_col2, status_col3, status_col4 = (
    st.columns(4)
)


# ============================================================
# TELEMETRY GRAPH STATUS
# ============================================================

with status_col1:

    if real_graph is not None:

        st.metric(
            "Telemetry Graph",
            "REAL CSV",
        )

        st.caption(
            f"{graph_metadata['nodes']} nodes / "
            f"{graph_metadata['edges']} edges"
        )

    else:

        st.metric(
            "Telemetry Graph",
            "Synthetic",
        )

        st.caption(
            "Deterministic graph generator"
        )


# ============================================================
# INFERENCE STATUS
# ============================================================

with status_col2:

    try:
        backend_health = api_request(
            "GET",
            "/health",
            timeout=5,
        )
        inference_status = "Online" if backend_health.get("status") == "ok" else "Unavailable"
    except RuntimeError:
        inference_status = "Unavailable"

    st.metric(
        "Inference Backend",
        inference_status,
    )

    st.caption(
        f"FastAPI · {API_BASE_URL}"
    )


# ============================================================
# SDN STATUS
# ============================================================

with status_col3:

    try:
        sdn_status = api_request(
            "GET",
            "/sdn/status",
            timeout=5,
        ).get(
            "controller_status",
            "Unavailable",
        )
    except RuntimeError:
        sdn_status = "Unavailable"

    if sdn_status == "Connected":

        st.metric(
            "SDN Enforcement",
            "Connected",
        )

        st.caption(
            "Controller reachable"
        )

    else:

        st.metric(
            "SDN Enforcement",
            "Simulation Mode",
        )

        st.caption(
            "Ryu controller not connected — no live rules will be pushed."
        )


# ============================================================
# TELEMETRY SOURCE STATUS
# ============================================================

with status_col4:

    if normalized_df is None:

        st.metric(
            "Telemetry Source",
            "Synthetic",
        )

        st.caption(
            "No external file loaded"
        )

    else:

        st.metric(
            "Telemetry Source",
            "CSV Loaded",
        )

        st.caption(
            f"{len(normalized_df):,} rows"
        )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.markdown("")

system_info_col1, system_info_col2 = (
    st.columns(
        [1, 1]
    )
)


# ============================================================
# MODEL PIPELINE
# ============================================================

with system_info_col1:

    with st.container(
        border=True
    ):

        st.subheader(
            "🔄 Model Pipeline"
        )

        if normalized_df is not None:

            st.write(
                "CSV Telemetry → Normalization → "
                "Temporal Window → Graph Construction → "
                "Graph Encoder → Temporal Forecasting → "
                "Risk Trajectory"
            )

        else:

            st.write(
                "Synthetic Graph → Graph Encoder → "
                "Temporal Forecasting → Risk Trajectory"
            )

        st.caption(
            f"Active target: {target_ip}"
        )


# ============================================================
# DECISION POLICY
# ============================================================

with system_info_col2:

    with st.container(
        border=True
    ):

        st.subheader(
            "🎯 Decision Policy"
        )

        st.write(
            "60% model risk is used as the "
            "defensive decision boundary."
        )

        st.caption(
            f"Forecast window: "
            f"{horizon_k * 2} seconds"
        )

        st.caption(
            "Enforcement requires explicit user approval."
        )


# ============================================================
# FINAL FOOTER
# ============================================================

st.markdown("---")

footer_mode = (
    "Real CSV telemetry"
    if normalized_df is not None
    else "Synthetic telemetry"
)

st.markdown(
    f"""
    <div class="footer-note">

        Vanguard-GWM Dashboard ·
        {footer_mode} ·
        FastAPI backend inference ·
        Defensive decision support

    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Configurable policy boundary for defensive decision support; "
    "not a statistically validated threat-severity cutoff."
)
