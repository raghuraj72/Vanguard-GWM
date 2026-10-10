import time

import altair as alt
import pandas as pd
import streamlit as st
import torch

from src.data_pipeline import StreamingTelemetryParser
from src.engine import VanguardEngine
from src.sdn_enforcer import SDNEnforcer


st.set_page_config(
    page_title="Vanguard-GWM | SIH Threat Forecasting SOC",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM UI STYLING
# ============================================================

st.markdown(
    """
    <style>
        div[data-testid="stMetric"] {
            background-color: #0f172a !important;
            border: 1px solid #334155 !important;
            border-radius: 10px !important;
            padding: 14px 18px !important;
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] label p {
            color: #94a3b8 !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }

        div[data-testid="stMetric"] div[data-testid="stMetricValue"],
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] div {
            color: #38bdf8 !important;
            font-weight: 700 !important;
            font-size: 1.6rem !important;
        }

        .risk-alert {
            padding: 12px 14px;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .risk-high {
            background-color: #450a0a;
            color: #fecaca;
            border: 1px solid #dc2626;
        }

        .risk-low {
            background-color: #064e3b;
            color: #a7f3d0;
            border: 1px solid #059669;
        }

        .info-box {
            background-color: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 12px 14px;
            margin-bottom: 10px;
        }

        .small-label {
            color: #94a3b8;
            font-size: 0.82rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ Vanguard-GWM: Generative Graph World Model")

st.caption(
    "SIH Project: AI-Based Network Attack Forecasting "
    "from Network Telemetry & World Models"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📁 Ingestion & Simulation Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CIC-IDS-2018 / PCAP / CSV Telemetry",
    type=["csv", "pcap"],
)

if uploaded_file is not None:
    st.sidebar.success(f"Loaded: {uploaded_file.name}")

    st.sidebar.caption(
        "Upload ingestion is currently reserved for the telemetry "
        "integration layer. The dashboard currently uses the "
        "deterministic synthetic graph generator."
    )


horizon_k = st.sidebar.slider(
    "Forecast Horizon (K-steps ahead)",
    min_value=1,
    max_value=8,
    value=5,
)


selected_window = st.sidebar.slider(
    "Temporal Slice Index (t)",
    min_value=0,
    max_value=20,
    value=8,
)


st.sidebar.markdown("---")

st.sidebar.subheader("🎯 Active Inspection Target")

target_node = st.sidebar.selectbox(
    "Target Host Node",
    [
        "192.168.1.105 (Web Server)",
        "192.168.1.110 (DB Server)",
        "10.0.0.5 (Domain Controller)",
        "10.0.0.22 (Dev Workstation)",
    ],
)

target_ip = target_node.split()[0]


# ============================================================
# VANGUARD ENGINE
# ============================================================

@st.cache_resource
def get_vanguard_engine(horizon: int):
    """
    Create and cache the actual Vanguard-GWM inference engine.

    The dashboard uses the model architecture implemented in
    src/engine.py rather than dashboard-only prediction logic.
    """

    return VanguardEngine(
        in_node_features=5,
        in_edge_features=5,
        latent_dim=64,
        horizon_k=horizon,
        risk_threshold=0.60,
    )


# ============================================================
# SYNTHETIC GRAPH GENERATOR
# ============================================================

@st.cache_data
def generate_graph_snapshot(window_index: int):
    """
    Generate a deterministic synthetic graph snapshot.

    The current repository uses synthetic telemetry generation
    rather than a production telemetry ingestion pipeline.
    """

    torch.manual_seed(42 + window_index)

    parser = StreamingTelemetryParser(
        window_size_sec=2.0,
    )

    graph = parser.synthesize_mock_window(
        num_nodes=12,
        num_edges=28,
    )

    return (
        graph.x,
        graph.edge_index,
        graph.edge_attr,
    )


# ============================================================
# VANGUARD INFERENCE
# ============================================================

def run_vanguard_inference(
    window_index: int,
    horizon: int,
    target: str,
):
    """
    Execute the actual Vanguard-GWM inference pipeline.

    Returns:
        result: Engine prediction dictionary
        latency_ms: Measured local inference latency
    """

    engine = get_vanguard_engine(horizon)

    x, edge_index, edge_attr = generate_graph_snapshot(
        window_index
    )

    start_time = time.perf_counter()

    result = engine.process_graph_snapshot(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        target_ip=target,
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000.0

    return result, latency_ms


# ============================================================
# RUN INFERENCE
# ============================================================

try:
    result, inference_latency_ms = run_vanguard_inference(
        selected_window,
        horizon_k,
        target_ip,
    )

    predicted_risk = result["risk_trajectory"]
    current_max_risk = result["peak_risk"]
    current_stage = result["current_stage"]

except Exception as exc:
    st.error(
        "Vanguard-GWM inference failed. "
        "Check the Python environment and model dependencies."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# COUNTERFACTUAL RISK VISUALIZATION
# ============================================================

future_steps = [
    f"t+{k * 2}s"
    for k in range(1, horizon_k + 1)
]


mitigated_risk = [
    max(0.05, risk * 0.28)
    for risk in predicted_risk
]


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:
    risk_delta = (
        current_max_risk - 0.20
    ) * 100

    st.metric(
        "Predicted Peak Risk",
        f"{current_max_risk * 100:.1f}%",
        delta=f"{risk_delta:+.1f}% vs reference",
        delta_color="inverse",
    )


with col2:
    st.metric(
        "MITRE ATT&CK Phase",
        current_stage,
    )


with col3:
    st.metric(
        "Forecast Horizon",
        f"{horizon_k} steps",
        delta="2s / step",
    )


with col4:
    st.metric(
        "Model Inference Latency",
        f"{inference_latency_ms:.2f} ms",
        delta="Measured locally",
    )


# ============================================================
# FORECAST + WHAT-IF SECTION
# ============================================================

st.markdown("---")

col_left, col_right = st.columns([3, 2])


# ============================================================
# FORECAST CHART
# ============================================================

with col_left:

    st.subheader(
        "📈 Autoregressive K-Step Risk Forecast Curve"
    )

    chart_df = pd.DataFrame(
        {
            "Horizon Step": (
                future_steps + future_steps
            ),

            "Risk Probability": (
                predicted_risk + mitigated_risk
            ),

            "Scenario": (
                ["Model Forecast"] * horizon_k
                +
                ["Counterfactual (Quarantine Active)"]
                * horizon_k
            ),
        }
    )


    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Horizon Step:N",
                title="Projected Future Steps (Δt = 2s)",
            ),

            y=alt.Y(
                "Risk Probability:Q",
                title="Model Risk Score",
                scale=alt.Scale(
                    domain=[0, 1]
                ),
            ),

            color=alt.Color(
                "Scenario:N",
                scale=alt.Scale(
                    domain=[
                        "Model Forecast",
                        "Counterfactual (Quarantine Active)",
                    ],
                    range=[
                        "#ef4444",
                        "#10b981",
                    ],
                ),
            ),

            tooltip=[
                "Horizon Step",
                "Scenario",
                alt.Tooltip(
                    "Risk Probability:Q",
                    format=".2%",
                ),
            ],
        )
        .properties(
            height=320
        )
    )


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


    st.caption(
        "The red trajectory is produced by the current "
        "Vanguard-GWM model. The green trajectory is a "
        "counterfactual quarantine visualization."
    )


# ============================================================
# DEFENSIVE POLICY SANDBOX
# ============================================================

with col_right:

    st.subheader(
        "🧪 'What-If' Defensive Policy Sandbox"
    )


    if current_max_risk >= 0.60:

        st.markdown(
            f"""
            <div class="risk-alert risk-high">
                ⚠️ HIGH THREAT: Elevated model risk for
                {target_ip}
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div class="risk-alert risk-low">
                ✅ NOMINAL: Model risk below enforcement threshold
                for {target_ip}
            </div>
            """,
            unsafe_allow_html=True,
        )


    try:
        isolate_clicked = st.button(
            "⚡ Simulate Host Isolation (Graph Pruning)",
            width="stretch",
        )

    except TypeError:
        isolate_clicked = st.button(
            "⚡ Simulate Host Isolation (Graph Pruning)",
            use_container_width=True,
        )


    if isolate_clicked:

        st.success(
            f"Isolation simulation requested for {target_ip}"
        )


        risk_reduction = max(
            0.0,
            (
                current_max_risk
                - mitigated_risk[-1]
            ) * 100,
        )


        st.metric(
            "Projected Risk Reduction",
            f"-{risk_reduction:.1f}%",
            delta="Counterfactual simulation",
        )


        try:

            enforcer = SDNEnforcer()

            enforcement_result = (
                enforcer.push_quarantine_rule(
                    target_ip=target_ip
                )
            )


            if enforcement_result:

                st.caption(
                    "✅ SDN quarantine action accepted by "
                    "the enforcement layer or simulation fallback."
                )

            else:

                st.caption(
                    "⚠️ SDN controller rejected the quarantine rule."
                )


        except Exception as exc:

            st.caption(
                "⚠️ SDN enforcement unavailable: "
                f"{type(exc).__name__}"
            )


# ============================================================
# MODEL CONFIGURATION + TELEMETRY
# ============================================================

st.markdown("---")

col_exp1, col_exp2 = st.columns([1, 1])


# ============================================================
# MODEL CONFIGURATION
# ============================================================

with col_exp1:

    st.subheader(
        "🧠 Vanguard-GWM Model Configuration"
    )

    st.caption(
        "Configuration of the model components used for "
        "the current inference request."
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
            ],

            "Value": [
                "5",
                "5",
                "64",
                "2",
                str(horizon_k),
                "8 snapshots",
            ],
        }
    )


    try:

        st.dataframe(
            model_config,
            hide_index=True,
            width="stretch",
        )

    except TypeError:

        st.dataframe(
            model_config,
            hide_index=True,
            use_container_width=True,
        )


    st.info(
        "The current model is an architecture-level inference "
        "pipeline operating on synthetic telemetry graphs. "
        "Risk scores should not be interpreted as calibrated "
        "real-world attack probabilities until trained and "
        "validated on representative data."
    )


# ============================================================
# TELEMETRY FEATURE OVERVIEW
# ============================================================

with col_exp2:

    st.subheader(
        "🔍 Telemetry Feature Overview"
    )

    st.caption(
        "Input features consumed by the graph encoder."
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


    try:

        st.dataframe(
            feature_df,
            hide_index=True,
            width="stretch",
        )

    except TypeError:

        st.dataframe(
            feature_df,
            hide_index=True,
            use_container_width=True,
        )


    st.caption(
        "Edge attributes additionally include flag bitmask, "
        "log bytes, inter-arrival-time mean, inter-arrival-time "
        "variance, and TTL variance."
    )


# ============================================================
# NETWORK TOPOLOGY
# ============================================================

st.markdown("---")

st.subheader(
    "🌐 Monitored Network Graph Topology ($G_t$)"
)


table_data = {
    "Source IP": [
        "192.168.1.105",
        "192.168.1.105",
        "10.0.0.5",
        "172.16.0.4",
        "192.168.1.200",
    ],

    "Destination IP": [
        "10.0.0.5",
        "192.168.1.110",
        "10.0.0.22",
        "192.168.1.105",
        "8.8.8.8",
    ],

    "Protocol": [
        "TCP / 445 (SMB)",
        "TCP / 3389 (RDP)",
        "TCP / 88 (Kerb)",
        "TCP / 443",
        "UDP / 53",
    ],

    "Flag Vector": [
        "[SYN, ACK]",
        "[SYN]",
        "[ACK, PSH]",
        "[ACK]",
        "[None]",
    ],

    "IAT Var (ms²)": [
        0.002,
        0.001,
        14.2,
        8.5,
        0.1,
    ],

    "Anomaly Score": [
        "0.94 (Critical)",
        "0.88 (High)",
        "0.12 (Low)",
        "0.04 (Low)",
        "0.01 (Low)",
    ],
}


topology_df = pd.DataFrame(
    table_data
)


try:

    st.dataframe(
        topology_df,
        hide_index=True,
        width="stretch",
    )

except TypeError:

    st.dataframe(
        topology_df,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

st.markdown("---")

st.subheader(
    "🟢 System Status"
)


status_col1, status_col2, status_col3 = st.columns(3)


# ============================================================
# TELEMETRY STATUS
# ============================================================

with status_col1:

    st.metric(
        "Telemetry Graph",
        "Synthetic",
    )


# ============================================================
# INFERENCE ENGINE STATUS
# ============================================================

with status_col2:

    st.metric(
        "Inference Engine",
        "Online",
    )


# ============================================================
# SDN CONTROLLER STATUS
# ============================================================

with status_col3:

    try:

        sdn_status = (
            SDNEnforcer()
            .check_controller_status()
        )

        if sdn_status == "Connected":

            status_label = "🟢 Connected"

        else:

            status_label = "🟡 Simulation"


    except Exception:

        status_label = "🟡 Simulation"


    st.metric(
        "SDN Enforcement",
        status_label,
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Vanguard-GWM dashboard | Synthetic telemetry mode | "
    "Model inference executed locally"
)