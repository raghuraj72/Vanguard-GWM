from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# ============================================================
# Network Event
# ============================================================

@dataclass
class NetworkEvent:
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int = 0
    dst_port: int = 0
    protocol: str = "UNKNOWN"
    packets: float = 1.0
    bytes: float = 0.0
    duration: float = 0.0
    anomaly_score: float = 0.0
    mitre_stage: str = "Unknown"


# ============================================================
# Streaming Telemetry Parser
# ============================================================

class StreamingTelemetryParser:
    """
    Converts network telemetry into temporal PyTorch Geometric
    graphs.

    Supported telemetry schemas include:

    timestamp, source_ip, destination_ip, protocol,
    packet_size, mitre_stage, anomaly_score

    and traditional network-flow schemas containing:

    src_ip, dst_ip, packets, bytes, duration, ports, etc.
    """

    MITRE_STAGES = [
        "Reconnaissance",
        "Resource Development",
        "Initial Access",
        "Execution",
        "Persistence",
        "Privilege Escalation",
        "Defense Evasion",
        "Credential Access",
        "Discovery",
        "Lateral Movement",
        "Collection",
        "Command and Control",
        "Exfiltration",
        "Impact",
        "Unknown",
    ]

    PROTOCOL_MAP = {
        "UNKNOWN": 0.0,
        "TCP": 1.0,
        "UDP": 2.0,
        "ICMP": 3.0,
        "HTTP": 4.0,
        "HTTPS": 5.0,
        "SSH": 6.0,
        "SMB": 7.0,
        "RDP": 8.0,
        "DNS": 9.0,
        "FTP": 10.0,
        "SMTP": 11.0,
        "LDAP": 12.0,
        "KERBEROS": 13.0,
    }

    MITRE_STAGE_MAP = {
        stage: float(index)
        for index, stage in enumerate(MITRE_STAGES)
    }

    def __init__(self, window_size_sec: float = 2.0):
        self.window_size_sec = float(window_size_sec)

    # ========================================================
    # Synthetic graph
    # ========================================================

    def synthesize_mock_window(
        self,
        num_nodes: int = 10,
        num_edges: int = 20,
        seed: Optional[int] = None,
    ) -> Data:
        """
        Generates a deterministic synthetic graph.

        The synthetic generator guarantees:

        - exactly num_nodes nodes
        - exactly num_edges directed edges
        - no self-loops
        - no duplicate directed edges
        - exactly one edge-attribute row per edge

        Used only when no external telemetry is supplied.
        """

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng(42)

        num_nodes = max(2, int(num_nodes))
        num_edges = max(1, int(num_edges))

        # Maximum number of unique directed edges without
        # self-loops:
        #
        # N * (N - 1)
        #
        max_possible_edges = num_nodes * (num_nodes - 1)

        if num_edges > max_possible_edges:
            raise ValueError(
                f"Cannot create {num_edges} unique directed "
                f"edges with {num_nodes} nodes. Maximum is "
                f"{max_possible_edges}."
            )

        # ----------------------------------------------------
        # Node features
        #
        # [in_degree,
        #  out_degree,
        #  avg_bytes,
        #  port_entropy,
        #  role_id]
        # ----------------------------------------------------

        x = rng.random(
            (num_nodes, 5),
            dtype=np.float32,
        )

        x[:, 0] *= 5.0
        x[:, 1] *= 5.0
        x[:, 2] *= 10.0
        x[:, 3] *= 3.0

        x[:, 4] = rng.integers(
            low=0,
            high=4,
            size=num_nodes,
        )

        # ----------------------------------------------------
        # Directed edges
        #
        # Generate every possible directed edge except
        # self-loops, then sample exactly num_edges without
        # replacement.
        # ----------------------------------------------------

        possible_edges = np.array(
            [
                (src_node, dst_node)
                for src_node in range(num_nodes)
                for dst_node in range(num_nodes)
                if src_node != dst_node
            ],
            dtype=np.int64,
        )

        selected_indices = rng.choice(
            len(possible_edges),
            size=num_edges,
            replace=False,
        )

        selected_edges = possible_edges[
            selected_indices
        ]

        src = selected_edges[:, 0]
        dst = selected_edges[:, 1]

        edge_index = torch.tensor(
            np.vstack([src, dst]),
            dtype=torch.long,
        )

        actual_edges = edge_index.shape[1]

        # Defensive invariant.
        if actual_edges != num_edges:
            raise RuntimeError(
                "Synthetic graph generation failed to "
                "produce the requested number of edges."
            )

        # ----------------------------------------------------
        # Edge features
        #
        # [protocol,
        #  log_bytes,
        #  iat_mean,
        #  iat_variance,
        #  ttl_variance]
        # ----------------------------------------------------

        edge_attr = rng.random(
            (actual_edges, 5),
            dtype=np.float32,
        )

        edge_attr[:, 0] *= 4.0
        edge_attr[:, 1] *= 10.0
        edge_attr[:, 2] *= 2.0
        edge_attr[:, 3] *= 2.0
        edge_attr[:, 4] *= 2.0

        graph = Data(
            x=torch.tensor(
                x,
                dtype=torch.float32,
            ),
            edge_index=edge_index,
            edge_attr=torch.tensor(
                edge_attr,
                dtype=torch.float32,
            ),
        )

        graph.node_ips = [
            f"10.0.0.{i + 1}"
            for i in range(num_nodes)
        ]

        graph.window_start = 0.0
        graph.window_end = self.window_size_sec

        graph.mitre_stage = "Unknown"

        return graph

    # ========================================================
    # CSV loading
    # ========================================================

    def load_csv(self, file_or_path) -> pd.DataFrame:
        """
        Load telemetry from a CSV file or Streamlit UploadedFile.
        """

        if isinstance(file_or_path, pd.DataFrame):
            return file_or_path.copy()

        if hasattr(file_or_path, "seek"):
            try:
                file_or_path.seek(0)
            except Exception:
                pass

        df = pd.read_csv(
            file_or_path,
            low_memory=False,
        )

        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        return df

    # ========================================================
    # Column discovery
    # ========================================================

    @staticmethod
    def _find_column(
        df: pd.DataFrame,
        candidates: list[str],
    ) -> Optional[str]:
        """
        Finds a dataframe column using case-insensitive
        normalized matching.
        """

        normalized = {
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "_"): column
            for column in df.columns
        }

        for candidate in candidates:
            key = (
                str(candidate)
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
                .replace(".", "_")
            )

            if key in normalized:
                return normalized[key]

        return None

    # ========================================================
    # Protocol normalization
    # ========================================================

    @classmethod
    def normalize_protocol(cls, value) -> float:
        """
        Converts protocol names into stable numeric IDs.
        """

        if pd.isna(value):
            return cls.PROTOCOL_MAP["UNKNOWN"]

        text = str(value).strip().upper()

        if text in cls.PROTOCOL_MAP:
            return cls.PROTOCOL_MAP[text]

        # Handle strings such as:
        # TCP/443, UDP/53, etc.
        if text.startswith("TCP"):
            return cls.PROTOCOL_MAP["TCP"]

        if text.startswith("UDP"):
            return cls.PROTOCOL_MAP["UDP"]

        if text.startswith("ICMP"):
            return cls.PROTOCOL_MAP["ICMP"]

        # Stable fallback for unknown protocols.
        return float(
            abs(hash(text)) % 1000
        ) / 1000.0

    # ========================================================
    # MITRE stage normalization
    # ========================================================

    @classmethod
    def normalize_mitre_stage(cls, value) -> str:
        if pd.isna(value):
            return "Unknown"

        text = str(value).strip()

        if not text:
            return "Unknown"

        # Exact match.
        for stage in cls.MITRE_STAGES:
            if text.lower() == stage.lower():
                return stage

        # Partial matching.
        lowered = text.lower()

        for stage in cls.MITRE_STAGES:
            if stage.lower() in lowered:
                return stage

        return "Unknown"

    @classmethod
    def mitre_stage_id(cls, value) -> float:
        stage = cls.normalize_mitre_stage(value)

        return cls.MITRE_STAGE_MAP.get(
            stage,
            cls.MITRE_STAGE_MAP["Unknown"],
        )

    # ========================================================
    # Normalize telemetry dataframe
    # ========================================================

    def normalize_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Converts different telemetry schemas into a canonical
        dataframe.

        Canonical columns:

        timestamp
        src_ip
        dst_ip
        src_port
        dst_port
        protocol
        packets
        bytes
        duration
        anomaly_score
        mitre_stage
        """

        if df is None or df.empty:
            raise ValueError(
                "Telemetry CSV is empty."
            )

        df = df.copy()

        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        # ----------------------------------------------------
        # Source / destination
        # ----------------------------------------------------

        src_col = self._find_column(
            df,
            [
                "source_ip",
                "src_ip",
                "source",
                "src",
                "Source IP",
                "Src IP",
                "Source",
            ],
        )

        dst_col = self._find_column(
            df,
            [
                "destination_ip",
                "dst_ip",
                "destination",
                "dst",
                "Destination IP",
                "Dst IP",
                "Destination",
            ],
        )

        if src_col is None:
            raise ValueError(
                "Could not find a source IP column. "
                "Expected source_ip or src_ip."
            )

        if dst_col is None:
            raise ValueError(
                "Could not find a destination IP column. "
                "Expected destination_ip or dst_ip."
            )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp_col = self._find_column(
            df,
            [
                "timestamp",
                "time",
                "datetime",
                "date_time",
                "flow_start",
                "start_time",
            ],
        )

        if timestamp_col is not None:

            parsed_timestamp = pd.to_datetime(
                df[timestamp_col],
                errors="coerce",
                utc=True,
            )

            if parsed_timestamp.notna().any():

                timestamp_seconds = (
                    parsed_timestamp.astype("int64")
                    / 1_000_000_000
                )

                timestamp_seconds = (
                    timestamp_seconds.where(
                        parsed_timestamp.notna(),
                        np.nan,
                    )
                )

            else:

                timestamp_seconds = pd.to_numeric(
                    df[timestamp_col],
                    errors="coerce",
                )

        else:

            timestamp_seconds = pd.Series(
                np.arange(
                    len(df),
                    dtype=float,
                )
            )

        # ----------------------------------------------------
        # Ports
        # ----------------------------------------------------

        src_port_col = self._find_column(
            df,
            [
                "source_port",
                "src_port",
                "sport",
                "source_port_number",
            ],
        )

        dst_port_col = self._find_column(
            df,
            [
                "destination_port",
                "dst_port",
                "dport",
                "destination_port_number",
            ],
        )

        # ----------------------------------------------------
        # Protocol
        # ----------------------------------------------------

        protocol_col = self._find_column(
            df,
            [
                "protocol",
                "proto",
                "protocol_name",
            ],
        )

        # ----------------------------------------------------
        # Packet count
        # ----------------------------------------------------

        packets_col = self._find_column(
            df,
            [
                "packets",
                "packet_count",
                "total_packets",
                "packet_count_total",
            ],
        )

        # ----------------------------------------------------
        # Bytes / packet size
        # ----------------------------------------------------

        bytes_col = self._find_column(
            df,
            [
                "bytes",
                "total_bytes",
                "flow_bytes",
                "packet_size",
                "length",
                "size",
            ],
        )

        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        duration_col = self._find_column(
            df,
            [
                "duration",
                "flow_duration",
                "duration_ms",
            ],
        )

        # ----------------------------------------------------
        # Anomaly score
        # ----------------------------------------------------

        anomaly_col = self._find_column(
            df,
            [
                "anomaly_score",
                "anomaly",
                "anomaly_probability",
                "risk_score",
            ],
        )

        # ----------------------------------------------------
        # MITRE stage
        # ----------------------------------------------------

        mitre_col = self._find_column(
            df,
            [
                "mitre_stage",
                "mitre_attack_stage",
                "attack_stage",
                "stage",
            ],
        )

        # ----------------------------------------------------
        # Build canonical dataframe
        # ----------------------------------------------------

        normalized = pd.DataFrame()

        normalized["timestamp"] = pd.to_numeric(
            timestamp_seconds,
            errors="coerce",
        )

        normalized["src_ip"] = (
            df[src_col]
            .astype(str)
            .str.strip()
        )

        normalized["dst_ip"] = (
            df[dst_col]
            .astype(str)
            .str.strip()
        )

        # ----------------------------------------------------
        # Ports
        # ----------------------------------------------------

        if src_port_col is not None:

            normalized["src_port"] = pd.to_numeric(
                df[src_port_col],
                errors="coerce",
            ).fillna(0)

        else:

            normalized["src_port"] = 0.0

        if dst_port_col is not None:

            normalized["dst_port"] = pd.to_numeric(
                df[dst_port_col],
                errors="coerce",
            ).fillna(0)

        else:

            normalized["dst_port"] = 0.0

        # ----------------------------------------------------
        # Protocol
        # ----------------------------------------------------

        if protocol_col is not None:

            normalized["protocol"] = df[
                protocol_col
            ].apply(
                self.normalize_protocol
            )

        else:

            normalized["protocol"] = 0.0

        # ----------------------------------------------------
        # Packets
        # ----------------------------------------------------

        if packets_col is not None:

            normalized["packets"] = pd.to_numeric(
                df[packets_col],
                errors="coerce",
            ).fillna(1.0)

        else:

            normalized["packets"] = 1.0

        # ----------------------------------------------------
        # Bytes / packet size
        # ----------------------------------------------------

        if bytes_col is not None:

            normalized["bytes"] = pd.to_numeric(
                df[bytes_col],
                errors="coerce",
            ).fillna(0.0)

        else:

            normalized["bytes"] = 0.0

        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        if duration_col is not None:

            normalized["duration"] = pd.to_numeric(
                df[duration_col],
                errors="coerce",
            ).fillna(0.0)

        else:

            normalized["duration"] = 0.0

        # ----------------------------------------------------
        # Anomaly score
        # ----------------------------------------------------

        if anomaly_col is not None:

            normalized["anomaly_score"] = pd.to_numeric(
                df[anomaly_col],
                errors="coerce",
            ).fillna(0.0)

        else:

            normalized["anomaly_score"] = 0.0

        normalized["anomaly_score"] = (
            normalized["anomaly_score"]
            .clip(0.0, 1.0)
        )

        # ----------------------------------------------------
        # MITRE stage
        # ----------------------------------------------------

        if mitre_col is not None:

            normalized["mitre_stage"] = df[
                mitre_col
            ].apply(
                self.normalize_mitre_stage
            )

        else:

            normalized["mitre_stage"] = "Unknown"

        # ----------------------------------------------------
        # Remove invalid IP rows
        # ----------------------------------------------------

        invalid_src = (
            normalized["src_ip"]
            .isin(
                [
                    "",
                    "nan",
                    "none",
                    "null",
                ]
            )
        )

        invalid_dst = (
            normalized["dst_ip"]
            .isin(
                [
                    "",
                    "nan",
                    "none",
                    "null",
                ]
            )
        )

        normalized = normalized[
            ~(invalid_src | invalid_dst)
        ].copy()

        # ----------------------------------------------------
        # Fix timestamps if necessary
        # ----------------------------------------------------

        if normalized["timestamp"].isna().all():

            normalized["timestamp"] = np.arange(
                len(normalized),
                dtype=float,
            )

        else:

            fallback_timestamp = (
                normalized["timestamp"]
                .dropna()
                .min()
            )

            normalized["timestamp"] = (
                normalized["timestamp"]
                .fillna(
                    fallback_timestamp
                )
            )

        # ----------------------------------------------------
        # Sort temporally
        # ----------------------------------------------------

        normalized = normalized.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        return normalized

    # ========================================================
    # Build graph from normalized dataframe
    # ========================================================

    def dataframe_to_graph(
        self,
        df: pd.DataFrame,
        window_index: int = 0,
    ) -> Data:
        """
        Converts one temporal slice into a graph.

        Node features:

        1. Incoming connection count
        2. Outgoing connection count
        3. Average bytes
        4. Protocol entropy
        5. Host role

        Edge features:

        1. Protocol
        2. log1p(bytes)
        3. Duration
        4. Packets
        5. Anomaly score

        Additional metadata contains MITRE stages and
        original telemetry information.
        """

        if df is None or df.empty:
            raise ValueError(
                "Cannot build graph from empty telemetry."
            )

        df = df.copy()

        required_columns = [
            "timestamp",
            "src_ip",
            "dst_ip",
            "protocol",
            "packets",
            "bytes",
            "duration",
            "anomaly_score",
            "mitre_stage",
        ]

        for column in required_columns:

            if column not in df.columns:

                if column == "protocol":
                    df[column] = 0.0

                elif column == "packets":
                    df[column] = 1.0

                elif column in [
                    "bytes",
                    "duration",
                    "anomaly_score",
                ]:
                    df[column] = 0.0

                elif column == "mitre_stage":
                    df[column] = "Unknown"

        # ----------------------------------------------------
        # Determine temporal window
        # ----------------------------------------------------

        min_timestamp = float(
            df["timestamp"].min()
        )

        requested_start = (
            min_timestamp
            + int(window_index)
            * self.window_size_sec
        )

        requested_end = (
            requested_start
            + self.window_size_sec
        )

        window_df = df[
            (df["timestamp"] >= requested_start)
            &
            (df["timestamp"] < requested_end)
        ].copy()

        # ----------------------------------------------------
        # If exact window has no events, use closest
        # available event window.
        # ----------------------------------------------------

        if window_df.empty:

            available_starts = (
                (
                    (
                        df["timestamp"]
                        - min_timestamp
                    )
                    // self.window_size_sec
                )
                .astype(int)
                .unique()
            )

            available_starts = sorted(
                available_starts.tolist()
            )

            if available_starts:

                closest_index = min(
                    available_starts,
                    key=lambda value: abs(
                        value
                        - int(window_index)
                    ),
                )

                requested_start = (
                    min_timestamp
                    + closest_index
                    * self.window_size_sec
                )

                requested_end = (
                    requested_start
                    + self.window_size_sec
                )

                window_df = df[
                    (df["timestamp"] >= requested_start)
                    &
                    (df["timestamp"] < requested_end)
                ].copy()

        # ----------------------------------------------------
        # Final fallback
        # ----------------------------------------------------

        if window_df.empty:

            window_df = (
                df.tail(
                    min(len(df), 1000)
                )
                .copy()
            )

            requested_start = float(
                window_df["timestamp"].min()
            )

            requested_end = float(
                window_df["timestamp"].max()
            )

        # ----------------------------------------------------
        # IP mapping
        # ----------------------------------------------------

        unique_ips = sorted(
            set(
                window_df["src_ip"].astype(str)
            )
            |
            set(
                window_df["dst_ip"].astype(str)
            )
        )

        if not unique_ips:
            raise ValueError(
                "No valid IP addresses found in telemetry."
            )

        node_to_id = {
            ip: index
            for index, ip in enumerate(unique_ips)
        }

        # ----------------------------------------------------
        # Node feature accumulation
        # ----------------------------------------------------

        num_nodes = len(unique_ips)

        incoming_count = np.zeros(
            num_nodes,
            dtype=np.float32,
        )

        outgoing_count = np.zeros(
            num_nodes,
            dtype=np.float32,
        )

        total_bytes = np.zeros(
            num_nodes,
            dtype=np.float32,
        )

        protocol_values = [
            []
            for _ in range(num_nodes)
        ]

        # ----------------------------------------------------
        # Edge construction
        # ----------------------------------------------------

        edge_sources = []
        edge_destinations = []
        edge_features = []
        edge_metadata = []

        for _, row in window_df.iterrows():

            src_ip = str(row["src_ip"])
            dst_ip = str(row["dst_ip"])

            if (
                src_ip not in node_to_id
                or dst_ip not in node_to_id
            ):
                continue

            src_id = node_to_id[src_ip]
            dst_id = node_to_id[dst_ip]

            packets = float(
                row.get(
                    "packets",
                    1.0,
                )
            )

            bytes_value = float(
                row.get(
                    "bytes",
                    0.0,
                )
            )

            duration = float(
                row.get(
                    "duration",
                    0.0,
                )
            )

            protocol = float(
                row.get(
                    "protocol",
                    0.0,
                )
            )

            anomaly_score = float(
                row.get(
                    "anomaly_score",
                    0.0,
                )
            )

            # ------------------------------------------------
            # Node aggregation
            # ------------------------------------------------

            outgoing_count[src_id] += 1.0
            incoming_count[dst_id] += 1.0

            total_bytes[src_id] += bytes_value
            total_bytes[dst_id] += bytes_value

            protocol_values[src_id].append(
                protocol
            )

            protocol_values[dst_id].append(
                protocol
            )

            # ------------------------------------------------
            # Edge
            # ------------------------------------------------

            edge_sources.append(src_id)
            edge_destinations.append(dst_id)

            edge_features.append(
                [
                    protocol,
                    np.log1p(
                        max(
                            0.0,
                            bytes_value,
                        )
                    ),
                    duration,
                    packets,
                    anomaly_score,
                ]
            )

            edge_metadata.append(
                {
                    "timestamp": float(
                        row["timestamp"]
                    ),
                    "source_ip": src_ip,
                    "destination_ip": dst_ip,
                    "anomaly_score": anomaly_score,
                    "mitre_stage": (
                        self.normalize_mitre_stage(
                            row.get(
                                "mitre_stage",
                                "Unknown",
                            )
                        )
                    ),
                    "protocol": protocol,
                    "packet_size": bytes_value,
                }
            )

        # ----------------------------------------------------
        # Protocol entropy
        # ----------------------------------------------------

        protocol_entropy = np.zeros(
            num_nodes,
            dtype=np.float32,
        )

        for node_id, values in enumerate(
            protocol_values
        ):

            if not values:
                continue

            _, counts = np.unique(
                values,
                return_counts=True,
            )

            probabilities = (
                counts / counts.sum()
            )

            entropy = -np.sum(
                probabilities
                * np.log2(
                    probabilities + 1e-12
                )
            )

            protocol_entropy[node_id] = float(
                entropy
            )

        # ----------------------------------------------------
        # Host role
        #
        # 0 = general host
        # 1 = source-heavy host
        # 2 = destination-heavy host
        # ----------------------------------------------------

        role_id = np.zeros(
            num_nodes,
            dtype=np.float32,
        )

        for node_id in range(num_nodes):

            if (
                outgoing_count[node_id]
                > incoming_count[node_id]
            ):

                role_id[node_id] = 1.0

            elif (
                incoming_count[node_id]
                > outgoing_count[node_id]
            ):

                role_id[node_id] = 2.0

        # ----------------------------------------------------
        # Node feature matrix
        # ----------------------------------------------------

        node_features = np.column_stack(
            [
                incoming_count,
                outgoing_count,
                total_bytes,
                protocol_entropy,
                role_id,
            ]
        ).astype(np.float32)

        # ----------------------------------------------------
        # Feature normalization
        # ----------------------------------------------------

        for column_index in range(
            node_features.shape[1]
        ):

            column = node_features[
                :,
                column_index,
            ]

            min_value = np.min(column)
            max_value = np.max(column)

            if max_value > min_value:

                node_features[
                    :,
                    column_index,
                ] = (
                    column - min_value
                ) / (
                    max_value - min_value
                )

        # ----------------------------------------------------
        # Edge tensors
        # ----------------------------------------------------

        if edge_sources:

            edge_index = torch.tensor(
                [
                    edge_sources,
                    edge_destinations,
                ],
                dtype=torch.long,
            )

            edge_attr = torch.tensor(
                edge_features,
                dtype=torch.float32,
            )

        else:

            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

            edge_attr = torch.empty(
                (0, 5),
                dtype=torch.float32,
            )

        # ----------------------------------------------------
        # Create graph
        # ----------------------------------------------------

        graph = Data(
            x=torch.tensor(
                node_features,
                dtype=torch.float32,
            ),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        graph.node_ips = unique_ips

        graph.window_start = float(
            requested_start
        )

        graph.window_end = float(
            requested_end
        )

        graph.telemetry_rows = int(
            len(window_df)
        )

        graph.edge_metadata = edge_metadata

        # ----------------------------------------------------
        # Dominant MITRE stage
        # ----------------------------------------------------

        stages = [
            item["mitre_stage"]
            for item in edge_metadata
            if item["mitre_stage"] != "Unknown"
        ]

        if stages:

            dominant_stage = (
                pd.Series(stages)
                .value_counts()
                .index[0]
            )

        else:

            dominant_stage = "Unknown"

        graph.mitre_stage = dominant_stage

        # ----------------------------------------------------
        # Maximum observed anomaly
        # ----------------------------------------------------

        if edge_metadata:

            graph.max_anomaly_score = float(
                max(
                    item["anomaly_score"]
                    for item in edge_metadata
                )
            )

            graph.mean_anomaly_score = float(
                np.mean(
                    [
                        item["anomaly_score"]
                        for item in edge_metadata
                    ]
                )
            )

        else:

            graph.max_anomaly_score = 0.0
            graph.mean_anomaly_score = 0.0

        return graph

    # ========================================================
    # CSV -> graph
    # ========================================================

    def csv_to_graph(
        self,
        file_or_path,
        window_index: int = 0,
    ) -> Data:
        """
        Complete ingestion pipeline:

        CSV
          ↓
        normalization
          ↓
        temporal graph
        """

        df = self.load_csv(
            file_or_path
        )

        normalized = self.normalize_dataframe(
            df
        )

        return self.dataframe_to_graph(
            normalized,
            window_index=window_index,
        )

    # ========================================================
    # Event extraction
    # ========================================================

    def dataframe_to_events(
        self,
        df: pd.DataFrame,
    ) -> list[NetworkEvent]:
        """
        Converts normalized telemetry into NetworkEvent
        objects.
        """

        normalized = self.normalize_dataframe(
            df
        )

        events = []

        for _, row in normalized.iterrows():

            events.append(
                NetworkEvent(
                    timestamp=float(
                        row["timestamp"]
                    ),
                    src_ip=str(
                        row["src_ip"]
                    ),
                    dst_ip=str(
                        row["dst_ip"]
                    ),
                    src_port=int(
                        row.get(
                            "src_port",
                            0,
                        )
                    ),
                    dst_port=int(
                        row.get(
                            "dst_port",
                            0,
                        )
                    ),
                    protocol=str(
                        row.get(
                            "protocol",
                            "UNKNOWN",
                        )
                    ),
                    packets=float(
                        row.get(
                            "packets",
                            1.0,
                        )
                    ),
                    bytes=float(
                        row.get(
                            "bytes",
                            0.0,
                        )
                    ),
                    duration=float(
                        row.get(
                            "duration",
                            0.0,
                        )
                    ),
                    anomaly_score=float(
                        row.get(
                            "anomaly_score",
                            0.0,
                        )
                    ),
                    mitre_stage=str(
                        row.get(
                            "mitre_stage",
                            "Unknown",
                        )
                    ),
                )
            )

        return events