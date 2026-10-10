"""
sdn_enforcer.py - Explicit SDN/OpenFlow Policy Dispatcher for Vanguard-GWM

This module provides an integration boundary between Vanguard-GWM and a
Ryu/OpenFlow controller.

Important:
    Model inference does NOT automatically trigger enforcement.

    Enforcement methods must be explicitly called by an approved workflow,
    operator action, or future policy engine.

When the controller is unavailable, the module can operate in simulation
mode so that demonstrations and tests do not pretend that a real network
change occurred.
"""

from __future__ import annotations

import ipaddress
import logging
from typing import Any

import requests


logger = logging.getLogger("SDN-Enforcer")


class SDNEnforcer:
    """
    SDN/OpenFlow policy dispatcher.

    The class is intentionally kept separate from model inference so that
    forecasting and network enforcement remain independently controllable.

    Args:
        ryu_host:
            Base URL of the Ryu controller.
        dpid:
            Datapath ID represented as a hexadecimal string.
        timeout:
            Default HTTP timeout in seconds.
        simulation_mode:
            When True, unavailable-controller actions are treated as
            simulation/staging operations rather than real enforcement.
    """

    def __init__(
        self,
        ryu_host: str = "http://localhost:8080",
        dpid: str = "0000000000000001",
        timeout: float = 2.0,
        simulation_mode: bool = True,
    ) -> None:
        if not isinstance(ryu_host, str) or not ryu_host.strip():
            raise ValueError("ryu_host must be a non-empty string.")

        if not isinstance(dpid, str) or not dpid.strip():
            raise ValueError("dpid must be a non-empty hexadecimal string.")

        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        try:
            parsed_dpid = int(dpid, 16)
        except ValueError as exc:
            raise ValueError(
                f"Invalid DPID '{dpid}'. Expected a hexadecimal value."
            ) from exc

        if parsed_dpid < 0:
            raise ValueError("DPID cannot be negative.")

        self.ryu_host = ryu_host.rstrip("/")
        self.dpid = parsed_dpid
        self.timeout = float(timeout)
        self.simulation_mode = bool(simulation_mode)

        self.flow_entry_url = (
            f"{self.ryu_host}/stats/flowentry/add"
        )

        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_ip(target_ip: str) -> str:
        """Validate and normalize an IPv4/IPv6 address."""
        if not isinstance(target_ip, str) or not target_ip.strip():
            raise ValueError("target_ip must be a non-empty string.")

        try:
            return str(ipaddress.ip_address(target_ip.strip()))
        except ValueError as exc:
            raise ValueError(
                f"Invalid target IP address: '{target_ip}'."
            ) from exc

    @staticmethod
    def _validate_positive_int(
        value: int,
        field_name: str,
    ) -> int:
        """Validate a positive integer parameter."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer.")

        if value <= 0:
            raise ValueError(
                f"{field_name} must be greater than zero."
            )

        return value

    # ------------------------------------------------------------------
    # Controller status
    # ------------------------------------------------------------------

    def check_controller_status(self) -> str:
        """
        Check whether the configured Ryu controller is reachable.

        Returns:
            "Connected" when the controller returns HTTP 200.

            "Simulation Fallback" when the controller cannot be reached
            or returns a non-success response.

        Note:
            This method only checks connectivity. It does not modify
            network state.
        """
        try:
            response = self._session.get(
                f"{self.ryu_host}/stats/switches",
                timeout=self.timeout,
            )

            if response.status_code == 200:
                return "Connected"

            logger.warning(
                "Ryu controller returned HTTP %s.",
                response.status_code,
            )
            return "Simulation Fallback"

        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Ryu controller is unavailable: %s",
                exc,
            )
            return "Simulation Fallback"

    # ------------------------------------------------------------------
    # Quarantine policy
    # ------------------------------------------------------------------

    def push_quarantine_rule(
        self,
        target_ip: str,
        priority: int = 50000,
        hard_timeout: int = 300,
    ) -> bool:
        """
        Request a DROP flow rule for a target IP.

        This method only performs an enforcement request when the Ryu
        controller is actually reachable.

        If the controller is unavailable and simulation_mode=True, the
        action is logged as a simulation and True is returned for
        backwards compatibility with the demonstration pipeline.

        Returns:
            True when the request was accepted or a simulation action
            was staged.

            False when the controller rejected the request or when
            simulation_mode=False and the controller is unavailable.

        Raises:
            ValueError:
                If the target IP or policy parameters are invalid.
        """
        target_ip = self._validate_ip(target_ip)
        priority = self._validate_positive_int(priority, "priority")
        hard_timeout = self._validate_positive_int(
            hard_timeout,
            "hard_timeout",
        )

        payload: dict[str, Any] = {
            "dpid": self.dpid,
            "cookie": 1,
            "cookie_mask": 1,
            "table_id": 0,
            "idle_timeout": 0,
            "hard_timeout": hard_timeout,
            "priority": priority,
            "flags": 1,
            "match": {
                "eth_type": 0x0800,
                "ipv4_src": target_ip,
            },
            "actions": [],
        }

        try:
            response = self._session.post(
                self.flow_entry_url,
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                logger.info(
                    "Quarantine rule accepted for %s on DPID %s.",
                    target_ip,
                    self.dpid,
                )
                return True

            logger.error(
                "Quarantine rule rejected for %s: HTTP %s.",
                target_ip,
                response.status_code,
            )
            return False

        except requests.exceptions.RequestException as exc:
            if self.simulation_mode:
                logger.info(
                    "[Simulation] DROP rule staged for %s on DPID %s.",
                    target_ip,
                    self.dpid,
                )
                logger.debug(
                    "Controller request failed during simulation: %s",
                    exc,
                )
                return True

            logger.error(
                "Unable to push quarantine rule for %s: %s",
                target_ip,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Honeynet redirect policy
    # ------------------------------------------------------------------

    def push_honeynet_redirect(
        self,
        target_ip: str,
        honeynet_port: int = 4,
        priority: int = 50000,
    ) -> bool:
        """
        Request redirection of traffic from a target IP to a honeynet port.

        This method does not automatically execute as part of model
        inference. It is an explicit policy-dispatch operation.

        Returns:
            True when the request was accepted or a simulation action
            was staged.

            False when the controller rejected the request or when
            simulation_mode=False and the controller is unavailable.
        """
        target_ip = self._validate_ip(target_ip)
        honeynet_port = self._validate_positive_int(
            honeynet_port,
            "honeynet_port",
        )
        priority = self._validate_positive_int(priority, "priority")

        payload: dict[str, Any] = {
            "dpid": self.dpid,
            "priority": priority,
            "match": {
                "eth_type": 0x0800,
                "ipv4_src": target_ip,
            },
            "actions": [
                {
                    "type": "OUTPUT",
                    "port": honeynet_port,
                }
            ],
        }

        try:
            response = self._session.post(
                self.flow_entry_url,
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                logger.info(
                    "Honeynet redirect accepted for %s -> port %s "
                    "on DPID %s.",
                    target_ip,
                    honeynet_port,
                    self.dpid,
                )
                return True

            logger.error(
                "Honeynet redirect rejected for %s: HTTP %s.",
                target_ip,
                response.status_code,
            )
            return False

        except requests.exceptions.RequestException as exc:
            if self.simulation_mode:
                logger.info(
                    "[Simulation] REDIRECT rule staged for %s -> "
                    "port %s on DPID %s.",
                    target_ip,
                    honeynet_port,
                    self.dpid,
                )
                logger.debug(
                    "Controller request failed during simulation: %s",
                    exc,
                )
                return True

            logger.error(
                "Unable to push honeynet redirect for %s: %s",
                target_ip,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "SDNEnforcer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()