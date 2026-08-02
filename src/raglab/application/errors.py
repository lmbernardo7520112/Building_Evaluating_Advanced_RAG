"""Application-level errors."""

from __future__ import annotations


class RagLabApplicationError(Exception):
    """Base for application-layer errors."""


class PortNotConfiguredError(RagLabApplicationError):
    """Raised when a required port has no implementation configured."""

    def __init__(self, port_name: str) -> None:
        super().__init__(f"Port '{port_name}' has no implementation configured")


class RunNotFoundError(RagLabApplicationError):
    """Raised when a requested run does not exist."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run '{run_id}' not found")
