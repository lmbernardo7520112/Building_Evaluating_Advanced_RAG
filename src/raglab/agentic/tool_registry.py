"""Tool registry — read-only capability registry with allowlist enforcement."""

from __future__ import annotations

from raglab.agentic.contracts import ToolSpecification, _canonical_json, _sha256
from raglab.agentic.errors import UnknownToolError
from raglab.agentic.router import ALLOWED_STRATEGIES


def _make_retrieval_tool_spec(strategy: str) -> ToolSpecification:
    """Create a retrieval tool specification for a given strategy."""
    impl_hash = _sha256(
        _canonical_json(
            {
                "type": "retrieval",
                "strategy": strategy,
                "version": "1.0.0",
            }
        )
    )
    return ToolSpecification(
        tool_id=f"retrieve_{strategy}",
        version="1.0.0",
        description=f"Retrieve evidence using the {strategy} strategy",
        read_only=True,
        network_access=False,
        deterministic=True,
        max_top_k=10,
        timeout_seconds=30.0,
        allowed_strategies=(strategy,),
        implementation_sha256=impl_hash,
    )


class ToolRegistry:
    """Governed, read-only registry of available tools.

    Tools are registered at initialization and cannot be modified.
    Only read-only tools without network access are permitted.

    Not exposed:
    - shell, filesystem, browser, network
    - qrels, gold answers, holdout, manifests
    - administrative or write tools
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpecification] = {}
        self._frozen = False

    def register(self, spec: ToolSpecification) -> None:
        """Register a tool specification.

        Raises ValueError if registry is frozen or tool violates constraints.
        """
        if self._frozen:
            raise ValueError("Registry is frozen — cannot register new tools")
        if not spec.read_only:
            raise ValueError(f"Tool '{spec.tool_id}' is not read-only")
        if spec.network_access:
            raise ValueError(f"Tool '{spec.tool_id}' requires network access")
        if spec.tool_id in self._tools:
            raise ValueError(f"Tool '{spec.tool_id}' already registered")
        self._tools[spec.tool_id] = spec

    def freeze(self) -> None:
        """Freeze the registry — no further registrations allowed."""
        self._frozen = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def get(self, tool_id: str) -> ToolSpecification:
        """Get a tool by ID. Raises UnknownToolError if not found."""
        if tool_id not in self._tools:
            raise UnknownToolError(tool_id)
        return self._tools[tool_id]

    def has(self, tool_id: str) -> bool:
        """Check if a tool is registered."""
        return tool_id in self._tools

    @property
    def tool_ids(self) -> frozenset[str]:
        """Return all registered tool IDs."""
        return frozenset(self._tools.keys())

    def registry_hash(self) -> str:
        """Compute deterministic hash of the entire registry state."""
        specs = {
            tid: {
                "version": s.version,
                "read_only": s.read_only,
                "network_access": s.network_access,
                "deterministic": s.deterministic,
                "max_top_k": s.max_top_k,
                "timeout_seconds": s.timeout_seconds,
                "allowed_strategies": list(s.allowed_strategies),
                "implementation_sha256": s.implementation_sha256,
            }
            for tid, s in sorted(self._tools.items())
        }
        return _sha256(_canonical_json(specs))


def build_default_registry() -> ToolRegistry:
    """Build the default tool registry with all seven retrieval strategies.

    Returns a FROZEN registry — no modifications allowed after creation.
    """
    registry = ToolRegistry()
    for strategy in sorted(ALLOWED_STRATEGIES):
        registry.register(_make_retrieval_tool_spec(strategy))
    registry.freeze()
    return registry
