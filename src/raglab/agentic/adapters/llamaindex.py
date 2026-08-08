"""Optional LlamaIndex adapter for the agentic track.

This adapter provides an OPTIONAL integration point with LlamaIndex.
In Slice 5A, it serves as an availability barrier only.

Design decisions:
- The agentic domain (contracts, enums, errors, router, etc.) has ZERO
  imports from llama-index-core. This adapter is the ONLY module that
  may import LlamaIndex, and only at function-call time.
- No Settings global is used.
- No nest_asyncio is used.
- No OpenAI dependency is introduced.
- No model execution occurs.
- No network calls are made.

If llama-index-core is not installed, this module raises
OptionalBackendNotAvailableError with a clear message.

Status: AVAILABILITY_BARRIER_ONLY (Slice 5A)
Future: Will provide RetrievalBackend implementation wrapping
        existing LlamaIndex retrieval adapters (Slice 5B+).
"""

from __future__ import annotations

from raglab.agentic.errors import OptionalBackendNotAvailableError

_BACKEND_NAME = "llama-index-core"


def check_llamaindex_available() -> bool:
    """Check if llama-index-core is importable.

    Returns True if available, False otherwise.
    Does NOT import at module level to avoid polluting the domain.
    """
    try:
        import importlib

        importlib.import_module("llama_index.core")
        return True
    except ImportError:
        return False


def get_llamaindex_version() -> str:
    """Return the installed llama-index-core version.

    Raises OptionalBackendNotAvailableError if not installed.
    """
    try:
        import importlib.metadata as meta

        return meta.version("llama-index-core")
    except Exception:
        raise OptionalBackendNotAvailableError(_BACKEND_NAME) from None


def require_llamaindex() -> None:
    """Assert that llama-index-core is available.

    Raises OptionalBackendNotAvailableError with a structured
    error message if the backend is not installed.

    Use this at the entry point of any LlamaIndex-dependent code path.
    """
    if not check_llamaindex_available():
        raise OptionalBackendNotAvailableError(_BACKEND_NAME)
