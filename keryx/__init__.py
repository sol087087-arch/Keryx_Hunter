# keryx/__init__.py
"""KeryxHunter - Sovereign air-gapped LLM vulnerability hunter.

PEP 562 lazy loading for heavy dependencies: importing keryx never loads
llama.cpp, FAISS, Docker, or any LLM library until they are actually used.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .core.agent import KeryxAgent
    from .core.orchestrator import KeryxOrchestrator
    from .core.shared_context import SharedContext

__version__       = "0.9.0"
__author__        = "KeryxHunter Contributors"
__license__       = "MIT"

_logger = logging.getLogger("keryx")


def get_version_tuple() -> tuple[int, int, int]:
    try:
        major, minor, patch = __version__.split(".")[:3]
        return int(major), int(minor), int(patch)
    except (ValueError, AttributeError):
        return (0, 0, 0)

__version_tuple__ = get_version_tuple()


# ---------------------------------------------------------------------------
# Eager imports — lightweight, always available
# FIX: ToolResult is NOT imported from schemas.
#      It lives in toolbox (dataclass) so the agent observation contract holds.
#      schemas exports input validators and typed result records only.
# ---------------------------------------------------------------------------

from .core.schemas import (       # noqa: E402
    GitBlameInput,
    HotspotEntry,
    MetricsSnapshot,
    RiskLevel,
)
from .core.shared_context import SharedContext  # noqa: E402

# ToolResult from toolbox — the one the whole pipeline uses.
from .tools.Toolbox import ToolResult           # noqa: E402


# ---------------------------------------------------------------------------
# Lazy loading map
# ---------------------------------------------------------------------------

_LAZY: dict[str, str] = {
    # Core
    "KeryxOrchestrator":     ".core.orchestrator",
    "KeryxAgent":            ".core.agent",
    "CapabilityRouter":      ".core.router",

    # Models
    "LocalModel":            ".models.local",
    "load_local_model":      ".models.local",
    "RemoteModel":           ".models.remote",
    "create_anthropic_model":".models.remote",
    "create_openai_model":   ".models.remote",
    "create_groq_model":     ".models.remote",
    "SwarmDebate":           ".models.swarm",
    "create_swarm_debate":   ".models.swarm",

    # Tools
    "GitBlameTool":          ".tools.git_blame",
    "create_git_blame_tool": ".tools.git_blame",
    "RagSearchTool":         ".tools.rag_search",
    "create_rag_search_tool":".tools.rag_search",
    "FuzzerTool":            ".tools.fuzzer",
    "GDBTool":               ".tools.gdb",
    "CodeQLTool":            ".tools.codeql",
    "ToolBox":               ".tools.Toolbox",

    # Advisors
    "AdvisorManager":        ".advisors.manager",
    "LocalAdvisor":          ".advisors.local_advisor",
    "CloudAdvisor":          ".advisors.cloud_advisor",

    # Knowledge
    "VulnerabilityGraph":    ".knowledge.graph",
    "LocalEmbedder":         ".knowledge.embeddings",
    "create_embedder":       ".knowledge.embeddings",

    # Sandbox
    "IsolatedContainer":     ".sandbox.container",
}


def __getattr__(name: str) -> Any:
    """PEP 562: resolve heavy symbols on first access."""
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name], package="keryx")
        obj    = getattr(module, name)
        # Cache in module globals so subsequent access is O(1).
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

async def hunt(
    target_path:  str,
    capability:   str = "deep_reasoning",
    mode:         str = "hybrid",
    models:       dict[str, Any] | None = None,
    toolbox:      Any | None = None,
    advisors:     Any | None = None,
    **kwargs:     Any,
) -> dict[str, Any]:
    """
    One-line sovereign entry point.

    Args:
        target_path: Path to the codebase to audit.
        capability:  Routing profile — "deep_reasoning" | "fast_scan" | ...
        mode:        "airgapped" | "hybrid" | "cloud"
        models:      Dict[name -> ModelInterface]. If None, auto-detected.
        toolbox:     Pre-built ToolBox. If None, default toolbox is created.
        advisors:    Pre-built AdvisorManager. If None, default is created.

    Example:
        result = await keryx.hunt("/path/to/firefox", mode="airgapped")
    """
    # FIX: lazy imports keep startup fast; imported here not at module level.
    from .core.orchestrator  import KeryxOrchestrator
    from .tools.Toolbox      import ToolBox
    from .advisors.manager   import AdvisorManager

    _toolbox  = toolbox  or ToolBox()
    _advisors = advisors or AdvisorManager()
    _models   = models   or {}

    # FIX: KeryxOrchestrator requires available_models as first positional arg.
    orchestrator = KeryxOrchestrator(
        available_models=_models,
        advisor_manager=_advisors,
        toolbox=_toolbox,
        **kwargs,
    )

    return await orchestrator.hunt(
        target_path=target_path,
        capability=capability,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Meta
    "__version__",
    "__version_tuple__",
    "__author__",
    "__license__",
    "get_version_tuple",

    # Eager — schemas
    "RiskLevel",
    "HotspotEntry",
    "GitBlameInput",
    "MetricsSnapshot",

    # Eager — runtime
    "SharedContext",
    "ToolResult",

    # Lazy — core
    "KeryxOrchestrator",
    "KeryxAgent",
    "CapabilityRouter",

    # Lazy — tools
    "GitBlameTool",
    "create_git_blame_tool",
    "RagSearchTool",
    "create_rag_search_tool",
    "ToolBox",

    # Lazy — knowledge
    "LocalEmbedder",
    "create_embedder",
    "VulnerabilityGraph",

    # Entry point
    "hunt",
]
