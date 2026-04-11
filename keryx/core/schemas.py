# keryx/core/schemas.py
"""Strict Pydantic schemas for KeryxHunter.

SCOPE: input validation and typed observability only.
Tool outputs use toolbox.ToolResult (dataclass) so the ToolBox dispatch
pipeline and agent observation contract remain intact.
Do NOT add ToolResult here — it lives in keryx.tools.toolbox.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


# ---------------------------------------------------------------------------
# Git tool input
# ---------------------------------------------------------------------------

class GitBlameInput(BaseModel):
    """Validated input for git operations."""

    command:     Literal["blame", "log", "hotspots"]
    file:        Optional[str]       = None
    line:        Optional[int]       = Field(None, ge=1)
    n:           int                 = Field(10, ge=1, le=1000)
    path:        Optional[str]       = None
    author:      Optional[str]       = None
    grep:        Optional[str]       = None
    since:       Optional[str]       = None
    extensions:  Optional[list[str]] = None
    min_changes: int                 = Field(2, ge=1)

    # FIX: @model_validator(mode="after") for cross-field validation.
    # @field_validator cannot reliably see sibling fields due to evaluation order.
    @model_validator(mode="after")
    def file_required_for_blame(self) -> "GitBlameInput":
        if self.command == "blame" and not self.file:
            raise ValueError("'file' is required when command='blame'")
        return self


# ---------------------------------------------------------------------------
# Hotspot entry (immutable result record)
# ---------------------------------------------------------------------------

class HotspotEntry(BaseModel):
    """A single hotspot file record. Frozen so callers cannot mutate scores."""

    file:             str
    changes:          int   = Field(..., ge=0)
    authors:          int   = Field(..., ge=0)
    complexity_score: float = Field(..., ge=0.0)
    risk_level:       RiskLevel
    # FIX: Field(max_length=3) applies to str, not list — silently ignored in Pydantic v2.
    # Use a validator to enforce the cap.
    top_authors: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}

    @field_validator("top_authors", mode="before")
    @classmethod
    def cap_top_authors(cls, v: list[str]) -> list[str]:
        return list(v)[:3]


# ---------------------------------------------------------------------------
# Metrics snapshot
# ---------------------------------------------------------------------------

class MetricsSnapshot(BaseModel):
    """Typed tool metrics. Replaces Dict[str, Any] in observability paths."""

    name:         str
    calls:        int   = Field(..., ge=0)
    successes:    int   = Field(..., ge=0)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    avg_time_ms:  float = Field(0.0, ge=0.0)

    @field_validator("success_rate", mode="before")
    @classmethod
    def round_rate(cls, v: float) -> float:
        return round(float(v), 4)
