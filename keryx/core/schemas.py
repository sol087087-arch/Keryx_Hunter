# keryx/core/schemas.py
"""Strict Pydantic schemas for KeryxHunter.

These schemas are used for INPUT VALIDATION only.
Tool outputs use toolbox.ToolResult (dataclass) so the ToolBox dispatch
pipeline and agent observation contract remain intact.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import FieldValidationInfo  # FIX: correct type for v2 validators


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


# ---------------------------------------------------------------------------
# Git tool input schema
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

    @model_validator(mode="after")
    def file_required_for_blame(self) -> "GitBlameInput":
        # FIX: use @model_validator(mode="after") — cleaner than @field_validator
        # for cross-field validation, and avoids the FieldValidationInfo.data
        # ordering issue (fields may not be populated yet in field validators).
        if self.command == "blame" and not self.file:
            raise ValueError("'file' is required for command='blame'")
        return self


# ---------------------------------------------------------------------------
# Hotspot entry (immutable)
# ---------------------------------------------------------------------------

class HotspotEntry(BaseModel):
    """A single hotspot file record."""

    file:             str
    changes:          int   = Field(..., ge=0)
    authors:          int   = Field(..., ge=0)
    complexity_score: float = Field(..., ge=0.0)
    risk_level:       RiskLevel
    # FIX: max_length on Field() applies to str, not list.
    # Use Annotated with a validator for list length enforcement.
    top_authors: Annotated[list[str], Field(default_factory=list)] = Field(
        default_factory=list
    )

    model_config = {"frozen": True}

    @field_validator("top_authors", mode="before")
    @classmethod
    def cap_top_authors(cls, v: list[str]) -> list[str]:
        return v[:3]


# ---------------------------------------------------------------------------
# Metrics snapshot (typed observability — replaces Dict[str, Any])
# ---------------------------------------------------------------------------

class MetricsSnapshot(BaseModel):
    """Tool metrics for observability."""

    name:         str
    calls:        int   = Field(..., ge=0)
    successes:    int   = Field(..., ge=0)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    avg_time_ms:  float = Field(0.0, ge=0.0)

    @field_validator("success_rate", mode="before")
    @classmethod
    def round_rate(cls, v: float) -> float:
        return round(float(v), 4)
