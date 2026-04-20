# keryx/core/_budget.py
"""BudgetController — tracks and enforces API/model spend limits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models.interface import ModelInterface


@dataclass
class BudgetController:
    """Track and enforce API/model budget for cloud models."""
    max_cost_usd: float = 10.0
    current_cost: float = 0.0
    max_calls:    int   = 500
    calls_made:   int   = 0

    def can_proceed(self, model: "ModelInterface") -> bool:
        if self.calls_made >= self.max_calls:
            return False
        return self.current_cost + self._estimate_call_cost(model) <= self.max_cost_usd

    def record_call(
        self,
        model:      "ModelInterface",
        tokens_in:  int = 1000,
        tokens_out: int = 500,
    ) -> None:
        cost_in  = model.cost_per_1k_input_tokens  or 0.0
        cost_out = model.cost_per_1k_output_tokens or 0.0
        self.current_cost += (tokens_in / 1000) * cost_in + (tokens_out / 1000) * cost_out
        self.calls_made   += 1

    def _estimate_call_cost(self, model: "ModelInterface") -> float:
        cost_in  = model.cost_per_1k_input_tokens  or 0.0
        cost_out = model.cost_per_1k_output_tokens or 0.0
        return (2.0 * cost_in) + (1.0 * cost_out)

    @property
    def remaining_budget(self) -> float:
        return self.max_cost_usd - self.current_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_cost_usd":      self.max_cost_usd,
            "current_cost":      self.current_cost,
            "calls_made":        self.calls_made,
            "remaining_budget":  self.remaining_budget,
        }
