from __future__ import annotations

import json
import re
from collections.abc import Iterator

from keryx.models.interface import (
    CostEstimate,
    GenerationConfig,
    GenerationResult,
    ModelCapabilities,
    ModelInterface,
    ToolCall,
    ToolDefinition,
)


class ScriptedModel(ModelInterface):
    """Deterministic fallback: read_file → codeql_query → FINISH, per-prompt target.

    Used when no API key is available (--scripted flag or missing ANTHROPIC_API_KEY).
    """

    model_name = "scripted-fallback"

    def __init__(self) -> None:
        self.main_calls = 0

    def generate(self, prompt: str, config: GenerationConfig | None = None,
                 *, grammar: str | None = None, max_tokens: int | None = None) -> str:
        if prompt.strip() == "ping":
            return "pong"
        if "vulnerability hypothesis" in prompt or "Does the evidence" in prompt:
            return "GENUINE CONCERN — AST finding appears valid."
        self.main_calls += 1
        target = self._target_from_prompt(prompt)
        step   = self._next_step(prompt, target)
        return json.dumps(step)

    @staticmethod
    def _target_from_prompt(prompt: str) -> str:
        m = re.search(r'Target file\s*:\s*(\S+)', prompt)
        return m.group(1) if m else "/tmp/unknown.py"

    @staticmethod
    def _next_step(prompt: str, target: str) -> dict:
        if ("] codeql_query" in prompt and "obs:" in prompt) or "[AST]" in prompt:
            return {"thought": "Analysis complete.",
                    "action": "FINISH", "action_input": {}, "confidence": 0.88}
        if "] read_file" in prompt and "obs:" in prompt:
            return {"thought": "Read target, running AST analysis.",
                    "action": "codeql_query",
                    "action_input": {"file_path": target}, "confidence": 0.82}
        return {"thought": "Starting: read target file.",
                "action": "read_file",
                "action_input": {"path": target}, "confidence": 0.75}

    async def generate_async(self, prompt: str, config: GenerationConfig | None = None,
                             *, grammar: str | None = None, max_tokens: int | None = None) -> str:
        return self.generate(prompt, config, grammar=grammar, max_tokens=max_tokens)

    def generate_result(self, prompt: str, config: GenerationConfig | None = None) -> GenerationResult:
        return GenerationResult(text=self.generate(prompt, config),
                                tokens_input=10, tokens_output=10,
                                duration_ms=1.0, finish_reason="stop")

    def generate_stream(self, prompt: str, config: GenerationConfig | None = None) -> Iterator[str]:
        yield self.generate(prompt, config)

    def generate_with_tools(self, prompt: str,
                            tools: list[ToolDefinition] | None = None,
                            config: GenerationConfig | None = None) -> str | ToolCall:
        return self.generate(prompt, config)

    def tokenize(self, text: str) -> list[int]:
        return [0] * max(1, len(text) // 4)

    def get_context_length(self) -> int:
        return 32_768

    def is_healthy(self) -> bool:
        return True

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> CostEstimate:
        return CostEstimate(input_cost_usd=0.0, output_cost_usd=0.0, total_cost_usd=0.0)

    def get_usage_cost(self) -> CostEstimate:
        return CostEstimate(input_cost_usd=0.0, output_cost_usd=0.0, total_cost_usd=0.0)

    def get_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            max_context_length=32_768, supports_tool_calling=False,
            supports_grammar=True, supports_batching=False, supports_streaming=False,
            supports_speculative=False, supports_min_p=False, requires_gpu=False,
            is_local=True, is_quantized=False,
        )

    def unload(self) -> None:
        pass
