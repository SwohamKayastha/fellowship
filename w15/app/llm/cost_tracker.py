"""Token and cost tracking for LLM calls."""

import logging
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (gpt-4o-mini as of 2026)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini-3.5-flash-lite": {"input": 0.05, "output": 0.20},
    "Qwen/Qwen3-0.6B": {"input": 0.0, "output": 0.0},  # self-hosted, no cost
}


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost(self) -> float:
        return 0.0  # cost calculated per model


@dataclass
class CostTracker:
    """Track tokens and cost across LLM calls."""

    budget_tokens: int = 0  # 0 = unlimited
    total_tokens: int = 0
    call_count: int = 0
    usage_by_model: Dict[str, TokenUsage] = field(default_factory=dict)
    cost_by_model: Dict[str, float] = field(default_factory=dict)

    def add_call(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Record a single LLM call."""
        total = prompt_tokens + completion_tokens
        self.total_tokens += total
        self.call_count += 1

        # Track by model
        if model not in self.usage_by_model:
            self.usage_by_model[model] = TokenUsage()
            self.cost_by_model[model] = 0.0

        self.usage_by_model[model].prompt_tokens += prompt_tokens
        self.usage_by_model[model].completion_tokens += completion_tokens

        # Calculate cost
        if model in PRICING:
            pricing = PRICING[model]
            cost = (prompt_tokens / 1_000_000) * pricing["input"] + (
                completion_tokens / 1_000_000
            ) * pricing["output"]
            self.cost_by_model[model] += cost

        logger.info(
            f"Tokens: {prompt_tokens} in + {completion_tokens} out = {total} | "
            f"Cost: ${self.cost_by_model[model]:.4f} | "
            f"Model: {model}"
        )

    def budget_remaining(self) -> int:
        """Tokens left before budget hit."""
        if self.budget_tokens == 0:
            return float("inf")
        return max(0, self.budget_tokens - self.total_tokens)

    def is_over_budget(self) -> bool:
        """Check if budget exceeded."""
        if self.budget_tokens == 0:
            return False
        return self.total_tokens >= self.budget_tokens

    def summary(self) -> str:
        """Print usage summary."""
        lines = [
            f"=== Token & Cost Summary ===",
            f"Total calls: {self.call_count}",
            f"Total tokens: {self.total_tokens}",
            f"Total cost: ${sum(self.cost_by_model.values()):.4f}",
        ]
        if self.budget_tokens > 0:
            remaining = self.budget_remaining()
            lines.append(f"Budget: {self.total_tokens}/{self.budget_tokens} tokens")
            if remaining == 0:
                lines.append("⚠️ BUDGET EXCEEDED - stopping early")
            else:
                lines.append(f"Remaining: {remaining} tokens")

        lines.append("\nBreakdown by model:")
        for model, usage in self.usage_by_model.items():
            cost = self.cost_by_model.get(model, 0.0)
            lines.append(
                f"  {model}: {usage.prompt_tokens} in, "
                f"{usage.completion_tokens} out, ${cost:.4f}"
            )
        return "\n".join(lines)


_global_tracker = CostTracker()


def get_tracker() -> CostTracker:
    return _global_tracker


def reset_tracker(budget_tokens: int = 0) -> None:
    global _global_tracker
    _global_tracker = CostTracker(budget_tokens=budget_tokens)
