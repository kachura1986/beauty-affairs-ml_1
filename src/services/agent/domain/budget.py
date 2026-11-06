import math
import time

from pydantic import BaseModel, ConfigDict, Field

from .reason import TerminationReason


class BudgetLimits(BaseModel):
    """
    Resource limits for an agent_graph run.

    CONTRACT: ``max_cost_usd == 0`` means a free model — the run proceeds as long as no nonzero cost is
    recorded. The extra finalization call after ``max_steps`` does not increment the step counter.
    """

    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(
        gt=0,
        description=(
            "Maximum number of LLM decision steps. When the limit is reached, one additional finalization call"
            "(tools disabled) may be made if allowed by policy and remaining wall-clock / token budgets."
        ),
    )
    max_seconds: float = Field(gt=0, description="Maximum wall-clock time in seconds.")
    max_tokens: int = Field(gt=0, description="Token budget for LLM input and output.")
    max_cost_usd: float = Field(
        ge=0,
        description=(
            "Cost budget for LLM usage, in USD. A value of 0 means a free model: "
            "the run proceeds as long as no nonzero cost is recorded."
        ),
    )


class Budget(BaseModel):
    """
    Tracks resource usage against the configured limits of an agent_graph run.

    CONTRACT: usage is checked before/after resource-consuming operations, so tokens, time, and cost may
    temporarily exceed their limits before the run is stopped.
    """

    model_config = ConfigDict(validate_assignment=True)

    limits: BudgetLimits
    started: float = Field(default_factory=time.monotonic, description="Monotonic start timestamp.")
    steps: int = Field(default=0, ge=0, description="LLM decision steps executed.")
    tokens: int = Field(default=0, ge=0, description="Total tokens consumed.")
    cost_usd: float = Field(default=0.0, ge=0, description="Accumulated cost in USD.")

    def elapsed(self) -> float:
        """
        Return the elapsed wall-clock time since the budget start time.

        Returns:
            float: Elapsed time in seconds.
        """
        return time.monotonic() - self.started

    def remaining_seconds(self) -> float:
        """
        Return the remaining wall-clock time before the configured limit is reached.

        Returns:
            float: Remaining time in seconds, clamped at ``0.0`` when the limit is reached or exceeded.
        """
        return max(0.0, self.limits.max_seconds - self.elapsed())

    def remaining_tokens(self) -> int:
        """
        Return the remaining token budget.

        Returns:
            int: Remaining tokens, clamped at ``0`` when the limit is reached or exceeded.
        """
        return max(0, self.limits.max_tokens - self.tokens)

    def remaining_cost_usd(self) -> float:
        """
        Return the remaining LLM cost budget.

        CONTRACT: ``max_cost_usd == 0`` means a free model → returns ``math.inf`` (never a hard stop
        on cost), so callers must not treat the result as a finite cap.

        Returns:
            float: Remaining budget in USD; ``math.inf`` for a free model, ``0.0`` when exhausted.
        """
        if self.limits.max_cost_usd == 0:
            return math.inf

        return max(0.0, self.limits.max_cost_usd - self.cost_usd)

    def remaining_steps(self) -> int:
        """
        Return the number of remaining LLM decision steps.

        Returns:
            int: Remaining steps, clamped at ``0`` when the limit is reached or exceeded.
        """
        return max(0, self.limits.max_steps - self.steps)

    def can_start_step(self) -> bool:
        """
        Check whether another LLM decision step can be started.

        Returns:
            bool: ``True`` when the step limit has not yet been reached, otherwise ``False``.
        """
        return self.steps < self.limits.max_steps

    def exceeded_without_steps(self) -> TerminationReason | None:
        """
        Check whether any budget limit other than steps has been reached or exceeded.

        CONTRACT: steps are intentionally not checked here — the final allowed LLM step must be processed
        before the run is stopped. Checks are ordered time → tokens → cost.
        ``max_cost_usd == 0`` means a free model: the run terminates as soon as any nonzero cost is reported.

        Returns:
            TerminationReason | None: A budget reason if a limit has been reached or exceeded, otherwise ``None``.
        """
        if self.elapsed() >= self.limits.max_seconds:
            return TerminationReason.BUDGET_TIME

        if self.tokens >= self.limits.max_tokens:
            return TerminationReason.BUDGET_TOKENS

        if self.limits.max_cost_usd > 0:
            if self.cost_usd >= self.limits.max_cost_usd:
                return TerminationReason.BUDGET_COST
        elif self.cost_usd > 0:
            return TerminationReason.BUDGET_COST

        return None

    def record_llm(self, in_tokens: int, out_tokens: int, cost_usd: float) -> None:
        """
        Record resource usage for a completed LLM call.

        Args:
            in_tokens (int): Number of input tokens consumed.
            out_tokens (int): Number of output tokens generated.
            cost_usd (float): Cost of the LLM call, in USD.
        """
        self.tokens += in_tokens + out_tokens
        self.cost_usd += cost_usd

    def ratios(self) -> dict[str, float]:
        """
        Calculate resource utilization relative to the configured limits.

        Returns:
            dict[str, float]: Utilization ratios for ``steps``, ``time``, ``tokens``, and ``cost``.
            Values may exceed ``1.0`` because usage is checked after resource-consuming operations.
        """
        return {
            'steps': self.steps / max(self.limits.max_steps, 1),
            'time': self.elapsed() / max(self.limits.max_seconds, 1e-9),
            'tokens': self.tokens / max(self.limits.max_tokens, 1),
            'cost': self.cost_usd / max(self.limits.max_cost_usd, 1e-9),
        }

    def soft_warning(self) -> str | None:
        """
        Return a warning when any budget utilization reaches 80%, so the agent can finish
        before a hard budget check stops the run.

        Returns:
            str | None: A warning message if any utilization ratio is at least ``0.8``, otherwise ``None``.
        """
        worst = max(self.ratios().values())

        if worst >= 0.8:
            return (
                "WARNING: more than 80% of the run budget has been used. Finish the task: "
                "provide the best available answer and explicitly list anything that remains unverified."
            )

        return None

    def snapshot(self) -> dict[str, str]:
        """
        Return a human-readable snapshot of the current budget usage.

        Returns:
            dict[str, str]: Usage and limits for steps, tokens, cost, and elapsed wall-clock time.
        """
        return {
            'steps': f'{self.steps}/{self.limits.max_steps}',
            'tokens': f'{self.tokens}/{self.limits.max_tokens}',
            'cost': f'{self.cost_usd:.4f}/{self.limits.max_cost_usd}',
            'seconds': f'{self.elapsed():.1f}/{self.limits.max_seconds}',
        }
