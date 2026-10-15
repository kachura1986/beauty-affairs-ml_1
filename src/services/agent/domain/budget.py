import time
from dataclasses import dataclass, field

from .reason import TerminationReason


@dataclass
class BudgetLimits:
    """
    Defines the resource limits for an agent_graph run.

    Attributes:
        max_steps (int): Maximum number of LLM decision steps allowed.
        max_seconds (float): Maximum wall-clock time allowed for the run, in seconds.
        max_tokens (int): Token budget for LLM input and output tokens.
        max_cost_usd (float): Cost budget for LLM usage, in USD.
    """

    max_steps: int
    max_seconds: float
    max_tokens: int
    max_cost_usd: float


@dataclass
class Budget:
    """
    Tracks resource usage against the configured limits of an agent_graph run.

    The budget tracks LLM decision steps, elapsed wall-clock time, consumed input and output tokens,
    and accumulated LLM cost. Limits are checked by the agent_graph before and after resource-consuming operations,
    so token, time, and cost usage may temporarily exceed their configured limits before execution is stopped.

    Attributes:
        limits (BudgetLimits): Resource limits configured for the run.
        started (float): Monotonic timestamp used as the budget start time. Defaults to the value of
        ``time.monotonic()`` when the budget is created.
        steps (int): Number of LLM decision steps executed so far.
        tokens (int): Total number of input and output LLM tokens consumed.
        cost_usd (float): Total accumulated LLM cost in USD.
    """

    limits: BudgetLimits
    started: float = field(default_factory=time.monotonic)
    steps: int = 0
    tokens: int = 0
    cost_usd: float = 0.0

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
            float: Remaining time in seconds. Returns ``0.0`` when the limit has been reached or exceeded.
        """
        return max(0.0, self.limits.max_seconds - self.elapsed())

    def remaining_tokens(self) -> int:
        """
        Return the remaining token budget.

        Returns:
            int: Remaining number of input and output tokens before the configured limit is reached.
            Returns ``0`` when the limit has been reached or exceeded.
        """
        return max(0, self.limits.max_tokens - self.tokens)

    def remaining_cost_usd(self) -> float:
        """
        Return the remaining LLM cost budget.

        Returns:
            float: Remaining budget in USD. Returns ``0.0`` when the limit has been reached or exceeded.
        """
        return max(0.0, self.limits.max_cost_usd - self.cost_usd)

    def remaining_steps(self) -> int:
        """
        Return the number of remaining LLM decision steps.

        Returns:
            int: Number of LLM decision steps that can still be executed. Returns ``0``
            when the step limit has been reached or exceeded.
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
        Check whether any configured budget limit other than steps has been reached or exceeded.

        Limits are checked in the following order: time, tokens, and cost.
        The step limit is intentionally not checked so that the final allowed LLM step
        can be processed before the run is stopped.

        Returns:
            TerminationReason | None: A budget reason if a limit has been reached or exceeded, otherwise ``None``.
        """
        if self.elapsed() >= self.limits.max_seconds:
            return TerminationReason.BUDGET_TIME

        if self.tokens >= self.limits.max_tokens:
            return TerminationReason.BUDGET_TOKENS

        if self.cost_usd >= self.limits.max_cost_usd:
            return TerminationReason.BUDGET_COST

        return None

    def exceeded(self) -> TerminationReason | None:
        """
        Check whether any budget limit has been reached or exceeded.

        Limits are checked in the following order: steps, time, tokens, and cost.
        The first reached limit determines the returned reason.

        Returns:
            TerminationReason | None: A budget reason if a limit has been reached or exceeded, otherwise ``None``.
        """
        if self.steps >= self.limits.max_steps:
            return TerminationReason.BUDGET_STEPS

        if self.elapsed() >= self.limits.max_seconds:
            return TerminationReason.BUDGET_TIME

        if self.tokens >= self.limits.max_tokens:
            return TerminationReason.BUDGET_TOKENS

        if self.cost_usd >= self.limits.max_cost_usd:
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

        A value of ``1.0`` means that the corresponding limit has been reached. Values above ``1.0`` are
        possible for time, tokens, and cost because usage is checked after resource-consuming operations.

        Returns:
            dict[str, float]: Mapping with utilization ratios for ``steps``, ``time``, ``tokens``, and ``cost``.
        """
        return {
            'steps': self.steps / max(self.limits.max_steps, 1),
            'time': self.elapsed() / max(self.limits.max_seconds, 1e-9),
            'tokens': self.tokens / max(self.limits.max_tokens, 1),
            'cost': self.cost_usd / max(self.limits.max_cost_usd, 1e-9),
        }

    def soft_warning(self) -> str | None:
        """
        Return a warning when any budget utilization reaches 80%.

        The warning is intended to give the agent_graph an opportunity to finish its work
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
            dict[str, str]: Mapping containing current usage and configured limits for steps, tokens, cost, and
            elapsed wall-clock time.
        """
        return {
            'steps': f'{self.steps}/{self.limits.max_steps}',
            'tokens': f'{self.tokens}/{self.limits.max_tokens}',
            'cost': f'{self.cost_usd:.4f}/{self.limits.max_cost_usd}',
            'seconds': f'{self.elapsed():.1f}/{self.limits.max_seconds}',
        }
