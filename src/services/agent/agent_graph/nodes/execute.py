import time
from typing import Any

from langgraph.runtime import Runtime

from ...context import build_tool_message
from ...domain import TerminationReason
from ...tools import ToolStatus
from ...tools.errors import already_failed_result, repeated_action_result, skipped_due_to_termination_result
from ..state import AgentContext, GraphState


async def execute_node(state: GraphState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    """
    Equivalent of Agent.execute_tools().

    Tool calls are still executed sequentially, exactly as in the original implementation.

    Args:
        state (GraphState): The current state of the graph.
        runtime (Runtime[AgentContext]): The runtime environment of the graph.

    Returns:
        dict[str, Any]: Result of the node.
    """
    ctx = runtime.context

    budget = state['budget']
    action_counts = dict(state['action_counts'])
    failed_actions = list(state['failed_actions'])
    failed_set = set(failed_actions)

    # Preserve the most recent successful tool result across execute rounds;
    # only overwrite it when a tool in this round returns ToolStatus.OK.
    last_tool_ok = state.get('last_tool_ok')

    history_updates: list[dict[str, Any]] = []
    termination_reason: TerminationReason | None = None

    # INVARIANT: len(history_updates) == len(state['pending_tool_calls']) on every exit path.
    # Never use `break` here — every call must receive a matching tool-message.
    for call in state['pending_tool_calls']:
        fp = ctx.tools.fingerprint(call)

        # Step 1. Run is already terminating: report this call and every
        # remaining one as skipped, so the assistant tool_calls stay paired.
        if termination_reason is not None:
            history_updates.append(build_tool_message(
                call,
                skipped_due_to_termination_result(call.get('name', '?'), termination_reason.value),
            ))
            continue

        # Step 2. This exact call already failed irrecoverably: return a
        # structured non-retryable error instead of an observation.
        if fp in failed_set:
            history_updates.append(build_tool_message(call, already_failed_result(call.get('name', '?'))))

            continue

        # Step 3. Count repeated identical actions; exceeding the limit
        # terminates the run but does not skip this call's response.
        count = action_counts.get(fp, 0) + 1
        action_counts[fp] = count

        if count > ctx.policy.repeat_limit:
            termination_reason = TerminationReason.REPEATED_ACTION
            history_updates.append(build_tool_message(call, repeated_action_result(call.get('name', '?'))))

            continue  # remaining calls are handled by branch (1)

        # Step 4. Pre-flight: budget may have expired since _request_llm.
        # `remaining` is computed once and reused for invoke() — no recomputation.
        remaining = budget.remaining_seconds()
        if remaining <= 0:
            termination_reason = TerminationReason.BUDGET_TIME
            history_updates.append(build_tool_message(
                call,
                skipped_due_to_termination_result(call.get('name', '?'), termination_reason.value),
            ))
            continue

        # Step 5. Execute the tool call with a timeout capped by both the remaining wall-clock budget and
        # the per-tool ceiling (the latter is applied inside invoke()).
        t0 = time.monotonic()
        res = await ctx.tools.invoke(call, timeout=remaining)
        latency_ms = int((time.monotonic() - t0) * 1000)
        ctx.trace.tool(budget.steps, call, res, latency_ms)

        # Track the most recent successful tool result explicitly, so the
        # finalization fallback does not have to guess from message content.
        if res.status is ToolStatus.OK:
            last_tool_ok = res.content

        # Remember irrecoverable failures so repeats are answered synthetically.
        if res.status is ToolStatus.ERROR and not res.retryable:
            failed_set.add(fp)
            if fp not in failed_actions:
                failed_actions.append(fp)

        history_updates.append(build_tool_message(call, res))

        # Step 6. Hard budget check after each call: record the reason but
        # keep iterating so every remaining call still gets a response.
        reason = budget.exceeded_without_steps()
        if reason:
            termination_reason = reason

    result: dict[str, Any] = {
        'history': history_updates,
        'action_counts': action_counts,
        'failed_actions': failed_actions,
        'last_tool_ok': last_tool_ok,
        'pending_tool_calls': [],
        'budget': budget,
    }

    if termination_reason:
        result['termination_reason'] = termination_reason

    return result
