from enum import StrEnum


class TerminationReason(StrEnum):
    # Task completed and the configured postcondition was verified successfully
    DONE = 'done'

    # Task completed, but no postcondition was configured, so the result is unverified
    DONE_UNVERIFIED = 'done_unverified'

    # The maximum number of LLM decision steps has been reached
    BUDGET_STEPS = 'budget_steps'

    # The maximum wall-clock execution time has been reached
    BUDGET_TIME = 'budget_time'

    # The maximum LLM token budget has been reached or exceeded
    BUDGET_TOKENS = 'budget_tokens'

    # The maximum allowed LLM cost has been reached or exceeded
    BUDGET_COST = 'budget_cost'

    # The agent_graph is no longer making useful progress, e.g. repeated empty responses
    NO_PROGRESS = 'no_progress'

    # The agent_graph attempted the same tool action too many times
    REPEATED_ACTION = 'repeated_action'

    # The LLM request failed while execution time remained
    LLM_ERROR = 'llm_error'