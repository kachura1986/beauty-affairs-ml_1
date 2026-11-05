"""Tests for ChatResponse schema: finalization is part of the public API contract."""
import pytest
from pydantic import ValidationError

from api.schemas.agent import ChatResponse, ChatUsage
from services.agent.domain import Finalization, TerminationReason


def test_chat_response_includes_finalization_field():
    """ChatResponse exposes finalization; clients must see how the answer was produced."""
    response = ChatResponse(
        run_id='r1',
        role='assistant',
        content='final',
        termination_reason=TerminationReason.BUDGET_STEPS,
        finalization=Finalization.FALLBACK,
        usage=ChatUsage(steps=1, tokens=10, cost_usd=0.0, seconds=0.1),
    )

    assert response.finalization == Finalization.FALLBACK


def test_chat_response_requires_finalization():
    """finalization has no default; omitting it is a schema error (contract, not accident)."""
    with pytest.raises(ValidationError):
        ChatResponse(
            run_id='r1',
            role='assistant',
            content=None,
            termination_reason=TerminationReason.BUDGET_STEPS,
            usage=ChatUsage(steps=1, tokens=10, cost_usd=0.0, seconds=0.1),
        )
