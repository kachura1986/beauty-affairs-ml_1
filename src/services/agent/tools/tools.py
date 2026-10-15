import math

from pydantic import BaseModel, Field

from .registry import ToolRegistry
from .tool import Tool


# DEMO TOOL
class FactorialArgs(BaseModel):
    n: int = Field(ge=0, le=2000)


def _factorial(n: int) -> str:
    """
    Calculate the factorial of a non-negative integer.

    Args:
        n: Non-negative integer.

    Returns:
        str: Decimal representation of the factorial result.

    Raises:
        ValueError: If n is negative or greater than 2000.
    """
    n = int(n)

    # Factorial is defined only for non-negative integers
    if n < 0:
        raise ValueError('Factorial is defined only for n >= 0.')

    # Limit the input to prevent unnecessarily expensive calculations and excessively large tool output
    if n > 2000:
        raise ValueError('n is too large; the maximum allowed value is 2000.')

    return str(math.factorial(n))


# TEST TOOL REGISTRY
DEMO_TOOLS = ToolRegistry([
    Tool(
        name='factorial',
        description=(
            'Exactly calculates the factorial of a non-negative integer n. '
            'Returns the decimal representation of the result. '
            'The maximum allowed value is n = 2000. '
            'Use this tool instead of calculating the result yourself.'
        ),
        parameters={
            'type': 'object',
            'properties': {
                'n': {
                    'type': 'integer',
                    'minimum': 0,
                    'maximum': 2000,
                    'description': 'Non-negative integer.',
                },
            },
            # The model must provide n when calling the tool
            'required': ['n'],
        },
        fn=_factorial,
        args_model=FactorialArgs,
    ),
])
