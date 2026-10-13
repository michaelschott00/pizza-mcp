"""Pizza MCP Server.

A toy MCP server that runs an interactive pizza-order interview. It demonstrates
two MCP features:

* elicitation - every prompt is asked through the SDK's elicitation support
  (via `Resolve`/`Elicit` resolvers, which work on every protocol version).
* progressive tool discovery - only the tools for the current stage are exposed
  to the client. When the stage changes the server swaps the registered tools
  and sends a `notifications/tools/list_changed` so the client re-fetches them.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Annotated, Any, Callable, Literal

from pydantic import BaseModel, Field

from mcp.server import MCPServer
from mcp.server.mcpserver import (
    AcceptedElicitation,
    ElicitationResult,
    Context,
    Elicit,
    Resolve,
)

server = MCPServer(
    "pizza",
    title="Pizza MCP Server",
    instructions=(
        "Guide the customer through an interactive pizza ordering interview. "
        "Only the tools for the current stage are available, so call the tool that "
        "is currently exposed and answer the prompts it presents. The stages are: "
        "greeting, ordering, pizza-making, and feedback. When a stage completes the "
        "tool list changes - re-fetch it to see the next stage's tools."
    ),
)


class Stage(Enum):
    GREETING = auto()
    ORDER = auto()
    PIZZA_MAKING = auto()
    FEEDBACK = auto()


current_stage = Stage.GREETING
order: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Stage 1: greeting
# --------------------------------------------------------------------------- #


class Greeting(BaseModel):
    feeling: str = Field(description="How are you doing today?")
    arrival: Literal["bike", "train", "car", "by foot"] = Field(description="How did you get here?")
    wants_pizza: bool = Field(description="Would you like to order a pizza?")


async def greet_resolver() -> Elicit[Greeting] | Greeting:
    return Elicit("Welcome to Pizza MCP! Before we begin, a few quick questions.", Greeting)


async def greet_customer(
    answers: Annotated[ElicitationResult[Greeting], Resolve(greet_resolver)],
    ctx: Context,
) -> str:
    """Greet the customer and ask how they are, how they arrived, and whether they want pizza."""
    if not isinstance(answers, AcceptedElicitation):
        return "No problem. Come back whenever you feel like a pizza!"

    data = answers.data
    if not data.wants_pizza:
        return (
            f"Thanks for stopping by and {data.feeling or 'take care'}! "
            "Maybe next time you'll want a pizza."
        )

    await _enter_stage(Stage.ORDER, ctx)
    return (
        f"Great, a pizza it is! Arriving by {data.arrival}, that works. "
        "Let's put your order together."
    )


# --------------------------------------------------------------------------- #
# Stage 2: order
# --------------------------------------------------------------------------- #


class Order(BaseModel):
    crust: Literal["cheese crust", "sausage crust", "no crust filling"] = Field(
        description="What kind of crust filling would you like?"
    )
    pizza_type: Literal["margarita", "speziale", "four cheese"] = Field(
        description="Which pizza would you like?"
    )
    extras: str = Field(default="", description="Anything else? Leave blank for none.")


async def order_resolver() -> Elicit[Order] | Order:
    return Elicit("Let's build your pizza.", Order)


async def take_order(
    answers: Annotated[ElicitationResult[Order], Resolve(order_resolver)],
    ctx: Context,
) -> str:
    """Take the pizza order: crust filling, pizza type, and any extras."""
    if not isinstance(answers, AcceptedElicitation):
        await _enter_stage(Stage.GREETING, ctx)
        return "Order cancelled. Let me know if you change your mind."

    data = answers.data
    order.update(crust=data.crust, pizza_type=data.pizza_type, extras=data.extras)

    await _enter_stage(Stage.PIZZA_MAKING, ctx)
    extras = f" plus {data.extras}" if data.extras else ""
    return f"One {data.pizza_type} with {data.crust}{extras}. Let's make it!"


# --------------------------------------------------------------------------- #
# Stage 3: pizza-making
# --------------------------------------------------------------------------- #


async def make_dough() -> str:
    """Make the dough for the pizza."""
    return f"Kneading the dough for a {order.get('pizza_type', 'pizza')}."


async def spread_dough() -> str:
    """Spread and shape the dough."""
    return "Spreading the dough into a nice round base."


async def add_topings() -> str:
    """Add the toppings to the pizza."""
    extras = order.get("extras")
    extra_note = f" Adding your extras: {extras}." if extras else ""
    return f"Adding tomato, mozzarella and friends.{extra_note}"


async def bake_pizza() -> str:
    """Bake the pizza in the oven."""
    return "Baking the pizza until the crust is golden."


async def serve_pizza(ctx: Context) -> str:
    """Serve the finished pizza and move on to the feedback stage."""
    await _enter_stage(Stage.FEEDBACK, ctx)
    return "Here is your pizza! Buon appetito."


# --------------------------------------------------------------------------- #
# Stage 4: feedback
# --------------------------------------------------------------------------- #


class Feedback(BaseModel):
    liked: bool = Field(description="Did you like your experience?")
    comments: str = Field(default="", description="Anything you'd like to say? Leave blank for none.")


async def feedback_resolver() -> Elicit[Feedback] | Feedback:
    return Elicit("Before you go, how was your experience?", Feedback)


async def collect_feedback(
    answers: Annotated[ElicitationResult[Feedback], Resolve(feedback_resolver)],
    ctx: Context,
) -> str:
    """Ask whether the customer liked their experience and collect any comments."""
    if not isinstance(answers, AcceptedElicitation):
        await _enter_stage(Stage.GREETING, ctx)
        return "Thanks for visiting!"

    data = answers.data
    closing = "Grazie! We're glad you enjoyed it." if data.liked else "Sorry to hear that - we'll do better."
    if data.comments:
        closing += f" Your feedback: {data.comments}"

    await _enter_stage(Stage.GREETING, ctx)
    return closing


# --------------------------------------------------------------------------- #
# Progressive tool discovery
# --------------------------------------------------------------------------- #

STAGE_TOOLS: dict[Stage, list[Callable[..., Any]]] = {
    Stage.GREETING: [greet_customer],
    Stage.ORDER: [take_order],
    Stage.PIZZA_MAKING: [make_dough, spread_dough, add_topings, bake_pizza, serve_pizza],
    Stage.FEEDBACK: [collect_feedback],
}


async def _enter_stage(stage: Stage, ctx: Context) -> None:
    """Swap the exposed tools to match `stage` and tell the client the list changed."""
    global current_stage

    for fn in STAGE_TOOLS[current_stage]:
        server.remove_tool(fn.__name__)
    current_stage = stage
    for fn in STAGE_TOOLS[stage]:
        server.add_tool(fn)

    await ctx.session.send_tool_list_changed()
    await ctx.notify_tools_changed()


def _register_initial_stage() -> None:
    order.clear()
    for fn in STAGE_TOOLS[Stage.GREETING]:
        server.add_tool(fn)


_register_initial_stage()


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
