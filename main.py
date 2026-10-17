"""Pizza MCP Server.

A toy MCP server that runs an interactive pizza-order interview. It demonstrates
two MCP features:

* elicitation, with a fallback - prompts are asked through the SDK's
  elicitation support (via `Resolve`/`Elicit` resolvers, which work on every
  protocol version). Clients that do not declare the elicitation capability
  are not asked directly: the tool returns a message telling the client to pose
  the question to the user, and accepts the answers back through its optional
  `answers` argument so the interview can continue.
* stage-gated tools - every tool is always exposed to the client, but each one
  refuses to run (with an informative error) unless the interview is currently
  in the stage it belongs to.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Annotated, Any, Callable, Literal

from pydantic import BaseModel, Field

from mcp.server import MCPServer
from mcp.server.mcpserver import (
    AcceptedElicitation,
    Context,
    ElicitationResult,
    Elicit,
    Resolve,
)
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer(
    "pizza",
    title="Pizza MCP Server",
    instructions=(
        "Guide the customer through an interactive pizza ordering interview. "
        "Every tool is always available, but each tool only works during the stage "
        "it belongs to. Calling a tool outside its stage returns an error telling "
        "you which stage the interview is currently in and which tools to call to "
        "reach the stage you need. Questions are asked through elicitation when the "
        "client supports it; otherwise the tool returns the questions, you ask the "
        "user, and you call the tool again with the answers in its `answers` argument. "
        "The stages are: greeting, ordering, pizza-making, and feedback."
    ),
)


class Stage(Enum):
    GREETING = auto()
    ORDER = auto()
    PIZZA_MAKING = auto()
    FEEDBACK = auto()


STAGE_LABELS: dict[Stage, str] = {
    Stage.GREETING: "greeting",
    Stage.ORDER: "ordering",
    Stage.PIZZA_MAKING: "pizza-making",
    Stage.FEEDBACK: "feedback",
}

STAGE_TRANSITIONS: dict[Stage, tuple[Stage, str]] = {
    Stage.GREETING: (Stage.ORDER, "greet_customer"),
    Stage.ORDER: (Stage.PIZZA_MAKING, "take_order"),
    Stage.PIZZA_MAKING: (Stage.FEEDBACK, "serve_pizza"),
    Stage.FEEDBACK: (Stage.GREETING, "collect_feedback"),
}

PIZZA_STEPS: list[str] = ["make_dough", "spread_dough", "add_topings", "bake_pizza", "serve_pizza"]

current_stage = Stage.GREETING
pizza_step = 0
order: dict[str, Any] = {}


def _enter_stage(stage: Stage) -> None:
    """Move the interview to `stage`."""
    global current_stage, pizza_step
    current_stage = stage
    if stage is Stage.PIZZA_MAKING:
        pizza_step = 0


def _advancing_tools(target: Stage, tool_name: str | None = None) -> list[str]:
    """The tools to call, in order, to move the interview from the current stage to `target`."""
    tools: list[str] = []
    stage = current_stage
    for _ in range(len(STAGE_TRANSITIONS)):
        if stage is target:
            break
        next_stage, tool = STAGE_TRANSITIONS[stage]
        if stage is Stage.PIZZA_MAKING:
            tools.extend(PIZZA_STEPS[:-1])
        tools.append(tool)
        stage = next_stage
    if target is Stage.PIZZA_MAKING and tool_name in PIZZA_STEPS:
        tools.extend(PIZZA_STEPS[: PIZZA_STEPS.index(tool_name)])
    return tools


def _require_stage(stage: Stage, tool_name: str | None = None) -> None:
    """Fail with an informative error unless the interview is in `stage`."""
    if current_stage is stage:
        return

    steps = _advancing_tools(stage, tool_name)
    path = ", then ".join(f"`{name}`" for name in steps)
    raise ToolError(
        f"This tool is only available during the {STAGE_LABELS[stage]} stage, "
        f"but the interview is currently in the {STAGE_LABELS[current_stage]} stage. "
        f"Call {path} to advance to where this tool can be used."
    )


def _require_pizza_step(name: str) -> None:
    """Fail unless the interview is making pizza and `name` is the next step."""
    _require_stage(Stage.PIZZA_MAKING, name)
    expected = PIZZA_STEPS[pizza_step]
    if name != expected:
        raise ToolError(
            f"The pizza-making steps must be called in order. The next step is `{expected}`."
        )


def _complete_pizza_step() -> None:
    global pizza_step
    pizza_step += 1


# --------------------------------------------------------------------------- #
# Elicitation support
# --------------------------------------------------------------------------- #


class ElicitationUnsupported(BaseModel):
    """Payload used in place of an elicitation when the client cannot answer one."""

    message: str


def _supports_elicitation(ctx: Context) -> bool:
    """Whether the connected client declared it can answer form elicitation requests.

    Mirrors the SDK's own capability gate: a bare `elicitation: {}` counts as form
    support, while URL-only elicitation does not.
    """
    try:
        capabilities = ctx.client_capabilities
    except ValueError:
        # Outside an active request there is no client to ask.
        return False
    elicitation = capabilities.elicitation if capabilities is not None else None
    return elicitation is not None and (elicitation.form is not None or elicitation.url is None)


def _render_questions(schema: type[BaseModel]) -> str:
    """The fields of `schema`, rendered as the questions to put to the user."""
    return "\n".join(f"- {field.description or name}" for name, field in schema.model_fields.items())


def _elicit_or_message(
    ctx: Context, prompt: str, schema: type[BaseModel], provided: BaseModel | None
) -> Elicit[Any] | BaseModel:
    """Ask `schema` through elicitation, use pre-supplied `provided` answers, or return a message.

    Clients that support elicitation are always asked directly. A client that cannot
    answer an elicitation can instead call the tool again with its `answers` argument;
    until it does, the tool returns a message telling it to pose the questions to the user.
    """
    if _supports_elicitation(ctx):
        return Elicit(prompt, schema)
    if provided is not None:
        return provided
    return ElicitationUnsupported(
        message=(
            "This client does not support MCP elicitation, so this server cannot ask "
            "the user directly. Please pose the following question(s) to the user, then "
            "call this tool again with the `answers` argument:\n"
            f"{prompt}\n{_render_questions(schema)}"
        )
    )


def _unsupported_message(result: ElicitationResult[Any]) -> str | None:
    """The fallback message when `result` is an `ElicitationUnsupported` payload."""
    if isinstance(result, AcceptedElicitation) and isinstance(result.data, ElicitationUnsupported):
        return result.data.message
    return None


# --------------------------------------------------------------------------- #
# Stage 1: greeting
# --------------------------------------------------------------------------- #


class Greeting(BaseModel):
    feeling: str = Field(description="How are you doing today?")
    arrival: Literal["bike", "train", "car", "by foot"] = Field(description="How did you get here?")
    wants_pizza: bool = Field(description="Would you like to order a pizza?")


async def greet_resolver(
    ctx: Context, answers: Greeting | None
) -> Elicit[Greeting] | Greeting | ElicitationUnsupported:
    _require_stage(Stage.GREETING)
    return _elicit_or_message(ctx, "Welcome to Pizza MCP!", Greeting, answers)


async def greet_customer(
    result: Annotated[ElicitationResult[Greeting], Resolve(greet_resolver)],
    answers: Greeting | None = None,
) -> str:
    """Greet the customer and ask how they are, how they arrived, and whether they want pizza.

    Available during the greeting stage. `answers` is only needed by clients without
    elicitation support: such a client asks the user the questions it is handed, then
    calls this tool again with `answers` filled in.
    """
    if (message := _unsupported_message(result)) is not None:
        return message

    if not isinstance(result, AcceptedElicitation):
        return "No problem. Come back whenever you feel like a pizza!"

    data = result.data
    if not data.wants_pizza:
        return (
            f"Thanks for stopping by and {data.feeling or 'take care'}! "
            "Maybe next time you'll want a pizza."
        )

    _enter_stage(Stage.ORDER)
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


async def order_resolver(
    ctx: Context, answers: Order | None
) -> Elicit[Order] | Order | ElicitationUnsupported:
    _require_stage(Stage.ORDER)
    return _elicit_or_message(ctx, "Let's build your pizza.", Order, answers)


async def take_order(
    result: Annotated[ElicitationResult[Order], Resolve(order_resolver)],
    answers: Order | None = None,
) -> str:
    """Take the pizza order: crust filling, pizza type, and any extras.

    Available during the ordering stage. `answers` is only needed by clients without
    elicitation support: such a client asks the user the questions it is handed, then
    calls this tool again with `answers` filled in.
    """
    if (message := _unsupported_message(result)) is not None:
        return message

    if not isinstance(result, AcceptedElicitation):
        _enter_stage(Stage.GREETING)
        return "Order cancelled. Let me know if you change your mind."

    data = result.data
    order.update(crust=data.crust, pizza_type=data.pizza_type, extras=data.extras)

    _enter_stage(Stage.PIZZA_MAKING)
    extras = f" plus {data.extras}" if data.extras else ""
    return f"One {data.pizza_type} with {data.crust}{extras}. Let's make it!"


# --------------------------------------------------------------------------- #
# Stage 3: pizza-making
# --------------------------------------------------------------------------- #


async def make_dough() -> str:
    """Make the dough for the pizza.

    Available during the pizza-making stage, as step 1 of 5.
    """
    _require_pizza_step("make_dough")
    _complete_pizza_step()
    return f"Kneading the dough for a {order.get('pizza_type', 'pizza')}."


async def spread_dough() -> str:
    """Spread and shape the dough.

    Available during the pizza-making stage, as step 2 of 5.
    """
    _require_pizza_step("spread_dough")
    _complete_pizza_step()
    return "Spreading the dough into a nice round base."


async def add_topings() -> str:
    """Add the toppings to the pizza.

    Available during the pizza-making stage, as step 3 of 5.
    """
    _require_pizza_step("add_topings")
    _complete_pizza_step()
    extras = order.get("extras")
    extra_note = f" Adding your extras: {extras}." if extras else ""
    return f"Adding tomato, mozzarella and friends.{extra_note}"


async def bake_pizza() -> str:
    """Bake the pizza in the oven.

    Available during the pizza-making stage, as step 4 of 5.
    """
    _require_pizza_step("bake_pizza")
    _complete_pizza_step()
    return "Baking the pizza until the crust is golden."


async def serve_pizza() -> str:
    """Serve the finished pizza and move on to the feedback stage.

    Available during the pizza-making stage, as step 5 of 5.
    """
    _require_pizza_step("serve_pizza")
    _complete_pizza_step()
    _enter_stage(Stage.FEEDBACK)
    return "Here is your pizza! Buon appetito."


# --------------------------------------------------------------------------- #
# Stage 4: feedback
# --------------------------------------------------------------------------- #


class Feedback(BaseModel):
    liked: bool = Field(description="Did you like your experience?")
    comments: str = Field(default="", description="Anything you'd like to say? Leave blank for none.")


async def feedback_resolver(
    ctx: Context, answers: Feedback | None
) -> Elicit[Feedback] | Feedback | ElicitationUnsupported:
    _require_stage(Stage.FEEDBACK)
    return _elicit_or_message(ctx, "Before you go, how was your experience?", Feedback, answers)


async def collect_feedback(
    result: Annotated[ElicitationResult[Feedback], Resolve(feedback_resolver)],
    answers: Feedback | None = None,
) -> str:
    """Ask whether the customer liked their experience and collect any comments.

    Available during the feedback stage. `answers` is only needed by clients without
    elicitation support: such a client asks the user the questions it is handed, then
    calls this tool again with `answers` filled in.
    """
    if (message := _unsupported_message(result)) is not None:
        return message

    if not isinstance(result, AcceptedElicitation):
        _enter_stage(Stage.GREETING)
        return "Thanks for visiting!"

    data = result.data
    closing = "Grazie! We're glad you enjoyed it." if data.liked else "Sorry to hear that - we'll do better."
    if data.comments:
        closing += f" Your feedback: {data.comments}"

    _enter_stage(Stage.GREETING)
    return closing


# --------------------------------------------------------------------------- #
# Tool registration
# --------------------------------------------------------------------------- #

STAGE_TOOLS: dict[Stage, list[Callable[..., Any]]] = {
    Stage.GREETING: [greet_customer],
    Stage.ORDER: [take_order],
    Stage.PIZZA_MAKING: [make_dough, spread_dough, add_topings, bake_pizza, serve_pizza],
    Stage.FEEDBACK: [collect_feedback],
}


def _register_tools() -> None:
    """Expose every tool up front; stage gating happens inside each tool."""
    order.clear()
    for tools in STAGE_TOOLS.values():
        for fn in tools:
            server.add_tool(fn)


_register_tools()


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
