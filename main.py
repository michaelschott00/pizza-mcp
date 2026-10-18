"""Pizza MCP Server.

A toy MCP server that runs an interactive pizza-order interview. It demonstrates
two MCP features:

* elicitation, with a capability-aware fallback - the interview questions are
  asked through the SDK's elicitation support (via `Resolve`/`Elicit` resolvers,
  which work on every protocol version). Clients that cannot answer elicitation
  requests are served a different set of tools whose questions travel through an
  `answers` argument instead. Which set a client receives is decided from the
  capabilities it declares when it connects, so neither the tools nor the
  instructions have to mention the distinction.
* stage-gated tools - every tool is always exposed to the client, but each one
  refuses to run (with an informative error) unless the interview is currently
  in the stage it belongs to.
"""

from __future__ import annotations

import os
from dataclasses import replace
from enum import Enum, auto
from typing import Annotated, Any, Callable, Literal

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel, Field

from mcp.server import MCPServer
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.mcpserver import (
    AcceptedElicitation,
    Elicit,
    ElicitationResult,
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
        "reach the stage you need. The stages are: greeting, ordering, "
        "pizza-making, and feedback."
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
# Capability-aware tool exposure
# --------------------------------------------------------------------------- #
#
# Every tool that asks the user a question is registered twice: once as a
# canonical tool that asks through elicitation, and once as a `<name>_answers`
# counterpart for clients that cannot answer elicitation requests (its questions
# come back through the tool's `answers` argument). The middleware below decides
# per connection which of the two a client is shown - and which one runs when it
# calls the canonical name - from the capabilities it declared on connect.

ANSWER_TOOL_SUFFIX = "_answers"

# The canonical names of the tools that have an `_answers` counterpart.
ELICITING_TOOL_NAMES = frozenset({"greet_customer", "take_order", "collect_feedback"})


def advertised_tool_name(name: str, supports_elicitation: bool) -> str | None:
    """The name a registered tool is exposed under to a client, or None when hidden.

    Both implementations of an eliciting tool are always registered, but a client
    only ever sees the one matching its capabilities, and always under the
    canonical name.
    """
    if name.endswith(ANSWER_TOOL_SUFFIX):
        return name[: -len(ANSWER_TOOL_SUFFIX)] if not supports_elicitation else None
    if name in ELICITING_TOOL_NAMES:
        return name if supports_elicitation else None
    return name


def _call_target(name: str, supports_elicitation: bool) -> str:
    """The registered implementation to run for an exposed tool `name`."""
    if name.endswith(ANSWER_TOOL_SUFFIX):
        name = name[: -len(ANSWER_TOOL_SUFFIX)]
    if name in ELICITING_TOOL_NAMES and not supports_elicitation:
        return name + ANSWER_TOOL_SUFFIX
    return name


def _relabel(result: HandlerResult, internal: str, public: str) -> HandlerResult:
    """Present an internal tool name to the client under its advertised name."""
    if not isinstance(result, dict):
        return result
    content = result.get("content")
    if not isinstance(content, list):
        return result
    relabelled: list[Any] = []
    for item in content:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            item = {**item, "text": item["text"].replace(internal, public)}
        relabelled.append(item)
    return {**result, "content": relabelled}


def _client_supports_elicitation(ctx: ServerRequestContext[Any, Any]) -> bool:
    """Whether the connected client declared form elicitation support.

    Mirrors the SDK's own capability gate: a bare `elicitation: {}` counts as form
    support, while URL-only elicitation does not.
    """
    capabilities = ctx.session.client_capabilities
    elicitation = capabilities.elicitation if capabilities is not None else None
    return elicitation is not None and (elicitation.form is not None or elicitation.url is None)


async def capability_middleware(ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
    """Expose each client only the tool variant its capabilities can run.

    Rewrites the advertised tool list and the tool name being called to the set
    matching the connection's capabilities, so the interview looks the same to an
    agent either way.
    """
    if ctx.method not in ("tools/list", "tools/call"):
        return await call_next(ctx)

    supports = _client_supports_elicitation(ctx)

    if ctx.method == "tools/call":
        params = dict(ctx.params or {})
        name = params.get("name")
        if isinstance(name, str):
            target = _call_target(name, supports)
            if target != name:
                result = await call_next(replace(ctx, params={**params, "name": target}))
                if target.endswith(ANSWER_TOOL_SUFFIX) and name == target[: -len(ANSWER_TOOL_SUFFIX)]:
                    result = _relabel(result, target, name)
                return result
        return await call_next(ctx)

    result = await call_next(ctx)
    if isinstance(result, dict) and isinstance(result.get("tools"), list):
        tools: list[Any] = []
        for tool in result["tools"]:
            advertised = advertised_tool_name(tool.get("name", ""), supports)
            if advertised is None:
                continue
            tools.append({**tool, "name": advertised} if advertised != tool.get("name") else tool)
        result = {**result, "tools": tools}
    return result


server.middleware.append(capability_middleware)


def _questions_message(prompt: str, schema: type[BaseModel]) -> str:
    """The question(s) of `schema` for a client that must relay them to the user."""
    questions = "\n".join(
        f"- {field.description or name}" for name, field in schema.model_fields.items()
    )
    return (
        "Please pose the following question(s) to the user, then call this tool again "
        f"with the `answers` argument:\n{prompt}\n{questions}"
    )


# --------------------------------------------------------------------------- #
# Stage 1: greeting
# --------------------------------------------------------------------------- #


class Greeting(BaseModel):
    feeling: str = Field(description="How are you doing today?")
    arrival: Literal["bike", "train", "car", "by foot"] = Field(description="How did you get here?")
    wants_pizza: bool = Field(description="Would you like to order a pizza?")


def _greet(data: Greeting) -> str:
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


async def greet_resolver() -> Elicit[Greeting]:
    _require_stage(Stage.GREETING)
    return Elicit("Welcome to Pizza MCP!", Greeting)


async def greet_customer(
    result: Annotated[ElicitationResult[Greeting], Resolve(greet_resolver)],
) -> str:
    """Greet the customer: ask how they are doing, how they arrived and whether they want pizza."""
    if not isinstance(result, AcceptedElicitation):
        return "No problem. Come back whenever you feel like a pizza!"
    return _greet(result.data)


async def greet_customer_answers(answers: Greeting | None = None) -> str:
    """Greet the customer: ask how they are doing, how they arrived and whether they want pizza.

    Ask the user the questions this tool returns, then call it again with `answers`.
    """
    _require_stage(Stage.GREETING)
    if answers is None:
        return _questions_message("Welcome to Pizza MCP!", Greeting)
    return _greet(answers)


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


def _take_order(data: Order) -> str:
    order.update(crust=data.crust, pizza_type=data.pizza_type, extras=data.extras)

    _enter_stage(Stage.PIZZA_MAKING)
    extras = f" plus {data.extras}" if data.extras else ""
    return f"One {data.pizza_type} with {data.crust}{extras}. Let's make it!"


async def order_resolver() -> Elicit[Order]:
    _require_stage(Stage.ORDER)
    return Elicit("Let's build your pizza.", Order)


async def take_order(
    result: Annotated[ElicitationResult[Order], Resolve(order_resolver)],
) -> str:
    """Take the pizza order: crust filling, pizza type and any extras."""
    if not isinstance(result, AcceptedElicitation):
        _enter_stage(Stage.GREETING)
        return "Order cancelled. Let me know if you change your mind."
    return _take_order(result.data)


async def take_order_answers(answers: Order | None = None) -> str:
    """Take the pizza order: crust filling, pizza type and any extras.

    Ask the user the questions this tool returns, then call it again with `answers`.
    """
    _require_stage(Stage.ORDER)
    if answers is None:
        return _questions_message("Let's build your pizza.", Order)
    return _take_order(answers)


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


def _collect_feedback(data: Feedback) -> str:
    closing = "Grazie! We're glad you enjoyed it." if data.liked else "Sorry to hear that - we'll do better."
    if data.comments:
        closing += f" Your feedback: {data.comments}"

    _enter_stage(Stage.GREETING)
    return closing


async def feedback_resolver() -> Elicit[Feedback]:
    _require_stage(Stage.FEEDBACK)
    return Elicit("Before you go, how was your experience?", Feedback)


async def collect_feedback(
    result: Annotated[ElicitationResult[Feedback], Resolve(feedback_resolver)],
) -> str:
    """Ask whether the customer liked their experience and collect any comments."""
    if not isinstance(result, AcceptedElicitation):
        _enter_stage(Stage.GREETING)
        return "Thanks for visiting!"
    return _collect_feedback(result.data)


async def collect_feedback_answers(answers: Feedback | None = None) -> str:
    """Ask whether the customer liked their experience and collect any comments.

    Ask the user the questions this tool returns, then call it again with `answers`.
    """
    _require_stage(Stage.FEEDBACK)
    if answers is None:
        return _questions_message("Before you go, how was your experience?", Feedback)
    return _collect_feedback(answers)


# --------------------------------------------------------------------------- #
# Tool registration
# --------------------------------------------------------------------------- #

TOOLS: list[Callable[..., Any]] = [
    greet_customer,
    take_order,
    collect_feedback,
    greet_customer_answers,
    take_order_answers,
    collect_feedback_answers,
    make_dough,
    spread_dough,
    add_topings,
    bake_pizza,
    serve_pizza,
]


def _register_tools() -> None:
    """Register every tool variant; `capability_middleware` narrows them per client."""
    order.clear()
    for fn in TOOLS:
        server.add_tool(fn)


_register_tools()


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #
#
# The SDK already emits one OpenTelemetry span per inbound message, but with no
# tracer provider registered those spans are discarded. Installing the SDK and
# the OTLP exporter (see `requirements.txt`) and registering them below ships
# the spans to the collector defined in `compose.yml`.

# The collector runs as the `otel` service, so it is reached by name from the
# compose network, not via localhost.
DEFAULT_OTLP_ENDPOINT = "http://otel:4317"
SERVICE_NAME = "pizza"


def _configure_telemetry() -> TracerProvider:
    """Ship the spans the SDK emits to the OTLP collector from `compose.yml`.

    The endpoint defaults to the collector's gRPC receiver and can be overridden
    with the standard `OTEL_EXPORTER_OTLP_ENDPOINT` (or
    `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) environment variables.
    """
    endpoint = os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT),
    )
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    return provider


def main() -> None:
    provider = _configure_telemetry()
    try:
        server.run()
    finally:
        # Flush the spans the batch processor has buffered before exiting.
        provider.shutdown()


if __name__ == "__main__":
    main()
