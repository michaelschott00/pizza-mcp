"""Pizza MCP Server.

A toy MCP server that runs a pizza-order interview. The interview asks its
questions up front - `greet_customer`, then `take_order`, then `choose_serving`,
always in that order - and then reads the answers back with `confirm_order`,
which asks the customer to sign off on the summarized order. Only once the order
is confirmed can the pizza be made. The server demonstrates three MCP features:

* elicitation, with a capability-aware fallback - the interview questions are
  asked through the SDK's elicitation support (via `Resolve`/`Elicit` resolvers,
  which work on every protocol version). Clients that cannot answer elicitation
  requests are served a different implementation of the questioning tools, whose
  questions live in the `answers` argument. Which implementation a client
  receives is decided from the capabilities it declares when it connects, so
  neither the tools nor the instructions have to mention the distinction.
* recommended answers - a question tool takes an optional `recommendation`, the
  answers the model derived from the prompt that started the conversation. They
  are offered to the customer alongside the normal options (as form defaults in
  the elicitation variant) so a correct guess never has to be repeated.
* stage-gated tools - every tool is always exposed to the client, but each one
  refuses to run (with an informative error) unless the interview is currently
  in the stage it belongs to.

Pass `--no-elicitation` to force the answers-based tools even for clients that
advertise elicitation support (useful for testing the fallback with clients
such as MCP Inspector). Elicitation support is enabled by default.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from enum import Enum, auto
from typing import Annotated, Any, Literal

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

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
        "Guide the customer through an interactive pizza ordering interview. Ask "
        "the questions in order: `greet_customer` (how they are doing and how they "
        "arrived), then `take_order` (the pizza), then `choose_serving` (eat here "
        "or to go). `choose_serving` replies with the full order and tells you to "
        "confirm it; do so with `confirm_order`, which summarizes every answer and "
        "asks the customer to sign off before the pizza is made. Each question tool "
        "optionally takes a `recommendation` - answers you derived from the "
        "customer's own words - which are offered alongside the normal options. "
        "Every tool is always available, but each tool only works during the stage "
        "it belongs to: the questions must be asked in order, and the pizza-making "
        "tools (`make_dough`, `spread_dough`, `add_topings`, `bake_pizza`, "
        "`serve_pizza`) cannot be used until the order is confirmed. Calling a tool "
        "outside its stage returns an error naming the current stage and the "
        "correct sequence of tools to call."
    ),
)


class Stage(Enum):
    GREETING = auto()
    ORDER = auto()
    SERVING = auto()
    CONFIRMATION = auto()
    PIZZA_MAKING = auto()


STAGE_LABELS: dict[Stage, str] = {
    Stage.GREETING: "greeting",
    Stage.ORDER: "ordering",
    Stage.SERVING: "serving",
    Stage.CONFIRMATION: "confirmation",
    Stage.PIZZA_MAKING: "pizza-making",
}

# Each stage tells us the question (or serving step) that moves the interview on
# to the next one. The questions are asked in this order and must not be
# reordered; `serve_pizza` closes the loop by starting a fresh interview.
STAGE_TRANSITIONS: dict[Stage, tuple[Stage, str]] = {
    Stage.GREETING: (Stage.ORDER, "greet_customer"),
    Stage.ORDER: (Stage.SERVING, "take_order"),
    Stage.SERVING: (Stage.CONFIRMATION, "choose_serving"),
    Stage.CONFIRMATION: (Stage.PIZZA_MAKING, "confirm_order"),
    Stage.PIZZA_MAKING: (Stage.GREETING, "serve_pizza"),
}

# The questions, in the order they have to be asked. Errors quote this so a tool
# called out of order always names the correct sequence.
QUESTION_SEQUENCE: tuple[str, ...] = ("greet_customer", "take_order", "choose_serving", "confirm_order")

PIZZA_STEPS: list[str] = ["make_dough", "spread_dough", "add_topings", "bake_pizza", "serve_pizza"]


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
#
# One model per questioning tool. The same model drives elicitation (as the
# requested schema) and the `answers` fallback (as the tool's `answers` argument).
# Each also has a matching `...Recommendation`: the same fields, but all optional,
# so the model may suggest only the answers it actually knows from the customer's
# prompt. The elicitation variant fills the recommendation in as a form default
# and names it in the prompt; the answers variant asks the customer itself, so it
# needs no recommendation argument.


class Greeting(BaseModel):
    feeling: str = Field(description="How are you doing today?")
    arrival: Literal["bike", "train", "car", "by foot"] = Field(description="How did you get here?")


class GreetingRecommendation(BaseModel):
    feeling: str | None = Field(default=None, description="How are you doing today?")
    arrival: Literal["bike", "train", "car", "by foot"] | None = Field(
        default=None, description="How did you get here?"
    )


class Order(BaseModel):
    crust: Literal["cheese crust", "sausage crust", "no crust filling"] = Field(
        description="What kind of crust filling would you like?"
    )
    pizza_type: Literal["margarita", "speziale", "four cheese"] = Field(
        description="Which pizza would you like?"
    )
    extras: str = Field(default="", description="Anything else? Leave blank for none.")


class OrderRecommendation(BaseModel):
    crust: Literal["cheese crust", "sausage crust", "no crust filling"] | None = Field(
        default=None, description="What kind of crust filling would you like?"
    )
    pizza_type: Literal["margarita", "speziale", "four cheese"] | None = Field(
        default=None, description="Which pizza would you like?"
    )
    extras: str | None = Field(default=None, description="Anything else? Leave blank for none.")


class Serving(BaseModel):
    serving: Literal["eat here", "to go"] = Field(description="Would you like to eat here or take it to go?")


class ServingRecommendation(BaseModel):
    serving: Literal["eat here", "to go"] | None = Field(
        default=None, description="Would you like to eat here or take it to go?"
    )


class Confirmation(BaseModel):
    confirmed: bool = Field(description="Do you confirm this order?")


def recommended_schema(schema: type[BaseModel], recommendation: BaseModel | None) -> type[BaseModel]:
    """`schema` with the recommended values as field defaults.

    The defaults are what a form-elicitation client offers the customer: every
    option is still listed, but the recommendation is the one pre-filled, so
    accepting a correct guess takes no typing. A recommendation may cover only
    some of the fields; the rest stay required.
    """
    if recommendation is None:
        return schema
    fields: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        value = getattr(recommendation, name, None)
        overrides = (FieldInfo(default=value),) if value is not None else ()
        fields[name] = (field.annotation, FieldInfo.merge_field_infos(field, *overrides))
    return create_model(schema.__name__, **fields)


def recommendation_note(schema: type[BaseModel], recommendation: BaseModel | None) -> str:
    """The recommendation spelled out for the prompt text, or an empty string."""
    if recommendation is None:
        return ""
    suggested = {
        name: getattr(recommendation, name)
        for name in schema.model_fields
        if getattr(recommendation, name, None) is not None
    }
    if not suggested:
        return ""
    lines = "\n".join(f"- {name}: {value}" for name, value in suggested.items())
    return f"\nSuggested answers (accept or choose your own):\n{lines}"


# --------------------------------------------------------------------------- #
# The interview
# --------------------------------------------------------------------------- #
#
# One interview backs every tool variant: the elicitation-based tools, the
# answer-based tools and the capability-independent pizza-making steps all
# operate on the same instance, so the interview stays consistent whichever
# variant a client happens to be served.


class Interview:
    """State, stage rules, prompt texts and domain logic for the pizza interview."""

    # Prompt texts for the elicitation resolvers; the answers fallback carries its
    # questions in the `answers` field descriptions instead.
    GREET_PROMPT = "Welcome to Pizza MCP!"
    ORDER_PROMPT = "Let's build your pizza."
    SERVING_PROMPT = "Almost there - how would you like your pizza?"

    def __init__(self) -> None:
        self.stage = Stage.GREETING
        self.pizza_step = 0
        self.order: dict[str, Any] = {}

    def reset(self) -> None:
        """Start a fresh interview, forgetting the previous order."""
        self.stage = Stage.GREETING
        self.pizza_step = 0
        self.order.clear()

    # -- stage gating ------------------------------------------------------- #

    def _advancing_tools(self, target: Stage, tool_name: str | None = None) -> list[str]:
        """The tools to call, in order, to move the interview from the current stage to `target`."""
        tools: list[str] = []
        stage = self.stage
        for _ in range(len(STAGE_TRANSITIONS)):
            if stage is target:
                break
            next_stage, tool = STAGE_TRANSITIONS[stage]
            tools.append(tool)
            stage = next_stage
        # The pizza-making steps are reached through `choose_serving`; when a
        # specific step is asked for, every earlier step comes before it too.
        if target is Stage.PIZZA_MAKING and tool_name in PIZZA_STEPS:
            tools.extend(PIZZA_STEPS[: PIZZA_STEPS.index(tool_name)])
        return tools

    def enter(self, stage: Stage) -> None:
        """Move the interview to `stage`."""
        self.stage = stage
        if stage is Stage.PIZZA_MAKING:
            self.pizza_step = 0

    def require_stage(self, stage: Stage, tool_name: str | None = None) -> None:
        """Fail with an informative error unless the interview is in `stage`."""
        if self.stage is stage:
            return

        steps = self._advancing_tools(stage, tool_name)
        path = ", then ".join(f"`{name}`" for name in steps)
        sequence = ", then ".join(f"`{name}`" for name in QUESTION_SEQUENCE)
        raise ToolError(
            f"This tool is only available during the {STAGE_LABELS[stage]} stage, "
            f"but the interview is currently in the {STAGE_LABELS[self.stage]} stage. "
            f"The questions must be asked in this order: {sequence}. "
            f"Call {path} to advance to where this tool can be used."
        )

    def require_pizza_step(self, name: str) -> None:
        """Fail unless the interview is making pizza and `name` is the next step."""
        self.require_stage(Stage.PIZZA_MAKING, name)
        expected = PIZZA_STEPS[self.pizza_step]
        if name != expected:
            raise ToolError(
                f"The pizza-making steps must be called in order. The next step is `{expected}`."
            )

    def complete_pizza_step(self) -> None:
        self.pizza_step += 1

    # -- domain logic ------------------------------------------------------- #
    #
    # The question tools only record what the customer said; none of them
    # answers. `choose_serving` is the last question, so it is the one that
    # replies, taking the whole order into account. `confirm_order` then reads
    # every answer back and asks the customer to sign off before the pizza is
    # made; declining sends the interview back to the ordering stage.

    def greet(self, data: Greeting) -> str:
        self.order.update(feeling=data.feeling, arrival=data.arrival)
        self.enter(Stage.ORDER)
        return f"Answer {data} locked in."

    def take_order(self, data: Order) -> str:
        self.order.update(crust=data.crust, pizza_type=data.pizza_type, extras=data.extras)
        self.enter(Stage.SERVING)
        return f"Answer {data} locked in."

    def choose_serving(self, data: Serving) -> str:
        self.order["serving"] = data.serving
        self.enter(Stage.CONFIRMATION)
        return (
            f"Answer {data} locked in.\n"
            "Order complete. Confirm it once more with the user using `confirm_order`."
        )

    def summary(self) -> str:
        """Every answer so far, laid out for the customer to confirm."""
        extras = self.order.get("extras")
        extra_note = f" plus {extras}" if extras else ""
        return (
            "Please confirm your order:\n"
            f"- feeling: {self.order.get('feeling', '?')}\n"
            f"- arrival: {self.order.get('arrival', '?')}\n"
            f"- pizza: one {self.order.get('pizza_type', '?')} "
            f"with {self.order.get('crust', '?')}{extra_note}\n"
            f"- serving: {self.order.get('serving', '?')}"
        )

    def confirm_order(self, data: Confirmation) -> str:
        if data.confirmed:
            self.enter(Stage.PIZZA_MAKING)
            return "The user confirmed the order. Proceed to `make_dough`"
        # Send the interview back to ordering so the customer can correct it.
        self.enter(Stage.ORDER)
        return "The user declined the order. Start again from `take_order`"

    # -- capability-independent tools --------------------------------------- #

    async def make_dough(self) -> str:
        """Make the dough for the pizza.

        Available during the pizza-making stage, as step 1 of 5.
        """
        self.require_pizza_step("make_dough")
        self.complete_pizza_step()
        return f"Kneading the dough for a {self.order.get('pizza_type', 'pizza')}."

    async def spread_dough(self) -> str:
        """Spread and shape the dough.

        Available during the pizza-making stage, as step 2 of 5.
        """
        self.require_pizza_step("spread_dough")
        self.complete_pizza_step()
        return "Spreading the dough into a nice round base."

    async def add_topings(self) -> str:
        """Add the toppings to the pizza.

        Available during the pizza-making stage, as step 3 of 5.
        """
        self.require_pizza_step("add_topings")
        self.complete_pizza_step()
        extras = self.order.get("extras")
        extra_note = f" Adding your extras: {extras}." if extras else ""
        return f"Adding tomato, mozzarella and friends.{extra_note}"

    async def bake_pizza(self) -> str:
        """Bake the pizza in the oven.

        Available during the pizza-making stage, as step 4 of 5.
        """
        self.require_pizza_step("bake_pizza")
        self.complete_pizza_step()
        return "Baking the pizza until the crust is golden."

    async def serve_pizza(self) -> str:
        """Serve the finished pizza and start a fresh interview.

        Available during the pizza-making stage, as step 5 of 5.
        """
        self.require_pizza_step("serve_pizza")
        self.complete_pizza_step()
        serving = self.order.get("serving", "to go")
        self.reset()
        if serving == "eat here":
            return "Here is your pizza! Enjoy it here. Buon appetito."
        return "Here is your pizza, packed up to go. Buon appetito."

    # -- elicitation resolvers ---------------------------------------------- #
    #
    # `@staticmethod` is deliberate: the SDK introspects a resolver referenced in
    # a `Resolve(...)` annotation as a plain function, so an ordinary method would
    # surface its `self` parameter as an unresolvable argument. The stage check has
    # to happen here - before the client is asked - so a tool called out of stage
    # fails without prompting. Shared state is reached through `INTERVIEW`.

    # A resolver may read a tool argument by name, which is how the optional
    # `recommendation` the model supplied reaches the question it belongs to.
    @staticmethod
    async def greet_resolver(recommendation: GreetingRecommendation | None = None) -> Elicit[Greeting]:
        INTERVIEW.require_stage(Stage.GREETING)
        message = Interview.GREET_PROMPT + recommendation_note(Greeting, recommendation)
        return Elicit(message, recommended_schema(Greeting, recommendation))

    @staticmethod
    async def order_resolver(recommendation: OrderRecommendation | None = None) -> Elicit[Order]:
        INTERVIEW.require_stage(Stage.ORDER)
        message = Interview.ORDER_PROMPT + recommendation_note(Order, recommendation)
        return Elicit(message, recommended_schema(Order, recommendation))

    @staticmethod
    async def serving_resolver(recommendation: ServingRecommendation | None = None) -> Elicit[Serving]:
        INTERVIEW.require_stage(Stage.SERVING)
        message = Interview.SERVING_PROMPT + recommendation_note(Serving, recommendation)
        return Elicit(message, recommended_schema(Serving, recommendation))

    @staticmethod
    async def confirm_resolver() -> Elicit[Confirmation]:
        INTERVIEW.require_stage(Stage.CONFIRMATION)
        return Elicit(INTERVIEW.summary(), Confirmation)


INTERVIEW = Interview()


# --------------------------------------------------------------------------- #
# Capability-aware tool exposure
# --------------------------------------------------------------------------- #
#
# Every tool that asks the customer a question is implemented twice, once per
# client capability, and the implementation lives in its own namespace:
#
#   * `ElicitationTools` (`elicit.*`) asks through the SDK's elicitation support,
#     and takes an optional `recommendation` whose answers become form defaults.
#   * `AnswersTools` (`answers.*`) takes the customer's reply through a single
#     required `answers` argument; the questions live in the parameter
#     descriptions, so the tool never has to be called twice.
#
# Both variants are always registered; `capability_middleware` decides per
# connection which namespace it exposes and routes calls into it, so the
# interview looks and behaves the same to an agent either way.


# TODO: make abstract (can't sensibly be initialized without setting namespace, for example)
class Toolset:
    """A namespace of questioning tools for one client capability.

    Both subclasses implement the same public tools and delegate to the shared
    `INTERVIEW`; the wrapper is a namespace, not a state holder.
    """

    namespace: str
    tool_names: tuple[str, ...]

    def __init__(self, interview: Interview) -> None:
        self.interview = interview

    @property
    def routes(self) -> dict[str, str]:
        """Public tool name -> the name this toolset registers it under."""
        return {public: f"{self.namespace}.{public}" for public in self.tool_names}

    @property
    def advertise(self) -> dict[str, str]:
        """Registered tool name -> the public name a client sees it under."""
        return {registered: public for public, registered in self.routes.items()}

    def register(self, server: MCPServer) -> None:
        """Register this toolset's tools under their namespaced names."""
        for public in self.tool_names:
            server.add_tool(getattr(self, public), name=f"{self.namespace}.{public}")


class ElicitationTools(Toolset):
    """The questioning tools for clients that can answer elicitation requests."""

    namespace = "elicit"
    tool_names = ("greet_customer", "take_order", "choose_serving", "confirm_order")

    async def greet_customer(
        self,
        result: Annotated[ElicitationResult[Greeting], Resolve(Interview.greet_resolver)],
        recommendation: GreetingRecommendation | None = None,
    ) -> str:
        """Ask how the customer is doing and how they arrived.

        Pass `recommendation` to suggest answers you derived from the customer's
        request; they are offered alongside the normal options.
        """
        if not isinstance(result, AcceptedElicitation):
            self.interview.reset()
            return "No problem. Come back whenever you feel like a pizza!"
        return self.interview.greet(result.data)

    async def take_order(
        self,
        result: Annotated[ElicitationResult[Order], Resolve(Interview.order_resolver)],
        recommendation: OrderRecommendation | None = None,
    ) -> str:
        """Ask which pizza the customer wants: crust filling, pizza type and extras.

        Pass `recommendation` to suggest answers you derived from the customer's
        request; they are offered alongside the normal options.
        """
        if not isinstance(result, AcceptedElicitation):
            self.interview.reset()
            return "Order cancelled. Let me know if you change your mind."
        return self.interview.take_order(result.data)

    async def choose_serving(
        self,
        result: Annotated[ElicitationResult[Serving], Resolve(Interview.serving_resolver)],
        recommendation: ServingRecommendation | None = None,
    ) -> str:
        """Ask whether the customer will eat here or take the pizza to go.

        Pass `recommendation` to suggest an answer you derived from the customer's
        request; it is offered alongside the normal options. This is the last
        question; confirm the complete order with `confirm_order` next.
        """
        if not isinstance(result, AcceptedElicitation):
            self.interview.reset()
            return "Order cancelled. Let me know if you change your mind."
        return self.interview.choose_serving(result.data)

    async def confirm_order(
        self,
        result: Annotated[ElicitationResult[Confirmation], Resolve(Interview.confirm_resolver)],
    ) -> str:
        """Read every answer back and ask the customer to confirm the order.

        Summarizes the whole interview and asks the customer to sign off. Call
        this after `choose_serving` and before the pizza-making steps.
        """
        if not isinstance(result, AcceptedElicitation):
            self.interview.reset()
            return "Order cancelled. Let me know if you change your mind."
        return self.interview.confirm_order(result.data)


class AnswersTools(Toolset):
    """The questioning tools for clients that cannot answer elicitation requests.

    The customer's reply travels through the tool's `answers` argument instead.
    The questions are the descriptions of that argument's fields, so the model
    can put every question to the customer up front and call each tool once.
    """

    namespace = "answers"
    tool_names = ("greet_customer", "take_order", "choose_serving", "confirm_order")

    async def greet_customer(self, answers: Greeting) -> str:
        """Ask how the customer is doing (`feeling`) and how they arrived (`arrival`).

        Ask the customer both questions before calling this tool.
        """
        self.interview.require_stage(Stage.GREETING)
        return self.interview.greet(answers)

    async def take_order(self, answers: Order) -> str:
        """Ask which pizza the customer wants: `crust`, `pizza_type` and `extras`.

        Ask the customer these questions before calling this tool.
        """
        self.interview.require_stage(Stage.ORDER)
        return self.interview.take_order(answers)

    async def choose_serving(self, answers: Serving) -> str:
        """Ask whether the customer will `eat here` or take the pizza `to go`.

        Ask the customer this question before calling this tool. This is the last
        question; confirm the complete order with `confirm_order` next.
        """
        self.interview.require_stage(Stage.SERVING)
        return self.interview.choose_serving(answers)

    async def confirm_order(self, answers: Confirmation) -> str:
        """Read every answer back and ask the customer to confirm the order.

        Before calling this tool, summarize the whole interview for the customer
        and ask them to sign off, then pass their reply in `confirmed`. Call this
        after `choose_serving` and before the pizza-making steps.
        """
        self.interview.require_stage(Stage.CONFIRMATION)
        return self.interview.confirm_order(answers)


# The registry: which toolset serves which client, keyed by elicitation support.
TOOLSETS: dict[bool, Toolset] = {
    True: ElicitationTools(INTERVIEW),
    False: AnswersTools(INTERVIEW),
}

# Every registered name that belongs to a capability-specific toolset. Lets the
# middleware tell a variant it must hide from one a client may call.
OWNERS: dict[str, Toolset] = {
    registered: toolset for toolset in TOOLSETS.values() for registered in toolset.advertise
}

# Whether the server may use elicitation at all. `--no-elicitation` clears this,
# forcing the answers-based toolset on every client - including those that
# advertise elicitation support, such as MCP Inspector.
ELICITATION_ENABLED = True


def advertised_name(name: str, supports_elicitation: bool) -> str | None:
    """The name a registered tool is exposed under to a client, or None when hidden."""
    active = TOOLSETS[supports_elicitation]
    owner = OWNERS.get(name)
    if owner is not None and owner is not active:
        return None
    return active.advertise.get(name, name)


def _client_supports_elicitation(ctx: ServerRequestContext[Any, Any]) -> bool:
    """Whether the connected client declared form elicitation support.

    Mirrors the SDK's own capability gate: a bare `elicitation: {}` counts as form
    support, while URL-only elicitation does not.
    """
    capabilities = ctx.session.client_capabilities
    elicitation = capabilities.elicitation if capabilities is not None else None
    return elicitation is not None and (elicitation.form is not None or elicitation.url is None)


def _relabel(result: HandlerResult, mapping: dict[str, str]) -> HandlerResult:
    """Present internal tool names to the client under their advertised names.

    The SDK prefixes tool failures with the *registered* name (e.g. `Error
    executing tool elicit.greet_customer: ...`), so content a caller can see is
    rewritten through the mapping before it leaves the server.
    """
    if not isinstance(result, dict):
        return result
    content = result.get("content")
    if not isinstance(content, list):
        return result
    relabelled: list[Any] = []
    for item in content:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            text = item["text"]
            for internal, public in mapping.items():
                text = text.replace(internal, public)
            item = {**item, "text": text}
        relabelled.append(item)
    return {**result, "content": relabelled}


async def capability_middleware(ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
    """Expose each connection only the tool variant its capabilities can run.

    Picks the toolset for the connection's capabilities, then rewrites the
    advertised tool list and the tool name being called to that toolset, so the
    interview looks the same to an agent either way.
    """
    if ctx.method not in ("tools/list", "tools/call"):
        return await call_next(ctx)

    supports = ELICITATION_ENABLED and _client_supports_elicitation(ctx)
    active = TOOLSETS[supports]

    if ctx.method == "tools/call":
        params = dict(ctx.params or {})
        name = params.get("name")
        if isinstance(name, str):
            # Map our internal names back to their public name, so a client can
            # never reach another capability's implementation by name either.
            owner = OWNERS.get(name)
            public = owner.advertise[name] if owner is not None else name
            target = active.routes.get(public)
            if target is not None:
                result = await call_next(replace(ctx, params={**params, "name": target}))
                return _relabel(result, active.advertise)
        return await call_next(ctx)

    result = await call_next(ctx)
    if isinstance(result, dict) and isinstance(result.get("tools"), list):
        tools: list[Any] = []
        for tool in result["tools"]:
            name = tool.get("name", "")
            advertised = advertised_name(name, supports)
            if advertised is None:
                continue
            tools.append({**tool, "name": advertised} if advertised != name else tool)
        result = {**result, "tools": tools}
    return result


server.middleware.append(capability_middleware)


# --------------------------------------------------------------------------- #
# Tool registration
# --------------------------------------------------------------------------- #


def _register_tools() -> None:
    """Register every tool variant; `capability_middleware` narrows them per client.

    The capability-independent pizza-making steps are registered once, under
    their own names; each toolset registers its variants under its namespace.
    """
    INTERVIEW.order.clear()
    for name in PIZZA_STEPS:
        server.add_tool(getattr(INTERVIEW, name))
    for toolset in TOOLSETS.values():
        toolset.register(server)


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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the server's command-line options."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run the Pizza MCP server over stdio.",
    )
    parser.add_argument(
        "--elicitation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use the elicitation-based tools with clients that support elicitation "
            "(default: enabled); --no-elicitation forces the answers-based tools"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global ELICITATION_ENABLED
    args = _parse_args(argv)
    ELICITATION_ENABLED = args.elicitation

    provider = _configure_telemetry()
    try:
        server.run()
    finally:
        # Flush the spans the batch processor has buffered before exiting.
        provider.shutdown()


if __name__ == "__main__":
    main()
