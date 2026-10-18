"""Pizza MCP Server.

A toy MCP server that runs an interactive pizza-order interview. It demonstrates
two MCP features:

* elicitation, with a capability-aware fallback - the interview questions are
  asked through the SDK's elicitation support (via `Resolve`/`Elicit` resolvers,
  which work on every protocol version). Clients that cannot answer elicitation
  requests are served a different implementation of the questioning tools, whose
  questions travel through an `answers` argument instead. Which implementation a
  client receives is decided from the capabilities it declares when it connects,
  so neither the tools nor the instructions have to mention the distinction.
* stage-gated tools - every tool is always exposed to the client, but each one
  refuses to run (with an informative error) unless the interview is currently
  in the stage it belongs to.
"""

from __future__ import annotations

import os
from dataclasses import replace
from enum import Enum, auto
from typing import Annotated, Any, Literal

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


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
#
# One model per questioning tool. The same model drives elicitation (as the
# requested schema) and the `answers` fallback (as the tool's `answers` argument).


class Greeting(BaseModel):
    feeling: str = Field(description="How are you doing today?")
    arrival: Literal["bike", "train", "car", "by foot"] = Field(description="How did you get here?")
    wants_pizza: bool = Field(description="Would you like to order a pizza?")


class Order(BaseModel):
    crust: Literal["cheese crust", "sausage crust", "no crust filling"] = Field(
        description="What kind of crust filling would you like?"
    )
    pizza_type: Literal["margarita", "speziale", "four cheese"] = Field(
        description="Which pizza would you like?"
    )
    extras: str = Field(default="", description="Anything else? Leave blank for none.")


class Feedback(BaseModel):
    liked: bool = Field(description="Did you like your experience?")
    comments: str = Field(default="", description="Anything you'd like to say? Leave blank for none.")


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

    # Prompt texts, shared by the elicitation resolvers and the `answers` fallback
    # (which renders the same text together with the field descriptions).
    GREET_PROMPT = "Welcome to Pizza MCP!"
    ORDER_PROMPT = "Let's build your pizza."
    FEEDBACK_PROMPT = "Before you go, how was your experience?"

    def __init__(self) -> None:
        self.stage = Stage.GREETING
        self.pizza_step = 0
        self.order: dict[str, Any] = {}

    # -- stage gating ------------------------------------------------------- #

    def _advancing_tools(self, target: Stage, tool_name: str | None = None) -> list[str]:
        """The tools to call, in order, to move the interview from the current stage to `target`."""
        tools: list[str] = []
        stage = self.stage
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
        raise ToolError(
            f"This tool is only available during the {STAGE_LABELS[stage]} stage, "
            f"but the interview is currently in the {STAGE_LABELS[self.stage]} stage. "
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

    # -- questions ---------------------------------------------------------- #

    @staticmethod
    def questions(prompt: str, schema: type[BaseModel]) -> str:
        """The question(s) of `schema` for a client that must relay them to the user."""
        questions = "\n".join(
            f"- {field.description or name}" for name, field in schema.model_fields.items()
        )
        return (
            "Please pose the following question(s) to the user, then call this tool again "
            f"with the `answers` argument:\n{prompt}\n{questions}"
        )

    # -- domain logic ------------------------------------------------------- #

    def greet(self, data: Greeting) -> str:
        if not data.wants_pizza:
            return (
                f"Thanks for stopping by and {data.feeling or 'take care'}! "
                "Maybe next time you'll want a pizza."
            )

        self.enter(Stage.ORDER)
        return (
            f"Great, a pizza it is! Arriving by {data.arrival}, that works. "
            "Let's put your order together."
        )

    def take_order(self, data: Order) -> str:
        self.order.update(crust=data.crust, pizza_type=data.pizza_type, extras=data.extras)

        self.enter(Stage.PIZZA_MAKING)
        extras = f" plus {data.extras}" if data.extras else ""
        return f"One {data.pizza_type} with {data.crust}{extras}. Let's make it!"

    def collect_feedback(self, data: Feedback) -> str:
        closing = "Grazie! We're glad you enjoyed it." if data.liked else "Sorry to hear that - we'll do better."
        if data.comments:
            closing += f" Your feedback: {data.comments}"

        self.enter(Stage.GREETING)
        return closing

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
        """Serve the finished pizza and move on to the feedback stage.

        Available during the pizza-making stage, as step 5 of 5.
        """
        self.require_pizza_step("serve_pizza")
        self.complete_pizza_step()
        self.enter(Stage.FEEDBACK)
        return "Here is your pizza! Buon appetito."

    # -- elicitation resolvers ---------------------------------------------- #
    #
    # `@staticmethod` is deliberate: the SDK introspects a resolver referenced in
    # a `Resolve(...)` annotation as a plain function, so an ordinary method would
    # surface its `self` parameter as an unresolvable argument. The stage check has
    # to happen here - before the client is asked - so a tool called out of stage
    # fails without prompting. Shared state is reached through `INTERVIEW`.

    @staticmethod
    async def greet_resolver() -> Elicit[Greeting]:
        INTERVIEW.require_stage(Stage.GREETING)
        return Elicit(Interview.GREET_PROMPT, Greeting)

    @staticmethod
    async def order_resolver() -> Elicit[Order]:
        INTERVIEW.require_stage(Stage.ORDER)
        return Elicit(Interview.ORDER_PROMPT, Order)

    @staticmethod
    async def feedback_resolver() -> Elicit[Feedback]:
        INTERVIEW.require_stage(Stage.FEEDBACK)
        return Elicit(Interview.FEEDBACK_PROMPT, Feedback)


INTERVIEW = Interview()


# --------------------------------------------------------------------------- #
# Capability-aware tool exposure
# --------------------------------------------------------------------------- #
#
# Every tool that asks the user a question is implemented twice, once per client
# capability, and the implementation lives in its own namespace:
#
#   * `ElicitationTools` (`elicit.*`) asks through the SDK's elicitation support.
#   * `AnswersTools` (`answers.*`) returns the questions for the client to relay
#     to the user and takes the user's reply back through an `answers` argument.
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
    tool_names = ("greet_customer", "take_order", "collect_feedback")

    async def greet_customer(
        self,
        result: Annotated[ElicitationResult[Greeting], Resolve(Interview.greet_resolver)],
    ) -> str:
        """Greet the customer: ask how they are doing, how they arrived and whether they want pizza."""
        if not isinstance(result, AcceptedElicitation):
            return "No problem. Come back whenever you feel like a pizza!"
        return self.interview.greet(result.data)

    async def take_order(
        self,
        result: Annotated[ElicitationResult[Order], Resolve(Interview.order_resolver)],
    ) -> str:
        """Take the pizza order: crust filling, pizza type and any extras."""
        if not isinstance(result, AcceptedElicitation):
            self.interview.enter(Stage.GREETING)
            return "Order cancelled. Let me know if you change your mind."
        return self.interview.take_order(result.data)

    async def collect_feedback(
        self,
        result: Annotated[ElicitationResult[Feedback], Resolve(Interview.feedback_resolver)],
    ) -> str:
        """Ask whether the customer liked their experience and collect any comments."""
        if not isinstance(result, AcceptedElicitation):
            self.interview.enter(Stage.GREETING)
            return "Thanks for visiting!"
        return self.interview.collect_feedback(result.data)


class AnswersTools(Toolset):
    """The questioning tools for clients that cannot answer elicitation requests.

    Their questions travel back through the tool's `answers` argument instead.
    """

    namespace = "answers"
    tool_names = ("greet_customer", "take_order", "collect_feedback")

    async def greet_customer(self, answers: Greeting | None = None) -> str:
        """Greet the customer: ask how they are doing, how they arrived and whether they want pizza.

        Ask the user the questions this tool returns, then call it again with `answers`.
        """
        self.interview.require_stage(Stage.GREETING)
        if answers is None:
            return self.interview.questions(Interview.GREET_PROMPT, Greeting)
        return self.interview.greet(answers)

    async def take_order(self, answers: Order | None = None) -> str:
        """Take the pizza order: crust filling, pizza type and any extras.

        Ask the user the questions this tool returns, then call it again with `answers`.
        """
        self.interview.require_stage(Stage.ORDER)
        if answers is None:
            return self.interview.questions(Interview.ORDER_PROMPT, Order)
        return self.interview.take_order(answers)

    async def collect_feedback(self, answers: Feedback | None = None) -> str:
        """Ask whether the customer liked their experience and collect any comments.

        Ask the user the questions this tool returns, then call it again with `answers`.
        """
        self.interview.require_stage(Stage.FEEDBACK)
        if answers is None:
            return self.interview.questions(Interview.FEEDBACK_PROMPT, Feedback)
        return self.interview.collect_feedback(answers)


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

    supports = _client_supports_elicitation(ctx)
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


def main() -> None:
    provider = _configure_telemetry()
    try:
        server.run()
    finally:
        # Flush the spans the batch processor has buffered before exiting.
        provider.shutdown()


if __name__ == "__main__":
    main()
