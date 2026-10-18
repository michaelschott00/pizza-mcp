# Pizza MCP Server

## Purpose of the mcp server

Please build a simple mcp server that implements a pizza order interview. The goal is to assemble exactly the pizza the customer wants and guide them through the process. As this setting suggests, this server is meant for understanding mcp features, not to ever go to production anywhere.

## Elicitation

Please use the mcp sdk's elicitation feature for all user prompts.

## Stage-gated tools

The server keeps internal state tracking which stage the interview is in. All
tools are exposed to the client at all times (progressive tool discovery via
`notifications/tools/list_changed` is unreliable in some clients). Instead, each
tool declares the stage it belongs to, fails with an informative error when
called outside that stage, and says which stage it is for in its description.

The stages for this server are

1) Greeting-stage: Ask the user how they are doing and how they got here (bike/train/car/by foot).
2) Order-stage: Ask the user a) what pizza type (margarita, speziale, four cheese), b) if they want cheese crust, sausage crust or no crust filling, c) if they would like anything else (free form entry).
3) Serving-stage: Ask the user whether they want to eat here or take the pizza to go.
4) Confirmation-stage: `confirm_order` summarizes every answer and asks the user to confirm the order. A confirmation proceeds to the pizza-making stage; a refusal returns the interview to the order stage so the user can make a corrected order.
5) Pizza-making-stage: Expose tools make_dough, spread_dough, add_topings, bake_pizza, serve_pizza. These must be called in that order; calling one out of order returns an error naming the next step. The pizza-making tools cannot be used until the order is confirmed, and serving the pizza ends the interview and starts a fresh one.

The questions are asked up front and in order: `greet_customer`, then
`take_order`, then `choose_serving`. `choose_serving`, the last question, answers
the customer, taking everything said so far into account for that one response,
and hands off to `confirm_order`; the order is only made once the customer has
confirmed it.

A tool called outside its stage returns a tool error naming the stage it expects,
the stage the interview is currently in, the correct sequence of questions, and
the tools to call (in order) to reach the point where it can be used.

## Recommended answers

A question tool optionally takes a `recommendation`: answers the model derived
from the prompt that started the conversation, filled in only where the model is
reasonably sure. The recommendation is offered to the user alongside the normal
options, never in place of them - as field defaults in the elicitation form, so
accepting a correct guess takes no typing.

## References

- MCP Python SDK: <https://py.sdk.modelcontextprotocol.io/>
