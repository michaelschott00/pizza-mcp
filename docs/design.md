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

1) Greeting-stage: Ask the user how they are doing, how they got here (bike/train/car/by foot) and if they'd like to order a pizza.
2) Order-stage: Ask the user a) if they want cheese crust, sausage crust or no crust filling, b) what pizza type (margarita, speziale, four cheese), c) if they would like anything else (free form entry)
3) Pizza-making-stage: Expose tools make_dough, spread_dough, add_topings, bake_pizza, serve_pizza. These must be called in that order; calling one out of order returns an error naming the next step.
4) Feedback-stage: Ask the user if they liked their experience (yes/no) and if they have anything to say (free form)

A tool called outside its stage returns a tool error naming the stage it expects,
the stage the interview is currently in, and the tools to call (in order) to
reach the point where it can be used.

## References

- MCP Python SDK: <https://py.sdk.modelcontextprotocol.io/>
