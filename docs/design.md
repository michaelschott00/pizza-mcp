# Pizza MCP Server

## Purpose of the mcp server

Please build a simple mcp server that implements a pizza order interview. The goal is to assemble exactly the pizza the customer wants and guide them through the process. As this setting suggests, this server is meant for understanding mcp features, not to ever go to production anywhere.

## Elicitation

Please use the mcp sdk's elicitation feature for all user prompts.

## Progressive tool discovery

I am not sure this feature can be implemented in the sdk so please check. The idea is that there are multiple stages and in each stage, only a subset of the tools are available. This means the mcp has to maintain internal state to tell which stage it is in and what tools to expose.
The stages for this server are

1) Greeting-stage: Ask the user how they are doing, how they got here (bike/train/car/by foot) and if they'd like to order a pizza.
2) Order-stage: Ask the user a) if they want cheese crust, sausage crust or no crust filling, b) what pizza type (margarita, speziale, four cheese), c) if they would like anything else (free form entry)
3) Pizza-making-stage: Expose tools make_dough, spread_dough, add_topings, bake_pizza, serve_pizza. Technically, these need to be called sequentially but for simplicity they may all be exposed in this stage.
4) Feedback-stage: Ask the user if they liked their experience (yes/no) and if they have anything to say (free form)

In each stage, only the tools for that stage should be available to the agent.

## References

- MCP Python SDK: <https://py.sdk.modelcontextprotocol.io/>
