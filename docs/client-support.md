@main.py this mcp server should support clients with and without elicitation capabilities.
there are two problems: 1) the way it's implemented currently leads to a double prompt if elicitation is supported.
the tool call itself requires answers to the questions and then the tool call additionally triggers elicitation, i.e. a prompt for the same inputs.
2) the tool prompts inform the model about the existence of the two supported types of clients.
this is nothing an agent should have to care or even know about.
it should be entirely up to the server to handle clients with various features.
I think it would be better to expose different tools based on whether the client supports elicitation or not.
this should be possible because afaik the client sends its capabilities to the server when it first connects.
the server should therefore be able to store inside whether the client supports it or not and only expose the appropriate version of the tools.
pls open a new git worktree for this
