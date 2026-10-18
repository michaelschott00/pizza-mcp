"""Inspect how the pizza MCP server's tools look once rendered into a chat prompt.

Loads the tool definitions straight from the server in `main.py` (the same ones
an MCP client would discover), converts them to the OpenAI function-calling
format, and renders the system prompt + tools + a sample user turn with the
Qwen chat template.

The tool JSON in the rendered prompt is re-indented and the escaped newlines in
the tool descriptions (`\n`) are turned into real line breaks, so the actual
payload the model sees is easier to read. The result is therefore a
visualisation rather than strictly valid JSON.
"""

import asyncio
import json

from transformers import AutoTokenizer

from main import advertised_name, server

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")  # tokenizer files only


def server_tools() -> list[dict]:
    """The server's registered tools, in OpenAI function-calling format.

    Shows the set an elicitation-capable client is exposed to; the server also
    registers an `answers` implementation of every questioning tool, but only
    advertises it to clients that cannot elicit.
    """
    exposed: list[dict] = []
    for tool in asyncio.run(server.list_tools()):
        name = advertised_name(tool.name, supports_elicitation=True)
        if name is None:
            continue
        exposed.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema or {"type": "object", "properties": {}},
                },
            }
        )
    return exposed


def format_tools(prompt: str) -> str:
    """Indent the `<tools>` block and render escaped newlines as real line breaks."""
    open_tag, close_tag = "<tools>", "</tools>"
    # The tags also appear once in the template's instructions, so take the last block.
    start = prompt.rindex(open_tag) + len(open_tag)
    end = prompt.index(close_tag, start)

    formatted = [
        json.dumps(json.loads(line), indent=2, ensure_ascii=False)
        for line in prompt[start:end].splitlines()
        if line.strip()
    ]
    tools_json = "\n".join(formatted).replace("\\n", "\n")

    return prompt[:start] + "\n" + tools_json + "\n" + prompt[end:]


messages = [
    {"role": "system", "content": server.instructions},
    {"role": "user", "content": "Hi, I'd like to order a pizza."},
]

prompt = tok.apply_chat_template(
    messages,
    tools=server_tools(),
    tokenize=False,
    add_generation_prompt=True,
)

print(format_tools(prompt))
