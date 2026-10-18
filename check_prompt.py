"""Inspect how the pizza MCP server's tools look once rendered into a chat prompt.

Loads the tool definitions straight from the server in `main.py` (the same ones
an MCP client would discover), converts them to the OpenAI function-calling
format, and renders the system prompt + tools + a sample user turn with the
Qwen chat template.

Which tools a client discovers depends on its capabilities: elicitation-capable
clients get the `ElicitationTools` variants, whose questions travel through
elicitation requests rather than the tool schema, while other clients get the
`AnswersTools` variants, whose questions live in the `answers` parameter
descriptions. Pass `--no-elicitation` to render the latter, or `--both` to
compare the two.

The tool JSON in the rendered prompt is re-indented and the escaped newlines in
the tool descriptions (`\n`) are turned into real line breaks, so the actual
payload the model sees is easier to read. The result is therefore a
visualisation rather than strictly valid JSON.
"""

import argparse
import asyncio
import json

from transformers import AutoTokenizer

from main import advertised_name, server

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")  # tokenizer files only


def server_tools(supports_elicitation: bool) -> list[dict]:
    """The server's registered tools, in OpenAI function-calling format.

    Shows the set a client with the given capability is exposed to; the server
    also registers an `answers` implementation of every questioning tool, but
    only advertises it to clients that cannot elicit.
    """
    exposed: list[dict] = []
    for tool in asyncio.run(server.list_tools()):
        name = advertised_name(tool.name, supports_elicitation=supports_elicitation)
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


def render(supports_elicitation: bool) -> str:
    """The chat prompt a client with the given capability would send."""
    prompt = tok.apply_chat_template(
        messages,
        tools=server_tools(supports_elicitation),
        tokenize=False,
        add_generation_prompt=True,
    )
    return format_tools(prompt)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the script's command-line options."""
    parser = argparse.ArgumentParser(
        prog="check_prompt.py",
        description="Render the pizza MCP server's tools into a Qwen chat prompt.",
    )
    parser.add_argument(
        "--elicitation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "render the tools of an elicitation-capable client (default: enabled); "
            "--no-elicitation renders the answers-based tools"
        ),
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="render both the elicitation and the answers variants",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    modes = [True, False] if args.both else [args.elicitation]
    for supports_elicitation in modes:
        if len(modes) > 1:
            print(f"########## supports_elicitation={supports_elicitation} ##########")
        print(render(supports_elicitation))


if __name__ == "__main__":
    main()
