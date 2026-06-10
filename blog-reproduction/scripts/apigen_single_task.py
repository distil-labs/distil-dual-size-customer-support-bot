"""Carve the single-task (airline) set from Salesforce/APIGen-MT-5k into traces.

Filters to the airline domain, converts ShareGPT -> OpenAI `messages`, and drops
malformed tool calls. Output is a single `traces.jsonl` (one OpenAI-messages trace
per line) + `job_description.json` (distil labs format) — i.e. the raw input for
the distil-lib trace-processing pipeline (observation_format: openai_messages). The
pipeline does its own train/test split, so we don't split here.

Two serving-driven transforms:
- Every assistant text reply is wrapped in a `respond_to_user(message)` tool call,
  so every assistant turn is EXACTLY one tool call and never free-form content.
  `respond_to_user` is terminal: it returns control to the user, so it is not
  followed by a tool result.
- Tools are identical across all airline conversations, so they live once in
  `job_description.json` rather than on every trace.
"""

import argparse
import json
from pathlib import Path

AIRLINE_SYSTEM_PREFIX = "# Airline"

ROLE_HUMAN = "human"
ROLE_GPT = "gpt"
ROLE_FUNCTION_CALL = "function_call"
ROLE_OBSERVATION = "observation"

# Tool-call names that are degenerate/hallucinated artifacts (empty args, not real
# tools). Conversations containing any of these are dropped.
MALFORMED_TOOL_NAMES = {"Adding", "Upgrade"}

# Synthetic tool that wraps every natural-language assistant reply, so the model's
# output is always exactly one tool call and never free-form text.
RESPOND_TO_USER_NAME = "respond_to_user"
RESPOND_TO_USER_TOOL = {
    "type": "function",
    "function": {
        "name": RESPOND_TO_USER_NAME,
        "description": (
            "Send a natural-language message to the user. This is the only way to "
            "communicate with the user. It ends your turn and returns control to the "
            "user, so do not make any other tool call in the same turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The message to send to the user."}
            },
            "required": ["message"],
        },
    },
}

def is_airline(example: dict) -> bool:
    return example["system"].startswith(AIRLINE_SYSTEM_PREFIX)


def conversation_has_malformed_call(conversation: list[dict]) -> bool:
    for turn in conversation:
        if turn["from"] == ROLE_FUNCTION_CALL:
            name = json.loads(turn["value"]).get("name")
            if name in MALFORMED_TOOL_NAMES:
                return True
    return False


def merge_consecutive_gpt(conversation: list[dict]) -> list[dict]:
    # The rare gpt->gpt case = two consecutive assistant text turns. Merge them so
    # they become a single respond_to_user call (one output, one returned turn).
    merged: list[dict] = []
    for turn in conversation:
        if turn["from"] == ROLE_GPT and merged and merged[-1]["from"] == ROLE_GPT:
            merged[-1] = {"from": ROLE_GPT, "value": merged[-1]["value"] + "\n\n" + turn["value"]}
        else:
            merged.append(turn)
    return merged


def stringify_arguments(arguments) -> str:
    # OpenAI tool_calls require `arguments` to be a JSON string. The source stores
    # it as a dict (usual) or already a string (rare).
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments)


def make_tool_call(call_id: str, name: str, arguments) -> dict:
    return {
        "role": "assistant",
        # Empty string (not null): distil-lib's ChatMessage types content as str.
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": stringify_arguments(arguments)},
            }
        ],
    }


def convert_conversation(example: dict) -> list[dict]:
    """ShareGPT turns -> OpenAI messages, with every assistant reply as a tool call.

    gpt           -> assistant respond_to_user(message=...) tool call (terminal).
    function_call -> assistant message with a single real tool_calls entry.
    observation   -> tool message linked by tool_call_id.
    """
    messages: list[dict] = [{"role": "system", "content": example["system"]}]
    call_counter = 0

    for turn in merge_consecutive_gpt(example["conversations"]):
        role, value = turn["from"], turn["value"]

        if role == ROLE_HUMAN:
            messages.append({"role": "user", "content": value})

        elif role == ROLE_GPT:
            messages.append(make_tool_call(f"call_{call_counter}", RESPOND_TO_USER_NAME, {"message": value}))
            call_counter += 1

        elif role == ROLE_FUNCTION_CALL:
            call = json.loads(value)
            messages.append(make_tool_call(f"call_{call_counter}", call["name"], call["arguments"]))
            call_counter += 1

        elif role == ROLE_OBSERVATION:
            # Always immediately follows its real function_call assistant message.
            tool_call_id = messages[-1]["tool_calls"][0]["id"]
            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": value})

        else:
            raise ValueError(f"Unexpected ShareGPT role: {role!r}")

    return messages


def build_traces(data: list[dict]) -> tuple[list[dict], dict]:
    traces: list[dict] = []
    stats = {"total_airline": 0, "dropped_malformed": 0, "ends_on_tool": 0}

    for example in data:
        if not is_airline(example):
            continue
        stats["total_airline"] += 1
        if conversation_has_malformed_call(example["conversations"]):
            stats["dropped_malformed"] += 1
            continue
        messages = convert_conversation(example)
        # ~90 transfer-to-human flows end on the tool result with no closing reply;
        # the trace-processing role validator wants the last message to be assistant,
        # so it will drop/fix these. Count them so the imperfection is visible.
        if messages[-1]["role"] == "tool":
            stats["ends_on_tool"] += 1
        traces.append({"messages": messages})
    return traces, stats


def build_job_description(airline_example: dict) -> dict:
    # All airline conversations share the same 14 tools; take them from any example
    # and wrap each in the OpenAI {"type": "function", ...} envelope. respond_to_user
    # is appended because it is now part of the model's output space.
    tools = [
        {"type": "function", "function": tool} for tool in json.loads(airline_example["tools"])
    ]
    tools.append(RESPOND_TO_USER_TOOL)
    return {"task_description": airline_example["system"], "tools": tools}


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main(input_path: str, output_dir: str) -> None:
    data = json.loads(Path(input_path).read_text())
    traces, stats = build_traces(data)
    job_description = build_job_description(next(e for e in data if is_airline(e)))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(traces, out / "traces.jsonl")
    (out / "job_description.json").write_text(json.dumps(job_description, indent=2) + "\n")

    print(f"Airline examples seen:    {stats['total_airline']}")
    print(f"Dropped (malformed call): {stats['dropped_malformed']}")
    print(f"Traces written:           {len(traces)}")
    print(f"  (of which end on a tool result, pipeline will drop/fix): {stats['ends_on_tool']}")
    print(f"Tools in job_description: {len(job_description['tools'])} (incl. respond_to_user)")
    print(f"Wrote -> {out / 'traces.jsonl'}")
    print(f"Wrote -> {out / 'job_description.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="slm-data/raw/apigen-mt_5k.json",
        help="Path to the raw apigen-mt_5k.json download.",
    )
    parser.add_argument(
        "--output-dir",
        default="slm-data/iteration-1",
        help="Directory to write train/eval JSONL + job_description.json into.",
    )
    args = parser.parse_args()
    main(input_path=args.input, output_dir=args.output_dir)
