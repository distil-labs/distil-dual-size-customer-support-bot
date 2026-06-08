"""Dual-size airline customer-support bot. Terminal CLI (SLM + deferral cascade).

A small fine-tuned SLM (Qwen3-1.7B class) handles most airline-support turns
itself. On a genuinely-hard turn it emits a ``defer_to_larger_model`` tool call;
the orchestrator then hands the *rest of the conversation* to a larger, pluggable
model (any OpenAI-compatible endpoint). Every assistant action is a single tool
call (including talking to the customer via ``respond_to_user``), matching how
the model was trained.

Usage:
    # 1. serve the SLM locally (see README), e.g. llama-server ... --port 8000
    # 2. point the large model at any OpenAI-compatible endpoint via env vars:
    export DEFER_BASE_URL=https://api.openai.com/v1
    export DEFER_API_KEY=sk-...
    export DEFER_MODEL=gpt-4o
    # 3. run:
    python orchestrator.py --port 8000 [--debug]
"""

import argparse
import json
import os

from openai import OpenAI

from tools import (
    DEFER_TOOL,
    RESPOND_TOOL,
    TRANSFER_TOOL,
    build_openai_tools,
    execute_tool,
    load_job_description,
)

# ---------------------------------------------------------------------------
# Display helpers (lightweight ANSI; honours the NO_COLOR convention)
# ---------------------------------------------------------------------------
_COLOR = os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


DIM = "2"
BOLD = "1"
CYAN = "36"
YELLOW = "33"
GREEN = "32"

MAX_STEPS = 12  # safety cap on tool calls within a single user turn

# ---------------------------------------------------------------------------
# System prompt: the distil tool-calling serving wrapper around the airline
# policy (job_description.task_description). Matches the trained model's format.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = (
    "You are a tool-calling model working on:\n"
    "<task_description>{task_description}</task_description>\n"
    "\n"
    "Respond to the conversation history by generating an appropriate tool call "
    "that satisfies the user request. Generate only the tool call according to the "
    "provided tool schema, do not generate anything else. Always respond with a tool call."
)


# ---------------------------------------------------------------------------
# Model client: stateless wrapper around an OpenAI-compatible endpoint
# ---------------------------------------------------------------------------
class ModelClient:
    """A tier in the cascade (the small SLM or the large deferral model)."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        tools: list[dict],
        label: str,
        extra_body: dict | None = None,
    ):
        self.model_name = model_name
        self.tools = tools
        self.label = label
        self.extra_body = extra_body
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def invoke(self, system_prompt: str, history: list[dict]) -> dict | str:
        """Return a parsed ``{"name", "arguments"}`` tool call, or an error string."""
        messages = [{"role": "system", "content": system_prompt}] + history
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=0,
            tools=self.tools,
            tool_choice="required",
        )
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        response = self.client.chat.completions.create(**kwargs).choices[0].message

        # Path A: proper tool_calls
        if response.tool_calls:
            fn = response.tool_calls[0].function
            arguments = fn.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            return {"name": fn.name, "arguments": arguments}

        # Path B: model returned a JSON tool call in content (fallback)
        if response.content:
            try:
                parsed = json.loads(response.content.strip())
                if "name" in parsed:
                    args = parsed.get("arguments", parsed.get("parameters", {}))
                    if isinstance(args, str):
                        args = json.loads(args or "{}")
                    return {"name": parsed["name"], "arguments": args}
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        return f"No valid tool call in response: {response}"


# ---------------------------------------------------------------------------
# Cascade orchestrator
# ---------------------------------------------------------------------------
class CascadeOrchestrator:
    """Runs the multi-turn agent loop and the sticky SLM -> large-model handoff."""

    def __init__(
        self,
        slm: ModelClient,
        large: ModelClient,
        system_prompt: str,
        debug: bool = False,
    ):
        self.slm = slm
        self.large = large
        self.system_prompt = system_prompt
        self.debug = debug
        self.history: list[dict] = []
        self.deferred = False  # sticky: once True, the large model runs the rest
        self._call_idx = 0
        self._last_call_id = ""

    def process_turn(self, user_text: str) -> str | None:
        """One user message -> the bot's full response. ``None`` ends the chat."""
        if user_text.lower() in ("quit", "exit"):
            return None

        self.history.append({"role": "user", "content": user_text})

        for _ in range(MAX_STEPS):
            client = self.large if self.deferred else self.slm
            call = client.invoke(self.system_prompt, self.history)

            if self.debug:
                print(_c(f"  [debug:{client.label}] {call}", DIM))

            if isinstance(call, str):  # no valid tool call
                fallback = "Sorry, I didn't catch that. Could you rephrase?"
                self.history.append({"role": "assistant", "content": fallback})
                return self._bot_line(client.label, fallback)

            name = call["name"]
            arguments = call.get("arguments") or {}

            # --- Deferral: switch to the large model for the rest of the convo ---
            if name == DEFER_TOOL and not self.deferred:
                self.deferred = True
                self._print_defer(arguments.get("reason", ""))
                continue  # re-run this same step on the large model

            self._record_tool_call(name, arguments)

            # --- Terminal tools (end the turn, no tool result follows) ---
            if name == RESPOND_TOOL:
                return self._bot_line(client.label, arguments.get("message", ""))
            if name == TRANSFER_TOOL:
                summary = arguments.get("summary", "")
                msg = "Let me connect you with a human agent."
                if summary:
                    msg += f" (Summary passed on: {summary})"
                return self._bot_line(client.label, msg)

            # --- Other tools: execute and feed the result back ---
            result = execute_tool(name, arguments)
            self._trace_tool(client.label, name, arguments, result)
            self._record_tool_result(result)

        return self._bot_line(client.label, "[reached max reasoning steps for this turn]")

    # ---- history bookkeeping --------------------------------------------------
    def _record_tool_call(self, name: str, arguments: dict) -> None:
        self._last_call_id = f"call_{self._call_idx}"
        self._call_idx += 1
        self.history.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": self._last_call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
            }
        )

    def _record_tool_result(self, result: dict) -> None:
        self.history.append(
            {
                "role": "tool",
                "tool_call_id": self._last_call_id,
                "content": json.dumps(result),
            }
        )

    # ---- display --------------------------------------------------------------
    def _bot_line(self, label: str, message: str) -> str:
        badge = _c(f"[{label}]", CYAN if label == "SLM" else GREEN)
        return f"{badge} {message}"

    def _trace_tool(self, label: str, name: str, arguments: dict, result: dict) -> None:
        args_preview = ", ".join(
            f"{k}={json.dumps(v)[:60]}" for k, v in arguments.items()
        )
        line = f"  · [{label}] {name}({args_preview}) → {json.dumps(result)}"
        print(_c(line, DIM))

    def _print_defer(self, reason: str) -> None:
        print(_c("\n  ⤴ SLM deferred to the large model.", YELLOW + ";" + BOLD))
        if reason:
            print(_c(f"     reason: {reason}", YELLOW))
        print(
            _c(
                "     → the large model now handles the rest of this conversation.\n",
                YELLOW,
            )
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(
    model: str,
    base_url: str,
    api_key: str,
    defer_base_url: str | None,
    defer_api_key: str,
    defer_model: str,
    debug: bool,
) -> None:
    job_description = load_job_description()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        task_description=job_description["task_description"]
    )

    # SLM gets the full 16-tool catalog (incl. defer). enable_thinking off, since the
    # model talks via respond_to_user / think, not the chat template's reasoning.
    slm = ModelClient(
        model_name=model,
        base_url=base_url,
        api_key=api_key,
        tools=build_openai_tools(job_description),
        label="SLM",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    # Large model gets every tool EXCEPT defer (it is the escalation tier).
    # No chat_template_kwargs, since the endpoint may be a vanilla OpenAI-compatible API.
    if not defer_base_url:
        raise SystemExit(
            "Set DEFER_BASE_URL (and DEFER_API_KEY / DEFER_MODEL) to the large "
            "model's OpenAI-compatible endpoint before running."
        )
    large = ModelClient(
        model_name=defer_model,
        base_url=defer_base_url,
        api_key=defer_api_key,
        tools=build_openai_tools(job_description, exclude=(DEFER_TOOL,)),
        label="LARGE",
    )

    orchestrator = CascadeOrchestrator(slm, large, system_prompt, debug=debug)

    print("Airline Support Bot, dual-size cascade (type 'quit' or 'exit' to stop)")
    print(_c(f"  SLM: {model} @ {base_url}", DIM))
    print(_c(f"  large (deferral): {defer_model} @ {defer_base_url}\n", DIM))
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            response = orchestrator.process_turn(user_input)
            if response is None:
                print("Bot: Thanks for contacting us. Safe travels!")
                break
            print(f"Bot: {response}")
    except (KeyboardInterrupt, EOFError):
        print("\nBot: Thanks for contacting us. Safe travels!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-size airline support bot")
    parser.add_argument("--model", default="model", help="SLM model name served locally")
    parser.add_argument("--port", type=int, default=8000, help="Port of the local SLM server")
    parser.add_argument("--api-key", default="EMPTY", help="SLM server API key (default EMPTY)")
    parser.add_argument(
        "--base-url", default=None, help="SLM base URL (overrides --port)"
    )
    parser.add_argument("--debug", action="store_true", help="Print raw model output each step")
    args = parser.parse_args()

    main(
        model=args.model,
        base_url=args.base_url or f"http://127.0.0.1:{args.port}/v1",
        api_key=args.api_key,
        # Large (deferral) model is configured purely via env vars; the user brings
        # their own OpenAI-compatible endpoint.
        defer_base_url=os.environ.get("DEFER_BASE_URL"),
        defer_api_key=os.environ.get("DEFER_API_KEY", "EMPTY"),
        defer_model=os.environ.get("DEFER_MODEL", "gpt-4o"),
        debug=args.debug,
    )
