"""Tool catalog + (dummy) tool execution for the airline support bot.

The 16 tools (14 tau-bench airline tools + ``respond_to_user`` +
``defer_to_larger_model``) and the airline policy are loaded from
``job_description.json`` — the same artifact used to train the SLM, so the demo
and the model never drift apart.

Execution is intentionally a *trivial success stub*: every backend tool returns
``{"status": "success"}``. Swap :func:`execute_tool` for a real backend (or the
tau-bench airline environment) to make the bot operate on real state.
"""

import json
from pathlib import Path

JOB_DESCRIPTION_PATH = Path(__file__).parent / "job_description.json"

# Tools with special control-flow meaning to the orchestrator (not backend calls).
RESPOND_TOOL = "respond_to_user"
DEFER_TOOL = "defer_to_larger_model"
TRANSFER_TOOL = "transfer_to_human_agents"


def load_job_description(path: Path = JOB_DESCRIPTION_PATH) -> dict:
    """Load the job description (airline policy + tool schemas)."""
    return json.loads(Path(path).read_text())


def _sanitize_parameters(parameters: dict | None) -> dict:
    """Make a job-description parameter schema safe for the OpenAI tools API.

    The job description stores ``additionalProperties: null`` (and a function-level
    ``strict: null``); ``null`` is not a valid JSON-schema value, so drop it.
    """
    if not parameters:
        return {"type": "object", "properties": {}}
    clean = dict(parameters)
    if clean.get("additionalProperties") is None:
        clean.pop("additionalProperties", None)
    return clean


def build_openai_tools(job_description: dict, exclude: tuple[str, ...] = ()) -> list[dict]:
    """Convert job-description tool defs into OpenAI ``tools`` array form."""
    tools = []
    for entry in job_description["tools"]:
        fn = entry.get("function", entry)
        if fn["name"] in exclude:
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": _sanitize_parameters(fn.get("parameters")),
                },
            }
        )
    return tools


def execute_tool(name: str, arguments: dict) -> dict:
    """Dummy backend execution — returns a generic success result for every tool.

    Replace this with real handlers (or wire to the tau-bench airline env) to give
    the bot real reservations, users, and flights to talk about.
    """
    return {"status": "success"}
