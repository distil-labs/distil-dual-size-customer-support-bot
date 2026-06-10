"""Prepare trace-processing inputs for the deferral dataset (data-only).

Takes the existing airline trace-processing input and produces a new input dir that
adds a `defer_to_larger_model` tool + a deferral guideline in the task description, so
the rewriting teacher inserts `defer_to_larger_model` on the hard turns. Traces and
config are copied unchanged. No codebase changes.

Deferral criteria come from the reference-free analysis (slm-data/analysis): read-only
lookups never fail (SLM-easy), while DB writes, compensation, and policy-judgement turns
fail often (SLM-hard) -> those defer.
"""

import argparse
import json
import shutil
from pathlib import Path

import yaml

# Dataset run: 100 train + 100 test seed traces (the rest become unstructured for synthgen).
# test_base > 0 re-enables the test-split evaluation (reference-free on the original traces +
# reference-based on the rewrite) as part of the job.
TRAIN_BASE = 100
TEST_BASE = 100
MIN_EXAMPLES = 1

# Appended to the airline policy (task_description) so the rewriter knows when to defer.
DEFERRAL_GUIDELINE = """

# Escalation: defer complex turns to a more capable model
Call `defer_to_larger_model` on turns that are NOT a single, direct, obviously-correct step, so
that a more capable model continues this same conversation with the same tools and policy. Judge
complexity in ABSOLUTE terms, by the reasoning the correct action depends on — not relative to
your own ability, and never merely because a particular tool (a write, a calculation, a sensitive
action) is requested.
A turn is complex — defer it — when producing the correct action requires any of:
- Verifying policy eligibility before acting, where the answer is not immediately obvious — e.g.
  whether a reservation can be modified or cancelled given basic-economy limits, travel insurance,
  membership tier, or partially-flown segments; or whether a user qualifies for compensation.
- Combining several rules or several pieces of information to decide what to do.
- A multi-step calculation whose result determines the action (fare differences, refund or
  compensation amounts, especially across multiple passengers).
- Resolving a genuinely ambiguous eligibility or policy question, or a judgement call about
  whether to refuse.
A turn is simple — do it yourself — when the correct action follows directly from the request and
the information already on hand, with no policy precondition to weigh: looking up a reservation or
user, searching flights, listing airports, asking for missing information, summarising or
confirming retrieved information, or a single unconditional action.
This is different from transfer_to_human_agents: deferring keeps the request automated (a more
capable model takes over); transferring to a human is only for requests outside the scope of
these tools, or when the user explicitly asks for a human.
"""

# Clarifies the human-transfer tool so it is distinct from defer_to_larger_model.
TRANSFER_DESCRIPTION = (
    "Transfer the user to a human agent, with a summary of the user's issue. Use this only "
    "when the request is outside the scope of the available tools, or the user explicitly asks "
    "for a human agent — NOT for requests that are merely complex (use defer_to_larger_model "
    "for those)."
)

DEFER_TO_LARGER_MODEL_TOOL = {
    "type": "function",
    "function": {
        "name": "defer_to_larger_model",
        "description": (
            "Escalate the current turn to a larger, more capable model that continues this "
            "conversation using the same tools and policy. Use it on turns that are not a single, "
            "direct, obviously-correct step — where the correct action depends on verifying "
            "non-obvious policy eligibility, combining several rules or pieces of information, a "
            "multi-step calculation, or resolving an ambiguous eligibility/policy question. Judge "
            "complexity by the structure of the problem in absolute terms, not relative to model "
            "capability, and do not defer merely because a write, calculation, or sensitive tool "
            "is requested. The user keeps being served automatically; this is a capability "
            "escalation, NOT a human hand-off (use transfer_to_human_agents for out-of-scope "
            "requests or explicit human requests)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": (
                        "The specific difficulty that makes this turn unsafe to handle yourself "
                        "(e.g. 'compensation eligibility + amount across 3 passengers')."
                    ),
                }
            },
            "required": ["reason"],
        },
    },
}


def build_job_description(source: dict) -> dict:
    tools = []
    for tool in source["tools"]:
        if tool["function"]["name"] == "transfer_to_human_agents":
            tool = {**tool, "function": {**tool["function"], "description": TRANSFER_DESCRIPTION}}
        tools.append(tool)
    tools.append(DEFER_TO_LARGER_MODEL_TOOL)
    return {**source, "task_description": source["task_description"] + DEFERRAL_GUIDELINE, "tools": tools}


def main(input_dir: str, output_dir: str) -> None:
    src, out = Path(input_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    source_jd = json.loads((src / "job_description.json").read_text())
    job_description = build_job_description(source_jd)
    (out / "job_description.json").write_text(json.dumps(job_description, indent=2) + "\n")

    shutil.copyfile(src / "traces.jsonl", out / "traces.jsonl")
    (out / "test.jsonl").unlink(missing_ok=True)  # ensure no pre-existing test set

    config = yaml.safe_load((src / "config.yaml").read_text())
    config["trace_processing"]["num_traces_as_training_base"] = TRAIN_BASE
    config["trace_processing"]["num_traces_as_testing_base"] = TEST_BASE
    config["trace_processing"]["min_generated_examples"] = MIN_EXAMPLES
    (out / "config.yaml").write_text(yaml.dump(config, sort_keys=False))

    print(f"Wrote -> {out / 'job_description.json'} ({len(job_description['tools'])} tools)")
    print(
        f"Wrote -> {out / 'config.yaml'} "
        f"(train_base={TRAIN_BASE}, test_base={TEST_BASE}, min_examples={MIN_EXAMPLES})"
    )
    print(f"Copied -> traces.jsonl from {src}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="slm-data/trace-processing/input")
    parser.add_argument("--output-dir", default="slm-data/trace-processing-deferral/input")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir)
