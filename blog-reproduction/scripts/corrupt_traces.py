"""Inject artificial noise into clean OpenAI-messages traces (iteration-2).

Three orthogonal corruptions, each applied INDEPENDENTLY to a fraction `rate` of
traces (a trace may receive any subset 0-3):

1. truncation   - cut the conversation at a random point (simulates dropped logs).
2. arg mutation - corrupt the arguments of one tool call (drop required field,
                  out-of-enum value, wrong type, corrupted string, or extra field).
3. schema drift - rename tool calls to an alias (camelCase, _v2, api_ prefix, ...),
                  consistently within a trace, while job_description.json keeps the
                  canonical names (simulates API evolution the logs didn't track).

Reads `traces.jsonl` + `job_description.json` from the input dir, writes the
corrupted `traces.jsonl`, a copy of `job_description.json` (canonical names), and a
`corruption_log.jsonl` sidecar recording exactly what was injected into each trace.
"""

import argparse
import json
import random
from pathlib import Path

DEFAULT_RATE = 0.2
DEFAULT_SEED = 7

# Minimum messages to keep when truncating: system + first user turn.
TRUNCATE_MIN_KEEP = 2

# Deterministic name transforms used for schema drift (6 alternatives per tool).
ALIAS_TRANSFORMS = [
    lambda s: s.split("_")[0] + "".join(p.title() for p in s.split("_")[1:]),  # camelCase
    lambda s: "".join(p.title() for p in s.split("_")),  # PascalCase
    lambda s: s + "_v2",
    lambda s: s + "_v3",
    lambda s: "api_" + s,
    lambda s: s.replace("_", ""),
]


def load_tool_schemas(job_description: dict) -> dict[str, dict]:
    schemas: dict[str, dict] = {}
    for tool in job_description["tools"]:
        fn = tool["function"]
        params = fn.get("parameters", {})
        schemas[fn["name"]] = {
            "required": set(params.get("required") or []),
            "properties": params.get("properties") or {},
        }
    return schemas


def truncate(messages: list[dict], rng: random.Random) -> tuple[list[dict], dict | None]:
    if len(messages) <= TRUNCATE_MIN_KEEP + 1:
        return messages, None
    keep = rng.randint(TRUNCATE_MIN_KEEP, len(messages) - 1)
    return messages[:keep], {"kept": keep, "dropped": len(messages) - keep}


def drift_schema(
    messages: list[dict], rng: random.Random, tool_names: list[str]
) -> tuple[list[dict], dict | None]:
    # One stable alias per tool for this trace (consistent within the conversation).
    alias_map = {name: rng.choice(ALIAS_TRANSFORMS)(name) for name in tool_names}
    renamed: dict[str, str] = {}
    out = []
    for m in messages:
        if m.get("tool_calls"):
            m = {**m, "tool_calls": [dict(tc) for tc in m["tool_calls"]]}
            for tc in m["tool_calls"]:
                name = tc["function"]["name"]
                if name in alias_map:
                    tc["function"] = {**tc["function"], "name": alias_map[name]}
                    renamed[name] = alias_map[name]
        out.append(m)
    return out, ({"renamed": renamed} if renamed else None)


def _mutate_one(args: dict, schema: dict, rng: random.Random) -> tuple[dict, str] | None:
    ops = ["drop_required", "out_of_enum", "wrong_type", "corrupt_string", "extra_field"]
    rng.shuffle(ops)
    for op in ops:
        if op == "drop_required":
            cand = [k for k in schema["required"] if k in args]
            if cand:
                k = rng.choice(cand)
                return {kk: vv for kk, vv in args.items() if kk != k}, f"dropped required {k!r}"
        if op == "out_of_enum":
            cand = [k for k in args if "enum" in schema["properties"].get(k, {})]
            if cand:
                k = rng.choice(cand)
                return {**args, k: "__invalid_enum__"}, f"out-of-enum {k!r}"
        if op == "wrong_type":
            if args:
                k = rng.choice(list(args))
                v = args[k]
                new = 99999 if isinstance(v, str) else str(v)
                return {**args, k: new}, f"wrong-type {k!r}"
        if op == "corrupt_string":
            cand = [k for k in args if isinstance(args[k], str) and args[k]]
            if cand:
                k = rng.choice(cand)
                return {**args, k: args[k] + "_XQ"}, f"corrupted string {k!r}"
        if op == "extra_field":
            return {**args, "__unexpected__": "x"}, "added unexpected field"
    return None


def mutate_arguments(
    messages: list[dict], rng: random.Random, schemas: dict[str, dict]
) -> tuple[list[dict], dict | None]:
    # Eligible = real tool calls (in schema) with a non-empty argument dict.
    candidates = []
    for i, m in enumerate(messages):
        for tc in m.get("tool_calls") or []:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            if name in schemas and isinstance(args, dict) and args:
                candidates.append((i, tc, name, args))
    if not candidates:
        return messages, None
    _, tc, name, args = rng.choice(candidates)
    result = _mutate_one(args, schemas[name], rng)
    if result is None:
        return messages, None
    new_args, desc = result
    tc["function"] = {**tc["function"], "arguments": json.dumps(new_args)}
    return messages, {"tool": name, "mutation": desc}


def corrupt(traces: list[dict], schemas: dict, rate: float, seed: int) -> tuple[list[dict], list[dict]]:
    tool_names = list(schemas)
    out_traces, log = [], []
    for i, trace in enumerate(traces):
        rng = random.Random(f"{seed}-{i}")
        messages = [dict(m) for m in trace["messages"]]
        entry: dict = {"index": i, "applied": []}

        if rng.random() < rate:
            messages, info = truncate(messages, rng)
            if info:
                entry["applied"].append("truncation")
                entry["truncation"] = info
        if rng.random() < rate:
            messages, info = drift_schema(messages, rng, tool_names)
            if info:
                entry["applied"].append("schema_drift")
                entry["schema_drift"] = info
        if rng.random() < rate:
            messages, info = mutate_arguments(messages, rng, schemas)
            if info:
                entry["applied"].append("arg_mutation")
                entry["arg_mutation"] = info

        out_traces.append({"messages": messages})
        if entry["applied"]:
            log.append(entry)
    return out_traces, log


def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main(input_dir: str, output_dir: str, rate: float, seed: int) -> None:
    src = Path(input_dir)
    traces = [json.loads(line) for line in (src / "traces.jsonl").read_text().splitlines()]
    job_description = json.loads((src / "job_description.json").read_text())
    schemas = load_tool_schemas(job_description)

    out_traces, log = corrupt(traces, schemas, rate, seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_traces, out / "traces.jsonl")
    write_jsonl(log, out / "corruption_log.jsonl")
    (out / "job_description.json").write_text(json.dumps(job_description, indent=2) + "\n")

    counts = {c: sum(c in e["applied"] for e in log) for c in ("truncation", "schema_drift", "arg_mutation")}
    print(f"Input traces:          {len(traces)}  (rate={rate}, seed={seed})")
    print(f"Traces corrupted (>=1): {len(log)}")
    for c, n in counts.items():
        print(f"  {c:14}: {n}")
    print(f"Wrote -> {out / 'traces.jsonl'}")
    print(f"Wrote -> {out / 'corruption_log.jsonl'}")
    print(f"Wrote -> {out / 'job_description.json'} (canonical names)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="slm-data/iteration-1")
    parser.add_argument("--output-dir", default="slm-data/iteration-2")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="Per-trace probability for EACH corruption.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir, rate=args.rate, seed=args.seed)
