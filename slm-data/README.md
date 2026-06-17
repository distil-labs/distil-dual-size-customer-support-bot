# Airline tool-calling SLM with learned deferral (reproduction package)

This directory holds the exact input used to train the shipped iteration-4 model: a small
task-specific model (**Qwen3-1.7B**) for multi-turn **airline customer-support tool calling**
that learns to **defer genuinely hard turns to a larger model** (`defer_to_larger_model`). It is
built with Distil Labs' *training-from-traces* pipeline: real conversation traces are repaired by
a teacher (**GLM-5**), expanded into synthetic data, and distilled into the student.

On a held-out set of airline support turns the distilled 1.7B reaches **~0.75** quality, up from
**0.42** untrained and approaching the **0.80** of its frontier-scale teacher, while running
locally.

```
slm-data/
├── README.md            you are here
├── traces.jsonl         1,587 airline traces (OpenAI chat format), the seed
├── job_description.json the airline policy + 16 tools (incl. defer_to_larger_model)
└── config.yaml          one config: trace-processing -> synthgen -> finetune
```

---

## (a) Where the data is from

The traces are derived from **[Salesforce/APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k)**,
filtered to the **airline** domain: multi-turn user/assistant conversations where the assistant
calls tools (search flights, book/cancel/update reservations, and so on).

> **License:** APIGen-MT-5k is **CC-BY-NC-4.0** (attribution, **non-commercial**). The
> `traces.jsonl` here is a derivative work and inherits that license. If you redistribute, keep
> the attribution and the non-commercial terms, or ship only `config.yaml` + `job_description.json`
> and have readers regenerate the traces from the public dataset.

## (b) What's in the input, and the deferral recipe

The pipeline turns these three files into the training set in three conceptual steps:

1. **Seed traces** (`traces.jsonl`): the airline APIGen conversations in OpenAI `messages`
   format. Every assistant turn is a single tool call (free-text replies are wrapped as
   `respond_to_user`), so the task is uniformly tool-calling.

2. **Trace processing**: the teacher (GLM-5) relabels each turn and **injects
   `defer_to_larger_model`** on genuinely-hard turns, then splits the data into train / test /
   unstructured. `job_description.json` carries the airline policy plus the 16 tools (14 tau-bench
   airline tools + `respond_to_user` + `defer_to_larger_model`).

3. **Defer-densified synthgen**: the seed turns are expanded into ~7,400 synthetic examples using
   the **defer-forcing mutation pool** in `config.yaml`. The pool merges defer-framed and normal
   topics so the natural escalation rate lands at **~74% conversation-level** defer in a single run
   (reproducing what was originally a two-run blend). This is the lever that teaches the model
   *when* to escalate, framed by the absolute structure of the problem rather than by which tool is
   involved.

The result: **1,587 seed traces**, a **16-tool** job description, and one `config.yaml` that drives
the whole pipeline.

## (c) Train the model to completion

Reproduce the model with the [Distil Labs CLI](https://docs.distillabs.ai/). The `slm-data/`
directory holds everything `upload-traces` needs: `traces.jsonl`, `job_description.json`, and
`config.yaml`. The config pins the student (`Qwen3-1.7B`), the teacher (`zai.glm-5`), the
trace-processing split, the ~7,400-example synthgen target, and the defer-forcing mutation pool, so
the commands below carry no extra flags.

**Install and authenticate**
```bash
curl -fsSL https://cli-assets.distillabs.ai/install.sh | sh
distil update      # the platform evolves quickly
distil login
```

**Create the model** (note the model ID it prints; use it as `<model-id>` below):
```bash
distil model create airline-support-deferral
```

**Upload and process the traces.** Trace processing relabels each turn, injects
`defer_to_larger_model` on hard turns, and produces the train / test / unstructured split. The
original model is also scored on the generated test set as a baseline.
```bash
distil model upload-traces <model-id> --data ./slm-data
distil model upload-status <model-id>      # poll until JOB_SUCCESS
```

**(Optional) teacher evaluation** is a feasibility check before the long run:
```bash
distil model run-teacher-evaluation <model-id>
distil model teacher-evaluation <model-id> --output json | jq '.aggregateMetrics'
```

**Train.** The teacher generates ~7,400 validated synthetic examples (~74% conversation-level
defer), then finetunes Qwen3-1.7B (LoRA) and evaluates base vs. tuned. This takes several hours.
```bash
distil model run-training <model-id>
distil model training <model-id>           # poll until JOB_SUCCESS
```

**Download and deploy.**
```bash
distil model download <model-id>           # merged fp16 weights, GGUF, LoRA adapter, eval metrics
distil model deploy local <model-id>
distil model invoke <model-id>             # prints the command to query your model
```

## Results

Running this recipe produces the shipped iteration-4 model. On a held-out set of airline support
turns, scored by an independent GLM-5 judge (fraction of responses rated correct):

| | Untrained Qwen3-1.7B | Distilled 1.7B (this recipe) | Frontier teacher (GLM-5) |
|---|:---:|:---:|:---:|
| Quality | 0.42 | **~0.75** | 0.80 |

Fine-tuning lifts the local 1.7B from **0.42 to ~0.75** (closing roughly 85% of the gap to its
frontier-scale teacher), while running ~96% of turns locally and escalating only the hardest ~4% to
the larger model.

> This config reproduces the iteration-4 training *distribution* in a single run. The shipped model
> was trained on the recovered/blended data (a contrastive run plus a pure-defer run); this is the
> recipe's character, not a byte-exact rebuild.
