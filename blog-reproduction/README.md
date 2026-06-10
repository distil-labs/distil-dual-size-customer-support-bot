# Airline tool-calling SLM with learned deferral (reproduction package)

This directory reproduces, end to end, a small task-specific model (**Qwen3-1.7B**) for
multi-turn **airline customer-support tool calling** that learns to **defer genuinely hard
turns to a larger model** (`defer_to_larger_model`). It is built with Distil Labs'
*training-from-traces* pipeline: real conversation traces are repaired by a teacher
(**GLM-5**), expanded into synthetic data, and distilled into the student.

The headline result: the distilled 1.7B student **matches its GLM-5 teacher** on the held-out
test set (LLM-as-a-judge 0.72 vs. teacher 0.70) while being ~40× smaller.

```
blog-reproduction/
├── README.md            ← you are here
├── input/               ← the trace-processing input (everything the pipeline needs)
│   ├── traces.jsonl         1,587 airline traces, OpenAI chat format, lightly corrupted
│   ├── job_description.json task description + 16 tools (incl. defer_to_larger_model)
│   └── config.yaml          one config for all three pipeline stages
└── scripts/             ← how input/traces.jsonl was built (provenance, optional to re-run)
    ├── apigen_single_task.py     APIGen-MT-5k (airline) -> OpenAI-messages traces
    ├── corrupt_traces.py         inject realistic production noise
    └── prepare_deferral_inputs.py add the deferral tool + guideline
```

---

## (a) Where the data is from

The traces are derived from **[Salesforce/APIGen-MT-5k](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k)**,
filtered to the **airline** domain: multi-turn user↔assistant conversations where the
assistant calls tools (search flights, book/cancel/update reservations, etc.).

> **License:** APIGen-MT-5k is **CC-BY-NC-4.0** (attribution, **non-commercial**). The traces
> in `input/traces.jsonl` are a derivative work and inherit that license. If you publish this
> package, keep the attribution and the non-commercial terms, or ship only `scripts/` +
> `input/{config.yaml,job_description.json}` and have readers regenerate the traces from the
> public dataset.

## (b) How it was preprocessed

`input/traces.jsonl` is the output of three small, independent steps (all in `scripts/`):

1. **Carve & convert** (`apigen_single_task.py`): filter APIGen-MT-5k to airline, convert
   ShareGPT → OpenAI `messages`. Also emits `job_description.json` (task description + the airline tool schemas).

2. **Corrupt** (`corrupt_traces.py`): inject realistic production noise. Three orthogonal corruptions, each applied independently to a fraction of traces: **random truncation**, **argument mutation**, and **schema drift**.

3. **Add deferral** (`prepare_deferral_inputs.py`): data-only change to
   `job_description.json`: add a 16th tool, **`defer_to_larger_model`**, plus a guideline that
   frames deferral in **absolute terms of task complexity** (defer when a turn is genuinely
   hard, *not* whenever a particular tool is involved). 

The result in `input/`: **1,587 traces**, a **16-tool** job description, and a `config.yaml`
that drives the whole pipeline.

## (c) Train the model to completion

Reproduce the model with the [Distil Labs CLI](https://docs.distillabs.ai/). The `input/`
directory already holds everything `upload-traces` needs: `traces.jsonl`,
`job_description.json`, and `config.yaml`. The config pins the student (`Qwen3-1.7B`),
the teacher (`zai.glm-5`), the trace-processing settings, the 5,000-example synthgen
target, and 3 LoRA epochs, so the commands below carry no extra flags.

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

**Upload and process the traces.** This runs the trace-processing pipeline: relevance
filtering, committee relabelling (which repairs each turn and inserts
`defer_to_larger_model` on genuinely hard turns), and the train / test / unstructured
split. The original model is also scored on the generated test set as a baseline.

```bash
distil model upload-traces <model-id> --data ./input
distil model upload-status <model-id>      # poll until JOB_SUCCESS
```

**(Optional) teacher evaluation** is a feasibility check before the long run:

```bash
distil model run-teacher-evaluation <model-id>
distil model teacher-evaluation <model-id> --output json | jq '.aggregateMetrics'
```

**Train.** Three stages run server-side: evaluate teacher, generate ~5,000 validated
synthetic examples, then finetune Qwen3-1.7B (LoRA, 3 epochs) and evaluate base vs.
tuned. This takes several hours.

```bash
distil model run-training <model-id>
distil model training <model-id>           # poll until JOB_SUCCESS
```

**Download and deploy.** `download` gives you the merged fp16 weights, the GGUF build,
the standalone LoRA adapter, an inference client, and the base-vs-tuned eval metrics.

```bash
distil model download <model-id>
distil model deploy local <model-id>
distil model invoke <model-id>             # prints the command to query your model
```

## Results (the run that produced the model)

Held-out test set (99 examples), base Qwen3-1.7B → tuned, vs. the GLM-5 teacher:

| metric | base student | **tuned student** | GLM-5 teacher |
|---|---|---|---|
| llm-as-a-judge | 0.42 | **0.72** | 0.70 |
| llm-as-a-judge (reference-free) | 0.50 | **0.79** | 0.66 |
| staged tool call | 0.49 | **0.71** | 0.67 |
| rouge | 0.48 | **0.62** | 0.60 |
| tool-call equivalence (exact) | 0.15 | **0.29** | 0.27 |

Tuning lifts every metric and brings the 1.7B student **level with, or slightly ahead of, its teacher**, including learning *when* to defer rather than guess.
