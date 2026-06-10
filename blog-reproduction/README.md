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
   ShareGPT → OpenAI `messages`. Every assistant turn is exactly **one tool call** (free-text
   replies are wrapped as `respond_to_user(message)`), so the task is uniformly tool-calling.
   Also emits `job_description.json` (task description + the airline tool schemas).

2. **Corrupt** (`corrupt_traces.py`): inject realistic production noise so the teacher has
   something to repair. Three orthogonal corruptions, each applied independently to a fraction
   of traces: **random truncation**, **argument mutation**, and **schema drift**.

3. **Add deferral** (`prepare_deferral_inputs.py`): data-only change to
   `job_description.json`: add a 16th tool, **`defer_to_larger_model`**, plus a guideline that
   frames deferral in **absolute terms of task complexity** (defer when a turn is genuinely
   hard, *not* whenever a particular tool is involved). The teacher inserts deferral during
   trace processing; it lands on **~3% of assistant turns**, a deliberately small fraction.

The result in `input/`: **1,587 traces**, a **16-tool** job description, and a `config.yaml`
that drives the whole pipeline.

## (c) Train the model to completion

**Prerequisites**

```bash
uv sync --extra full --extra dev        # torch/transformers/peft + CLI entrypoints
# Teacher access (GLM-5 via AWS Bedrock by default). Any supported tool-calling teacher works.
export DISTIL_LIB_LLM_PROVIDER=bedrock   # or: vertex_ai | together_ai
# ...plus that provider's credentials. A GPU is required for stage 3.
```

**Three stages**: each stage's output directory is the next stage's input:

```bash
# 1) Trace processing: repair/clean traces, insert deferral on hard turns,
#    split into train (100) / test (100) / unstructured (rest), and score the originals.
uv run process-traces          --input-dir input        --output-dir 1-processed

# 2) Synthetic data generation: the teacher (GLM-5) expands the 100 seed traces
#    into ~5,000 validated synthetic examples (final-synthetic-dataset/).
uv run generate-synthetic-data --input-dir 1-processed  --output-dir 2-synthetic

# 3) Finetune the student (Qwen3-1.7B, LoRA, 3 epochs) and evaluate base vs. tuned.
uv run finetune-student        --input-dir 2-synthetic  --output-dir 3-model
```

Add `--dryrun` to any stage for a fast, no-teacher smoke test of the wiring first.

**What you get** in `3-model/`:

```
3-model/
├── model/                merged fp16 weights (Qwen3-1.7B + adapter)
├── model-adapter/        standalone LoRA adapter
├── model.gguf            quantized build for llama.cpp / Ollama
├── model_client.py       inference client
├── README.md             model card
└── eval/
    ├── base-model/model-eval/metrics-eval-aggregated.json   # before tuning
    └── tuned-model/model-eval/metrics-eval-aggregated.json  # after tuning
```

> Running on a cluster instead of a laptop? The same three stages map to the Argo tasks
> `run-trace-processing`, `run-synthetic-data-generation`, and `run-finetune`
> (or `run-e2e-distillation` to chain synthgen + finetune in one job).

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
