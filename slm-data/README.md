# iteration-4 reproduction uploads object

A single self-contained distil-lib input whose pipeline output **approximates the blended
training set** used for the iteration-4 deferral SLM (Qwen3-1.7B, airline support + `defer_to_larger_model`).

## Contents
- `traces.jsonl`: the original seed: 1,587 APIGen-MT **airline** traces (OpenAI-message format).
- `job_description.json`: 16 tools (14 tau-bench airline + `respond_to_user` + `defer_to_larger_model`).
- `config.yaml`: one config driving all stages (trace-processing → synthgen → finetune).

## What it reproduces (and the caveat)
The actual iter-4 data was a **blend of two synthgen runs**, a *contrastive* run (~49% conv-defer)
and a *pure-defer* run (~100% conv-defer), combined to **~74% conv-level / ~14% turn-level defer**.

This config reproduces that **distribution in a single run**: the `synthgen.mutation_topics` pool is
~87% defer-forcing (13 of 15 topics) + 2 same-theme hard-negatives, which yields ~74% conv-defer with
GLM-5. It is an **approximation of the blend's character**, not a byte-exact rebuild of the two runs.

## How to run (teacher = GLM-5 on together.ai)
The e2e template does **not** run trace-processing, so it's two steps:
```bash
export DISTIL_LIB_LLM_PROVIDER=together_ai
export TOGETHER_API_KEY=...   # from AWS Secrets Manager: together-ai-api-key (eu-central-1)

# 1) traces -> seed train/test/unstructured (teacher relabels + injects defers)
process-traces --input-dir . --output-dir ./_seeds

# 2) seeds + this config -> synthgen (~74% defer) -> finetune  (Argo run-e2e-distillation, or local)
#    point the synthgen/finetune stage at ./_seeds with this config.yaml
```
NOTE: `generation_target: 7000` is a large run (hours of teacher calls); this is the *recipe*; the
shipped iter-4 model was trained on the recovered/blended data, not by re-running this from scratch.

## Licensing
`traces.jsonl` is derived from APIGen-MT-5k (**CC-BY-NC-4.0, non-commercial**), not covered by the
repo's code license. Flag before any redistribution; or ship scripts-only and regenerate the seed.
