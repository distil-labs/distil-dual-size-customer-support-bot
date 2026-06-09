# Flexible Customer Support Bot 🛫

*A fine-tuned small language model that resolves most airline customer-support turns itself and **defers genuinely-hard turns to a larger model**, a cascade that reserves the big model (and its cost) for the cases that actually need it.*

You're building a customer-support assistant. A frontier LLM handles every conversation well, but you're paying frontier prices for "look up my reservation" and "what's my baggage allowance," which are the overwhelming majority of turns. The hard turns (refund eligibility under fare rules, compensation math across passengers, multi-constraint rebooking) are a small minority, but they're the ones where a small model quietly gets it *wrong*.

This demo wires up a **two-tier cascade**. A **Qwen3-1.7B** model, fine-tuned on airline support, handles the bulk of turns locally. When it hits a turn whose correct action depends on non-obvious policy eligibility, combining several rules, or a multi-step calculation, it emits a single `defer_to_larger_model` tool call and the orchestrator hands the conversation to a larger, pluggable model. The small model learns *when it is out of its depth*, and that judgment is the whole point.

[Distil Labs](https://www.distillabs.ai/) is a platform for training task-specific small language models via knowledge distillation: models 50-400x smaller than current state-of-the-art LLMs that maintain comparable accuracy on a bounded task and run on your own machine. Check out [our docs](https://docs.distillabs.ai/) to dive deeper.

> **Trained weights available.** The model repos now ship the distilled Qwen3-1.7B (GLM-5 teacher). On the held-out airline test set the tuned 1.7B model edges out its roughly 40x larger teacher on llm-as-a-judge and staged tool calling (see Results), and it defers on the genuinely-hard turns.

## How the cascade works

```
You ── user message ──▶  ┌──────────────┐   defer_to_larger_model   ┌──────────────┐
                         │  SLM (local) │ ────────────────────────▶ │ Large model  │
   respond_to_user ◀──── │  Qwen3-1.7B  │   (sticky: rest of conv)  │ (pluggable)  │
                         └──────┬───────┘                           └──────┬───────┘
                                │  tool calls + results  ◀─────────────────┘
                                ▼
                     get_reservation_details, book_reservation,
                     send_certificate, think, ... (16 tools)
```

Every assistant action is a **single tool call**, including talking to the customer via `respond_to_user`. This keeps the orchestrator thin and deterministic: it routes tool calls, executes them, and feeds results back. On a hard turn the SLM calls `defer_to_larger_model`, and from that point the larger model handles the rest of the conversation. This is a *capability* escalation, distinct from `transfer_to_human_agents`, which is for out-of-scope or explicit human requests.

## Results

Held-out airline test set. The tuned 1.7B model beats its roughly 40x larger GLM-5 teacher on llm-as-a-judge (0.722 vs 0.697) and staged tool calling (0.707 vs 0.667), and lifts every metric well above the base model.

| Model | llm-as-a-judge | llm-judge (ref-free) | staged_tool_call | ROUGE | tool_call_equiv |
|---|:---:|:---:|:---:|:---:|:---:|
| GLM-5 teacher (~40x larger) | 0.697 | - | 0.667 | - | - |
| **Distil Qwen3-1.7B (tuned)** | **0.722** | **0.794** | **0.707** | **0.616** | **0.290** |
| Qwen3-1.7B (base) | 0.422 | 0.502 | 0.487 | 0.482 | 0.154 |

## Quick Start

### Prerequisites
- Python 3.10+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) with `llama-server` on your PATH
- A large-model endpoint for the deferral tier: **any OpenAI-compatible API** (OpenAI, Together, a vLLM server, a Bedrock proxy, and so on)

### 1. Install
```bash
./install.sh   # installs the Python deps (openai, huggingface_hub)
```

### 2. Serve the SLM
Download the GGUF (currently a base Qwen3-1.7B placeholder; the trained weights replace it in place in the same repo) and serve it:
```bash
hf download distil-labs/distil-qwen3-1.7b-customer-support-deferral-gguf \
  distil-qwen3-1.7b-customer-support-deferral.gguf --local-dir models

llama-server --model models/distil-qwen3-1.7b-customer-support-deferral.gguf \
  --port 8000 --jinja
```

### 3. Configure the large (deferral) model
Any OpenAI-compatible endpoint, via environment variables:
```bash
export DEFER_BASE_URL=https://api.openai.com/v1     # or https://api.together.xyz/v1, your vLLM, etc.
export DEFER_API_KEY=sk-...
export DEFER_MODEL=gpt-4o
```

### 4. Run
```bash
python orchestrator.py --port 8000        # add --debug to see raw model output each step
```

## Usage Examples

The SLM resolves ordinary turns itself, making tool calls and replying via `respond_to_user`:

```
You: Hi! Can you pull up my reservation, the ID is 8JX2WO?
  · [SLM] get_reservation_details(reservation_id="8JX2WO")
Bot: [SLM] Here are the details of your reservation 8JX2WO. Anything else I can help with?
```

On a hard eligibility turn, the trained SLM defers, and the larger model takes over with a policy-grounded answer:

```
You: I want a full refund. It's a basic economy ticket and I've already flown the first leg. Am I eligible?

  ⤴ SLM deferred to the large model.
     reason: refund eligibility depends on fare class (basic economy) + partially-flown segments
     → the large model now handles the rest of this conversation.

Bot: [LARGE] You are not eligible for a full refund because you have already flown the first leg
            of your basic economy ticket. Let me connect you with a human agent for next steps.
```

Each turn is badged `[SLM]` or `[LARGE]`, and tool calls are traced inline so you can see exactly where the conversation is handled.

## How the Orchestrator Works

`orchestrator.py` runs a small agent loop per user message. It calls the active model with `tool_choice="required"`, gets back one tool call, and acts on it:

| Tool call | Orchestrator action |
|---|---|
| `respond_to_user(message)` | Print the reply, end the turn (terminal, no result follows) |
| `think(thought)` | Record reasoning, continue |
| `defer_to_larger_model(reason)` | **Switch to the large model for the rest of the conversation**, re-run the step |
| `transfer_to_human_agents(summary)` | Print a hand-off line, end the turn |
| any other tool | Execute it, feed the result back to the model, continue |

The tools and the airline policy (the system prompt) are loaded from `job_description.json`, the *same* artifact the model is trained on, so the demo and the model never drift apart. Tool execution lives behind a single integration point, `execute_tool` in `tools.py`: connect it to your reservation systems (or the tau-bench airline environment) and the orchestrator and model stay unchanged.

```
.
├── orchestrator.py      # the cascade: model clients + agent loop + sticky deferral + CLI
├── tools.py             # 16-tool catalog (from job_description.json) + tool execution
├── job_description.json # airline policy + tool schemas (the training artifact)
├── install.sh
├── requirements.txt
└── README.md
```

## How We Built the Model

The model is distilled with the [Distil Labs](https://www.distillabs.ai/) platform:

1. **Traces**: airline customer-support conversations from a public dataset (tau-bench airline tool set), filtered to a single shared policy and converted to OpenAI-style tool-calling traces.
2. **Trace processing**: the raw traces are cleaned, normalized, and relabeled by a teacher model through the distil trace-processing pipeline.
3. **Deferral signal**: a `defer_to_larger_model` tool plus explicit policy guidance is added, so the teacher marks the genuinely-hard turns (compensation eligibility, multi-constraint changes) for escalation while the student learns to handle the rest.
4. **Synthetic expansion + fine-tuning**: the dataset is expanded and distilled onto Qwen3-1.7B, with **GLM-5** as the teacher.

The resulting model is published in two formats:
- [`distil-qwen3-1.7b-customer-support-deferral`](https://huggingface.co/distil-labs/distil-qwen3-1.7b-customer-support-deferral): transformers / safetensors (vLLM, `AutoModel`)
- [`distil-qwen3-1.7b-customer-support-deferral-gguf`](https://huggingface.co/distil-labs/distil-qwen3-1.7b-customer-support-deferral-gguf): GGUF for llama.cpp (used by this demo)

### Supported Functions (16 tools)

| Function | Description |
|---|---|
| `book_reservation` | Book a new flight reservation |
| `cancel_reservation` | Cancel an existing reservation |
| `get_reservation_details` | Look up a reservation |
| `get_user_details` | Look up a user / profile |
| `list_all_airports` | List supported airports |
| `search_direct_flight` | Search direct flights |
| `search_onestop_flight` | Search one-stop flights |
| `update_reservation_flights` | Change flights on a reservation |
| `update_reservation_baggages` | Update baggage on a reservation |
| `update_reservation_passengers` | Update passengers on a reservation |
| `send_certificate` | Issue a travel certificate / compensation |
| `calculate` | Perform an arithmetic calculation |
| `think` | Private step-by-step reasoning (no side effects) |
| `respond_to_user` | Send a natural-language message to the customer (ends the turn) |
| `transfer_to_human_agents` | Hand off to a human agent (out-of-scope / explicit request) |
| `defer_to_larger_model` | Escalate this turn to a larger model (capability escalation) |

## Train Your Own Model

The workflow is generic across multi-turn tool-calling tasks. To train a deferral SLM for your own support domain:

```bash
curl -fsSL https://cli-assets.distillabs.ai/install.sh | sh
distil login

distil model create my-support-deferral
distil model upload-data <model-id> --data ./data    # job_description.json + traces
distil model run-training <model-id>
distil model download <model-id>
```

You can also use the [Distil CLI Claude Code skill](https://github.com/distil-labs/distil-cli-skill) to drive training directly from Claude Code.

## FAQ

**Q: Why a cascade instead of just using the big model for everything?**
Most support turns are simple lookups and confirmations a small local model handles well and cheaply. A cascade reserves the large model for the hard minority, so you pay frontier prices only when the problem actually warrants it.

**Q: How is `defer_to_larger_model` different from `transfer_to_human_agents`?**
Deferring is a *capability* escalation: a larger model continues the same automated conversation with the same tools and policy. Transferring is for requests outside the tools' scope, or when the user explicitly asks for a person.

**Q: The base model doesn't defer. Is that a bug?**
No. The published weights are currently base Qwen3-1.7B placeholders. Knowing *when* to defer is exactly what the distillation teaches; the trained weights make the SLM defer on the hard turns.

**Q: How do I connect real systems?**
Tool execution is a single integration point, `execute_tool` in `tools.py`. Point it at your reservation backend (or the tau-bench airline environment); the orchestrator and the model do not change.

**Q: Can I use a different large model?**
Yes. It's any OpenAI-compatible endpoint, set via `DEFER_BASE_URL` / `DEFER_API_KEY` / `DEFER_MODEL`. Likewise, swap the SLM by pointing `llama-server` at a different GGUF.

## Links

- [Transformers model](https://huggingface.co/distil-labs/distil-qwen3-1.7b-customer-support-deferral)
- [GGUF model](https://huggingface.co/distil-labs/distil-qwen3-1.7b-customer-support-deferral-gguf)
- [Distil Labs Website](https://www.distillabs.ai/)
- [GitHub](https://github.com/distil-labs)
- [Hugging Face](https://huggingface.co/distil-labs)
```
