#!/usr/bin/env bash
set -e

# Python dependencies (the OpenAI client + HF hub for downloading the GGUF).
pip install -r requirements.txt

cat <<'EOF'

Python deps installed.

Next:
  1. Install llama.cpp so that `llama-server` is on your PATH:
       https://github.com/ggerganov/llama.cpp
  2. Download the SLM GGUF (the distilled Qwen3-1.7B):
       hf download distil-labs/distil-qwen3-1.7b-customer-support-deferral-gguf \
         distil-qwen3-1.7b-customer-support-deferral.gguf --local-dir models
  3. Serve it:
       llama-server --model models/distil-qwen3-1.7b-customer-support-deferral.gguf \
         --port 8000 --jinja
  4. Point the large (deferral) model at any OpenAI-compatible endpoint:
       export DEFER_BASE_URL=https://api.openai.com/v1
       export DEFER_API_KEY=sk-...
       export DEFER_MODEL=gpt-4o
  5. Run:  python orchestrator.py --port 8000

See README.md for details.
EOF
