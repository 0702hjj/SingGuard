#!/usr/bin/env bash
# Serve the converted LLaVAShield HF checkpoint on GPU 6 for validation.
#
# GPU 6 is NOT exclusively ours: another user's training job sits on it (~18.8GB of 48GB),
# so gpu-memory-utilization is pinned well below what the repo's VLLMServer uses (0.90) to
# leave that job headroom.  --enforce-eager drops the cudagraph pool for the same reason.
set -euo pipefail

EVAL=/mnt/data/intern3/research/SingGuard/eval
PY=/mnt/data/intern3/research/SingGuard/.venv/bin/python
LOG=$EVAL/logs/vllm_llavashield_hf.log

# MMDS-Q test rows reach ~38k prompt tokens (multi-image anyres + long policy/query), and
# vLLM hard-400s anything past max-model-len rather than truncating.  The repo's production
# entry for this model says 16384, so raise it here to check the model itself on those rows.
MAXLEN=${MAX_MODEL_LEN:-16384}
# vLLM refuses MAXLEN > config.max_position_embeddings (32768 here) unless this is set.
# Only do it to probe a known-oversized sample: past 32768 the RoPE positions leave the
# model's trained range, so any verdict from such a request is out-of-spec evidence.
if [ "${ALLOW_LONG:-0}" = "1" ]; then
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
fi

cd "$EVAL"
exec env CUDA_VISIBLE_DEVICES=6 "$PY" -u -m vllm.entrypoints.openai.api_server \
  --model models/llavashield-v1-7b-hf \
  --served-model-name guard \
  --port 8199 \
  --dtype bfloat16 \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.42 \
  --max-num-seqs 4 \
  --enforce-eager \
  --limit-mm-per-prompt '{"image": 8}' \
  --chat-template models/llavashield-v1-7b-hf/_chat_template.resolved.jinja \
  --trust-remote-code \
  --disable-mm-preprocessor-cache
