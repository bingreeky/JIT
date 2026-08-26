#!/usr/bin/env bash
# Serve the JIT meta model as an OpenAI-compatible endpoint with vLLM.
#
# The pipeline only ever talks HTTP, so this script is a convenience, not a
# requirement: point --meta-base at any OpenAI-compatible endpoint instead
# (SGLang, TGI, a hosted API) and the pipeline behaves the same. Use this when
# you are serving a local JIT checkpoint yourself.
#
#   MODEL=/path/to/checkpoint-70-merged bash scripts/serve_meta_model.sh
#
# Environment:
#   MODEL                   model dir or HF repo id                  (required)
#   SERVED_NAME             --meta-model name clients use            (default: jit)
#   PORT                    listen port                              (default: 8000)
#   TP                      tensor-parallel size                     (default: 4)
#   MAX_MODEL_LEN           context window                           (default: 163840)
#   GPU_MEM_UTIL            KV-cache fraction                        (default: 0.90)
#   MAX_NUM_SEQS            concurrent sequences                     (default: 32)
#   MAX_NUM_BATCHED_TOKENS  prefill chunk cap                        (default: 4096)
set -euo pipefail

MODEL="${MODEL:?set MODEL=/path/to/jit/checkpoint (or a HF repo id)}"
SERVED_NAME="${SERVED_NAME:-jit}"
PORT="${PORT:-8000}"
TP="${TP:-4}"

# The generation prompt carries the reference catalogue, the full tool
# descriptions and the task; generation itself is a whole 5-file harness. The
# window has to hold both -- 32k is not enough and silently truncates harnesses.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-163840}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"

# The logprob selector scores candidates on this same server with
# prompt_logprobs, which transiently materializes (chunk x vocab) logits. An
# uncapped prefill chunk over a ~30k-token candidate costs several GiB and OOMs
# at util 0.9, so cap the chunk. Set to "" if you score on a separate endpoint.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"

EXTRA=()
[ -n "$MAX_NUM_BATCHED_TOKENS" ] && EXTRA+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")

echo "[serve] model=$MODEL served-as=$SERVED_NAME tp=$TP len=$MAX_MODEL_LEN port=$PORT"
echo "[serve] clients: --meta-base http://<host>:$PORT/v1 --meta-model $SERVED_NAME"

# NB: do NOT enable --reasoning-parser qwen3. It splits the assistant turn at
# the FIRST </think>, and a generated harness legitimately contains a literal
# </think> (e.g. re.sub(r'<think>.*?</think>', '', x) used to strip the
# executor's own reasoning). The parser then splits INSIDE the generated code
# and drops memory.py / planning.py into reasoning_content, which surfaces as a
# spurious "did not output all required files". With the parser off the full raw
# output lands in `content`; the model's own draft skeleton is handled by
# _parse_harness_response, which takes the LAST <<<TAG>>> block.
exec vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  ${EXTRA[@]+"${EXTRA[@]}"}
