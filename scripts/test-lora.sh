#!/bin/bash
# End-to-end test of LoRA merging with the real model.
# Runs generate_av.py directly (current source) with the real LTX-2.3 Q4 model
# and the realisdance LoRA. Output: ./tmp/test_lora_out.mp4
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p tmp

PYTHON="${PYTHON:-/opt/homebrew/opt/python@3.11/bin/python3.11}"
LORA="${LORA:-/Volumes/T7/ComfyUI-Shared/models/loras/realisdance_ltx2.3_ic-lora_step_02000.safetensors}"

cd LTXVideoGenerator/Resources
"$PYTHON" generate_av.py \
  --prompt "A small fluffy kitten playing with a ball of yarn on a wooden floor, soft natural lighting, close-up shot." \
  --height 320 --width 320 --num-frames 25 --seed 42 --fps 24 --steps 30 --cfg-scale 3.0 \
  --output-path ../../tmp/test_lora_out.mp4 \
  --model-repo notapalindrome/ltx23-mlx-av-q4 \
  --text-encoder-repo mlx-community/gemma-3-12b-it-4bit \
  --tiling auto \
  --lora-path "$LORA" \
  --lora-strength 1.0
