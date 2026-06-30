#!/usr/bin/env bash
# install.sh — Set up the environment for panoramic-360-eval
#
# This script:
#   1. Installs base dependencies via pip
#   2. Applies panoramic Rolling RoPE patches to vLLM and transformers
#
# Requirements: Python 3.10+, CUDA 12.x, pip
#
# Usage:
#   bash install.sh

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================"
echo "  panoramic-360-eval — Environment Setup"
echo "========================================================"

# ── Step 1: Install vLLM and transformers (exact versions) ───────────────────
# vLLM 0.11.0 pins torch==2.8.0 (CUDA 12.8 build) and installs it automatically,
# so we do NOT pre-install torch with a fixed CUDA index (that can mismatch).
echo ""
echo "[1/2] Installing vLLM 0.11.0 and transformers 4.57.0 (pulls torch 2.8.0)..."
pip install vllm==0.11.0 transformers==4.57.0

# Install remaining dependencies (quote specs so bash does not treat > as redirection)
pip install "opencv-python>=4.8.0" "Pillow>=10.0.0" "numpy>=1.24.0" "tqdm>=4.65.0"

# ── Step 2: Apply panoramic patches ──────────────────────────────────────────
echo ""
echo "[2/2] Applying panoramic Rolling RoPE patches..."

VLLM_DIR=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
TRANS_DIR=$(python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")

echo "  vLLM site-packages : $VLLM_DIR"
echo "  transformers site-packages : $TRANS_DIR"

# vLLM patches
install -D "$REPO_DIR/patches/vllm/model_executor/layers/rotary_embedding/mrope.py" \
           "$VLLM_DIR/model_executor/layers/rotary_embedding/mrope.py"
install -D "$REPO_DIR/patches/vllm/model_executor/models/qwen3_vl.py" \
           "$VLLM_DIR/model_executor/models/qwen3_vl.py"
install -D "$REPO_DIR/patches/vllm/multimodal/parse.py" \
           "$VLLM_DIR/multimodal/parse.py"
install -D "$REPO_DIR/patches/vllm/v1/worker/gpu_model_runner.py" \
           "$VLLM_DIR/v1/worker/gpu_model_runner.py"

# transformers patches
install -D "$REPO_DIR/patches/transformers/models/qwen3_vl/configuration_qwen3_vl.py" \
           "$TRANS_DIR/models/qwen3_vl/configuration_qwen3_vl.py"
install -D "$REPO_DIR/patches/transformers/models/qwen3_vl/modeling_qwen3_vl.py" \
           "$TRANS_DIR/models/qwen3_vl/modeling_qwen3_vl.py"
install -D "$REPO_DIR/patches/transformers/models/qwen3_vl/processing_qwen3_vl.py" \
           "$TRANS_DIR/models/qwen3_vl/processing_qwen3_vl.py"
install -D "$REPO_DIR/patches/transformers/processing_utils.py" \
           "$TRANS_DIR/processing_utils.py"

echo ""
echo "========================================================"
echo "  Setup complete! Panoramic Rolling RoPE patches applied."
echo ""
echo "  Next steps:"
echo "    1. Download Matterport3D images (see data/README.md)"
echo "    2. Run evaluation:"
echo "       python eval.py \\"
echo "         --model your-org/panoramic-360-grpo-qwen3vl-4b \\"
echo "         --pano_dir /path/to/matterport3d \\"
echo "         --n_samples 50"
echo "========================================================"
