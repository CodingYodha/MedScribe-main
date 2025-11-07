#!/bin/bash

# WSL2 cuDNN Compatibility Script
# This script sets critical environment variables BEFORE Python starts
# to prevent cuDNN from loading in a broken state

echo "============================================================"
echo "MedScribe - WSL2 cuDNN Compatibility Mode"
echo "============================================================"
echo ""
echo "Applying WSL2 fixes..."

# CRITICAL: Disable cuDNN v8 API (WSL2 incompatible)
export TORCH_CUDNN_V8_API_ENABLED=0

# Force lazy CUDA module loading to prevent early initialization
export CUDA_MODULE_LOADING=LAZY

# Prevent CUDA memory caching issues
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Limit GPU visibility (optional, use if multi-GPU system)
export CUDA_VISIBLE_DEVICES=0

# Disable TensorFlow GPU (if installed, to avoid conflicts)
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Log level for debugging (optional)
export PYTHONUNBUFFERED=1

echo "✓ Environment variables configured:"
echo "  - TORCH_CUDNN_V8_API_ENABLED=0 (cuDNN disabled)"
echo "  - CUDA_MODULE_LOADING=LAZY (delayed loading)"
echo "  - PYTORCH_NO_CUDA_MEMORY_CACHING=1 (no caching)"
echo ""
echo "Device Strategy:"
echo "  - Whisper: CPU (avoid cuDNN crashes)"
echo "  - Gemma: GPU (cuDNN not required)"
echo ""
echo "============================================================"
echo "Starting MedScribe..."
echo "============================================================"
echo ""

# Activate conda environment if needed
# Uncomment and modify if using conda:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate medd

# Run Python with all environment variables set
python main.py
