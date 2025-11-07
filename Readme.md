# MedScribe - Voice-Assisted Prescription System

An AI-powered real-time transcription and prescription extraction system for medical consultations.

## Quick Start

### Using Existing Conda Environment

```bash
cd /mnt/e/Projects/Med_Scribe/MedScribe

# Install system dependencies (one-time only)
chmod +x install_system_deps.sh
./install_system_deps.sh

# Activate environment
conda activate medd

# Install missing Python packages
pip install sounddevice tqdm

# Run (if cuDNN issues, use workaround script)
chmod +x run_with_workaround.sh
./run_with_workaround.sh
```

**Note**: If you encounter cuDNN errors, the workaround script handles them automatically.

### Fresh Installation

```bash
cd /mnt/e/Projects/Med_Scribe/MedScribe
conda env create -f environment_linux.yml
conda activate medd
python test_system.py  # Verify setup
python main.py
```

## Features

- Real-time transcription to English using Faster Whisper with BatchedInferencePipeline
- 4-bit quantized Gemma model for efficient prescription extraction
- Continuous audio processing with no model reloading
- Comprehensive edge case handling
- Automatic JSON output with timestamps
- Custom stopping criteria for clean output

## Key Technologies

- **Faster Whisper 1.2.0**: BatchedInferencePipeline for efficient transcription
- **Fine-tuned Gemma**: 4-bit quantized with BitsAndBytesConfig (bitsandbytes 0.48.1)
- **PyTorch 2.8.0**: CUDA 12.4 support
- **Multi-threaded**: Parallel audio capture, transcription, and processing
- **pysoundfile 0.9.0**: Audio backend for torchaudio
- **Local models only**: No internet required for inference (`local_files_only=True`)

## Documentation

- **`QUICK_START.md`** - Quick reference guide for immediate setup
- **`USAGE_INSTRUCTIONS.md`** - Detailed setup and usage instructions
- **`CONDA_SETUP.md`** - Comprehensive conda environment guide
- **`LOCAL_MODELS_SETUP.md`** - Local model configuration (no downloads)
- **`UPDATE_SUMMARY.md`** - Changes from notebook implementation
- **`NOTEBOOK_COMPARISON.md`** - Side-by-side comparison with your notebook
- **`environment_linux.yml`** - Complete conda environment specification

## Files

- `main.py` - Main application with threading architecture
- `requirements.txt` - Python dependencies
- `config.ini` - Configuration settings
- `test_system.py` - System verification script

## Requirements

- Python 3.11.14 (from conda environment)
- CUDA 12.4 GPU (required for 4-bit quantization)
- 16GB RAM minimum
- Microphone
- WSL or Linux environment (recommended)
- Local model files (no downloads during runtime):
  - `E:\Projects\Med_Scribe\Medscribe_testing\models\large-v3`
  - `E:\Projects\Med_Scribe\Medscribe_testing\models\finetuned\gemma-prescription-finetuned-it-merged_final`

## Model Paths

Update these in `main.py` if your paths differ:
```python
WHISPER_MODEL_PATH = r"E:\Projects\Med_Scribe\Medscribe_testing\models\large-v3"
GEMMA_MODEL_PATH = r"E:\Projects\Med_Scribe\Medscribe_testing\models\finetuned\gemma-prescription-finetuned-it-merged_final"
```