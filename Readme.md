# MedScribe - Voice-Assisted Prescription System

An AI-powered real-time transcription and prescription extraction system for medical consultations.

## Quick Start

1. **Install Dependencies**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Test System**
   ```powershell
   python test_system.py
   ```

3. **Run Application**
   ```powershell
   python main.py
   ```

## Features

- Real-time Hindi/Marathi to English transcription using Faster Whisper
- Automatic prescription detail extraction using fine-tuned Gemma model
- Continuous audio processing with no model reloading
- Comprehensive edge case handling
- Automatic JSON output with timestamps

## Documentation

See `USAGE_INSTRUCTIONS.md` for detailed setup and usage instructions.

## Files

- `main.py` - Main application with threading architecture
- `requirements.txt` - Python dependencies
- `config.ini` - Configuration settings
- `test_system.py` - System verification script
- `USAGE_INSTRUCTIONS.md` - Comprehensive documentation

## Requirements

- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- Microphone
- CUDA GPU (recommended for performance)