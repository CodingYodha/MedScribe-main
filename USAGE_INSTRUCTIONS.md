# MedScribe - Voice-Assisted Prescription System

A real-time voice-to-prescription system that transcribes doctor-patient conversations in Hindi/Marathi and automatically extracts prescription details using AI models.

## System Architecture

The system consists of three main components running concurrently:

1. **Audio Capture Module**: Continuously captures audio from microphone in 3-second chunks
2. **Transcription Engine**: Uses Faster Whisper to translate Hindi/Marathi audio to English text
3. **Gemma Processor**: Extracts structured prescription data (medicines, tests, dosages) from transcriptions

All three components run in parallel threads with proper queue management to handle processing delays.

## Key Features

- Continuous audio streaming with buffer management
- Real-time Hindi/Marathi to English translation
- Automatic prescription detail extraction
- Models remain loaded throughout the session (no reloading)
- Edge case handling for silence detection, queue overflow, and processing delays
- Automatic prescription saving to timestamped JSON files
- Comprehensive logging to file and console

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- Microphone connected to the system
- Windows OS with PowerShell

## Installation

### Step 1: Set Up Python Environment

Open PowerShell and navigate to the project directory:

```powershell
cd E:\Projects\Med_Scribe\MedScribe
```

Create a virtual environment:

```powershell
python -m venv venv
```

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

If you encounter execution policy errors, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 2: Install Dependencies

Install required packages:

```powershell
pip install -r requirements.txt
```

For CUDA support (recommended), ensure you have CUDA toolkit installed and install PyTorch with CUDA:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only installation:

```powershell
pip install torch torchvision torchaudio
```

### Step 3: Verify Model Paths

Ensure the following paths exist and contain the models:

- Faster Whisper Model: `E:\Projects\Med_Scribe\Medscribe_testing\models\large-v3`
- Gemma Model: `E:\Projects\Med_Scribe\Medscribe_testing\models\finetuned\gemma-prescription-finetuned-it-merged_final`

If your model paths are different, update the paths in `main.py`:

```python
class Config:
    WHISPER_MODEL_PATH = r"YOUR_WHISPER_PATH"
    GEMMA_MODEL_PATH = r"YOUR_GEMMA_PATH"
```

## Usage

### Running the System

1. Ensure your virtual environment is activated
2. Connect your microphone
3. Run the main script:

```powershell
python main.py
```

### What Happens When Running

1. **Initialization Phase**: Both Whisper and Gemma models are loaded (this may take 30-60 seconds)
2. **Active Listening Phase**: System starts capturing audio and displays real-time transcriptions
3. **Processing Phase**: Extracted prescription details are logged and saved automatically
4. **Stopping**: Press `Ctrl+C` to stop the system gracefully

### Output Files

- **Prescription JSON Files**: `prescription_YYYYMMDD_HHMMSS.json` - Created each time new data is extracted
- **Log File**: `medscribe.log` - Contains detailed system logs

### Sample Prescription JSON Output

```json
{
  "patient_name": "Ramesh Kumar",
  "medicines": [
    {
      "name": "Paracetamol",
      "dosage": "500mg",
      "frequency": "3 times daily",
      "duration": "5 days"
    }
  ],
  "tests": ["Blood Sugar", "CBC"],
  "diagnosis": "Fever",
  "instructions": "Take medicine after meals"
}
```

## System Configuration

You can adjust these parameters in `main.py` under the `Config` class:

- `AUDIO_CHUNK_DURATION`: Duration of each audio chunk in seconds (default: 3)
- `SAMPLE_RATE`: Audio sample rate in Hz (default: 16000)
- `MIN_AUDIO_LENGTH`: Minimum audio length to process (default: 1.0 seconds)
- `SILENCE_THRESHOLD`: RMS threshold for silence detection (default: 0.01)
- `MAX_QUEUE_SIZE`: Maximum queue size for audio/transcription buffers (default: 100)

## Edge Cases Handled

1. **Audio Buffer Overflow**: If audio is captured faster than processing, oldest chunks are dropped with warnings
2. **Silence Detection**: Silent audio chunks are filtered out to avoid unnecessary processing
3. **Empty Transcriptions**: Empty or null transcriptions are skipped
4. **Queue Overflow**: Both transcription and processing queues have overflow protection
5. **Model Loading Errors**: Graceful error handling with detailed logging
6. **Transcription Delays**: Threaded architecture ensures audio capture continues while transcription processes
7. **Microphone Disconnection**: Audio stream errors are logged without crashing the system

## Architecture Details

### Threading Model

```
Main Thread
    |
    +-- Audio Capture Thread (captures microphone input continuously)
    |        |
    |        v
    |   Audio Queue (3-second chunks)
    |        |
    |        v
    +-- Transcription Thread (processes audio chunks)
    |        |
    |        v
    |   Transcription Queue (English text)
    |        |
    |        v
    +-- Gemma Processing Thread (extracts JSON data)
             |
             v
        Prescription JSON Files
```

### Key Design Decisions

1. **No Model Reloading**: Models are loaded once during initialization and kept in memory until the program exits
2. **Queue-Based Architecture**: Decouples audio capture from processing to handle Whisper's processing time
3. **Accumulation Strategy**: Keeps last 10 transcriptions for context when extracting prescription details
4. **Incremental JSON Merging**: New prescription data is merged with existing data to build complete prescription

## Troubleshooting

### Issue: "Audio capture failed"
- Check if microphone is connected and recognized by Windows
- Verify microphone permissions in Windows Settings
- Try listing available devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`

### Issue: "CUDA out of memory"
- Reduce batch processing or use CPU mode
- Close other GPU-intensive applications
- Update `Config.DEVICE = "cpu"` to force CPU usage

### Issue: "Models not loading"
- Verify model paths are correct and accessible
- Check if you have sufficient disk space and RAM
- Ensure all dependencies are installed correctly

### Issue: "Poor transcription quality"
- Ensure microphone is close to the speaker
- Reduce background noise
- Adjust `SILENCE_THRESHOLD` if too sensitive
- Check if the correct language is set (currently Hindi)

### Issue: "No JSON output"
- Verify the Gemma model is trained to output JSON format
- Check logs for JSON parsing errors
- Ensure transcriptions contain prescription-related information

## Performance Optimization

- **GPU Usage**: System automatically uses CUDA if available
- **CPU Threads**: Whisper uses 4 CPU threads for faster processing
- **VAD Filtering**: Voice Activity Detection reduces unnecessary transcription
- **Chunk Size**: 3-second chunks balance latency and accuracy

## Future Enhancements

- Speaker diarization to differentiate doctor and patient
- Integration with prescription software via UI automation
- Support for multiple languages
- Real-time UI for displaying extracted information
- Confirmation and editing interface
- Direct printing integration

## License

This project is for educational and research purposes.

## Support

For issues or questions, check the log files first:
- `medscribe.log` - System logs with detailed error messages

## Technical Requirements

- Minimum 8GB RAM (16GB recommended)
- 4GB free disk space for models
- Microphone with decent quality
- Windows 10 or higher
- Python 3.8 or higher
