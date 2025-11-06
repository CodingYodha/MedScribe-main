import sys
import json
from pathlib import Path

def test_imports():
    print("Testing imports...")
    try:
        import numpy
        print(f"  numpy: {numpy.__version__}")
    except ImportError as e:
        print(f"  numpy: FAILED - {e}")
        return False
    
    try:
        import sounddevice
        print(f"  sounddevice: {sounddevice.__version__}")
    except ImportError as e:
        print(f"  sounddevice: FAILED - {e}")
        return False
    
    try:
        import faster_whisper
        print(f"  faster-whisper: OK")
    except ImportError as e:
        print(f"  faster-whisper: FAILED - {e}")
        return False
    
    try:
        import transformers
        print(f"  transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"  transformers: FAILED - {e}")
        return False
    
    try:
        import torch
        print(f"  torch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  torch: FAILED - {e}")
        return False
    
    return True

def test_audio_devices():
    print("\nTesting audio devices...")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"  Found {len(devices)} audio devices:")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"    [{i}] {device['name']} (Input)")
        default_input = sd.query_devices(kind='input')
        print(f"  Default input device: {default_input['name']}")
        return True
    except Exception as e:
        print(f"  Audio device test FAILED - {e}")
        return False

def test_model_paths():
    print("\nTesting model paths...")
    
    whisper_path = Path(r"E:\Projects\Med_Scribe\Medscribe_testing\models\large-v3")
    gemma_path = Path(r"E:\Projects\Med_Scribe\Medscribe_testing\models\finetuned\gemma-prescription-finetuned-it-merged_final")
    
    if whisper_path.exists():
        print(f"  Whisper model path: OK")
    else:
        print(f"  Whisper model path: NOT FOUND - {whisper_path}")
        return False
    
    if gemma_path.exists():
        print(f"  Gemma model path: OK")
    else:
        print(f"  Gemma model path: NOT FOUND - {gemma_path}")
        return False
    
    return True

def main():
    print("=" * 60)
    print("MedScribe System Test")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Audio Devices", test_audio_devices()))
    results.append(("Model Paths", test_model_paths()))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\nAll tests passed! System is ready to run.")
        print("Run 'python main.py' to start the MedScribe system.")
        return 0
    else:
        print("\nSome tests failed. Please fix the issues before running the system.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
