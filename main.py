#main.py
# CRITICAL: Set environment variables BEFORE any CUDA imports
# This must be at the absolute top to prevent cuDNN from loading
import os
import sys

# WSL2 cuDNN Compatibility Fix - MUST be before torch import
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

# Now import torch and disable cuDNN globally
import torch
# Force disable cuDNN - critical for WSL2 stability
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Now safe to import other CUDA-dependent libraries
import json
import time
import queue
import threading
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel, BatchedInferencePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm
import requests  # For model server communication


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('medscribe.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Config:
    # Detect platform and set paths accordingly
    if sys.platform == "linux":
        WHISPER_MODEL_PATH = "/mnt/e/Projects/Med_Scribe/Medscribe_testing/models/large-v3"
        GEMMA_MODEL_PATH = "/mnt/e/Projects/Med_Scribe/Medscribe_testing/models/finetuned/gemma-prescription-finetuned-it-merged_final"
    else:
        WHISPER_MODEL_PATH = r"E:\Projects\Med_Scribe\Medscribe_testing\models\large-v3"
        GEMMA_MODEL_PATH = r"E:\Projects\Med_Scribe\Medscribe_testing\models\finetuned\gemma-prescription-finetuned-it-merged_final"
    
    # Audio configuration
    SAMPLE_RATE = 16000
    AUDIO_CHUNK_DURATION = 3
    AUDIO_CHUNK_SAMPLES = SAMPLE_RATE * AUDIO_CHUNK_DURATION
    
    MIN_AUDIO_LENGTH = 1.0
    SILENCE_THRESHOLD = 0.01
    MAX_QUEUE_SIZE = 100
    
    # CRITICAL: Separate device configuration for Whisper and Gemma
    # Whisper uses CPU to avoid WSL2 cuDNN crashes
    # Gemma uses GPU for speed (doesn't use problematic cuDNN operations)
    WHISPER_DEVICE = "cpu"  # Force CPU - WSL2 cuDNN is broken for Whisper
    WHISPER_COMPUTE_TYPE = "int8"  # CPU-optimized precision
    
    GEMMA_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GEMMA_COMPUTE_TYPE = "float16"
    
    # Whisper inference parameters
    WHISPER_BEAM_SIZE = 4
    WHISPER_BATCH_SIZE = 8
    WHISPER_VAD_FILTER = True
    
    # GPU memory management for 6GB VRAM
    MAX_MEMORY_ALLOCATION = {0: "5GB"}  # Reserve 5GB for GPU, leave 1GB buffer


class AudioCapture:
    def __init__(self, sample_rate=Config.SAMPLE_RATE, chunk_duration=Config.AUDIO_CHUNK_DURATION):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.audio_queue = queue.Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.is_running = False
        self.stream = None
        self.buffer = []
        self.lock = threading.Lock()
        
        # WSL2-specific: Increase latency to prevent timeout
        self.latency = 'high'  # High latency mode for WSL2 compatibility
        
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        try:
            audio_data = indata.copy().flatten()
            
            with self.lock:
                self.buffer.extend(audio_data)
                
                while len(self.buffer) >= self.chunk_samples:
                    chunk = np.array(self.buffer[:self.chunk_samples], dtype=np.float32)
                    self.buffer = self.buffer[self.chunk_samples:]
                    
                    if not self.audio_queue.full():
                        self.audio_queue.put(chunk)
                    else:
                        logger.warning("Audio queue full, dropping oldest chunk")
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put(chunk)
                        except queue.Empty:
                            pass
                            
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
    
    def verify_audio_system(self):
        """Verify audio system is working before starting stream"""
        try:
            logger.info("Verifying audio system...")
            
            # Check if devices exist
            devices = sd.query_devices()
            logger.info(f"  Found {len(devices)} audio devices")
            
            # Check default input
            default_input = sd.default.device[0]
            logger.info(f"  Default input device: {default_input}")
            
            # Try a very short test recording (non-blocking)
            logger.info("  Testing audio capture (1 second test)...")
            test_duration = 1.0
            test_recording = sd.rec(
                int(test_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocking=False
            )
            
            # Wait with timeout
            sd.wait(timeout=5.0)  # 5 second timeout for 1 second recording
            
            if test_recording is not None and len(test_recording) > 0:
                logger.info("  ✓ Audio system test passed")
                return True
            else:
                logger.warning("  ⚠ Audio system test produced no data")
                return False
                
        except Exception as e:
            logger.error(f"  ✗ Audio system test failed: {e}")
            return False
    
    def start(self):
        if self.is_running:
            logger.warning("Audio capture already running")
            return
        
        try:
            self.is_running = True
            
            # WSL2 Fix: Use larger blocksize and explicit latency
            blocksize = int(self.sample_rate * 0.2)  # 200ms blocks instead of 100ms
            
            logger.info("Initializing audio stream (WSL2 compatibility mode)...")
            logger.info(f"  Sample rate: {self.sample_rate} Hz")
            logger.info(f"  Block size: {blocksize} samples ({blocksize/self.sample_rate:.3f}s)")
            logger.info(f"  Latency: {self.latency}")
            
            # Check if PulseAudio is responsive first
            try:
                sd.query_devices()
            except Exception as e:
                logger.error(f"Audio device query failed: {e}")
                raise RuntimeError("Audio system not responsive. Is PulseAudio running?")
            
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=blocksize,  # Larger blocks for WSL2
                latency=self.latency,  # Explicit high latency
                callback=self.audio_callback,
                # WSL2 critical: Don't request real-time priority
                prime_output_buffers_using_stream_callback=False
            )
            
            # Start with timeout handling
            logger.info("Starting audio stream (this may take 10-15 seconds in WSL2)...")
            self.stream.start()
            
            # Verify stream is actually running
            time.sleep(0.5)
            if not self.stream.active:
                raise RuntimeError("Stream started but not active")
            
            logger.info(f"✓ Audio capture started successfully")
            
        except sd.PortAudioError as e:
            logger.error(f"PortAudio error: {e}")
            logger.error("Troubleshooting steps:")
            logger.error("  1. Run: pulseaudio --check")
            logger.error("  2. Run: pulseaudio --start")
            logger.error("  3. Check WSL2 audio passthrough is enabled")
            self.is_running = False
            raise
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            self.is_running = False
            raise
    
    def stop(self):
        self.is_running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                logger.info("Audio capture stopped")
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
    
    def get_audio_chunk(self, timeout=1):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class TranscriptionEngine:
    def __init__(self, model_path=Config.WHISPER_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.batched_model = None
        self.transcription_queue = queue.Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.is_running = False
        
    def load_model(self):
        try:
            logger.info(f"Loading Faster Whisper model from {self.model_path}")
            logger.info(f"Device: {Config.WHISPER_DEVICE} (CPU mode for WSL2 cuDNN compatibility)")
            
            with tqdm(total=100, desc="Loading Whisper Model", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                pbar.update(20)
                # Force Whisper to CPU to avoid cuDNN crashes in WSL2
                self.model = WhisperModel(
                    self.model_path,
                    device=Config.WHISPER_DEVICE,  # CPU only
                    compute_type=Config.WHISPER_COMPUTE_TYPE,  # int8 for CPU efficiency
                    local_files_only=True
                )
                pbar.update(60)
                
                self.batched_model = BatchedInferencePipeline(model=self.model)
                pbar.update(20)
            
            logger.info(f"✓ Whisper model loaded successfully on {Config.WHISPER_DEVICE}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_data):
        if self.batched_model is None:
            logger.error("Model not loaded")
            return None
        
        try:
            # Check for silence
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms < Config.SILENCE_THRESHOLD:
                logger.debug("Silence detected, skipping transcription")
                return None
            
            # Direct numpy array transcription - safe on CPU
            # (cuDNN is disabled globally, so this won't crash)
            segments, info = self.batched_model.transcribe(
                audio_data,  # Direct numpy array - no temp file needed
                language='en',
                beam_size=Config.WHISPER_BEAM_SIZE,
                vad_filter=Config.WHISPER_VAD_FILTER,
                batch_size=Config.WHISPER_BATCH_SIZE,
                word_timestamps=False
            )
            
            transcript_chunks = [segment.text.strip() for segment in segments if segment.text.strip()]
            transcription = " ".join(transcript_chunks)
            
            if transcription:
                logger.info(f"Transcription: {transcription}")
                return transcription
            else:
                logger.debug("Empty transcription result")
                return None
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    def process_audio_queue(self, audio_queue):
        self.is_running = True
        logger.info("Transcription thread started")
        
        while self.is_running:
            try:
                audio_chunk = audio_queue.get(timeout=1)
                
                if audio_chunk is None:
                    continue
                
                # Transcribe with proper error handling
                try:
                    transcription = self.transcribe_audio(audio_chunk)
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"CUDA OOM error: {e}")
                    logger.info("Clearing CUDA cache...")
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error during transcription: {e}")
                    continue
                
                if transcription and not self.transcription_queue.full():
                    self.transcription_queue.put(transcription)
                elif transcription and self.transcription_queue.full():
                    logger.warning("Transcription queue full, dropping oldest entry")
                    try:
                        self.transcription_queue.get_nowait()
                        self.transcription_queue.put(transcription)
                    except queue.Empty:
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in transcription thread: {e}")
        
        logger.info("Transcription thread stopped")
    
    def stop(self):
        self.is_running = False


class GemmaProcessor:
    """Lightweight client for Gemma Model Server (Ollama-style)"""
    
    def __init__(self, server_url="http://127.0.0.1:5000", model_path=None):
        self.server_url = server_url
        self.model_path = model_path  # Not used, kept for compatibility
        self.is_running = False
        self.accumulated_text = []
        self.last_json_output = {}
        
    def load_model(self):
        """Check if model server is ready (model already loaded in VRAM)"""
        try:
            logger.info(f"Connecting to Gemma model server at {self.server_url}")
            
            response = requests.get(f"{self.server_url}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('model_loaded'):
                    vram = data.get('vram_allocated_gb', 0)
                    logger.info("✓ Connected to model server")
                    logger.info(f"  Model already in VRAM: {vram:.2f}GB")
                    logger.info(f"  Inference will be INSTANT - no loading time!")
                    return True
                else:
                    logger.error("Model server found but model not loaded")
                    logger.error("Wait for model server to finish loading...")
                    return False
            else:
                logger.error(f"Model server returned status {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to model server!")
            logger.error("\n" + "="*60)
            logger.error("Please start the model server first:")
            logger.error("")
            logger.error("  Option 1 (Foreground):")
            logger.error("    bash start_model_server.sh")
            logger.error("")
            logger.error("  Option 2 (Background):")
            logger.error("    bash start_model_server.sh background")
            logger.error("")
            logger.error("  Option 3 (Manual):")
            logger.error("    python model_server.py --model-path /path/to/gemma")
            logger.error("="*60 + "\n")
            return False
        except requests.exceptions.Timeout:
            logger.error("Model server health check timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to model server: {e}")
            return False
    
    def extract_json_from_text(self, text):
        """Send text to model server for processing"""
        try:
            response = requests.post(
                f"{self.server_url}/generate",
                json={'text': text},
                timeout=30  # Inference should be fast (model in VRAM)
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    return result.get('data')
                else:
                    logger.warning(f"Server returned no JSON: {result.get('error')}")
                    return None
            else:
                logger.error(f"Server error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("Model server request timed out (>30 sec)")
            logger.warning("This shouldn't happen with model in VRAM - check server logs")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Lost connection to model server!")
            logger.error("Check if server is still running: curl http://127.0.0.1:5000/health")
            return None
        except Exception as e:
            logger.error(f"Error calling model server: {e}")
            return None
    
    def merge_json_data(self, new_data):
        if not new_data:
            return
        
        for key, value in new_data.items():
            if key not in self.last_json_output:
                self.last_json_output[key] = value
            elif isinstance(value, list) and isinstance(self.last_json_output[key], list):
                for item in value:
                    if item not in self.last_json_output[key]:
                        self.last_json_output[key].append(item)
            elif isinstance(value, dict) and isinstance(self.last_json_output[key], dict):
                self.last_json_output[key].update(value)
            else:
                self.last_json_output[key] = value
    
    def process_transcription_queue(self, transcription_queue):
        self.is_running = True
        logger.info("Gemma processing thread started")
        
        while self.is_running:
            try:
                transcription = transcription_queue.get(timeout=1)
                
                if not transcription:
                    continue
                
                self.accumulated_text.append(transcription)
                
                combined_text = " ".join(self.accumulated_text[-10:])
                
                json_data = self.extract_json_from_text(combined_text)
                
                if json_data:
                    self.merge_json_data(json_data)
                    logger.info(f"Extracted JSON: {json.dumps(self.last_json_output, indent=2)}")
                    self.save_prescription()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in Gemma processing thread: {e}")
        
        logger.info("Gemma processing thread stopped")
    
    def save_prescription(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prescription_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.last_json_output, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Prescription saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving prescription: {e}")
    
    def stop(self):
        self.is_running = False
    
    def get_current_prescription(self):
        return self.last_json_output.copy()


class MedScribeSystem:
    def __init__(self):
        self.audio_capture = AudioCapture()
        self.transcription_engine = TranscriptionEngine()
        self.gemma_processor = GemmaProcessor()
        self.threads = []
        self.is_running = False
        
    def initialize(self):
        print("\n" + "="*60)
        print("Initializing MedScribe System")
        print("="*60 + "\n")
        
        # Log device configuration
        print("Device Configuration:")
        print(f"  Whisper: {Config.WHISPER_DEVICE} ({Config.WHISPER_COMPUTE_TYPE})")
        print(f"  Gemma: Model Server (HTTP) - model stays in VRAM")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  cuDNN: DISABLED (WSL2 compatibility mode)")
        else:
            print("  GPU: Not available")
        print()
        
        try:
            print("Step 1/2: Loading Whisper Model...")
            self.transcription_engine.load_model()
            print("✓ Whisper model loaded\n")
            
            print("Step 2/2: Connecting to Gemma Model Server...")
            if not self.gemma_processor.load_model():
                print("\n" + "="*60)
                print("⚠ Model server not ready")
                print("="*60)
                print("\nPLEASE START MODEL SERVER FIRST:")
                print("  bash start_model_server.sh")
                print("\nOR in background:")
                print("  bash start_model_server.sh background")
                print("\nWait 4-5 minutes for model to load, then run main.py again")
                print("="*60 + "\n")
                return False
            print("✓ Connected to model server\n")
            
            print("="*60)
            print("All systems ready!")
            print("="*60 + "\n")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    def start(self):
        if self.is_running:
            logger.warning("System already running")
            return
        
        try:
            # WSL2: Verify audio before starting
            logger.info("Running pre-flight audio check...")
            if not self.audio_capture.verify_audio_system():
                logger.error("Audio system verification failed. Cannot start system.")
                logger.error("\nWSL2 Audio Setup Instructions:")
                logger.error("1. Install PulseAudio in WSL2: sudo apt install pulseaudio")
                logger.error("2. Start PulseAudio: pulseaudio --start")
                logger.error("3. Check status: pulseaudio --check")
                logger.error("4. In Windows: Enable WSL2 audio in Settings > System > Sound")
                return
            
            self.is_running = True
            
            self.audio_capture.start()
            
            transcription_thread = threading.Thread(
                target=self.transcription_engine.process_audio_queue,
                args=(self.audio_capture.audio_queue,),
                daemon=True
            )
            transcription_thread.start()
            self.threads.append(transcription_thread)
            
            gemma_thread = threading.Thread(
                target=self.gemma_processor.process_transcription_queue,
                args=(self.transcription_engine.transcription_queue,),
                daemon=True
            )
            gemma_thread.start()
            self.threads.append(gemma_thread)
            
            logger.info("MedScribe system started successfully")
            logger.info("Listening for audio input... Press Ctrl+C to stop")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            self.stop()
            raise
    
    def stop(self):
        if not self.is_running:
            return
        
        logger.info("Stopping MedScribe system...")
        self.is_running = False
        
        self.audio_capture.stop()
        self.transcription_engine.stop()
        self.gemma_processor.stop()
        
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2)
        
        final_prescription = self.gemma_processor.get_current_prescription()
        if final_prescription:
            logger.info("Final prescription data:")
            print(json.dumps(final_prescription, indent=2, ensure_ascii=False))
        
        logger.info("MedScribe system stopped")
    
    def run(self):
        try:
            self.start()
            
            while self.is_running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.stop()


def main():
    print("=" * 60)
    print("MedScribe - Voice-Assisted Prescription System")
    print("=" * 60)
    print()
    
    system = MedScribeSystem()
    
    if not system.initialize():
        logger.error("Failed to initialize system. Exiting.")
        return 1
    
    print()
    print("System initialized successfully!")
    print("The system will now:")
    print("  1. Capture audio from your microphone")
    print("  2. Transcribe Hindi/Marathi to English")
    print("  3. Extract prescription details using Gemma")
    print("  4. Save results to JSON files")
    print()
    print("Press Ctrl+C to stop the system")
    print("=" * 60)
    print()
    
    system.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
