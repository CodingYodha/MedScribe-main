import os
import sys
import json
import time
import queue
import threading
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
    WHISPER_MODEL_PATH = r"E:\Projects\Med_Scribe\Medscribe_testing\models\large-v3"
    GEMMA_MODEL_PATH = r"E:\Projects\Med_Scribe\Medscribe_testing\models\finetuned\gemma-prescription-finetuned-it-merged_final"
    
    SAMPLE_RATE = 16000
    AUDIO_CHUNK_DURATION = 3
    AUDIO_CHUNK_SAMPLES = SAMPLE_RATE * AUDIO_CHUNK_DURATION
    
    MIN_AUDIO_LENGTH = 1.0
    SILENCE_THRESHOLD = 0.01
    MAX_QUEUE_SIZE = 100
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"


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
    
    def start(self):
        if self.is_running:
            logger.warning("Audio capture already running")
            return
        
        try:
            self.is_running = True
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=int(self.sample_rate * 0.1),
                callback=self.audio_callback
            )
            self.stream.start()
            logger.info(f"Audio capture started - Sample rate: {self.sample_rate} Hz")
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
        self.transcription_queue = queue.Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.is_running = False
        
    def load_model(self):
        try:
            logger.info(f"Loading Faster Whisper model from {self.model_path}")
            self.model = WhisperModel(
                self.model_path,
                device=Config.DEVICE,
                compute_type=Config.COMPUTE_TYPE,
                cpu_threads=4,
                num_workers=2
            )
            logger.info("Faster Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_data):
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        try:
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms < Config.SILENCE_THRESHOLD:
                logger.debug("Silence detected, skipping transcription")
                return None
            
            segments, info = self.model.transcribe(
                audio_data,
                task="translate",
                language="hi",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400
                )
            )
            
            transcription = ""
            for segment in segments:
                transcription += segment.text + " "
            
            transcription = transcription.strip()
            
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
                
                transcription = self.transcribe_audio(audio_chunk)
                
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
    def __init__(self, model_path=Config.GEMMA_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.is_running = False
        self.accumulated_text = []
        self.last_json_output = {}
        
    def load_model(self):
        try:
            logger.info(f"Loading Gemma model from {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
                device_map="auto" if Config.DEVICE == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if Config.DEVICE == "cpu":
                self.model = self.model.to(Config.DEVICE)
            
            self.model.eval()
            
            logger.info("Gemma model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Gemma model: {e}")
            raise
    
    def extract_json_from_text(self, text):
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            start_idx = generated_text.find('{')
            end_idx = generated_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = generated_text[start_idx:end_idx + 1]
                json_data = json.loads(json_str)
                return json_data
            else:
                logger.warning("No JSON found in generated text")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
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
        logger.info("Initializing MedScribe system...")
        
        try:
            logger.info("Loading Whisper model (this may take a few moments)...")
            self.transcription_engine.load_model()
            
            logger.info("Loading Gemma model (this may take a few moments)...")
            self.gemma_processor.load_model()
            
            logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            return False
    
    def start(self):
        if self.is_running:
            logger.warning("System already running")
            return
        
        try:
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
