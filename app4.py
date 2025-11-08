import streamlit as st
import json
import torch
import os
import re
import numpy as np
import io
import wave
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import tempfile
import threading
import queue
from collections import deque
import sounddevice as sd
import scipy.io.wavfile as wavfile

# --- Model Loading ---

@st.cache_resource
def load_whisper_model():
    """Loads the Faster-Whisper model on CPU to save VRAM."""
    model_path = "/mnt/e/Projects/Med_Scribe/MedScribe/large-v3"
    
    if not os.path.exists(model_path):
        st.error(f"Whisper model not found at: {model_path}")
        st.stop()
    
    model = WhisperModel(
        model_path,
        device="cpu",
        compute_type="int8",
        num_workers=4
    )
    return model

@st.cache_resource
def load_gemma_model():
    """Loads the fine-tuned Gemma model and tokenizer on GPU with 4-bit quantization."""
    model_dir = "/mnt/e/Projects/Med_Scribe/MedScribe/gemma-prescription-finetuned-it-merged_final"
    
    if not os.path.exists(model_dir):
        st.error(f"Gemma model not found at: {model_dir}")
        st.stop()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True
    )
    
    return tokenizer, model

# --- Audio Processing ---

class AudioRecorder:
    """Real-time audio recorder using sounddevice."""
    
    def __init__(self, sample_rate=16000, chunk_duration=30):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = chunk_duration * sample_rate
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_buffer = []
        self.stream = None
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream."""
        if status:
            print(f"Audio status: {status}")
        
        # Add audio data to buffer
        self.audio_buffer.extend(indata[:, 0].copy())
        
        # Check if we have enough data for a chunk
        if len(self.audio_buffer) >= self.chunk_samples:
            # Extract chunk
            chunk = np.array(self.audio_buffer[:self.chunk_samples], dtype=np.float32)
            self.audio_queue.put(chunk)
            
            # Keep overlap (last 2 seconds) for continuity
            overlap_samples = 2 * self.sample_rate
            self.audio_buffer = self.audio_buffer[self.chunk_samples - overlap_samples:]
    
    def start_recording(self):
        """Start recording audio."""
        self.is_recording = True
        self.audio_buffer = []
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=int(self.sample_rate * 0.5)  # 0.5 second blocks
        )
        self.stream.start()
    
    def stop_recording(self):
        """Stop recording and return any remaining audio."""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        # Process remaining buffer
        if len(self.audio_buffer) > 0:
            remaining = np.array(self.audio_buffer, dtype=np.float32)
            self.audio_queue.put(remaining)
            self.audio_buffer = []
    
    def get_audio_chunk(self):
        """Get next audio chunk from queue (non-blocking)."""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

def save_audio_chunk_to_wav(audio_data, sample_rate=16000):
    """Convert numpy audio array to WAV file in memory."""
    # Ensure audio is in correct format
    audio_data = np.array(audio_data, dtype=np.float32)
    
    # Normalize and convert to int16
    audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
    
    # Create WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    buffer.seek(0)
    return buffer

# --- JSON Extraction ---

def extract_json(text):
    """Extract and parse JSON from model output with robust error handling."""
    text = re.sub(r"```[a-z]*\n?", "", text)
    text = re.sub(r"```\n?", "", text)
    text = re.sub(r"Output structure:.*?Output:\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?(?=\{)", "", text, flags=re.DOTALL)
    
    json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    
    for json_str in json_matches:
        try:
            json_str = json_str.strip()
            parsed = json.loads(json_str)
            
            if isinstance(parsed, dict) and any(k in parsed for k in ['medicines', 'diseases', 'symptoms']):
                return clean_json_structure(parsed)
        except json.JSONDecodeError:
            continue
    
    try:
        match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}(?=[^}]*$)', text)
        if match:
            json_str = match.group(0)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            parsed = json.loads(json_str)
            return clean_json_structure(parsed)
    except:
        pass
    
    return None

def clean_json_structure(data):
    """Clean and validate the JSON structure."""
    cleaned = {
        "medicines": [],
        "diseases": [],
        "symptoms": [],
        "tests": [],
        "instructions": []
    }
    
    if "medicines" in data and isinstance(data["medicines"], list):
        for item in data["medicines"]:
            if isinstance(item, dict) and "name" in item:
                name_lower = item.get("name", "").lower()
                if not any(word in name_lower for word in ["walk", "ultrasound", "control", "follow"]):
                    cleaned["medicines"].append({
                        "name": item.get("name", ""),
                        "dosage": item.get("dosage", "unspecified"),
                        "frequency": item.get("frequency", "unspecified"),
                        "duration": item.get("duration", "unspecified"),
                        "route": item.get("route", "oral"),
                        "timing": item.get("timing", "unspecified")
                    })
    
    if "diseases" in data:
        if isinstance(data["diseases"], list):
            cleaned["diseases"] = [str(d) for d in data["diseases"] if d]
        elif isinstance(data["diseases"], str):
            cleaned["diseases"] = [data["diseases"]]
    
    if "symptoms" in data:
        if isinstance(data["symptoms"], list):
            cleaned["symptoms"] = [str(s) for s in data["symptoms"] if s]
        elif isinstance(data["symptoms"], str):
            cleaned["symptoms"] = [data["symptoms"]]
    
    if "tests" in data and isinstance(data["tests"], list):
        for item in data["tests"]:
            if isinstance(item, dict) and "name" in item:
                cleaned["tests"].append({
                    "name": item.get("name", ""),
                    "timing": item.get("timing", "unspecified")
                })
            elif isinstance(item, str):
                cleaned["tests"].append({"name": item, "timing": "unspecified"})
    
    if "instructions" in data:
        if isinstance(data["instructions"], list):
            cleaned["instructions"] = list(set([str(i) for i in data["instructions"] if i]))
        elif isinstance(data["instructions"], str):
            cleaned["instructions"] = [data["instructions"]]
    
    return cleaned

def merge_json_outputs(json_list):
    """Merge multiple JSON outputs into a single comprehensive output."""
    merged = {
        "medicines": [],
        "diseases": [],
        "symptoms": [],
        "tests": [],
        "instructions": []
    }
    
    med_names = set()
    
    for json_obj in json_list:
        if not json_obj:
            continue
        
        for med in json_obj.get("medicines", []):
            med_name = med.get("name", "").lower()
            if med_name and med_name not in med_names:
                merged["medicines"].append(med)
                med_names.add(med_name)
        
        for disease in json_obj.get("diseases", []):
            if disease and disease not in merged["diseases"]:
                merged["diseases"].append(disease)
        
        for symptom in json_obj.get("symptoms", []):
            if symptom and symptom not in merged["symptoms"]:
                merged["symptoms"].append(symptom)
        
        test_names = {t.get("name", "").lower() for t in merged["tests"]}
        for test in json_obj.get("tests", []):
            test_name = test.get("name", "").lower()
            if test_name and test_name not in test_names:
                merged["tests"].append(test)
                test_names.add(test_name)
        
        for instruction in json_obj.get("instructions", []):
            if instruction and instruction not in merged["instructions"]:
                merged["instructions"].append(instruction)
    
    return merged

# --- Processing Functions ---

def process_audio_chunk(audio_chunk, whisper_model, gemma_tokenizer, gemma_model):
    """Process a single audio chunk and return extracted entities."""
    try:
        # Convert to WAV format
        wav_buffer = save_audio_chunk_to_wav(audio_chunk)
        
        # Transcribe
        segments, _ = whisper_model.transcribe(
            wav_buffer,
            language='en',
            beam_size=4,
            vad_filter=True
        )
        transcript_chunks = [segment.text.strip() for segment in segments if segment.text.strip()]
        transcript = " ".join(transcript_chunks)
        
        if not transcript.strip():
            return None, ""
        
        # Extract entities with Gemma
        system_prompt = """You are a medical prescription parser. Extract ONLY information explicitly stated.

Rules:
1. Extract medicines with EXACT dosages mentioned
2. If dosage/frequency unclear, mark as "unspecified"
3. Do NOT infer or assume any information
4. Only extract actual medicines in the medicines array
5. Walking, exercise, tests go in instructions or tests, NOT medicines
6. Output ONLY a single valid JSON object, nothing else

Output format:
{
  "medicines": [{"name": "MedicineName", "dosage": "500mg", "frequency": "twice daily", "duration": "30 days"}],
  "diseases": ["disease1", "disease2"],
  "symptoms": ["symptom1", "symptom2"],
  "tests": [{"name": "test name", "timing": "when"}],
  "instructions": ["instruction1", "instruction2"]
}"""
        
        user_prompt = f"""{system_prompt}

Extract from this prescription conversation:
{transcript}

Output only valid JSON:"""

        inputs = gemma_tokenizer(user_prompt, return_tensors='pt').to(gemma_model.device)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        outputs = gemma_model.generate(
            **inputs,
            max_new_tokens=512,
            pad_token_id=gemma_tokenizer.pad_token_id,
            eos_token_id=gemma_tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None
        )
        
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_len:]
        result_text = gemma_tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        result_text = result_text.replace("<start_of_turn>user", "").replace("<start_of_turn>model", "")
        result_text = result_text.split("AAA")[0].strip()

        json_output = extract_json(result_text)
        
        return json_output, transcript
        
    except Exception as e:
        print(f"Chunk processing error: {str(e)}")
        return None, ""

# --- Display Functions ---

def display_prescription_form(json_output):
    """Display the prescription form with extracted data."""
    st.subheader("Prescription Form:")
    
    if json_output.get("medicines"):
        st.markdown("**Medicines:**")
        for idx, med in enumerate(json_output["medicines"]):
            with st.expander(f"Medicine {idx + 1}: {med.get('name', 'N/A')}"):
                st.text_input("Name", value=med.get("name", ""), key=f"name_{idx}_{time.time()}")
                st.text_input("Dosage", value=med.get("dosage", "unspecified"), key=f"dosage_{idx}_{time.time()}")
                st.text_input("Frequency", value=med.get("frequency", "unspecified"), key=f"frequency_{idx}_{time.time()}")
                st.text_input("Duration", value=med.get("duration", "unspecified"), key=f"duration_{idx}_{time.time()}")
                st.text_input("Route", value=med.get("route", "oral"), key=f"route_{idx}_{time.time()}")
                st.text_input("Timing", value=med.get("timing", "unspecified"), key=f"timing_{idx}_{time.time()}")

    if json_output.get("diseases"):
        st.markdown("**Diseases:**")
        diseases_text = ", ".join(json_output["diseases"])
        st.text_area("Diagnosed Diseases", value=diseases_text, height=100, key=f"diseases_{time.time()}")
    
    if json_output.get("symptoms"):
        st.markdown("**Symptoms:**")
        symptoms_text = ", ".join(json_output["symptoms"])
        st.text_area("Reported Symptoms", value=symptoms_text, height=100, key=f"symptoms_{time.time()}")
        
    if json_output.get("tests"):
        st.markdown("**Tests Ordered:**")
        for idx, test in enumerate(json_output["tests"]):
            with st.expander(f"Test {idx + 1}: {test.get('name', 'N/A')}"):
                st.text_input("Test Name", value=test.get("name", ""), key=f"test_name_{idx}_{time.time()}")
                st.text_input("Timing", value=test.get("timing", "unspecified"), key=f"test_timing_{idx}_{time.time()}")
    
    if json_output.get("instructions"):
        st.markdown("**Instructions:**")
        instructions_text = "\n".join([f"• {instr}" for instr in json_output["instructions"]])
        st.text_area("Doctor's Instructions", value=instructions_text, height=150, key=f"instructions_{time.time()}")

# --- Main Application ---

def main():
    st.title("Doctor's AI Assistant - Real-Time Streaming")
    
    # Initialize session state
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'recorder' not in st.session_state:
        st.session_state.recorder = None
    if 'accumulated_transcripts' not in st.session_state:
        st.session_state.accumulated_transcripts = []
    if 'accumulated_jsons' not in st.session_state:
        st.session_state.accumulated_jsons = []
    if 'chunks_processed' not in st.session_state:
        st.session_state.chunks_processed = 0
    
    # Sidebar
    with st.sidebar:
        st.subheader("System Info")
        if torch.cuda.is_available():
            st.write(f"GPU: {torch.cuda.get_device_name(0)}")
            st.write(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            st.write("GPU: Not available")
        st.write("Whisper: CPU (int8)")
        st.write("Gemma: GPU (4-bit)")
        st.write("---")
        st.write(f"**Models Status:** {'✅ Loaded' if st.session_state.models_loaded else 'Not loaded'}")
        st.write(f"**Recording:** {'🔴 Active' if st.session_state.recording else 'Inactive'}")
        st.write(f"**Chunks Processed:** {st.session_state.chunks_processed}")

    # Load models
    if not st.session_state.models_loaded:
        try:
            with st.spinner("Loading Whisper model on CPU..."):
                whisper_model = load_whisper_model()
            
            with st.spinner("Loading Gemma model on GPU (4-bit)..."):
                gemma_tokenizer, gemma_model = load_gemma_model()
            
            st.session_state.models_loaded = True
            st.success("✅ Models loaded successfully!")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.stop()
    else:
        whisper_model = load_whisper_model()
        gemma_tokenizer, gemma_model = load_gemma_model()

    # Mode selection
    st.subheader("Select Mode:")
    mode = st.radio("Processing Mode", ["Real-Time Microphone", "Upload Audio File"], horizontal=True)
    
    if mode == "Real-Time Microphone":
        st.info("Click 'Start Recording' to begin speaking. The system will continuously transcribe and analyze as you speak.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎤 Start Recording", type="primary", disabled=st.session_state.recording):
                st.session_state.recording = True
                st.session_state.accumulated_transcripts = []
                st.session_state.accumulated_jsons = []
                st.session_state.chunks_processed = 0
                st.session_state.recorder = AudioRecorder(sample_rate=16000, chunk_duration=30)
                st.session_state.recorder.start_recording()
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Recording", disabled=not st.session_state.recording):
                if st.session_state.recorder:
                    st.session_state.recorder.stop_recording()
                st.session_state.recording = False
                st.rerun()
        
        if st.session_state.recording:
            st.markdown("### 🔴 Recording Active - Speak Now")
            st.warning("Processing happens in background every 30 seconds. Keep speaking naturally.")
            
            # Create placeholders for live updates
            transcript_placeholder = st.empty()
            json_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Process audio chunks in real-time
            while st.session_state.recording:
                if st.session_state.recorder:
                    audio_chunk = st.session_state.recorder.get_audio_chunk()
                    
                    if audio_chunk is not None:
                        status_placeholder.info(f"Processing chunk {st.session_state.chunks_processed + 1}...")
                        
                        # Process chunk
                        json_output, transcript = process_audio_chunk(
                            audio_chunk,
                            whisper_model,
                            gemma_tokenizer,
                            gemma_model
                        )
                        
                        if transcript:
                            st.session_state.accumulated_transcripts.append(transcript)
                        
                        if json_output:
                            st.session_state.accumulated_jsons.append(json_output)
                        
                        st.session_state.chunks_processed += 1
                        
                        # Update displays
                        if st.session_state.accumulated_transcripts:
                            with transcript_placeholder.container():
                                st.markdown("**Live Transcript:**")
                                st.text_area(
                                    "Transcript",
                                    value=" ".join(st.session_state.accumulated_transcripts),
                                    height=200,
                                    key=f"live_trans_{st.session_state.chunks_processed}",
                                    label_visibility="collapsed"
                                )
                        
                        if st.session_state.accumulated_jsons:
                            merged = merge_json_outputs(st.session_state.accumulated_jsons)
                            with json_placeholder.container():
                                st.markdown("**Extracted Information (Live):**")
                                st.json(merged)
                        
                        status_placeholder.success(f"✅ Processed chunk {st.session_state.chunks_processed}")
                
                time.sleep(0.5)  # Check for new chunks every 0.5 seconds
        
        else:
            # Recording stopped - show final results
            if st.session_state.accumulated_jsons:
                # Process any remaining audio
                if st.session_state.recorder:
                    remaining_chunk = st.session_state.recorder.get_audio_chunk()
                    if remaining_chunk is not None:
                        with st.spinner("Processing final chunk..."):
                            json_output, transcript = process_audio_chunk(
                                remaining_chunk,
                                whisper_model,
                                gemma_tokenizer,
                                gemma_model
                            )
                            if transcript:
                                st.session_state.accumulated_transcripts.append(transcript)
                            if json_output:
                                st.session_state.accumulated_jsons.append(json_output)
                
                st.success(f"✅ Recording completed! Processed {st.session_state.chunks_processed} chunks.")
                
                # Merge all results
                final_json = merge_json_outputs(st.session_state.accumulated_jsons)
                
                st.divider()
                
                with st.expander("Full Transcript", expanded=False):
                    st.write(" ".join(st.session_state.accumulated_transcripts))
                
                with st.expander("Complete Extracted Data (JSON)", expanded=False):
                    st.json(final_json)
                
                display_prescription_form(final_json)
    
    else:  # File upload mode
        st.info("Upload an audio file for batch processing.")
        
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            
            if st.button("Transcribe and Analyze", type="primary"):
                st.divider()
                
                with st.spinner("Transcribing audio..."):
                    try:
                        segments, _ = whisper_model.transcribe(
                            uploaded_file,
                            language='en',
                            beam_size=4,
                            vad_filter=True
                        )
                        transcript_chunks = [segment.text.strip() for segment in segments if segment.text.strip()]
                        full_transcript = " ".join(transcript_chunks)
                        
                        st.success("✅ Transcription completed!")
                        with st.expander("View Full Transcription", expanded=False):
                            st.write(full_transcript)
                    except Exception as e:
                        st.error(f"Transcription error: {str(e)}")
                        st.stop()

                with st.spinner("Analyzing prescription..."):
                    try:
                        system_prompt = """You are a medical prescription parser. Extract ONLY information explicitly stated.

Rules:
1. Extract medicines with EXACT dosages mentioned
2. If dosage/frequency unclear, mark as "unspecified"
3. Do NOT infer or assume any information
4. Only extract actual medicines in the medicines array
5. Walking, exercise, tests go in instructions or tests, NOT medicines
6. Output ONLY a single valid JSON object, nothing else

Output format:
{
  "medicines": [{"name": "MedicineName", "dosage": "500mg", "frequency": "twice daily", "duration": "30 days"}],
  "diseases": ["disease1", "disease2"],
  "symptoms": ["symptom1", "symptom2"],
  "tests": [{"name": "test name", "timing": "when"}],
  "instructions": ["instruction1", "instruction2"]
}"""
                        
                        user_prompt = f"""{system_prompt}

Extract from this prescription conversation:
{full_transcript}

Output only valid JSON:"""

                        inputs = gemma_tokenizer(user_prompt, return_tensors='pt').to(gemma_model.device)
                        
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        outputs = gemma_model.generate(
                            **inputs,
                            max_new_tokens=512,
                            pad_token_id=gemma_tokenizer.pad_token_id,
                            eos_token_id=gemma_tokenizer.eos_token_id,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            top_k=None
                        )
                        
                        input_len = inputs.input_ids.shape[1]
                        generated_tokens = outputs[0][input_len:]
                        result_text = gemma_tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        
                        result_text = result_text.replace("<start_of_turn>user", "").replace("<start_of_turn>model", "")
                        result_text = result_text.split("AAA")[0].strip()

                        json_output = extract_json(result_text)
                        
                        if not json_output:
                            st.error("Could not extract valid JSON from model output")
                            st.subheader("Raw model output:")
                            st.text(result_text)
                            st.stop()
                        
                        st.success("✅ Analysis completed!")
                        
                        with st.expander("View Extracted Data (JSON)", expanded=False):
                            st.json(json_output)
                        
                        display_prescription_form(json_output)
                        
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()