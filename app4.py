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
from audiorecorder import audiorecorder
import time

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

def save_audio_to_wav(audio_data, sample_rate=16000):
    """Convert audio data to WAV format in memory."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    buffer.seek(0)
    return buffer

def detect_silence(audio_chunk, threshold=500, min_silence_duration=1.5):
    """Detect if audio chunk contains significant silence at the end."""
    if len(audio_chunk) == 0:
        return False
    
    # Check last portion of audio for silence
    sample_rate = 16000
    silence_samples = int(min_silence_duration * sample_rate)
    
    if len(audio_chunk) < silence_samples:
        return False
    
    last_portion = audio_chunk[-silence_samples:]
    rms = np.sqrt(np.mean(last_portion**2))
    
    return rms < threshold

def split_audio_intelligently(audio_data, sample_rate=16000, chunk_duration=30, overlap_duration=2):
    """Split audio into chunks with overlap to preserve context at boundaries."""
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    
    chunks = []
    start = 0
    
    while start < len(audio_data):
        end = min(start + chunk_samples, len(audio_data))
        chunk = audio_data[start:end]
        
        # If this is not the last chunk, try to find a silence point for better split
        if end < len(audio_data):
            # Look for silence in the last 3 seconds of the chunk
            search_start = max(0, len(chunk) - 3 * sample_rate)
            silence_found = False
            
            for i in range(search_start, len(chunk) - sample_rate):
                window = chunk[i:i + int(0.5 * sample_rate)]
                rms = np.sqrt(np.mean(window**2))
                
                if rms < 300:  # Silence threshold
                    chunk = chunk[:i + int(0.5 * sample_rate)]
                    silence_found = True
                    break
            
            if not silence_found:
                # No silence found, use overlap
                start = end - overlap_samples
            else:
                start = start + len(chunk) - overlap_samples
        else:
            start = end
        
        chunks.append(chunk)
    
    return chunks

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
    
    # Track unique items
    med_names = set()
    
    for json_obj in json_list:
        if not json_obj:
            continue
        
        # Merge medicines (avoid duplicates by name)
        for med in json_obj.get("medicines", []):
            med_name = med.get("name", "").lower()
            if med_name and med_name not in med_names:
                merged["medicines"].append(med)
                med_names.add(med_name)
        
        # Merge diseases (unique)
        for disease in json_obj.get("diseases", []):
            if disease and disease not in merged["diseases"]:
                merged["diseases"].append(disease)
        
        # Merge symptoms (unique)
        for symptom in json_obj.get("symptoms", []):
            if symptom and symptom not in merged["symptoms"]:
                merged["symptoms"].append(symptom)
        
        # Merge tests (unique by name)
        test_names = {t.get("name", "").lower() for t in merged["tests"]}
        for test in json_obj.get("tests", []):
            test_name = test.get("name", "").lower()
            if test_name and test_name not in test_names:
                merged["tests"].append(test)
                test_names.add(test_name)
        
        # Merge instructions (unique)
        for instruction in json_obj.get("instructions", []):
            if instruction and instruction not in merged["instructions"]:
                merged["instructions"].append(instruction)
    
    return merged

# --- Streaming Processing ---

def process_audio_chunk(audio_chunk, whisper_model, gemma_tokenizer, gemma_model, sample_rate=16000):
    """Process a single audio chunk and return extracted entities."""
    try:
        # Convert to WAV format
        wav_buffer = save_audio_to_wav(audio_chunk, sample_rate)
        
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
        st.error(f"Chunk processing error: {str(e)}")
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
    st.title("Doctor's AI Assistant - Streaming Mode")
    
    # Initialize session state
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'streaming_active' not in st.session_state:
        st.session_state.streaming_active = False
    if 'accumulated_transcript' not in st.session_state:
        st.session_state.accumulated_transcript = []
    if 'accumulated_json' not in st.session_state:
        st.session_state.accumulated_json = []
    if 'audio_buffer' not in st.session_state:
        st.session_state.audio_buffer = np.array([], dtype=np.int16)
    
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
        st.write(f"**Streaming Status:** {'🔴 Active' if st.session_state.streaming_active else 'Inactive'}")

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
    mode = st.radio("", ["Streaming Mode (Live Recording)", "Upload Audio File"], horizontal=True)
    
    if mode == "Streaming Mode (Live Recording)":
        st.info("Click 'Start Recording' to begin live transcription and analysis. Speak naturally, and the system will process audio in intelligent chunks.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Start Recording", type="primary", disabled=st.session_state.streaming_active):
                st.session_state.streaming_active = True
                st.session_state.accumulated_transcript = []
                st.session_state.accumulated_json = []
                st.rerun()
        
        with col2:
            if st.button("Stop Recording", disabled=not st.session_state.streaming_active):
                st.session_state.streaming_active = False
                st.rerun()
        
        if st.session_state.streaming_active:
            st.markdown("### 🔴 Recording Active")
            
            # Audio recorder
            audio = audiorecorder("", "", key="recorder")
            
            if len(audio) > 0:
                # Convert audio to numpy array
                audio_array = np.array(audio.get_array_of_samples(), dtype=np.int16)
                
                # Add to buffer
                st.session_state.audio_buffer = np.concatenate([st.session_state.audio_buffer, audio_array])
                
                # Process if buffer is large enough (e.g., 30 seconds of audio)
                sample_rate = audio.frame_rate
                chunk_size = 30 * sample_rate
                
                if len(st.session_state.audio_buffer) >= chunk_size:
                    with st.spinner("Processing audio chunk..."):
                        # Split intelligently
                        chunks = split_audio_intelligently(st.session_state.audio_buffer, sample_rate)
                        
                        for chunk in chunks[:-1]:  # Process all but last chunk
                            json_output, transcript = process_audio_chunk(
                                chunk, whisper_model, gemma_tokenizer, gemma_model, sample_rate
                            )
                            
                            if transcript:
                                st.session_state.accumulated_transcript.append(transcript)
                            if json_output:
                                st.session_state.accumulated_json.append(json_output)
                        
                        # Keep last chunk in buffer
                        st.session_state.audio_buffer = chunks[-1] if chunks else np.array([], dtype=np.int16)
            
            # Display accumulated results
            if st.session_state.accumulated_transcript:
                with st.expander("Live Transcript", expanded=True):
                    st.write(" ".join(st.session_state.accumulated_transcript))
            
            if st.session_state.accumulated_json:
                merged = merge_json_outputs(st.session_state.accumulated_json)
                with st.expander("Extracted Information (Live)", expanded=True):
                    st.json(merged)
        
        else:
            # Streaming stopped - show final results
            if st.session_state.accumulated_json:
                st.success("✅ Recording completed!")
                
                # Process any remaining buffer
                if len(st.session_state.audio_buffer) > 0:
                    with st.spinner("Processing final audio chunk..."):
                        json_output, transcript = process_audio_chunk(
                            st.session_state.audio_buffer, whisper_model, 
                            gemma_tokenizer, gemma_model, 16000
                        )
                        if transcript:
                            st.session_state.accumulated_transcript.append(transcript)
                        if json_output:
                            st.session_state.accumulated_json.append(json_output)
                    st.session_state.audio_buffer = np.array([], dtype=np.int16)
                
                # Merge all results
                final_json = merge_json_outputs(st.session_state.accumulated_json)
                
                st.divider()
                with st.expander("Full Transcript", expanded=False):
                    st.write(" ".join(st.session_state.accumulated_transcript))
                
                with st.expander("Complete Extracted Data (JSON)", expanded=False):
                    st.json(final_json)
                
                display_prescription_form(final_json)
    
    else:  # Upload mode
        st.info("Upload an audio file for batch processing.")
        
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
        
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