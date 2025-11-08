import streamlit as st
import json
import torch
import os
import re
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Model Loading ---

@st.cache_resource
def load_whisper_model():
    """Loads the Faster-Whisper model on CPU to save VRAM."""
    # Convert Windows path to WSL path if needed
    model_path = "/mnt/e/Projects/Med_Scribe/MedScribe/large-v3"
    
    if not os.path.exists(model_path):
        st.error(f"Whisper model not found at: {model_path}")
        st.stop()
    
    # Load Whisper on CPU to save VRAM for Gemma
    model = WhisperModel(
        model_path,
        device="cpu",  # Running on CPU to save VRAM
        compute_type="int8",  # Use int8 for CPU efficiency
        num_workers=4  # Multi-threading for CPU
    )
    return model

@st.cache_resource
def load_gemma_model():
    """Loads the fine-tuned Gemma model and tokenizer on GPU with 4-bit quantization."""
    # Convert Windows path to WSL path
    model_dir = "/mnt/e/Projects/Med_Scribe/MedScribe/gemma-prescription-finetuned-it-merged_final"
    
    if not os.path.exists(model_dir):
        st.error(f"Gemma model not found at: {model_dir}")
        st.stop()
    
    # 4-bit quantization to fit in 6GB VRAM
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
        dtype=torch.float16,  # Changed from torch_dtype
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True
    )
    
    return tokenizer, model

# --- Robust JSON Extraction ---

def extract_json(text):
    """Extract and parse JSON from model output with robust error handling."""
    # Remove code fences and language markers
    text = re.sub(r"```[a-z]*\n?", "", text)
    text = re.sub(r"```\n?", "", text)
    
    # Remove common prefixes the model might add
    text = re.sub(r"Output structure:.*?Output:\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?(?=\{)", "", text, flags=re.DOTALL)
    
    # Find all JSON-like structures
    json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    
    for json_str in json_matches:
        try:
            # Clean up common issues
            json_str = json_str.strip()
            
            # Try to parse
            parsed = json.loads(json_str)
            
            # Validate it has the expected structure
            if isinstance(parsed, dict) and any(k in parsed for k in ['medicines', 'diseases', 'symptoms']):
                return clean_json_structure(parsed)
        except json.JSONDecodeError:
            continue
    
    # If no valid JSON found, try more aggressive cleaning
    try:
        # Extract the last occurrence of {...}
        match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}(?=[^}]*$)', text)
        if match:
            json_str = match.group(0)
            # Fix common issues
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
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
    
    # Clean medicines - ensure they're actually medicines
    if "medicines" in data and isinstance(data["medicines"], list):
        for item in data["medicines"]:
            if isinstance(item, dict) and "name" in item:
                # Filter out items that are actually tests or instructions
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
    
    # Clean diseases
    if "diseases" in data:
        if isinstance(data["diseases"], list):
            cleaned["diseases"] = [str(d) for d in data["diseases"] if d]
        elif isinstance(data["diseases"], str):
            cleaned["diseases"] = [data["diseases"]]
    
    # Clean symptoms
    if "symptoms" in data:
        if isinstance(data["symptoms"], list):
            cleaned["symptoms"] = [str(s) for s in data["symptoms"] if s]
        elif isinstance(data["symptoms"], str):
            cleaned["symptoms"] = [data["symptoms"]]
    
    # Clean tests
    if "tests" in data and isinstance(data["tests"], list):
        for item in data["tests"]:
            if isinstance(item, dict) and "name" in item:
                cleaned["tests"].append({
                    "name": item.get("name", ""),
                    "timing": item.get("timing", "unspecified")
                })
            elif isinstance(item, str):
                cleaned["tests"].append({"name": item, "timing": "unspecified"})
    
    # Clean instructions
    if "instructions" in data:
        if isinstance(data["instructions"], list):
            # Deduplicate instructions
            cleaned["instructions"] = list(set([str(i) for i in data["instructions"] if i]))
        elif isinstance(data["instructions"], str):
            cleaned["instructions"] = [data["instructions"]]
    
    return cleaned

# --- Main Application Logic ---

def main():
    """The main function for the Streamlit application."""
    st.title("Doctor's AI Assistant")
    
    # Initialize session state for tracking processed files
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}
    if 'current_file_key' not in st.session_state:
        st.session_state.current_file_key = None
    
    # Display system info
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
        st.write(f"**Models Status:** {'✅ Loaded' if 'models_loaded' in st.session_state else 'Not loaded'}")
        st.write("---")
        st.subheader("Processing History")
        if st.session_state.processed_files:
            for idx, (file_key, file_info) in enumerate(st.session_state.processed_files.items()):
                st.write(f"{idx + 1}. {file_info['name']}")
        else:
            st.write("No files processed yet")

    # Load models ONCE with error handling (cached with @st.cache_resource)
    if 'models_loaded' not in st.session_state:
        try:
            with st.spinner("Loading Whisper model on CPU..."):
                whisper_model = load_whisper_model()
            
            with st.spinner("Loading Gemma model on GPU (4-bit)..."):
                gemma_tokenizer, gemma_model = load_gemma_model()
            
            # Store in session state to track loading status
            st.session_state.models_loaded = True
            st.success("✅ Models loaded successfully! Ready to process multiple audio files.")
            st.info("Models will remain loaded until server stops. You can process multiple files without reloading.")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.stop()
    else:
        # Models already loaded, just retrieve from cache
        whisper_model = load_whisper_model()
        gemma_tokenizer, gemma_model = load_gemma_model()
        st.success("✅ Models ready! Upload a new audio file to process.")

    # Audio file uploader
    uploaded_file = st.file_uploader(
        "Upload an audio file", 
        type=["wav", "mp3", "m4a"],
        help="Upload a new consultation recording. You can process multiple files without reloading models."
    )

    if uploaded_file is not None:
        # Create unique key for this file
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # Check if this is a new file
        is_new_file = file_key != st.session_state.current_file_key
        
        if is_new_file:
            st.session_state.current_file_key = file_key
            st.info(f"New file detected: {uploaded_file.name}")
        
        st.audio(uploaded_file, format="audio/wav")

        # Show if this file was already processed
        if file_key in st.session_state.processed_files:
            st.warning("This file has already been processed. Click 'Transcribe and Analyze' to process again.")
        
        if st.button("Transcribe and Analyze", type="primary"):
            # Store processing timestamp
            st.session_state.processed_files[file_key] = {
                'name': uploaded_file.name,
                'timestamp': st.session_state.get('processing_count', 0) + 1
            }
            st.session_state.processing_count = st.session_state.get('processing_count', 0) + 1
            
            st.divider()
            st.subheader(f"Processing: {uploaded_file.name}")
            
            with st.spinner("Transcribing audio on CPU..."):
                try:
                    # Transcribe audio
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

            with st.spinner("Analyzing prescription on GPU..."):
                try:
                    # System prompt for Gemma
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

                    # Run Gemma model for entity extraction
                    inputs = gemma_tokenizer(user_prompt, return_tensors='pt').to(gemma_model.device)
                    
                    # Clear CUDA cache before generation
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
                    
                    # Decode only the generated tokens
                    input_len = inputs.input_ids.shape[1]
                    generated_tokens = outputs[0][input_len:]
                    result_text = gemma_tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    
                    # Clean up the output
                    result_text = result_text.replace("<start_of_turn>user", "").replace("<start_of_turn>model", "")
                    result_text = result_text.split("AAA")[0].strip()

                    # Extract JSON
                    json_output = extract_json(result_text)
                    
                    if not json_output:
                        st.error("Could not extract valid JSON from model output")
                        st.subheader("Raw model output:")
                        st.text(result_text)
                        st.stop()
                    
                    st.success("✅ Prescription analysis completed!")
                    
                    with st.expander("View Extracted Data (JSON)", expanded=False):
                        st.json(json_output)
                    
                    # --- Simulate filling a form ---
                    st.subheader("Simulated Prescription Form:")
                    
                    # Medicines section
                    if json_output.get("medicines"):
                        st.markdown("**Medicines:**")
                        for idx, med in enumerate(json_output["medicines"]):
                            with st.expander(f"Medicine {idx + 1}: {med.get('name', 'N/A')}"):
                                st.text_input("Name", value=med.get("name", ""), key=f"name_{idx}")
                                st.text_input("Dosage", value=med.get("dosage", "unspecified"), key=f"dosage_{idx}")
                                st.text_input("Frequency", value=med.get("frequency", "unspecified"), key=f"frequency_{idx}")
                                st.text_input("Duration", value=med.get("duration", "unspecified"), key=f"duration_{idx}")
                                st.text_input("Route", value=med.get("route", "oral"), key=f"route_{idx}")
                                st.text_input("Timing", value=med.get("timing", "unspecified"), key=f"timing_{idx}")

                    # Diseases section
                    if json_output.get("diseases"):
                        st.markdown("**Diseases:**")
                        diseases_text = ", ".join(json_output["diseases"])
                        st.text_area("Diagnosed Diseases", value=diseases_text, height=100)
                    
                    # Symptoms section
                    if json_output.get("symptoms"):
                        st.markdown("**Symptoms:**")
                        symptoms_text = ", ".join(json_output["symptoms"])
                        st.text_area("Reported Symptoms", value=symptoms_text, height=100)
                        
                    # Tests section
                    if json_output.get("tests"):
                        st.markdown("**Tests Ordered:**")
                        for idx, test in enumerate(json_output["tests"]):
                            with st.expander(f"Test {idx + 1}: {test.get('name', 'N/A')}"):
                                st.text_input("Test Name", value=test.get("name", ""), key=f"test_name_{idx}")
                                st.text_input("Timing", value=test.get("timing", "unspecified"), key=f"test_timing_{idx}")
                    
                    # Instructions section
                    if json_output.get("instructions"):
                        st.markdown("**Instructions:**")
                        instructions_text = "\n".join([f"• {instr}" for instr in json_output["instructions"]])
                        st.text_area("Doctor's Instructions", value=instructions_text, height=150)

                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    import traceback
                    st.text("Full error traceback:")
                    st.code(traceback.format_exc())
                    st.stop()

            # Generate visit summary after main prescription analysis
            with st.spinner("Generating visit summary..."):
                try:
                    summary_prompt = f"""You are a medical documentation assistant. Create a comprehensive visit summary.

Extract and summarize from this consultation:
{full_transcript}

Output ONLY valid JSON in this exact format:
{{
  "patient_complaint": "brief chief complaint",
  "visit_summary": "2-3 sentence summary of the consultation",
  "diagnosis": "primary diagnosis or condition discussed",
  "treatment_plan": "brief overview of treatment approach",
  "follow_up": "follow-up instructions or next visit details"
}}

Output only the JSON object, nothing else:"""

                    # Run Gemma for summary generation
                    summary_inputs = gemma_tokenizer(summary_prompt, return_tensors='pt').to(gemma_model.device)
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    summary_outputs = gemma_model.generate(
                        **summary_inputs,
                        max_new_tokens=384,
                        pad_token_id=gemma_tokenizer.pad_token_id,
                        eos_token_id=gemma_tokenizer.eos_token_id,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None
                    )
                    
                    # Decode summary
                    summary_input_len = summary_inputs.input_ids.shape[1]
                    summary_generated_tokens = summary_outputs[0][summary_input_len:]
                    summary_text = gemma_tokenizer.decode(summary_generated_tokens, skip_special_tokens=True)
                    
                    # Clean up
                    summary_text = summary_text.replace("<start_of_turn>user", "").replace("<start_of_turn>model", "")
                    summary_text = summary_text.split("AAA")[0].strip()
                    
                    # Extract summary JSON
                    summary_json = extract_json(summary_text)
                    
                    if summary_json:
                        st.success("✅ Visit summary generated!")
                        
                        with st.expander("View Summary Data (JSON)", expanded=False):
                            st.json(summary_json)
                        
                        # Simulate visit summary form
                        st.subheader("Visit Summary Form:")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.text_input(
                                "Chief Complaint", 
                                value=summary_json.get("patient_complaint", ""), 
                                key="chief_complaint"
                            )
                            st.text_area(
                                "Diagnosis", 
                                value=summary_json.get("diagnosis", ""), 
                                height=100,
                                key="diagnosis"
                            )
                        
                        with col2:
                            st.text_input(
                                "Follow-up", 
                                value=summary_json.get("follow_up", ""), 
                                key="follow_up"
                            )
                            st.text_area(
                                "Treatment Plan", 
                                value=summary_json.get("treatment_plan", ""), 
                                height=100,
                                key="treatment_plan"
                            )
                        
                        st.text_area(
                            "Visit Summary", 
                            value=summary_json.get("visit_summary", ""), 
                            height=120,
                            key="visit_summary_text"
                        )
                    else:
                        st.warning("Could not generate visit summary")
                        st.text("Raw summary output:")
                        st.text(summary_text)
                        
                except Exception as e:
                    st.error(f"Summary generation error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # Final success message
            st.divider()
            st.success(f"Processing complete for: {uploaded_file.name}")
            st.info("You can now upload another audio file to process. Models remain loaded!")

if __name__ == "__main__":
    main()