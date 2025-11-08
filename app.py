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

    # Load models with error handling
    try:
        with st.spinner("Loading Whisper model on CPU..."):
            whisper_model = load_whisper_model()
        
        with st.spinner("Loading Gemma model on GPU (4-bit)..."):
            gemma_tokenizer, gemma_model = load_gemma_model()
        
        st.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

    # Audio file uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("Transcribe and Analyze"):
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
                    st.subheader("Transcription:")
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
                    
                    st.subheader("Prescription Analysis:")
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

if __name__ == "__main__":
    main()