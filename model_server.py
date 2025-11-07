"""
Gemma Model Server - Keeps model loaded in VRAM like Ollama
Runs as a separate process, serves inference requests via HTTP
"""

import os
import sys
import json
import logging
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

# CRITICAL: Set environment variables BEFORE torch uses CUDA
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

# Disable cuDNN
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model storage (stays in memory)
MODEL = None
TOKENIZER = None
STOPPING_CRITERIA = None
MODEL_LOADED = False

class StopOnBackticks(StoppingCriteria):
    def __init__(self, tokenizer, stop_sequence="AAA"):
        self.tokenizer = tokenizer
        self.stop_sequence = stop_sequence
        self.stop_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        if len(input_ids[0]) >= len(self.stop_ids):
            if (input_ids[0][-len(self.stop_ids):] == torch.tensor(self.stop_ids, device=input_ids.device)).all():
                return True
        return False

class GemmaServer:
    def __init__(self, model_path):
        self.model_path = model_path
        
    def load_model(self):
        global MODEL, TOKENIZER, STOPPING_CRITERIA, MODEL_LOADED
        
        if MODEL_LOADED:
            logger.info("Model already loaded in VRAM")
            return True
        
        logger.info(f"Loading Gemma model (one-time operation)...")
        logger.info(f"Path: {self.model_path}")
        logger.info(f"Device: GPU (cuda)")
        
        try:
            # 4-bit quantization for 6GB VRAM
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            logger.info("Loading tokenizer...")
            TOKENIZER = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            
            logger.info("Loading model (this takes 4-5 minutes)...")
            MODEL = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory={0: "5GB"},  # 6GB GPU limit
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
            
            MODEL.eval()
            
            # Setup stopping criteria
            STOPPING_CRITERIA = StoppingCriteriaList([StopOnBackticks(TOKENIZER)])
            
            MODEL_LOADED = True
            
            # Log VRAM usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"✓ Model loaded successfully")
                logger.info(f"  VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                logger.info(f"  Model will stay in VRAM until server stops")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

@app.route('/health', methods=['GET'])
def health_check():
    """Check if model server is alive"""
    return jsonify({
        'status': 'running',
        'model_loaded': MODEL_LOADED,
        'cuda_available': torch.cuda.is_available(),
        'vram_allocated_gb': torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Generate prescription JSON from transcription text"""
    global MODEL, TOKENIZER, STOPPING_CRITERIA, MODEL_LOADED
    
    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 503
    
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        system_prompt = """
You are a medical prescription parser. Extract ONLY information explicitly stated.

Rules:
1. Extract medicines with EXACT dosages mentioned
2. If dosage/frequency unclear, mark as "unspecified"
3. Do NOT infer or assume any information
4. If doctor says "continue previous meds", extract NOTHING
5. Output only one valid JSON object and stop
6. At the end of the Output print AAA

Output format:
{
  "medicines": [{"name": str, "dosage": str, "frequency": str, "duration": str}],
  "diseases": [str],
  "tests": [{"name": str, "timing": str}]
}
"""
        
        user_prompt = f"""
{system_prompt}
Extract from this prescription conversation:
{text}

Remember: Only extract explicitly stated information. No assumptions.
"""
        
        inputs = TOKENIZER(user_prompt, return_tensors='pt').to(MODEL.device)
        
        with torch.no_grad():
            outputs = MODEL.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.01,
                pad_token_id=TOKENIZER.pad_token_id,
                eos_token_id=TOKENIZER.eos_token_id,
                stopping_criteria=STOPPING_CRITERIA,
                do_sample=False
            )
        
        result_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_str = result_text[start_idx:end_idx + 1]
            json_data = json.loads(json_str)
            return jsonify({'success': True, 'data': json_data})
        else:
            logger.warning("No JSON found in generated text")
            return jsonify({'success': False, 'error': 'No JSON found'})
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return jsonify({'error': f'JSON decode error: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/unload', methods=['POST'])
def unload_model():
    """Free VRAM by unloading model"""
    global MODEL, TOKENIZER, STOPPING_CRITERIA, MODEL_LOADED
    
    if MODEL_LOADED:
        del MODEL
        del TOKENIZER
        del STOPPING_CRITERIA
        torch.cuda.empty_cache()
        MODEL_LOADED = False
        logger.info("Model unloaded from VRAM")
        return jsonify({'success': True, 'message': 'Model unloaded'})
    
    return jsonify({'success': False, 'message': 'Model not loaded'})

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Gemma Model Server')
    parser.add_argument('--model-path', type=str, required=True, help='Path to Gemma model')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Gemma Model Server (Ollama-style)")
    print("=" * 60)
    
    server = GemmaServer(args.model_path)
    
    print("\nLoading model (this will take 4-5 minutes one time)...")
    if not server.load_model():
        print("\n✗ Failed to load model. Exiting.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Model Server Ready!")
    print("=" * 60)
    print(f"Listening on http://{args.host}:{args.port}")
    print("\nEndpoints:")
    print("  GET  /health      - Check server status")
    print("  POST /generate    - Generate prescription JSON")
    print("  POST /unload      - Unload model from VRAM")
    print("\nModel will stay in VRAM until server stops")
    print("Press Ctrl+C to stop server")
    print("=" * 60 + "\n")
    
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == '__main__':
    main()
