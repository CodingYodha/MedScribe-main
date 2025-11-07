#!/bin/bash

# Gemma Model Server Startup Script
# Keeps model loaded in VRAM for instant inference

if [ "$1" == "background" ]; then
    BACKGROUND=true
else
    BACKGROUND=false
fi

# Detect platform and set model path
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    MODEL_PATH="/mnt/e/Projects/Med_Scribe/Medscribe_testing/models/finetuned/gemma-prescription-finetuned-it-merged_final"
else
    MODEL_PATH="E:/Projects/Med_Scribe/Medscribe_testing/models/finetuned/gemma-prescription-finetuned-it-merged_final"
fi

echo "============================================================"
echo "Gemma Model Server - Ollama-style Persistence"
echo "============================================================"
echo ""
echo "Model Path: $MODEL_PATH"
echo "Port: 5000"
echo ""

if [ "$BACKGROUND" = true ]; then
    echo "Starting model server in background..."
    nohup python model_server.py \
        --model-path "$MODEL_PATH" \
        --port 5000 \
        --host 127.0.0.1 > model_server.log 2>&1 &
    
    PID=$!
    echo $PID > model_server.pid
    
    echo "✓ Model server started in background"
    echo "  PID: $PID (saved to model_server.pid)"
    echo "  Logs: tail -f model_server.log"
    echo ""
    echo "Wait 4-5 minutes for model to load, then check:"
    echo "  curl http://127.0.0.1:5000/health"
    echo ""
    echo "To stop server:"
    echo "  kill \$(cat model_server.pid)"
else
    echo "Starting model server (foreground)..."
    echo "This will load the model ONCE and keep it in VRAM"
    echo "Press Ctrl+C to stop server"
    echo ""
    
    python model_server.py \
        --model-path "$MODEL_PATH" \
        --port 5000 \
        --host 127.0.0.1
fi
