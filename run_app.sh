#!/bin/bash

# DeepSeek-OCR Streamlit App Launcher
# This script sets up the environment and launches the Streamlit application

set -e

echo "🚀 Starting DeepSeek-OCR Studio..."
echo ""

# Check if we're in the correct directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found in current directory"
    echo "Please run this script from the deepseek-ocr directory"
    exit 1
fi

# Set CUDA environment if needed
if [ -d "/usr/local/cuda-11.8" ]; then
    export TRITON_PTXAS_PATH="/usr/local/cuda-11.8/bin/ptxas"
    echo "✅ CUDA 11.8 environment configured"
fi

# Set vLLM environment
export VLLM_USE_V1=0

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "⚠️  Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "📄 DeepSeek-OCR Studio"
echo "======================================"
echo "Opening in your browser..."
echo ""
echo "Press Ctrl+C to stop the server"
echo "======================================"
echo ""

# Launch Streamlit with optimal settings
streamlit run app.py \
    --server.port 8501 \
    --server.address localhost \
    --server.maxUploadSize 200 \
    --browser.gatherUsageStats false \
    --theme.base light
