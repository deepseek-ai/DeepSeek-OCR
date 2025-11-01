#!/bin/bash

# ==============================================================================
# DeepSeek-OCR ROCm Verification Script
# ==============================================================================
#
# This script downloads a test image and runs the ROCm-compatible
# DeepSeek-OCR script using the python interpreter from the activated
# virtual environment.
#
# ==============================================================================

echo "### DeepSeek-OCR ROCm Verification Script ###"
echo "Timestamp: $(date)"
echo ""

# --- Section 1: Download Test Image ---
echo "--- 1. Download Test Image ---"
wget https://static.simonwillison.net/static/2025/ft.jpeg -O test_image.jpeg
echo ""
echo "----------------------------------------"
echo ""

# --- Section 2: Run OCR ---
echo "--- 2. Run OCR ---"
python3 run_ocr_amd.py test_image.jpeg
echo ""
echo "----------------------------------------"
echo ""

echo "### Verification Complete ###"
