#!/bin/bash

# ==============================================================================
# AMD ROCm and DeepSeek-OCR Setup Script
# ==============================================================================
#
# This script installs the necessary dependencies to run DeepSeek-OCR on an
# AMD ROCm-enabled system.
#
# NOTE: This script assumes you have a compatible ROCm-enabled PyTorch
# installed in your virtual environment.
#
# ==============================================================================

echo "### AMD ROCm and DeepSeek-OCR Setup Script ###"
echo "Timestamp: $(date)"
echo ""

# --- Section 1: Project Repositories ---
echo "--- 1. Project Repositories ---"
echo "Installing Git LFS..."
sudo apt-get update
sudo apt-get install -y git-lfs
git-lfs install
echo ""
echo "Cloning DeepSeek-OCR source code..."
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
echo ""
echo "Cloning DeepSeek-OCR model..."
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR DeepSeek-OCR-model
echo ""
echo "----------------------------------------"
echo ""

# --- Section 2: Dependencies ---
echo "--- 2. Dependencies ---"
echo "Installing Python dependencies into the virtual environment..."
pip3 install -r DeepSeek-OCR/requirements.txt
echo ""
echo "----------------------------------------"
echo ""

echo "### Setup Complete ###"
