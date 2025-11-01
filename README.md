# DeepSeek-OCR on AMD Strix Halo (gfx1151)

This project provides a set of scripts and documentation to run the DeepSeek-OCR model on an AMD Strix Halo (gfx1151) GPU with ROCm.

## Project Overview

The goal of this project is to port the DeepSeek-OCR model, which is designed for NVIDIA CUDA, to run on AMD's ROCm platform. This involves:
1.  **System Diagnosis:** A script to gather information about the target system.
2.  **Environment Setup:** A script to install the correct ROCm-enabled PyTorch and other dependencies.
3.  **Code Adaptation:** A patched Python script that is compatible with ROCm.
4.  **Verification:** A script to run the OCR on a test image.

## Getting Started

### 1. System Diagnosis

Before you begin, run the `diagnose.sh` script to gather information about your system. This will help in troubleshooting any issues that may arise.

```bash
chmod +x diagnose.sh
./diagnose.sh > diagnostics.log 2>&1
```

### 2. Environment Setup

The `setup-rocm.sh` script will install all the necessary dependencies, including a ROCm-enabled version of PyTorch, and clone the DeepSeek-OCR code and model repositories.

```bash
chmod +x setup-rocm.sh
./setup-rocm.sh
```

### 3. Run the OCR

The `run_ocr.sh` script will download a test image and run the ROCm-compatible OCR script.

```bash
chmod +x run_ocr.sh
./run_ocr.sh
```

## Technical Details

### ROCm-Compatible Code

The original `run_dpsk_ocr.py` script contained CUDA-specific code. The `run_ocr_amd.py` script has been modified to be device-agnostic:

*   **Device Handling:** The `.cuda()` call has been replaced with a more flexible `.to(device)` call, where `device` is determined at runtime.
*   **Flash Attention:** The `flash-attn` library, which is CUDA-specific, has been removed. The model will fall back to a standard attention mechanism.

### Scripts

*   **`diagnose.sh`:** Gathers system information.
*   **`setup-rocm.sh`:** Installs dependencies and clones repositories.
*   **`run_ocr.sh`:** Runs the OCR on a test image.
*   **`run_ocr_amd.py`:** The ROCm-compatible OCR script.
*   **`plan.md`:** The plan followed to create this project.
*   **`notes.md`:** Detailed notes and findings from the development process.
