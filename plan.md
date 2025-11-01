# Plan: Porting DeepSeek-OCR to AMD Strix Halo (gfx1151)

This document outlines the plan to get the DeepSeek-OCR model running on an AMD Strix Halo (gfx1151) GPU using ROCm 7.1.0.

### 1. Environment Research and ROCm 7.1.0 Installation
*   **Action:** Inspect the system (Ubuntu 24.04.3 LTS, Kernel 6.8.0) to confirm prerequisites for ROCm 7.1.0.
*   **Action:** Add the official AMD ROCm 7.1.0 `apt` repository.
*   **Action:** Install the ROCm toolkit and drivers using `apt-get install rocm-dev`.
*   **Verification:** Use `rocminfo` and `rocm-smi` to confirm the `gfx1151` GPU is detected and the driver is functional.
*   **Documentation:** Log all commands, outputs, and findings in `notes.md`.

### 2. Install ROCm-Enabled PyTorch
*   **Action:** Identify and install the correct PyTorch wheel for ROCm 7.1.0 and the system's Python version.
*   **Verification:** Execute a Python script to confirm `torch.cuda.is_available()` returns `True` (for ROCm) and that the GPU is recognized by PyTorch.

### 3. Set Up Project Repositories
*   **Action:** Install `git-lfs` to handle large file downloads.
*   **Action:** `git clone` the DeepSeek-OCR source code repository.
*   **Action:** `git clone` the 6.3 GB model files from Hugging Face into a `DeepSeek-OCR-model` directory.

### 4. Adapt Dependencies and Create Scripts
*   **Action:** Create a `setup.sh` script to automate the installation of the ROCm-enabled PyTorch and all Python dependencies from `requirements.txt`.
*   **Action:** Modify the setup to handle CUDA-specific dependencies like `flash-attn` by excluding them and ensuring the code can fall back to a standard attention mechanism.
*   **Action:** Create a `run_ocr.py` script, adapted from the original, to be device-agnostic (run on `hip` instead of `cuda`).
*   **Action:** Create a `run_ocr.sh` wrapper for easy execution.

### 5. Test OCR Inference
*   **Action:** Execute `setup.sh` to prepare the full environment.
*   **Action:** Execute `run_ocr.sh` to run the OCR task on the test image.
*   **Action:** Debug any CUDA-specific code that causes errors. If a hard blocker is found (e.g., an irreplaceable CUDA-only function), I will implement a detailed report on the incompatibility and provide it to you, while also attempting to find a viable alternative.
*   **Verification:** Confirm that the OCR text is successfully extracted and saved to a file.

### 6. Create Final Documentation
*   **Action:** Consolidate all findings from `notes.md` into a final, comprehensive `README.md`.
*   **Content:** The `README.md` will include:
    *   A summary of the project goal (porting to AMD).
    *   The challenges encountered and the solutions developed.
    *   Step-by-step instructions on how to run the provided `setup.sh` and `run_ocr.sh` scripts.
    *   A clear explanation of the final working state.

### 7. Complete pre commit steps
*   **Action:** Complete pre commit steps to make sure proper testing, verifications, reviews and reflections are done.

### 8. Submit the changes
*   **Action:** Once all tests pass and the documentation is complete, I will submit all scripts and markdown files.
