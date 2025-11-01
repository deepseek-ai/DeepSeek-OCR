# DeepSeek-OCR on AMD Strix Halo (gfx1151) - Notes

## Phase 1: Environment Research and ROCm 7.1.0 Installation

**Date:** 2025-10-31

### 1.1 System Analysis

*   **Operating System:** Ubuntu 24.04.3 LTS (Noble Numbat)
*   **Kernel:** Linux 6.8.0
*   **Architecture:** x86_64
*   **Target GPU:** AMD Strix Halo (gfx1151)
*   **Target ROCm Version:** 7.1.0

### 1.2 ROCm Installation Research

Based on the official ROCm documentation, the installation process for Ubuntu involves adding the AMD ROCm package repository and then installing the required packages.

**Next Steps:**

1.  Add the AMD GPU repository.
2.  Install the ROCm toolkit.
3.  Verify the installation and check if the `gfx1151` is detected.
