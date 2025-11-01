#!/bin/bash

# ==============================================================================
# AMD ROCm and System Diagnostic Toolkit
# ==============================================================================
#
# This script gathers essential information about your system's hardware,
# operating system, and existing drivers. It does NOT install or modify
# anything on your system.
#
# Instructions:
# 1. Run this script on your local machine with the Strix Halo GPU.
#    chmod +x diagnose.sh
#    ./diagnose.sh > diagnostics.log 2>&1
# 2. Provide the generated 'diagnostics.log' file back to me.
#
# ==============================================================================

echo "### AMD ROCm and System Diagnostic Toolkit ###"
echo "Timestamp: $(date)"
echo ""

# --- Section 1: Operating System and Kernel ---
echo "--- 1. Operating System and Kernel ---"
if [ -f /etc/os-release ]; then
    cat /etc/os-release
else
    echo "Could not determine OS version."
fi
echo ""
echo "Kernel Version:"
uname -a
echo ""
echo "----------------------------------------"
echo ""

# --- Section 2: GPU Hardware Information ---
echo "--- 2. GPU Hardware Information ---"
echo "Listing PCI devices (looking for VGA/Display controllers)..."
if command -v lspci &> /dev/null; then
    lspci -nn | grep -E 'VGA|Display'
    if [ $? -ne 0 ]; then
        echo "No VGA or Display controller found with lspci. This is unexpected."
        echo "Full lspci output:"
        lspci -nn
    fi
else
    echo "Warning: lspci not found. Please install pciutils."
fi
echo ""
echo "----------------------------------------"
echo ""

# --- Section 3: ROCm and AMD Driver Status ---
echo "--- 3. ROCm and AMD Driver Status ---"
echo "Checking for ROCm info tools..."
if command -v rocminfo &> /dev/null; then
    echo "rocminfo found. Running..."
    rocminfo
else
    echo "rocminfo not found. ROCm is likely not installed or not in PATH."
fi
echo ""

echo "Checking for rocm-smi..."
if command -v rocm-smi &> /dev/null; then
    echo "rocm-smi found. Running..."
    rocm-smi
else
    echo "rocm-smi not found. ROCm is likely not installed or not in PATH."
fi
echo ""

echo "Checking for /dev/kfd and /dev/dri..."
ls -l /dev/kfd /dev/dri
echo ""

echo "Checking loaded kernel modules for amdgpu..."
lsmod | grep amdgpu
echo ""
echo "----------------------------------------"
echo ""


# --- Section 4: Python and PyTorch Environment ---
echo "--- 4. Python and PyTorch Environment ---"
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    python3 --version
else
    echo "python3 not found."
fi
echo ""

echo "Checking pip version..."
if command -v pip3 &> /dev/null; then
    pip3 --version
else
    echo "pip3 not found."
fi
echo ""

echo "Checking for existing PyTorch installation..."
python3 -c "
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'PyTorch installation path: {torch.__file__}')
    is_rocm = torch.version.hip is not None
    print(f'Built with ROCm support: {is_rocm}')
    if is_rocm:
        print(f'ROCm version used by PyTorch: {torch.version.hip}')
    is_cuda = torch.cuda.is_available()
    print(f'torch.cuda.is_available(): {is_cuda}')
    if is_cuda:
        print(f'Number of GPUs detected by PyTorch: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'  - GPU {i}: {torch.cuda.get_device_name(i)}')
except ImportError:
    print('PyTorch is not installed.')
except Exception as e:
    print(f'An error occurred while checking PyTorch: {e}')
"
echo ""
echo "----------------------------------------"
echo ""

echo "### Diagnostic Complete ###"
echo "Please provide the complete output of this script."
