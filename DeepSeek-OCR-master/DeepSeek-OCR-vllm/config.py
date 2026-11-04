"""DeepSeek-OCR vLLM pipeline configuration.

Platform support: upstream vLLM runs on Linux with CUDA/ROCm GPUs only, so
this whole directory (vLLM engine + CUDA kernels) cannot run on macOS. The
check below (issue #319) fails fast with guidance on macOS and is a no-op on
Linux. macOS users should use ../DeepSeek-OCR-hf/run_dpsk_ocr.py instead.

Inference MODE: see MODE_PRESETS below (issue #300).
"""

import platform as _platform
import sys as _sys

# ---------------------------------------------------------------------------
# Platform support check (GitHub issue #319: "VLLM 能否支持 Mac OS系统")
# ---------------------------------------------------------------------------
# Upstream vLLM only supports Linux (x86_64 / aarch64) with CUDA/ROCm GPUs and
# publishes Linux-only wheels, so `pip install vllm` on macOS fails with
# "xxx is not a supported wheel on this platform" and the engine can never
# start on macOS (Intel or Apple Silicon). This check raises a clear error
# instead of a cryptic wheel/import failure. It is a no-op on Linux: the
# Linux/CUDA path below is completely unchanged.

# Upstream vLLM platform/installation requirements.
VLLM_DOCS_INSTALL_URL = 'https://docs.vllm.ai/en/latest/getting_started/installation/'
# Transformers-based inference entry in this repo that works on macOS
# (CPU/MPS) without vLLM.
HF_RUNNER_REL_PATH = '../DeepSeek-OCR-hf/run_dpsk_ocr.py'


def _macos_unsupported_message(machine):
    return (
        '[DeepSeek-OCR-vllm] Unsupported platform: macOS (%s).\n'
        'Upstream vLLM only supports Linux (x86_64/aarch64) with CUDA/ROCm '
        'GPUs and publishes Linux-only wheels, so vLLM can neither be '
        'installed nor run on macOS (Intel or Apple Silicon) '
        '(GitHub issue #319).\n'
        'Options:\n'
        '  1. Run this vLLM pipeline on a Linux machine with an NVIDIA GPU, or\n'
        '  2. On macOS, use the transformers-based pipeline in this repo instead:\n'
        '     DeepSeek-OCR-hf/run_dpsk_ocr.py (no vLLM required; CPU/MPS supported).\n'
        'Upstream platform requirements: %s'
        % (machine, VLLM_DOCS_INSTALL_URL)
    )


def check_vllm_platform_support(system=None, machine=None):
    """Fail fast on platforms upstream vLLM cannot run on.

    No-op on Linux (the CUDA path is unchanged). Raises RuntimeError on
    macOS with guidance (issue #319). `system`/`machine` overrides exist so
    the Linux no-op branch can be verified on non-Linux hosts.
    """
    system = (system or _platform.system()).lower()
    machine = (machine or _platform.machine()).lower()
    if _sys.platform == 'darwin' or system == 'darwin':
        raise RuntimeError(_macos_unsupported_message(machine))
    return {'system': system, 'machine': machine}


check_vllm_platform_support()


# ---------------------------------------------------------------------------
# Inference MODE presets (GitHub issue #300:
# "How to set MODE (base, large, gundam...) in upstream vllm?")
# ---------------------------------------------------------------------------
# IMPORTANT: BASE_SIZE / IMAGE_SIZE / CROP_MODE are *not* upstream vLLM
# parameters. Upstream vLLM (AsyncEngineArgs / LLM(...) kwargs, SamplingParams)
# has no "MODE" knob; these three values are consumed by this repo's own
# preprocessing/model code:
#   - process/image_process.py: DeepseekOCRProcessor reads IMAGE_SIZE,
#     BASE_SIZE, CROP_MODE (+ MIN_CROPS/MAX_CROPS) for tiling/resize/padding;
#   - deepseek_ocr.py: reads them to estimate dummy-image shapes and the
#     vision token count (get_image_size_with_most_features);
#   - run_dpsk_ocr_image.py / run_dpsk_ocr_pdf.py / run_dpsk_ocr_eval_batch.py:
#     pass cropping=CROP_MODE into tokenize_with_images(...).
# So "setting MODE for upstream vLLM" just means picking one preset below in
# this file. The engine args that DO map to upstream vLLM
# (model/block_size/max_model_len/gpu_memory_utilization/tensor_parallel_size,
# SamplingParams temperature/max_tokens, ...) live in the run_*.py scripts.
#
# Preset table (same values as the transformers entry
# DeepSeek-OCR-hf/run_dpsk_ocr.py `model.infer(..., base_size=...,
# image_size=..., crop_mode=...)`):
MODE_PRESETS = {
    # Tiny: smallest footprint, weakest detail.
    'tiny': {'base_size': 512, 'image_size': 512, 'crop_mode': False},
    # Small: light and fast.
    'small': {'base_size': 640, 'image_size': 640, 'crop_mode': False},
    # Base: balanced single-scale inference, no tiling.
    'base': {'base_size': 1024, 'image_size': 1024, 'crop_mode': False},
    # Large: highest single-scale resolution, needs the most GPU memory.
    'large': {'base_size': 1280, 'image_size': 1280, 'crop_mode': False},
    # Gundam (default): tiled/crop inference for dense documents; base view at
    # 1024 plus 640px crops (MIN_CROPS..MAX_CROPS tiles).
    'gundam': {'base_size': 1024, 'image_size': 640, 'crop_mode': True},
}

# TODO: change modes -> set one of 'tiny' / 'small' / 'base' / 'large' / 'gundam'.
MODE = 'gundam'

_preset = MODE_PRESETS[MODE]
BASE_SIZE = _preset['base_size']
IMAGE_SIZE = _preset['image_size']
CROP_MODE = _preset['crop_mode']
MIN_CROPS= 2
MAX_CROPS= 6 # max:9; If your GPU memory is small, it is recommended to set it to 6.
MAX_CONCURRENCY = 100 # If you have limited GPU memory, lower the concurrency count.
NUM_WORKERS = 64 # image pre-process (resize/padding) workers 
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR' # change to your model path

# TODO: change INPUT_PATH
# .pdf: run_dpsk_ocr_pdf.py; 
# .jpg, .png, .jpeg: run_dpsk_ocr_image.py; 
# Omnidocbench images path: run_dpsk_ocr_eval_batch.py

INPUT_PATH = '' 
OUTPUT_PATH = ''

PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
# PROMPT = '<image>\nFree OCR.'
# TODO commonly used prompts
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'
# .......


from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
