# DeepSeek-OCR Setup Guide for PDF Parsing

## Prerequisites

- CUDA 11.8+ (GPU required)
- Python 3.12.9
- At least 40GB GPU memory (A100 recommended) or adjust MAX_CONCURRENCY for smaller GPUs

## Installation Steps

### 1. Create Conda Environment

```bash
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Download vLLM wheel from: https://github.com/vllm-project/vllm/releases/tag/v0.8.5
# Then install it:
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# Install other requirements
pip install -r requirements.txt

# Install flash attention
pip install flash-attn==2.7.3 --no-build-isolation
```

### 3. Configure for PDF Parsing

Edit `/home/user/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`:

```python
# Set your input PDF path
INPUT_PATH = '/path/to/your/document.pdf'

# Set output directory
OUTPUT_PATH = '/path/to/output/directory'

# Model will be auto-downloaded from HuggingFace
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'
```

### 4. Run PDF Parsing

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_pdf.py
```

## Output Files

The script generates three files:

1. **`*_det.mmd`**: Raw output with detection bounding boxes
2. **`*.mmd`**: Clean markdown with images extracted
3. **`*_layouts.pdf`**: PDF with visual layout annotations
4. **`images/`**: Extracted images from the document

## Configuration Options

### Resolution Modes (in config.py)

```python
# Tiny: Fast, lower quality
BASE_SIZE = 512
IMAGE_SIZE = 512
CROP_MODE = False

# Small: Balanced
BASE_SIZE = 640
IMAGE_SIZE = 640
CROP_MODE = False

# Base: Good quality
BASE_SIZE = 1024
IMAGE_SIZE = 1024
CROP_MODE = False

# Large: Best quality
BASE_SIZE = 1280
IMAGE_SIZE = 1280
CROP_MODE = False

# Gundam: Dynamic resolution (Recommended)
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
```

### Performance Tuning

```python
# Reduce for limited GPU memory
MAX_CONCURRENCY = 100  # Lower this for smaller GPUs (e.g., 10-20)
MAX_CROPS = 6          # Max 9, use 6 for smaller GPUs
NUM_WORKERS = 64       # CPU workers for image preprocessing
```

### Prompts

```python
# For documents (recommended)
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# For general images
PROMPT = '<image>\n<|grounding|>OCR this image.'

# Without layout preservation
PROMPT = '<image>\nFree OCR.'

# For figures
PROMPT = '<image>\nParse the figure.'
```

## Quick Start Example

See `example_pdf_parse.py` for a simple usage example.

## Troubleshooting

### Out of Memory Error
- Reduce `MAX_CONCURRENCY` in config.py (try 10-20)
- Reduce `MAX_CROPS` to 4 or lower
- Use smaller resolution mode (Small or Tiny)

### Model Download Issues
- The model (~10GB) auto-downloads from HuggingFace on first run
- Ensure you have stable internet connection
- Can manually download: `huggingface-cli download deepseek-ai/DeepSeek-OCR`

### CUDA Version Mismatch
- Ensure CUDA 11.8 is installed: `nvcc --version`
- Set environment variable if needed: `export CUDA_HOME=/usr/local/cuda-11.8`
