# Quick Start Guide for PDF Parsing

Get started with DeepSeek-OCR for PDF parsing in 3 easy steps!

## Step 1: Install Dependencies

```bash
# Create and activate conda environment
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr

# Install PyTorch with CUDA
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Download vLLM wheel from: https://github.com/vllm-project/vllm/releases/tag/v0.8.5
# Then install:
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# Install other requirements
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

## Step 2: Configure Your Paths

### Option A: Use the Example Script (Easiest)

1. Edit `example_pdf_parse.py`:
   ```python
   INPUT_PDF = '/path/to/your/document.pdf'
   OUTPUT_DIR = './output'
   ```

2. Run it:
   ```bash
   python example_pdf_parse.py
   ```

### Option B: Configure Manually

1. Copy the example config:
   ```bash
   cp example_config.py DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py
   ```

2. Edit `DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py`:
   ```python
   INPUT_PATH = '/path/to/your/document.pdf'
   OUTPUT_PATH = '/path/to/output/directory'
   ```

3. Run the parser:
   ```bash
   cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
   python run_dpsk_ocr_pdf.py
   ```

## Step 3: View Results

After processing, you'll find in your output directory:

- **`*.mmd`**: Markdown version of your PDF
- **`*_layouts.pdf`**: PDF with visual annotations showing detected elements
- **`images/`**: Extracted images from the document
- **`*_det.mmd`**: Raw output with bounding box coordinates

## What This Does

DeepSeek-OCR will:
1. Convert each PDF page to high-quality images
2. Use AI vision model to extract text, tables, and layout
3. Generate markdown with preserved structure
4. Extract images and figures
5. Create annotated PDF showing detected regions

## Performance Notes

- **First run**: Model (~10GB) downloads from HuggingFace
- **Speed**: ~2500 tokens/s on A100-40G GPU
- **Memory**: Requires at least 40GB GPU memory (or reduce MAX_CONCURRENCY)

## Troubleshooting

### Out of Memory?
Edit config.py:
```python
MAX_CONCURRENCY = 10  # Reduce from 100
MAX_CROPS = 4         # Reduce from 6
```

### GPU Issues?
Check CUDA:
```bash
nvidia-smi
nvcc --version  # Should be 11.8+
```

### Need More Help?

See `SETUP_GUIDE.md` for detailed instructions and troubleshooting.

## Example Output

Input: `research_paper.pdf`

Output:
```
output/
├── research_paper.mmd              # Clean markdown
├── research_paper_det.mmd          # With bounding boxes
├── research_paper_layouts.pdf      # Annotated PDF
└── images/
    ├── 0_0.jpg                     # Extracted figures
    ├── 0_1.jpg
    └── ...
```

The `.mmd` markdown file can be viewed in any markdown viewer or converted to other formats.

## Advanced Usage

### Different Document Types

```python
# For academic papers (default)
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# For invoices/forms
PROMPT = '<image>\n<|grounding|>OCR this image.'

# For text-only extraction
PROMPT = '<image>\nFree OCR.'

# For charts/figures
PROMPT = '<image>\nParse the figure.'
```

### Quality vs Speed

```python
# Fastest (lower quality)
BASE_SIZE = 512
IMAGE_SIZE = 512
CROP_MODE = False

# Best quality (slower)
BASE_SIZE = 1280
IMAGE_SIZE = 1280
CROP_MODE = False

# Balanced (recommended)
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
```

---

**Ready to start?** Run `python example_pdf_parse.py` after configuring your paths!
