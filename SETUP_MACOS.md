# DeepSeek-OCR Setup Guide for macOS (Apple Silicon)

## Important Notes for Apple Silicon

**Limitations:**
- ❌ **vLLM is NOT supported** on macOS (requires CUDA/NVIDIA GPU)
- ❌ **Flash Attention is NOT supported** on macOS
- ✅ **Transformers approach WORKS** with MPS (Metal Performance Shaders)
- ⚠️ **Performance**: Slower than GPU-accelerated Linux (~10-20x slower)
- ✅ **Your 24GB RAM**: Sufficient for the model (~10GB) with room for processing

**Recommended Approach:** Use the Transformers-based implementation, not vLLM.

## Setup for Apple Silicon MacBook

### 1. Create Virtual Environment (No Conda Needed)

```bash
# In VSCode terminal, navigate to the repository
cd /path/to/DeepSeek-OCR

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install PyTorch for Apple Silicon

```bash
# Install PyTorch with MPS (Metal) support
pip install torch torchvision torchaudio
```

### 3. Install Dependencies (macOS Compatible)

```bash
# Install requirements WITHOUT vLLM and flash-attn
pip install transformers tokenizers PyMuPDF img2pdf einops easydict addict Pillow numpy
```

**DO NOT install:**
- ❌ vllm (CUDA only)
- ❌ flash-attn (CUDA only)

### 4. VSCode Setup

1. Open the repository in VSCode
2. Press `Cmd+Shift+P` and select "Python: Select Interpreter"
3. Choose the interpreter from your `venv` folder
4. Install Python extension if not already installed

## Using DeepSeek-OCR on macOS

### Option 1: Simple Image/PDF Parsing (Recommended)

Use the Transformers-based approach which works on Apple Silicon:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```

But first, edit `run_dpsk_ocr.py` to configure your paths (see below).

### Option 2: Use the macOS Example Script

I'll create a simplified script that works on Apple Silicon without vLLM.

## Configuration

Edit `DeepSeek-OCR-master/DeepSeek-OCR-hf/run_dpsk_ocr.py`:

```python
import os
from transformers import AutoModel, AutoTokenizer
import torch

# Use MPS (Metal) for Apple Silicon GPU acceleration
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Load model WITHOUT flash_attention_2 (not supported on macOS)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True
)

# Move to MPS device and use float16 to save memory
model = model.eval().to(device).to(torch.float16)

# Configure your paths
prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = '/path/to/your/image.jpg'  # Change this
output_path = './output'                 # Change this

# Parse the document
res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True
)

print(f"Results saved to: {output_path}")
```

## Performance Expectations

On Apple Silicon M-series with 24GB RAM:

- **Model loading**: 2-3 minutes (first time downloads ~10GB)
- **Processing speed**:
  - Single image: 30-60 seconds
  - PDF page: 1-2 minutes per page
- **Memory usage**: ~12-15GB during processing

**For PDFs**: Process will be significantly slower than on GPU. Consider:
- Processing one page at a time
- Using lower resolution modes for faster processing
- Starting with small test documents

## Resolution Modes for Better Performance

Edit these parameters in your script for speed/quality tradeoff:

```python
# Fastest (use for testing)
res = model.infer(
    tokenizer, prompt=prompt, image_file=image_file,
    base_size=512, image_size=512, crop_mode=False,
    output_path=output_path
)

# Balanced (recommended for 24GB RAM)
res = model.infer(
    tokenizer, prompt=prompt, image_file=image_file,
    base_size=640, image_size=640, crop_mode=False,
    output_path=output_path
)

# Best quality (slowest)
res = model.infer(
    tokenizer, prompt=prompt, image_file=image_file,
    base_size=1024, image_size=1024, crop_mode=False,
    output_path=output_path
)
```

## Troubleshooting

### Memory Issues

If you run out of memory:
- Use smaller resolution: `base_size=512, image_size=512`
- Close other applications
- Process one image at a time

### MPS Not Available

If MPS is not detected:
```bash
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

If False, update macOS and PyTorch to latest versions.

### Model Download Issues

```bash
# Pre-download the model
pip install huggingface-hub
huggingface-cli download deepseek-ai/DeepSeek-OCR
```

### Import Errors

Make sure you're using the virtual environment:
```bash
which python  # Should show path to venv/bin/python
```

## VSCode Tips

### Integrated Terminal
- Use VSCode's integrated terminal (`` Ctrl+` ``)
- Make sure venv is activated (you should see `(venv)` in prompt)

### Running Scripts
- Right-click on a .py file → "Run Python File in Terminal"
- Or use the play button in top-right corner

### Jupyter Notebooks
If you prefer notebooks:
```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=deepseek-ocr
```

Then create a `.ipynb` file and select the `deepseek-ocr` kernel.

## Next Steps

See `example_macos_pdf_parse.py` for a complete working example tailored for Apple Silicon.

## Alternative: Use Online GPU Services

For faster processing, consider:
- Google Colab (free T4 GPU)
- Kaggle Notebooks (free GPU)
- Paperspace Gradient (free tier available)

These provide NVIDIA GPUs where vLLM will work at full speed.
