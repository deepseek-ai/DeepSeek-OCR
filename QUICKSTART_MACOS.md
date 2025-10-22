# Quick Start for macOS (Apple Silicon)

Get DeepSeek-OCR running on your MacBook in 3 easy steps!

## ⚠️ Important: macOS Limitations

- **vLLM NOT supported** (requires NVIDIA GPU)
- **Flash Attention NOT supported** (requires CUDA)
- Uses **Transformers** library with MPS (Metal GPU) acceleration
- **Slower** than NVIDIA GPU (~10-20x) but works great for small batches

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB+ RAM (24GB recommended)
- VSCode installed
- Python 3.9 or later

---

## Step 1: Setup Environment

Open Terminal or VSCode's integrated terminal:

```bash
# Navigate to the repository
cd /path/to/DeepSeek-OCR

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements_macos.txt
```

**Expected time**: 5-10 minutes

---

## Step 2: Configure Your File

Edit `example_macos_pdf_parse.py`:

```python
# Change these two lines:
INPUT_FILE = '/Users/yourname/Documents/myfile.pdf'  # Your file
OUTPUT_DIR = './output'                               # Output folder
```

**Supported formats**: PDF, JPG, PNG, JPEG, BMP, TIFF

---

## Step 3: Run It!

```bash
python example_macos_pdf_parse.py
```

**First run**: Downloads ~10GB model (5-10 minutes)
**Processing**: 1-2 minutes per PDF page

---

## Using in VSCode

### Setup

1. Open folder in VSCode: `File > Open Folder` → select `DeepSeek-OCR`
2. Select Python interpreter:
   - Press `Cmd+Shift+P`
   - Type "Python: Select Interpreter"
   - Choose `./venv/bin/python`

3. Open `example_macos_pdf_parse.py`
4. Edit `INPUT_FILE` path
5. Click the "Run" button (▶️) in top-right

### Terminal in VSCode

- Open terminal: `` Cmd+` ``
- Should show `(venv)` in prompt
- Run: `python example_macos_pdf_parse.py`

---

## Output Files

After processing, check your output directory:

```
output/
├── myfile_combined.md    # All pages in one markdown file
└── images/               # Extracted images (if any)
```

---

## Performance Tips

### For Faster Processing (Lower Quality)

Edit `example_macos_pdf_parse.py`:

```python
BASE_SIZE = 512      # Smaller = faster
IMAGE_SIZE = 512
CROP_MODE = False
```

### For Best Quality (Slower)

```python
BASE_SIZE = 1024     # Larger = better quality
IMAGE_SIZE = 1024
CROP_MODE = False
```

### Expected Speed (24GB M-series Mac)

| Mode  | Size | Speed/Page | Quality |
|-------|------|------------|---------|
| Tiny  | 512  | ~30 sec    | Good    |
| Small | 640  | ~45 sec    | Better  |
| Base  | 1024 | ~90 sec    | Best    |

---

## Troubleshooting

### "MPS not available"

Update macOS and PyTorch:
```bash
pip install --upgrade torch torchvision
```

Check with:
```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

### "Out of memory"

Use smaller size:
```python
BASE_SIZE = 512
IMAGE_SIZE = 512
```

### "ModuleNotFoundError"

Make sure venv is activated:
```bash
source venv/bin/activate
which python  # Should show: .../venv/bin/python
```

### "File not found"

Use absolute paths:
```python
INPUT_FILE = '/Users/yourname/Documents/file.pdf'  # Full path
```

### Slow Processing

This is expected on Apple Silicon vs NVIDIA GPUs. For faster processing:
- Use smaller resolution mode
- Process fewer pages
- Consider using cloud GPU (Google Colab)

---

## What's Happening?

1. **Model Download** (first run only): Downloads 10GB AI model
2. **PDF Conversion**: Converts PDF pages to images
3. **OCR Processing**: Uses AI vision model to extract text
4. **Markdown Generation**: Creates structured markdown with layout
5. **Image Extraction**: Saves figures and images separately

---

## Next Steps

- See `SETUP_MACOS.md` for detailed documentation
- Try different prompts for different document types
- Adjust quality settings for your needs

## Alternative Prompts

Edit in `example_macos_pdf_parse.py`:

```python
# For documents (default)
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

# For simple text extraction
PROMPT = "<image>\nFree OCR."

# For general OCR
PROMPT = "<image>\n<|grounding|>OCR this image."

# For charts/figures
PROMPT = "<image>\nParse the figure."
```

---

## Need Faster Processing?

For production use with many PDFs, consider:
- **Google Colab** (free T4 GPU): 10-20x faster
- **Paperspace** (free tier available)
- **AWS/Azure** cloud GPUs

The vLLM approach will work there with full speed!

---

**Ready?** Just run: `python example_macos_pdf_parse.py`
