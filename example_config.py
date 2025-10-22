"""
Example Configuration for DeepSeek-OCR PDF Parsing

Copy this configuration to DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py
and adjust the paths according to your needs.
"""

# ============================================
# RESOLUTION MODE SETTINGS
# ============================================
# Choose one of the following preset modes:

# Gundam Mode (Recommended): Dynamic resolution for best quality/speed balance
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True

# Uncomment below for other modes:
# ---------------------------------------------
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# ---------------------------------------------

# ============================================
# PERFORMANCE SETTINGS
# ============================================
MIN_CROPS = 2
MAX_CROPS = 6  # Reduce to 4 if you have limited GPU memory (max: 9)
MAX_CONCURRENCY = 100  # Reduce to 10-20 for GPUs with <40GB memory
NUM_WORKERS = 64  # CPU workers for image preprocessing
PRINT_NUM_VIS_TOKENS = False
SKIP_REPEAT = True

# ============================================
# MODEL SETTINGS
# ============================================
# Model will be auto-downloaded from HuggingFace on first run (~10GB)
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'

# Or use a local path if you've pre-downloaded the model:
# MODEL_PATH = '/path/to/local/model/directory'

# ============================================
# INPUT/OUTPUT PATHS
# ============================================
# TODO: CHANGE THESE PATHS!

# Input file path
# For PDF: Use with run_dpsk_ocr_pdf.py
# For images (.jpg, .png, .jpeg): Use with run_dpsk_ocr_image.py
INPUT_PATH = '/path/to/your/input.pdf'

# Output directory (will be created if it doesn't exist)
OUTPUT_PATH = '/path/to/output/directory'

# ============================================
# PROMPT SETTINGS
# ============================================
# Choose the appropriate prompt for your use case

# For documents (most common - preserves layout and structure)
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# Uncomment below for other use cases:
# ---------------------------------------------
# For general OCR on images:
# PROMPT = '<image>\n<|grounding|>OCR this image.'

# For text-only extraction without layout:
# PROMPT = '<image>\nFree OCR.'

# For parsing figures/charts:
# PROMPT = '<image>\nParse the figure.'

# For detailed image description:
# PROMPT = '<image>\nDescribe this image in detail.'

# For locating specific text (replace 'xxxx' with your text):
# PROMPT = '<image>\nLocate <|ref|>xxxx<|/ref|> in the image.'
# ---------------------------------------------

# ============================================
# TOKENIZER INITIALIZATION
# ============================================
from transformers import AutoTokenizer
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
