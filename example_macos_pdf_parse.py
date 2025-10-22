#!/usr/bin/env python3
"""
DeepSeek-OCR for macOS (Apple Silicon)

This script is optimized for Apple Silicon Macs using the Transformers library.
It does NOT use vLLM (which requires CUDA/NVIDIA GPU).

Requirements:
- macOS with Apple Silicon (M1/M2/M3)
- 24GB+ RAM recommended
- Python 3.9+
"""

import os
import sys
from pathlib import Path
import torch
from PIL import Image
import fitz  # PyMuPDF

# ============================================
# CONFIGURATION - CHANGE THESE VALUES
# ============================================

# Input: Can be a PDF or image file
INPUT_FILE = '/path/to/your/document.pdf'  # TODO: Change this
OUTPUT_DIR = './output'                      # TODO: Change this

# Model configuration
MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'

# Processing mode (adjust for speed/quality tradeoff)
# Tiny:   base_size=512,  image_size=512,  crop_mode=False  (fastest)
# Small:  base_size=640,  image_size=640,  crop_mode=False  (balanced)
# Base:   base_size=1024, image_size=1024, crop_mode=False  (best quality)
# Gundam: base_size=1024, image_size=640,  crop_mode=True   (dynamic, slower)

BASE_SIZE = 640      # Recommended for 24GB RAM
IMAGE_SIZE = 640     # Recommended for 24GB RAM
CROP_MODE = False    # Set True for better quality but slower

# Prompt options
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
# PROMPT = "<image>\nFree OCR."  # Alternative: simpler text extraction

# ============================================
# SCRIPT STARTS HERE
# ============================================

def check_environment():
    """Check if the environment is properly set up"""
    print("Checking environment...")

    # Check PyTorch
    print(f"✓ PyTorch version: {torch.__version__}")

    # Check for MPS (Metal Performance Shaders) support
    if torch.backends.mps.is_available():
        print("✓ MPS (Metal GPU) available")
        return "mps"
    else:
        print("⚠ MPS not available, using CPU (will be slower)")
        print("  To enable MPS: update macOS and PyTorch to latest versions")
        return "cpu"

def pdf_to_images(pdf_path, dpi=144):
    """Convert PDF pages to images"""
    print(f"Converting PDF to images (DPI={dpi})...")
    images = []

    try:
        pdf_document = fitz.open(pdf_path)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_num in range(pdf_document.page_count):
            print(f"  Processing page {page_num + 1}/{pdf_document.page_count}")
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)

            import io
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background

            images.append(img)

        pdf_document.close()
        print(f"✓ Converted {len(images)} pages to images")
        return images

    except Exception as e:
        print(f"Error converting PDF: {e}")
        return None

def load_model(device):
    """Load the DeepSeek-OCR model"""
    print(f"\nLoading model from HuggingFace...")
    print("Note: First run will download ~10GB model (may take 5-10 minutes)")

    try:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print("✓ Tokenizer loaded")

        # Load model WITHOUT flash_attention_2 (not supported on macOS)
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_safetensors=True
        )

        # Move to device (MPS or CPU)
        if device == "mps":
            model = model.eval().to(device).to(torch.float16)
        else:
            model = model.eval().to(device)

        print(f"✓ Model loaded on {device}")

        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Try: huggingface-cli login (if model requires authentication)")
        print("3. Manually download: huggingface-cli download deepseek-ai/DeepSeek-OCR")
        return None, None

def process_image(model, tokenizer, image, output_path):
    """Process a single image with the model"""
    try:
        res = model.infer(
            tokenizer,
            prompt=PROMPT,
            image_file=image if isinstance(image, str) else None,
            image=None if isinstance(image, str) else image,
            output_path=output_path,
            base_size=BASE_SIZE,
            image_size=IMAGE_SIZE,
            crop_mode=CROP_MODE,
            save_results=True,
            test_compress=True
        )
        return res
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def process_pdf(model, tokenizer, pdf_path, output_dir):
    """Process an entire PDF"""
    # Convert PDF to images
    images = pdf_to_images(pdf_path)
    if not images:
        return False

    print(f"\nProcessing {len(images)} pages...")
    print("This may take several minutes on Apple Silicon\n")

    all_results = []

    for idx, image in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] Processing page {idx}...")

        # Save temp image
        temp_image_path = os.path.join(output_dir, f'temp_page_{idx}.png')
        image.save(temp_image_path)

        # Process
        result = process_image(model, tokenizer, temp_image_path, output_dir)

        if result:
            all_results.append(result)
            print(f"  ✓ Page {idx} completed")
        else:
            print(f"  ✗ Page {idx} failed")

        # Clean up temp file
        try:
            os.remove(temp_image_path)
        except:
            pass

    # Combine results
    if all_results:
        combined_output = os.path.join(
            output_dir,
            Path(pdf_path).stem + '_combined.md'
        )

        with open(combined_output, 'w', encoding='utf-8') as f:
            for idx, result in enumerate(all_results, 1):
                f.write(f"\n\n## Page {idx}\n\n")
                f.write(result)
                f.write("\n\n---\n")

        print(f"\n✓ Combined results saved to: {combined_output}")
        return True

    return False

def main():
    print("=" * 70)
    print("DeepSeek-OCR for macOS (Apple Silicon)")
    print("=" * 70)

    # Check configuration
    if INPUT_FILE == '/path/to/your/document.pdf':
        print("\n⚠ WARNING: Please configure INPUT_FILE in the script!")
        print("Edit this file and set INPUT_FILE to your document path.\n")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(0)

    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"\n✗ Error: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check environment
    device = check_environment()

    # Load model
    model, tokenizer = load_model(device)
    if model is None:
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Starting OCR Processing")
    print("=" * 70)

    # Determine file type and process
    file_ext = Path(INPUT_FILE).suffix.lower()

    if file_ext == '.pdf':
        print(f"Input: PDF file ({INPUT_FILE})")
        success = process_pdf(model, tokenizer, INPUT_FILE, OUTPUT_DIR)
    elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        print(f"Input: Image file ({INPUT_FILE})")
        result = process_image(model, tokenizer, INPUT_FILE, OUTPUT_DIR)
        success = result is not None
    else:
        print(f"✗ Unsupported file type: {file_ext}")
        print("Supported: .pdf, .jpg, .jpeg, .png, .bmp, .tiff")
        sys.exit(1)

    # Summary
    print("\n" + "=" * 70)
    if success:
        print("✓ Processing completed successfully!")
        print(f"Output directory: {OUTPUT_DIR}")
    else:
        print("✗ Processing failed")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
