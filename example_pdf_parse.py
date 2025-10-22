#!/usr/bin/env python3
"""
Simple Example: Parse a PDF with DeepSeek-OCR

This is a minimal example showing how to parse a single PDF file.
For production use, use the full run_dpsk_ocr_pdf.py script in DeepSeek-OCR-vllm/
"""

import os
import sys

# Example usage configuration
INPUT_PDF = '/path/to/your/document.pdf'  # TODO: Change this to your PDF path
OUTPUT_DIR = './output'  # TODO: Change this to your desired output directory

def check_setup():
    """Check if basic requirements are met"""
    print("Checking setup...")

    # Check if we're in the right directory
    if not os.path.exists('DeepSeek-OCR-master'):
        print("ERROR: DeepSeek-OCR-master directory not found!")
        print("Please run this script from the repository root directory.")
        return False

    # Check if input file exists
    if not os.path.exists(INPUT_PDF):
        print(f"ERROR: Input PDF not found: {INPUT_PDF}")
        print("Please update INPUT_PDF in this script with your PDF path.")
        return False

    # Check if dependencies are installed
    try:
        import torch
        import transformers
        import fitz  # PyMuPDF
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Please install dependencies following SETUP_GUIDE.md")
        return False

    return True

def configure_and_run():
    """Configure the model and run PDF parsing"""

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Update the config file
    config_path = 'DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py'

    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        return False

    print(f"\nParsing PDF: {INPUT_PDF}")
    print(f"Output will be saved to: {OUTPUT_DIR}")
    print("\nNOTE: First run will download the model (~10GB) from HuggingFace")
    print("This may take several minutes depending on your internet connection.\n")

    # Read current config
    with open(config_path, 'r') as f:
        config_content = f.read()

    # Update paths
    config_content = config_content.replace(
        "INPUT_PATH = ''",
        f"INPUT_PATH = '{os.path.abspath(INPUT_PDF)}'"
    )
    config_content = config_content.replace(
        "OUTPUT_PATH = ''",
        f"OUTPUT_PATH = '{os.path.abspath(OUTPUT_DIR)}'"
    )

    # Write updated config
    with open(config_path, 'w') as f:
        f.write(config_content)

    print("✓ Configuration updated")

    # Run the PDF parsing script
    print("\nStarting PDF parsing...")
    os.chdir('DeepSeek-OCR-master/DeepSeek-OCR-vllm')

    try:
        import subprocess
        result = subprocess.run(['python', 'run_dpsk_ocr_pdf.py'],
                              capture_output=False,
                              text=True)

        if result.returncode == 0:
            print("\n✓ PDF parsing completed successfully!")
            print(f"\nOutput files:")
            print(f"  - Markdown: {OUTPUT_DIR}/*.mmd")
            print(f"  - Layout PDF: {OUTPUT_DIR}/*_layouts.pdf")
            print(f"  - Images: {OUTPUT_DIR}/images/")
        else:
            print(f"\nERROR: PDF parsing failed with return code {result.returncode}")

    except Exception as e:
        print(f"\nERROR: {e}")
        return False

    return True

def main():
    print("=" * 60)
    print("DeepSeek-OCR PDF Parser - Simple Example")
    print("=" * 60)

    if not check_setup():
        print("\nSetup check failed. Please fix the issues above and try again.")
        print("See SETUP_GUIDE.md for detailed instructions.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Starting PDF Parsing")
    print("=" * 60)

    success = configure_and_run()

    if success:
        print("\n" + "=" * 60)
        print("Done! Check the output directory for results.")
        print("=" * 60)
    else:
        print("\nFailed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    # Quick configuration reminder
    if INPUT_PDF == '/path/to/your/document.pdf':
        print("\n" + "!" * 60)
        print("REMINDER: Please update INPUT_PDF and OUTPUT_DIR")
        print("in this script before running!")
        print("!" * 60 + "\n")

        user_input = input("Continue anyway? (y/N): ")
        if user_input.lower() != 'y':
            print("Exiting. Please update the paths and try again.")
            sys.exit(0)

    main()
