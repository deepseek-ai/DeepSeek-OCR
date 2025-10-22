# 🐳 DeepSeek-OCR Gradio Interface 🐳

This is a Gradio-based interactive interface for the DeepSeek-OCR model, providing a friendly Web UI to utilize the powerful features of DeepSeek-OCR.



## 🛠️ Installation Instructions


### Installation Steps

1. **Clone the project**
   ```bash
   git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
   cd DeepSeek-OCR/DeepSeek-OCR-gradio
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Main Dependencies

- **PyTorch 2.6.0** and **torchvision**: Deep learning framework
- **transformers 4.46.3**: Hugging Face model library
- **gradio 4.0.0+**: Web UI framework
- **flash-attn**: Flash Attention 2 implementation for performance optimization
- **Pillow**: Image processing
- **einops**: Tensor operation simplification
- Other auxiliary libraries: tokenizers, addict, easydict, safetensors, accelerate, sentencepiece, protobuf

## 🚀 Startup Steps

1. **Ensure environment is activated**
   ```bash
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

2. **Launch the application**
   ```bash
   python app.py
   ```

3. **Wait for model loading**
   - The application will automatically download and load the DeepSeek-OCR model on startup
   - First run may take considerable time to download model files
   - "✅ Model loaded successfully." message will be displayed after successful model loading

4. **Access the interface**
   - Local access: http://localhost:7860
   - Public access: Through the public link provided by Gradio (automatically generated with share=True)

## 📖 Usage Instructions

### Basic Operation Flow

1. **Upload Image**
   - Click the upload area or drag image files
   - Support pasting images from clipboard
   - Support common image formats (PNG, JPG, JPEG, etc.)

2. **Select Resolution**
   - **Tiny**: 512x512, fastest processing speed
   - **Small**: 640x640, balanced speed and quality
   - **Base**: 1024x1024, high-quality processing
   - **Large**: 1280x1280, highest quality
   - **Gundam (Recommended)**: 1024x640, recommended for most documents

3. **Select Task Type**
   - **📝 Free OCR**: Extract all text from the image
   - **📄 Convert to Markdown**: Convert documents to Markdown format
   - **📈 Parse Chart**: Parse data in charts
   - **🔍 Locate Object by Reference**: Locate objects based on reference text

4. **Enter Reference Text** (Only for localization tasks)
   - When selecting "Locate Object by Reference", you need to enter the text to search for
   - For example: "teacher", "20-10", "a red car", etc.

5. **Process Image**
   - Click the "Process Image" button to start processing
   - Wait for processing to complete, results will be displayed on the right side
