# DeepSeek-OCR Studio 📄

A comprehensive Streamlit-based application for **DeepSeek-OCR** that provides an intuitive interface for extracting information from presentations, PDFs, and documents with tables and graphics.

## Features ✨

### 🎯 Core Capabilities

- **Drag-and-Drop File Upload**: Easy file upload interface supporting PDF, PNG, JPG, and JPEG formats
- **Multi-Page Processing**: Handle multi-page PDFs with individual page navigation
- **Real-Time Processing**: Process documents with visual progress tracking
- **Multiple Output Formats**: Markdown, annotated images, and raw text exports

### 🔧 Configuration Options

#### Resolution Modes
- **Tiny**: 512×512 (64 vision tokens) - Fastest, basic quality
- **Small**: 640×640 (100 vision tokens) - Fast, good quality
- **Base**: 1024×1024 (256 vision tokens) - Balanced quality/speed
- **Large**: 1280×1280 (400 vision tokens) - High quality
- **Gundam**: Dynamic resolution with n×640×640 + 1×1024×1024 - Best quality

#### Prompt Templates
Pre-configured prompts for different use cases:
- **Document to Markdown**: Convert documents with layout preservation
- **OCR Image**: General OCR with grounding
- **Free OCR**: OCR without layout structure
- **Parse Figure**: Extract information from charts and diagrams
- **Describe Image**: General image description
- **Custom**: Define your own prompts

#### Advanced Settings
- **Max Crops**: Control dynamic resolution cropping (2-9)
- **Max Concurrency**: Batch processing concurrency (1-200)
- **GPU Memory**: Memory utilization control (0.5-0.95)
- **N-gram Settings**: Prevent repetitive text generation
  - N-gram Size: 10-50
  - Window Size: 30-150
- **Pre-processing Workers**: Parallel image processing (1-128)
- **PDF DPI**: Control PDF rendering quality (72-300)

### 📊 Results View

Three comprehensive views for each processed document:

1. **Markdown Output**
   - Clean markdown conversion
   - Raw output inspection
   - Syntax-highlighted display

2. **Visualized**
   - Original image view
   - Annotated image with bounding boxes
   - Color-coded element detection

3. **Downloads**
   - Download markdown files
   - Download annotated images
   - Download raw text output
   - Batch ZIP download for multi-page documents

## Installation 🛠️

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ GPU memory recommended

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
   cd deepseek-ocr
   ```

2. **Install dependencies**:
   ```bash
   # Install PyTorch with CUDA support
   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

   # Install vLLM (download the wheel from releases)
   pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

   # Install other requirements
   pip install -r requirements.txt

   # Install flash-attention
   pip install flash-attn==2.7.3 --no-build-isolation
   ```

3. **Download the model** (automatic on first run):
   The app will automatically download `deepseek-ai/DeepSeek-OCR` from HuggingFace on first use.

## Usage 🚀

### Quick Start

Run the launcher script:
```bash
./run_app.sh
```

Or manually:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Step-by-Step Guide

1. **Configure Settings** (Sidebar)
   - Select your desired resolution mode
   - Choose a prompt template or create custom
   - Adjust advanced settings if needed

2. **Upload Files** (Upload & Process Tab)
   - Drag and drop your files or click to browse
   - Supports multiple files at once
   - Click "Process Files" to start

3. **View Results** (Results Tab)
   - Select file and page to view
   - Toggle between markdown, visualized, and download views
   - Download individual pages or all pages as ZIP

### Example Use Cases

#### Academic Papers
```
Resolution: Base (1024×1024)
Prompt: Document to Markdown
Use Case: Extract text, tables, and formulas from research papers
```

#### Presentations with Graphics
```
Resolution: Gundam (Dynamic)
Prompt: Parse Figure
Use Case: Extract charts, diagrams, and slide content
```

#### Financial Reports
```
Resolution: Large (1280×1280)
Prompt: Document to Markdown
Use Case: Extract tables, numbers, and structured data
```

#### Technical Diagrams
```
Resolution: Gundam (Dynamic)
Prompt: Parse Figure
PDF DPI: 300
Use Case: High-quality extraction from complex technical drawings
```

## Configuration Details ⚙️

### Model Settings

- **Model Path**: HuggingFace model ID or local path
  - Default: `deepseek-ai/DeepSeek-OCR`
  - Can use local path for offline usage

### Performance Tuning

For **limited GPU memory**:
```
Resolution: Small or Base
Max Concurrency: 50
GPU Memory: 0.7
Max Crops: 4
```

For **maximum quality**:
```
Resolution: Gundam or Large
Max Concurrency: 100
GPU Memory: 0.9
Max Crops: 9
PDF DPI: 300
```

For **speed priority**:
```
Resolution: Tiny or Small
Max Concurrency: 200
Pre-processing Workers: 128
Max Crops: 2
```

### N-gram No-Repeat Settings

Prevents the model from generating repetitive text:

- **N-gram Size**: Larger values catch longer repetitive patterns
- **Window Size**: Larger windows check more context
- **Whitelist Tokens**: Special tokens allowed to repeat (e.g., `<td>`, `</td>` for tables)

## Technical Architecture 🏗️

### Components

1. **Frontend**: Streamlit web interface
2. **Model Engine**: vLLM for efficient inference
3. **Image Processing**: Concurrent preprocessing with ThreadPoolExecutor
4. **OCR Engine**: DeepSeek-OCR vision-language model
5. **Post-Processing**: Bounding box extraction and visualization

### Data Flow

```
Upload → Pre-process → Model Inference → Post-process → Display
   ↓         ↓              ↓                ↓            ↓
 Files    Resize       vLLM Generate    Extract BB    Markdown
         Padding      DeepSeek-OCR      Clean Text   Visualize
```

## Troubleshooting 🔧

### Common Issues

**Model fails to load**:
- Check GPU memory availability
- Reduce `GPU Memory Utilization` setting
- Verify CUDA installation: `nvidia-smi`

**Out of memory errors**:
- Use smaller resolution mode
- Reduce `Max Concurrency`
- Lower `Max Crops` for Gundam mode
- Close other GPU applications

**Slow processing**:
- Increase `Pre-processing Workers`
- Use lower resolution for faster processing
- Reduce PDF DPI for large documents

**Poor quality results**:
- Increase resolution mode
- Use higher PDF DPI
- Try different prompt templates
- For figures, use "Parse Figure" prompt

**File upload fails**:
- Check file size (max 200MB per file)
- Verify file format (PDF, PNG, JPG, JPEG only)
- Try processing fewer files at once

## API Reference 📚

### Key Functions

#### `load_model(model_path, max_concurrency, gpu_memory_util)`
Loads the DeepSeek-OCR model with vLLM engine.

**Parameters**:
- `model_path`: HuggingFace model ID or local path
- `max_concurrency`: Maximum concurrent sequences
- `gpu_memory_util`: GPU memory fraction (0.0-1.0)

**Returns**: vLLM LLM instance

#### `pdf_to_images(pdf_bytes, dpi=144)`
Converts PDF to high-quality images.

**Parameters**:
- `pdf_bytes`: PDF file bytes
- `dpi`: Resolution for rendering (default: 144)

**Returns**: List of PIL Images

#### `process_ocr(images, llm, sampling_params, prompt, crop_mode, num_workers=4)`
Processes images with DeepSeek-OCR.

**Parameters**:
- `images`: List of PIL Images
- `llm`: vLLM model instance
- `sampling_params`: Sampling parameters
- `prompt`: Prompt template
- `crop_mode`: Enable dynamic cropping
- `num_workers`: Parallel workers

**Returns**: List of generation outputs

## Performance Benchmarks ⚡

Approximate processing speeds (on A100-40G):

| Mode | Tokens/s | Pages/min | Quality |
|------|----------|-----------|---------|
| Tiny | ~3500 | 30-40 | ⭐⭐ |
| Small | ~3000 | 25-35 | ⭐⭐⭐ |
| Base | ~2500 | 20-30 | ⭐⭐⭐⭐ |
| Large | ~2000 | 15-25 | ⭐⭐⭐⭐⭐ |
| Gundam | ~2500 | 20-30 | ⭐⭐⭐⭐⭐ |

*Actual performance varies based on document complexity and hardware*

## Contributing 🤝

Contributions welcome! Areas for improvement:

- [ ] Add support for more file formats (DOCX, PPTX)
- [ ] Implement batch folder processing
- [ ] Add OCR accuracy metrics
- [ ] Support for custom model fine-tuning
- [ ] Multi-language UI support
- [ ] Cloud deployment templates

## License 📄

This application uses DeepSeek-OCR. Please refer to the [DeepSeek-OCR repository](https://github.com/deepseek-ai/DeepSeek-OCR) for licensing information.

## Citation 📖

If you use this application in your research, please cite:

```bibtex
@article{wei2024deepseek-ocr,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2510.18234},
  year={2025}
}
```

## Resources 🔗

- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)
- [Model on HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [GitHub Repository](https://github.com/deepseek-ai/DeepSeek-OCR)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Support 💬

For issues and questions:
- GitHub Issues: [DeepSeek-OCR Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
- Discord: [DeepSeek AI Community](https://discord.gg/Tc7c45Zzu5)

---

**Built with ❤️ using Streamlit and DeepSeek-OCR**
