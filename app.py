import streamlit as st
import os
import io
import sys
import tempfile
import torch
from PIL import Image, ImageOps
import fitz
import img2pdf
import re
from pathlib import Path
import base64
import zipfile

# Set page config
st.set_page_config(
    page_title="DeepSeek-OCR Studio",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 16px;
    }
    .upload-box {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f0f8ff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Add DeepSeek-OCR modules to path
sys.path.insert(0, '/home/user/deepseek-ocr/DeepSeek-OCR-master/DeepSeek-OCR-vllm')

def load_model(model_path, max_concurrency, gpu_memory_util):
    """Load the DeepSeek-OCR model with vLLM"""
    try:
        from deepseek_ocr import DeepseekOCRForCausalLM
        from vllm.model_executor.models.registry import ModelRegistry
        from vllm import LLM

        ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

        if torch.version.cuda == '11.8':
            os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
        os.environ['VLLM_USE_V1'] = '0'

        llm = LLM(
            model=model_path,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            enforce_eager=False,
            trust_remote_code=True,
            max_model_len=8192,
            swap_space=0,
            max_num_seqs=max_concurrency,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_util,
            disable_mm_preprocessor_cache=True
        )
        return llm
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def pdf_to_images(pdf_bytes, dpi=144):
    """Convert PDF to images"""
    images = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)

    pdf_document.close()
    return images

def load_image(image_bytes):
    """Load and correct image orientation"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image.convert('RGB')
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def extract_bounding_boxes(text):
    """Extract bounding box information from OCR output"""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image = []
    matches_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other

def draw_bounding_boxes(image, refs):
    """Draw bounding boxes on image"""
    from PIL import ImageDraw, ImageFont
    import numpy as np

    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    for i, ref in enumerate(refs):
        try:
            label_type = ref[1]
            cor_list = eval(ref[2])

            color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))
            color_a = color + (20,)

            for points in cor_list:
                x1, y1, x2, y2 = points
                x1 = int(x1 / 999 * image_width)
                y1 = int(y1 / 999 * image_height)
                x2 = int(x2 / 999 * image_width)
                y2 = int(y2 / 999 * image_height)

                if label_type == 'title':
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                    draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                else:
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                text_x = x1
                text_y = max(0, y1 - 15)
                text_bbox = draw.textbbox((0, 0), label_type, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height],
                            fill=(255, 255, 255, 30))
                draw.text((text_x, text_y), label_type, font=font, fill=color)
        except:
            continue

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw

def process_ocr(images, llm, sampling_params, prompt, crop_mode, num_workers=4):
    """Process images with DeepSeek-OCR"""
    from process.image_process import DeepseekOCRProcessor
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    def process_single_image(image):
        cache_item = {
            "prompt": prompt,
            "multi_modal_data": {"image": DeepseekOCRProcessor().tokenize_with_images(
                images=[image], bos=True, eos=True, cropping=crop_mode
            )},
        }
        return cache_item

    # Process images in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        batch_inputs = list(executor.map(process_single_image, images))

    # Generate outputs
    outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

    return outputs_list

# Header
st.markdown('<h1 class="main-header">📄 DeepSeek-OCR Studio</h1>', unsafe_allow_html=True)
st.markdown("### Extract information from presentations, PDFs, and documents with tables and graphics")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Model Settings
    st.subheader("Model Settings")
    model_path = st.text_input("Model Path", value="deepseek-ai/DeepSeek-OCR",
                               help="HuggingFace model path or local path")

    # Resolution Mode
    st.subheader("Resolution Mode")
    resolution_mode = st.selectbox(
        "Select Resolution",
        ["Gundam (Dynamic)", "Tiny (512×512)", "Small (640×640)", "Base (1024×1024)", "Large (1280×1280)"],
        help="Higher resolution = more vision tokens but better quality"
    )

    # Map resolution mode to settings
    resolution_settings = {
        "Gundam (Dynamic)": {"base_size": 1024, "image_size": 640, "crop_mode": True},
        "Tiny (512×512)": {"base_size": 512, "image_size": 512, "crop_mode": False},
        "Small (640×640)": {"base_size": 640, "image_size": 640, "crop_mode": False},
        "Base (1024×1024)": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
        "Large (1280×1280)": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    }

    settings = resolution_settings[resolution_mode]

    # Advanced Settings
    with st.expander("🔧 Advanced Settings", expanded=False):
        max_crops = st.slider("Max Crops (for Gundam mode)", 2, 9, 6,
                             help="Maximum number of crops for dynamic resolution")
        max_concurrency = st.slider("Max Concurrency", 1, 200, 100,
                                   help="Maximum concurrent sequences")
        gpu_memory = st.slider("GPU Memory Utilization", 0.5, 0.95, 0.9, 0.05,
                              help="Fraction of GPU memory to use")

        st.subheader("N-gram No-Repeat Settings")
        ngram_size = st.slider("N-gram Size", 10, 50, 20,
                              help="Size of n-grams to check for repetition")
        window_size = st.slider("Window Size", 30, 150, 50,
                               help="Window size for repetition checking")

        num_workers = st.slider("Pre-processing Workers", 1, 128, 64,
                               help="Number of parallel workers for image preprocessing")

    # Prompt Templates
    st.subheader("📝 Prompt Template")
    prompt_type = st.selectbox(
        "Select Prompt Type",
        [
            "Document to Markdown",
            "OCR Image",
            "Free OCR (No Layout)",
            "Parse Figure",
            "Describe Image",
            "Custom"
        ]
    )

    prompt_templates = {
        "Document to Markdown": "<image>\n<|grounding|>Convert the document to markdown.",
        "OCR Image": "<image>\n<|grounding|>OCR this image.",
        "Free OCR (No Layout)": "<image>\nFree OCR.",
        "Parse Figure": "<image>\nParse the figure.",
        "Describe Image": "<image>\nDescribe this image in detail.",
    }

    if prompt_type == "Custom":
        prompt = st.text_area("Custom Prompt", value="<image>\n<|grounding|>Convert the document to markdown.",
                             help="Use <image> placeholder for image position")
    else:
        prompt = prompt_templates[prompt_type]
        st.info(f"Prompt: `{prompt}`")

    # PDF Settings
    st.subheader("📄 PDF Settings")
    pdf_dpi = st.slider("PDF DPI", 72, 300, 144,
                       help="Resolution for PDF to image conversion")

# Main Content Area
tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Results", "ℹ️ About"])

with tab1:
    st.header("Upload Files")

    # File uploader with drag and drop
    uploaded_files = st.file_uploader(
        "Drag and drop files here or click to browse",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload PDF, PNG, or JPG files"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")

        # Display uploaded files
        cols = st.columns(min(len(uploaded_files), 4))
        for idx, file in enumerate(uploaded_files[:4]):
            with cols[idx]:
                if file.type == "application/pdf":
                    st.info(f"📄 {file.name}")
                else:
                    st.image(Image.open(file), caption=file.name, use_container_width=True)

        if len(uploaded_files) > 4:
            st.info(f"... and {len(uploaded_files) - 4} more file(s)")

        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_button = st.button("🚀 Process Files", type="primary", use_container_width=True)

        if process_button:
            st.session_state.processing = True

            with st.spinner("Loading model... This may take a few minutes on first run."):
                try:
                    # Load model
                    llm = load_model(model_path, max_concurrency, gpu_memory)

                    if llm is None:
                        st.error("Failed to load model. Please check the model path and settings.")
                        st.session_state.processing = False
                    else:
                        # Setup sampling params
                        from vllm import SamplingParams
                        from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

                        logits_processors = [NoRepeatNGramLogitsProcessor(
                            ngram_size=ngram_size,
                            window_size=window_size,
                            whitelist_token_ids={128821, 128822}
                        )]

                        sampling_params = SamplingParams(
                            temperature=0.0,
                            max_tokens=8192,
                            logits_processors=logits_processors,
                            skip_special_tokens=False,
                            include_stop_str_in_output=True,
                        )

                        all_results = []

                        # Process each file
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for file_idx, uploaded_file in enumerate(uploaded_files):
                            status_text.text(f"Processing {uploaded_file.name} ({file_idx + 1}/{len(uploaded_files)})...")

                            # Convert to images
                            if uploaded_file.type == "application/pdf":
                                pdf_bytes = uploaded_file.read()
                                images = pdf_to_images(pdf_bytes, dpi=pdf_dpi)
                            else:
                                image_bytes = uploaded_file.read()
                                image = load_image(image_bytes)
                                images = [image] if image else []

                            if not images:
                                st.warning(f"Skipping {uploaded_file.name} - could not load images")
                                continue

                            # Process with OCR
                            outputs = process_ocr(
                                images, llm, sampling_params, prompt,
                                settings['crop_mode'], num_workers
                            )

                            # Store results
                            file_results = {
                                'filename': uploaded_file.name,
                                'images': images,
                                'outputs': outputs,
                                'type': uploaded_file.type
                            }
                            all_results.append(file_results)

                            progress_bar.progress((file_idx + 1) / len(uploaded_files))

                        st.session_state.processed_results = all_results
                        status_text.text("✅ Processing complete!")
                        st.success("All files processed successfully!")
                        st.balloons()

                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    st.session_state.processing = False

with tab2:
    st.header("Results")

    if st.session_state.processed_results:
        # File selector
        file_names = [r['filename'] for r in st.session_state.processed_results]
        selected_file_idx = st.selectbox("Select File", range(len(file_names)),
                                         format_func=lambda x: file_names[x])

        result = st.session_state.processed_results[selected_file_idx]

        # Page selector for multi-page documents
        if len(result['images']) > 1:
            page_idx = st.slider("Select Page", 0, len(result['images']) - 1, 0)
        else:
            page_idx = 0

        # Display tabs for different views
        result_tab1, result_tab2, result_tab3 = st.tabs(["📝 Markdown Output", "🖼️ Visualized", "💾 Downloads"])

        with result_tab1:
            st.subheader(f"Page {page_idx + 1} - Markdown Output")

            # Extract and clean output
            output_text = result['outputs'][page_idx].outputs[0].text

            # Remove special tokens
            if '<｜end▁of▁sentence｜>' in output_text:
                output_text = output_text.replace('<｜end▁of▁sentence｜>', '')

            # Extract bounding boxes
            matches, matches_images, matches_other = extract_bounding_boxes(output_text)

            # Show raw output
            with st.expander("🔍 Raw Output", expanded=False):
                st.code(output_text, language="markdown")

            # Clean output for display
            clean_output = output_text
            for match in matches_other:
                clean_output = clean_output.replace(match, '')
            clean_output = clean_output.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

            st.markdown(clean_output)

        with result_tab2:
            st.subheader(f"Page {page_idx + 1} - Visualized")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Original Image**")
                st.image(result['images'][page_idx], use_container_width=True)

            with col2:
                st.write("**With Bounding Boxes**")
                if matches:
                    annotated_image = draw_bounding_boxes(result['images'][page_idx], matches)
                    st.image(annotated_image, use_container_width=True)
                else:
                    st.info("No bounding boxes detected")
                    st.image(result['images'][page_idx], use_container_width=True)

        with result_tab3:
            st.subheader("Download Results")

            # Download markdown
            output_text = result['outputs'][page_idx].outputs[0].text
            clean_output = output_text
            for match in matches_other:
                clean_output = clean_output.replace(match, '')

            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    label="📄 Download Markdown",
                    data=clean_output,
                    file_name=f"{Path(result['filename']).stem}_page{page_idx+1}.md",
                    mime="text/markdown"
                )

            with col2:
                # Download annotated image
                if matches:
                    annotated_image = draw_bounding_boxes(result['images'][page_idx], matches)
                    buf = io.BytesIO()
                    annotated_image.save(buf, format='PNG')
                    st.download_button(
                        label="🖼️ Download Annotated",
                        data=buf.getvalue(),
                        file_name=f"{Path(result['filename']).stem}_page{page_idx+1}_annotated.png",
                        mime="image/png"
                    )

            with col3:
                # Download raw output
                st.download_button(
                    label="📋 Download Raw Text",
                    data=output_text,
                    file_name=f"{Path(result['filename']).stem}_page{page_idx+1}_raw.txt",
                    mime="text/plain"
                )

            # Download all pages
            if len(result['images']) > 1:
                st.divider()
                st.subheader("Download All Pages")

                if st.button("📦 Prepare ZIP Download"):
                    with st.spinner("Creating ZIP file..."):
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for idx, (img, output) in enumerate(zip(result['images'], result['outputs'])):
                                # Add markdown
                                output_text = output.outputs[0].text
                                clean_text = output_text
                                matches_all, _, matches_other_all = extract_bounding_boxes(output_text)
                                for match in matches_other_all:
                                    clean_text = clean_text.replace(match, '')

                                zip_file.writestr(
                                    f"page_{idx+1}.md",
                                    clean_text
                                )

                                # Add annotated image
                                if matches_all:
                                    annotated = draw_bounding_boxes(img, matches_all)
                                    buf = io.BytesIO()
                                    annotated.save(buf, format='PNG')
                                    zip_file.writestr(
                                        f"page_{idx+1}_annotated.png",
                                        buf.getvalue()
                                    )

                        st.download_button(
                            label="⬇️ Download ZIP",
                            data=zip_buffer.getvalue(),
                            file_name=f"{Path(result['filename']).stem}_all_pages.zip",
                            mime="application/zip"
                        )
    else:
        st.info("👈 Upload and process files in the 'Upload & Process' tab to see results here")

with tab3:
    st.header("About DeepSeek-OCR Studio")

    st.markdown("""
    ### 🎯 Features

    This application provides a comprehensive interface to **DeepSeek-OCR**, a powerful vision-language model for optical character recognition and document understanding.

    #### 📋 Supported Features:

    - **Multi-format Support**: Process PDFs, images (PNG, JPG, JPEG)
    - **Multiple Resolution Modes**:
      - Tiny: 512×512 (64 vision tokens)
      - Small: 640×640 (100 vision tokens)
      - Base: 1024×1024 (256 vision tokens)
      - Large: 1280×1280 (400 vision tokens)
      - Gundam: Dynamic resolution with crops

    - **Versatile Prompts**:
      - Document to Markdown conversion
      - OCR with layout preservation
      - Figure parsing
      - Image description
      - Custom prompts

    - **Advanced Capabilities**:
      - Table extraction and recognition
      - Bounding box detection
      - Layout analysis
      - Figure extraction
      - Multi-page PDF processing

    - **Visualization & Export**:
      - Markdown output
      - Annotated images with bounding boxes
      - Batch downloads
      - ZIP export for multi-page documents

    ### 🔧 Configuration Options

    - **N-gram No-Repeat**: Prevents repetitive text generation
    - **GPU Memory Control**: Optimize for your hardware
    - **Concurrent Processing**: Batch process multiple pages
    - **DPI Settings**: Control PDF rendering quality

    ### 📚 Use Cases

    Perfect for:
    - Academic paper analysis
    - Presentation extraction
    - Financial reports with tables
    - Technical documentation
    - Scientific figures and charts
    - Meeting notes and slides

    ### 🔗 Resources

    - [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)
    - [Model on HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
    - [GitHub Repository](https://github.com/deepseek-ai/DeepSeek-OCR)

    ### ⚙️ Technical Details

    - **Model**: DeepSeek-OCR
    - **Engine**: vLLM for efficient inference
    - **Framework**: Streamlit for interactive UI
    - **Processing**: Concurrent image preprocessing with ThreadPoolExecutor
    """)

    st.divider()
    st.caption("Built with Streamlit • Powered by DeepSeek-OCR")

# Footer
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption("💡 Tip: Use higher resolution modes for better quality on complex documents with tables and graphics")
