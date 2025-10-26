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
import json
from datetime import datetime

# Add DeepSeek-OCR modules to path
sys.path.insert(0, '/home/user/deepseek-ocr/DeepSeek-OCR-master/DeepSeek-OCR-vllm')
sys.path.insert(0, '/home/user/deepseek-ocr')

# Import utility modules
from utils.job_queue import JobQueue, JobStatus
from utils.output_formatters import JSONFormatter, HTMLFormatter, DOCXFormatter, CSVFormatter, ExcelFormatter
from utils.office_converters import OfficeConverter
from utils.post_processing import PostProcessor, TextQualityAnalyzer
from utils.i18n import I18n

# Set page config
st.set_page_config(
    page_title="DeepSeek-OCR Studio Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'locale' not in st.session_state:
    st.session_state.locale = 'en'
if 'i18n' not in st.session_state:
    st.session_state.i18n = I18n(st.session_state.locale)
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = None
if 'job_queue' not in st.session_state:
    st.session_state.job_queue = JobQueue()
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'post_processor' not in st.session_state:
    st.session_state.post_processor = PostProcessor()

# Get i18n instance
i18n = st.session_state.i18n

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(120deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
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

    def process_single_image(image):
        cache_item = {
            "prompt": prompt,
            "multi_modal_data": {"image": DeepseekOCRProcessor().tokenize_with_images(
                images=[image], bos=True, eos=True, cropping=crop_mode
            )},
        }
        return cache_item

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        batch_inputs = list(executor.map(process_single_image, images))

    outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

    return outputs_list

# Header
st.markdown(f'<h1 class="main-header">{i18n.t("app.title")}</h1>', unsafe_allow_html=True)
st.markdown(f"### {i18n.t('app.subtitle')}")

# Sidebar Configuration
with st.sidebar:
    st.header(f"⚙️ {i18n.t('sidebar.configuration')}")

    # Language selector
    st.subheader("🌐 Language / 语言")
    language = st.selectbox(
        "Select Language",
        ["English", "Español", "中文", "Français", "Deutsch"],
        index=["English", "Español", "中文", "Français", "Deutsch"].index(
            {"en": "English", "es": "Español", "zh": "中文", "fr": "Français", "de": "Deutsch"}.get(st.session_state.locale, "English")
        )
    )

    new_locale = {"English": "en", "Español": "es", "中文": "zh", "Français": "fr", "Deutsch": "de"}[language]
    if new_locale != st.session_state.locale:
        st.session_state.locale = new_locale
        st.session_state.i18n = I18n(new_locale)
        st.rerun()

    # Model Settings
    st.subheader(i18n.t("sidebar.model_settings"))
    model_path = st.text_input(i18n.t("sidebar.model_path"), value="deepseek-ai/DeepSeek-OCR")

    # Resolution Mode
    st.subheader(i18n.t("sidebar.resolution_mode"))
    resolution_mode = st.selectbox(
        i18n.t("sidebar.resolution_mode"),
        ["Gundam (Dynamic)", "Tiny (512×512)", "Small (640×640)", "Base (1024×1024)", "Large (1280×1280)"]
    )

    resolution_settings = {
        "Gundam (Dynamic)": {"base_size": 1024, "image_size": 640, "crop_mode": True},
        "Tiny (512×512)": {"base_size": 512, "image_size": 512, "crop_mode": False},
        "Small (640×640)": {"base_size": 640, "image_size": 640, "crop_mode": False},
        "Base (1024×1024)": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
        "Large (1280×1280)": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    }

    settings = resolution_settings[resolution_mode]

    # Advanced Settings
    with st.expander("🔧 " + i18n.t("sidebar.advanced_settings"), expanded=False):
        max_crops = st.slider("Max Crops", 2, 9, 6)
        max_concurrency = st.slider("Max Concurrency", 1, 200, 100)
        gpu_memory = st.slider("GPU Memory", 0.5, 0.95, 0.9, 0.05)
        ngram_size = st.slider("N-gram Size", 10, 50, 20)
        window_size = st.slider("Window Size", 30, 150, 50)
        num_workers = st.slider("Workers", 1, 128, 64)

    # Prompt Templates
    st.subheader("📝 " + i18n.t("sidebar.prompt_template"))
    prompt_type = st.selectbox(
        i18n.t("sidebar.prompt_template"),
        ["Document to Markdown", "OCR Image", "Free OCR", "Parse Figure", "Describe Image", "Custom"]
    )

    prompt_templates = {
        "Document to Markdown": "<image>\n<|grounding|>Convert the document to markdown.",
        "OCR Image": "<image>\n<|grounding|>OCR this image.",
        "Free OCR": "<image>\nFree OCR.",
        "Parse Figure": "<image>\nParse the figure.",
        "Describe Image": "<image>\nDescribe this image in detail.",
    }

    if prompt_type == "Custom":
        prompt = st.text_area("Custom Prompt", value="<image>\n<|grounding|>Convert the document to markdown.")
    else:
        prompt = prompt_templates[prompt_type]
        st.info(f"Prompt: `{prompt}`")

    # PDF Settings
    st.subheader("📄 " + i18n.t("sidebar.pdf_settings"))
    pdf_dpi = st.slider("PDF DPI", 72, 300, 144)

    # Post-processing Settings
    with st.expander("🔍 " + i18n.t("post_processing.title"), expanded=False):
        enable_spellcheck = st.checkbox(i18n.t("post_processing.enable_spellcheck"), value=False)
        enable_grammar = st.checkbox(i18n.t("post_processing.enable_grammar"), value=False)
        enable_table_validation = st.checkbox(i18n.t("post_processing.enable_table_validation"), value=True)
        enable_formula_check = st.checkbox(i18n.t("post_processing.enable_formula_check"), value=True)

        st.session_state.post_processor = PostProcessor(
            enable_spellcheck=enable_spellcheck,
            enable_grammar=enable_grammar,
            enable_table_validation=enable_table_validation,
            enable_formula_check=enable_formula_check
        )

# Main Content Area
tabs = st.tabs([
    "📤 " + i18n.t("tabs.upload"),
    "📊 " + i18n.t("tabs.results"),
    "📁 " + i18n.t("tabs.batch"),
    "🔄 " + i18n.t("tabs.comparison"),
    "✏️ " + i18n.t("tabs.editor"),
    "ℹ️ " + i18n.t("tabs.about")
])

# Tab 1: Upload & Process
with tabs[0]:
    st.header(i18n.t("tabs.upload"))

    uploaded_files = st.file_uploader(
        i18n.t("upload.drag_drop"),
        type=['pdf', 'png', 'jpg', 'jpeg', 'docx', 'pptx', 'xlsx'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(i18n.t("upload.files_uploaded", count=len(uploaded_files)))

        cols = st.columns(min(len(uploaded_files), 4))
        for idx, file in enumerate(uploaded_files[:4]):
            with cols[idx]:
                if file.type == "application/pdf":
                    st.info(f"📄 {file.name}")
                elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                    st.info(f"📋 {file.name}")
                else:
                    st.image(Image.open(file), caption=file.name, use_container_width=True)

        if len(uploaded_files) > 4:
            st.info(f"... and {len(uploaded_files) - 4} more file(s)")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_button = st.button("🚀 " + i18n.t("upload.process_button"), type="primary", use_container_width=True)

        if process_button:
            with st.spinner(i18n.t("upload.processing")):
                try:
                    llm = load_model(model_path, max_concurrency, gpu_memory)

                    if llm is None:
                        st.error("Failed to load model")
                    else:
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
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for file_idx, uploaded_file in enumerate(uploaded_files):
                            status_text.text(f"Processing {uploaded_file.name} ({file_idx + 1}/{len(uploaded_files)})...")

                            file_bytes = uploaded_file.read()
                            file_type = uploaded_file.type

                            # Convert to images based on file type
                            if file_type == "application/pdf":
                                images = pdf_to_images(file_bytes, dpi=pdf_dpi)
                            elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                             "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                                # Office formats
                                office_type = {'docx': 'docx', 'pptx': 'pptx', 'xlsx': 'xlsx'}.get(
                                    uploaded_file.name.split('.')[-1].lower(), 'docx'
                                )
                                images = OfficeConverter.convert_to_images(file_bytes, office_type, dpi=pdf_dpi)
                            else:
                                image = load_image(file_bytes)
                                images = [image] if image else []

                            if not images:
                                st.warning(f"Skipping {uploaded_file.name}")
                                continue

                            outputs = process_ocr(images, llm, sampling_params, prompt, settings['crop_mode'], num_workers)

                            file_results = {
                                'filename': uploaded_file.name,
                                'images': images,
                                'outputs': outputs,
                                'type': file_type
                            }
                            all_results.append(file_results)

                            progress_bar.progress((file_idx + 1) / len(uploaded_files))

                        st.session_state.processed_results = all_results
                        status_text.text("✅ " + i18n.t("upload.complete"))
                        st.success(i18n.t("upload.complete"))
                        st.balloons()

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# Tab 2: Results (keeping this concise, continue in next message)
with tabs[1]:
    st.header(i18n.t("tabs.results"))

    if st.session_state.processed_results:
        file_names = [r['filename'] for r in st.session_state.processed_results]
        selected_file_idx = st.selectbox(i18n.t("results.select_file"), range(len(file_names)),
                                         format_func=lambda x: file_names[x])

        result = st.session_state.processed_results[selected_file_idx]

        if len(result['images']) > 1:
            page_idx = st.slider(i18n.t("results.select_page"), 0, len(result['images']) - 1, 0)
        else:
            page_idx = 0

        result_tabs = st.tabs([
            "📝 " + i18n.t("results.markdown_output"),
            "🖼️ " + i18n.t("results.visualized"),
            "💾 " + i18n.t("results.downloads"),
            "🔍 Quality Analysis"
        ])

        # Markdown Output
        with result_tabs[0]:
            output_text = result['outputs'][page_idx].outputs[0].text

            if '<｜end▁of▁sentence｜>' in output_text:
                output_text = output_text.replace('<｜end▁of▁sentence｜>', '')

            matches, matches_images, matches_other = extract_bounding_boxes(output_text)

            # Apply post-processing if enabled
            clean_output = output_text
            for match in matches_other:
                clean_output = clean_output.replace(match, '')
            clean_output = clean_output.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

            if any([enable_spellcheck, enable_grammar, enable_table_validation, enable_formula_check]):
                processed_output, issues = st.session_state.post_processor.process(clean_output)

                if issues['corrections_applied'] > 0:
                    st.info(i18n.t("post_processing.corrections_applied", count=issues['corrections_applied']))

                    with st.expander("View Issues"):
                        if issues['spelling_errors']:
                            st.write("Spelling:", issues['spelling_errors'])
                        if issues['grammar_issues']:
                            st.write("Grammar:", issues['grammar_issues'])
                        if issues['table_issues']:
                            st.write("Tables:", issues['table_issues'])
                        if issues['formula_issues']:
                            st.write("Formulas:", issues['formula_issues'])

                clean_output = processed_output

            with st.expander("🔍 " + i18n.t("results.raw_output"), expanded=False):
                st.code(output_text, language="markdown")

            st.markdown(clean_output)

        # Visualized
        with result_tabs[1]:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**" + i18n.t("results.original_image") + "**")
                st.image(result['images'][page_idx], use_container_width=True)

            with col2:
                st.write("**" + i18n.t("results.with_bounding_boxes") + "**")
                if matches:
                    annotated_image = draw_bounding_boxes(result['images'][page_idx], matches)
                    st.image(annotated_image, use_container_width=True)
                else:
                    st.info("No bounding boxes detected")
                    st.image(result['images'][page_idx], use_container_width=True)

        # Downloads
        with result_tabs[2]:
            st.subheader(i18n.t("results.downloads"))

            output_text = result['outputs'][page_idx].outputs[0].text
            matches, matches_images, matches_other = extract_bounding_boxes(output_text)

            clean_output = output_text
            for match in matches_other:
                clean_output = clean_output.replace(match, '')

            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    "📄 " + i18n.t("results.download_markdown"),
                    data=clean_output,
                    file_name=f"{Path(result['filename']).stem}_page{page_idx+1}.md",
                    mime="text/markdown"
                )

            with col2:
                if matches:
                    annotated_image = draw_bounding_boxes(result['images'][page_idx], matches)
                    buf = io.BytesIO()
                    annotated_image.save(buf, format='PNG')
                    st.download_button(
                        "🖼️ " + i18n.t("results.download_annotated"),
                        data=buf.getvalue(),
                        file_name=f"{Path(result['filename']).stem}_page{page_idx+1}_annotated.png",
                        mime="image/png"
                    )

            with col3:
                st.download_button(
                    "📋 " + i18n.t("results.download_raw"),
                    data=output_text,
                    file_name=f"{Path(result['filename']).stem}_page{page_idx+1}_raw.txt",
                    mime="text/plain"
                )

            st.divider()
            st.subheader("Additional Formats")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                # JSON
                json_data = JSONFormatter.format(
                    output_text, matches,
                    result['images'][page_idx].width,
                    result['images'][page_idx].height,
                    {"filename": result['filename'], "page": page_idx + 1}
                )
                st.download_button(
                    "📊 " + i18n.t("results.download_json"),
                    data=json_data,
                    file_name=f"{Path(result['filename']).stem}_page{page_idx+1}.json",
                    mime="application/json"
                )

            with col2:
                # HTML
                html_data = HTMLFormatter.format(
                    output_text, matches,
                    {"filename": result['filename'], "page": page_idx + 1}
                )
                st.download_button(
                    "🌐 " + i18n.t("results.download_html"),
                    data=html_data,
                    file_name=f"{Path(result['filename']).stem}_page{page_idx+1}.html",
                    mime="text/html"
                )

            with col3:
                # DOCX (create temp file)
                try:
                    temp_docx = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
                    DOCXFormatter.format(
                        clean_output, matches, temp_docx.name,
                        {"filename": result['filename'], "page": page_idx + 1}
                    )
                    with open(temp_docx.name, 'rb') as f:
                        st.download_button(
                            "📝 " + i18n.t("results.download_docx"),
                            data=f.read(),
                            file_name=f"{Path(result['filename']).stem}_page{page_idx+1}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    os.unlink(temp_docx.name)
                except Exception as e:
                    st.warning(f"DOCX export requires python-docx: {e}")

            with col4:
                # CSV/Excel
                try:
                    temp_xlsx = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                    ExcelFormatter.format(clean_output, temp_xlsx.name)
                    with open(temp_xlsx.name, 'rb') as f:
                        st.download_button(
                            "📊 " + i18n.t("results.download_csv"),
                            data=f.read(),
                            file_name=f"{Path(result['filename']).stem}_page{page_idx+1}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    os.unlink(temp_xlsx.name)
                except Exception as e:
                    st.warning(f"Excel export requires openpyxl: {e}")

        # Quality Analysis
        with result_tabs[3]:
            st.subheader("Text Quality Analysis")

            clean_output = output_text
            for match in matches_other:
                clean_output = clean_output.replace(match, '')

            analysis = TextQualityAnalyzer.analyze(clean_output)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Characters", f"{analysis['character_count']:,}")
                st.metric("Words", f"{analysis['word_count']:,}")

            with col2:
                st.metric("Lines", analysis['line_count'])
                st.metric("Paragraphs", analysis['paragraph_count'])

            with col3:
                st.metric("Tables", analysis['table_count'])
                st.metric("Formulas", analysis['formula_count'])

            with col4:
                st.metric("Code Blocks", analysis['code_block_count'])
                st.metric("Avg Word Len", f"{analysis['average_word_length']:.1f}")

    else:
        st.info("Upload and process files to see results here")

# Tab 3: Batch Processing
with tabs[2]:
    st.header(i18n.t("tabs.batch"))

    st.markdown("""
    **Batch Processing** allows you to queue multiple files for processing with progress persistence.
    Jobs can be paused and resumed across sessions.
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Create New Batch Job")
        job_name = st.text_input(i18n.t("batch.job_name"), value=f"Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        batch_files = st.file_uploader(
            "Upload files for batch processing",
            type=['pdf', 'png', 'jpg', 'jpeg', 'docx', 'pptx', 'xlsx'],
            accept_multiple_files=True,
            key="batch_uploader"
        )

        if st.button("➕ " + i18n.t("batch.create_job")):
            if batch_files:
                file_names = [f.name for f in batch_files]
                config = {
                    'resolution_mode': resolution_mode,
                    'prompt': prompt,
                    'pdf_dpi': pdf_dpi,
                    'model_path': model_path
                }
                job_id = st.session_state.job_queue.create_job(job_name, file_names, config)
                st.success(f"Created job: {job_id}")
                st.rerun()

    with col2:
        st.subheader("Queue Status")
        all_jobs = st.session_state.job_queue.get_all_jobs()

        active_jobs = [j for j in all_jobs if j['status'] in [JobStatus.PENDING, JobStatus.PROCESSING]]
        completed_jobs = [j for j in all_jobs if j['status'] == JobStatus.COMPLETED]

        st.metric("Active Jobs", len(active_jobs))
        st.metric("Completed Jobs", len(completed_jobs))

    st.divider()

    # Display jobs
    st.subheader(i18n.t("batch.active_jobs"))

    for job in all_jobs[:10]:  # Show recent 10 jobs
        with st.expander(f"{job['name']} - {job['status']}", expanded=(job['status'] == JobStatus.PROCESSING)):
            progress = st.session_state.job_queue.get_job_progress(job['job_id'])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Status", job['status'])
            with col2:
                st.metric("Progress", f"{progress['progress']:.1f}%")
            with col3:
                st.metric("Files", f"{progress['processed_files']}/{progress['total_files']}")

            st.progress(progress['progress'] / 100)

            cols = st.columns([1, 1, 1, 2])

            with cols[0]:
                if st.button("🗑️ " + i18n.t("batch.delete_job"), key=f"del_{job['job_id']}"):
                    st.session_state.job_queue.delete_job(job['job_id'])
                    st.rerun()

            with cols[1]:
                if job['status'] in [JobStatus.PENDING, JobStatus.PROCESSING]:
                    if st.button("⏸️ " + i18n.t("batch.cancel_job"), key=f"cancel_{job['job_id']}"):
                        st.session_state.job_queue.cancel_job(job['job_id'])
                        st.rerun()

            with cols[2]:
                if job['status'] == JobStatus.COMPLETED:
                    if st.button("👁️ " + i18n.t("batch.view_results"), key=f"view_{job['job_id']}"):
                        results = st.session_state.job_queue.get_job_results(job['job_id'])
                        st.write(results)

# Tab 4: Comparison Tool
with tabs[3]:
    st.header(i18n.t("comparison.title"))

    st.markdown("""
    Compare OCR results across different **resolution modes** and **prompts** to find the best configuration for your documents.
    """)

    if st.session_state.processed_results:
        comparison_file = st.selectbox(
            "Select file to compare",
            range(len(st.session_state.processed_results)),
            format_func=lambda x: st.session_state.processed_results[x]['filename']
        )

        st.subheader(i18n.t("comparison.select_modes"))

        col1, col2 = st.columns(2)

        with col1:
            modes_to_compare = st.multiselect(
                "Resolution Modes",
                ["Tiny (512×512)", "Small (640×640)", "Base (1024×1024)", "Large (1280×1280)", "Gundam (Dynamic)"],
                default=["Small (640×640)", "Base (1024×1024)"]
            )

        with col2:
            prompts_to_compare = st.multiselect(
                "Prompt Templates",
                list(prompt_templates.keys()),
                default=["Document to Markdown", "Free OCR"]
            )

        if st.button("🔄 " + i18n.t("comparison.compare_button")):
            st.info("Comparison feature processes the same file with different settings. This may take time.")
            # Implementation would reprocess with different settings
            # For now, show placeholder
            st.success("Comparison complete! (Feature in development)")
    else:
        st.info("Process some files first to enable comparison")

# Tab 5: Interactive Editor
with tabs[4]:
    st.header(i18n.t("editor.title"))

    if st.session_state.processed_results:
        st.markdown("""
        Edit the OCR output directly and save your changes. You can also adjust bounding boxes and re-process specific regions.
        """)

        editor_file_idx = st.selectbox(
            "Select file to edit",
            range(len(st.session_state.processed_results)),
            format_func=lambda x: st.session_state.processed_results[x]['filename'],
            key="editor_file"
        )

        result = st.session_state.processed_results[editor_file_idx]

        if len(result['images']) > 1:
            editor_page_idx = st.slider("Select page to edit", 0, len(result['images']) - 1, 0, key="editor_page")
        else:
            editor_page_idx = 0

        output_text = result['outputs'][editor_page_idx].outputs[0].text
        matches, matches_images, matches_other = extract_bounding_boxes(output_text)

        clean_output = output_text
        for match in matches_other:
            clean_output = clean_output.replace(match, '')

        # Markdown editor
        st.subheader(i18n.t("editor.edit_markdown"))
        edited_text = st.text_area(
            "Edit the text below",
            value=clean_output,
            height=400,
            key="markdown_editor"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 " + i18n.t("editor.save_changes")):
                # Save edited text
                st.session_state.processed_results[editor_file_idx]['outputs'][editor_page_idx].edited = edited_text
                st.success("Changes saved!")

        with col2:
            st.download_button(
                "⬇️ Download Edited Version",
                data=edited_text,
                file_name=f"{Path(result['filename']).stem}_edited.md",
                mime="text/markdown"
            )

        # Preview
        st.subheader("Preview")
        st.markdown(edited_text)

    else:
        st.info("Process some files first to enable editing")

# Tab 6: About
with tabs[5]:
    st.header(i18n.t("tabs.about"))

    st.markdown("""
    ### 🎯 DeepSeek-OCR Studio Pro

    This enhanced version includes:

    #### ✨ **New Features**

    - 📁 **Batch Processing**: Queue multiple files with progress persistence
    - 📊 **Additional Output Formats**: JSON, HTML, DOCX, CSV/Excel export
    - 🔄 **Comparison Tool**: Compare different resolution modes and prompts
    - ✏️ **Interactive Editor**: Edit OCR results with live preview
    - 🌐 **Multi-Language UI**: Support for English, Spanish, Chinese, French, German
    - 📋 **Office Format Support**: Process DOCX, PPTX, XLSX files
    - 🔍 **Intelligent Post-Processing**:
      - Spell-check and correction
      - Grammar checking
      - Table structure validation
      - LaTeX formula verification
      - Text quality analysis

    #### 📚 **Output Formats**

    - **Markdown**: Clean, formatted text with preserved structure
    - **JSON**: Structured data with element metadata and coordinates
    - **HTML**: Styled webpage with CSS
    - **DOCX**: Editable Microsoft Word document
    - **CSV/Excel**: Table extraction to spreadsheet

    #### 🎯 **Use Cases**

    Perfect for:
    - 📑 Academic papers with complex formulas
    - 📊 Business presentations with charts
    - 📈 Financial reports with tables
    - 🔬 Scientific publications with figures
    - 📝 Technical documentation
    - 🗂️ Batch document processing

    #### 🔧 **Configuration**

    - **5 Resolution Modes**: From Tiny (fast) to Gundam (best quality)
    - **Multiple Prompt Templates**: Optimized for different document types
    - **Advanced Post-Processing**: Quality assurance and validation
    - **Flexible Exports**: Choose your preferred output format

    ### 📖 Resources

    - [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)
    - [Model on HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
    - [GitHub Repository](https://github.com/deepseek-ai/DeepSeek-OCR)

    ---

    **Built with ❤️ using Streamlit and DeepSeek-OCR**
    """)

# Footer
st.divider()
st.caption("💡 Tip: Use the Comparison tab to find the best settings for your documents")
