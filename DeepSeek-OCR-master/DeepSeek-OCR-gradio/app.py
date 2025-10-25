import gradio as gr
import torch
from transformers import AutoModel, AutoTokenizer
import os
import tempfile
from PIL import Image, ImageDraw
import re
from typing import Tuple, Optional, Dict, Any

# --- Constants and Configuration ---
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
MODEL_SIZE_CONFIGS = {
    "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "Gundam (Recommended)": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}

TASK_PROMPTS = {
    "📝 Free OCR": "<image>\nFree OCR.",
    "📄 Convert to Markdown": "<image>\n<|grounding|>Convert document to markdown.",
    "📈 Parse Chart": "<image>\nParse chart.",
}

DEFAULT_MODEL_SIZE = "Gundam (Recommended)"
DEFAULT_TASK_TYPE = "📄 Convert to Markdown"
BOUNDING_BOX_PATTERN = re.compile(r"<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>")
BOUNDING_BOX_COLOR = "red"
BOUNDING_BOX_WIDTH = 3
NORMALIZATION_FACTOR = 1000

# --- Global Variables ---
model = None
tokenizer = None
model_gpu = None


def load_model_and_tokenizer() -> None:
    """Load DeepSeek-OCR model and tokenizer at startup."""
    global model, tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval()
    print("✅ Model loaded successfully.")


def move_model_to_gpu() -> None:
    """Move model to GPU if it's not already there."""
    global model_gpu
    if model_gpu is None:
        print("🚀 Moving model to GPU...")
        # Use non-blocking transfer for better performance
        model_gpu = model.cuda().to(torch.bfloat16, non_blocking=True)
        print("✅ Model is now on GPU.")


def find_result_image(path: str) -> Optional[Image.Image]:
    """
    Find pre-generated result image in the specified path.
    
    Args:
        path: Directory path to search for result image
        
    Returns:
        PIL image if found, otherwise None
    """
    for filename in os.listdir(path):
        if "grounding" in filename or "result" in filename:
            try:
                image_path = os.path.join(path, filename)
                return Image.open(image_path)
            except Exception as e:
                print(f"Error opening result image {filename}: {e}")
    return None


def build_prompt(task_type: str, ref_text: str) -> str:
    """
    Build appropriate prompt based on task type and reference text.
    
    Args:
        task_type: OCR task type
        ref_text: Reference text for localization task
        
    Returns:
        Formatted prompt string
    """
    if task_type == "🔍 Locate Object by Reference":
        if not ref_text or ref_text.strip() == "":
            raise gr.Error("For 'localization' task, you must provide reference text to find!")
        return f"<image>\nLocate <|ref|>{ref_text.strip()}<|/ref|> in the image."
    
    return TASK_PROMPTS.get(task_type, TASK_PROMPTS["📝 Free OCR"])


def extract_and_draw_bounding_boxes(text_result: str, original_image: Image.Image) -> Optional[Image.Image]:
    """
    Extract bounding box coordinates from text result and draw them on the image.
    
    Args:
        text_result: OCR text result containing bounding box coordinates
        original_image: Original PIL image to draw on
        
    Returns:
        PIL image with bounding boxes drawn, or None if no coordinates found
    """
    # Use iterator directly to avoid unnecessary list creation
    matches = list(BOUNDING_BOX_PATTERN.finditer(text_result))
    
    if not matches:
        return None
    
    print(f"✅ Found {len(matches)} bounding boxes. Drawing on original image.")
    
    # Create a copy of the original image for drawing
    image_with_bboxes = original_image.copy()
    draw = ImageDraw.Draw(image_with_bboxes)
    w, h = original_image.size
    
    # Pre-calculate scale factors for better performance
    w_scale = w / NORMALIZATION_FACTOR
    h_scale = h / NORMALIZATION_FACTOR
    
    for match in matches:
        # Extract and scale coordinates more efficiently
        coords = tuple(int(c) for c in match.groups())
        x1_norm, y1_norm, x2_norm, y2_norm = coords
        
        # Scale normalized coordinates using pre-calculated factors
        x1 = int(x1_norm * w_scale)
        y1 = int(y1_norm * h_scale)
        x2 = int(x2_norm * w_scale)
        y2 = int(y2_norm * h_scale)
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=BOUNDING_BOX_COLOR, width=BOUNDING_BOX_WIDTH)
    
    return image_with_bboxes


def run_inference(prompt: str, image_path: str, output_path: str, config: Dict[str, Any]) -> str:
    """
    Run model inference with given parameters.
    
    Args:
        prompt: Formatted prompt for the model
        image_path: Path to input image
        output_path: Directory path for output files
        config: Model configuration dictionary
        
    Returns:
        Text result from the model
    """
    print(f"🏃 Running inference with prompt: {prompt}")
    text_result = model_gpu.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=output_path,
        base_size=config["base_size"],
        image_size=config["image_size"],
        crop_mode=config["crop_mode"],
        save_results=True,
        test_compress=True,
        eval_mode=True,
    )
    print(f"====\n📄 Text Result: {text_result}\n====")
    return text_result


def process_ocr_task(image: Optional[Image.Image], model_size: str, task_type: str, ref_text: str) -> Tuple[str, Optional[Image.Image]]:
    """
    Process image with DeepSeek-OCR to support all tasks.
    
    Args:
        image: Input PIL image
        model_size: Model size configuration
        task_type: OCR task type
        ref_text: Reference text for localization task
        
    Returns:
        (text_result, result_image) tuple
    """
    if image is None:
        return "Please upload an image first.", None
    
    # Ensure model is on GPU
    move_model_to_gpu()
    
    # Build prompt based on task type
    prompt = build_prompt(task_type, ref_text)
    
    # Get model configuration
    config = MODEL_SIZE_CONFIGS.get(model_size, MODEL_SIZE_CONFIGS[DEFAULT_MODEL_SIZE])
    
    with tempfile.TemporaryDirectory() as output_path:
        # Save temporary image with optimized format
        temp_image_path = os.path.join(output_path, "temp_image.png")
        # Use optimize=True for better compression
        image.save(temp_image_path, optimize=True)
        
        # Run inference
        text_result = run_inference(prompt, temp_image_path, output_path, config)
        
        # Try to extract and draw bounding boxes from text result
        result_image = extract_and_draw_bounding_boxes(text_result, image)
        
        # If no bounding boxes found, fall back to pre-generated result image
        if result_image is None:
            print("⚠️ No bounding box coordinates found in text result. Falling back to searching for result image file.")
            result_image = find_result_image(output_path)
        
        return text_result, result_image


def toggle_ref_text_visibility(task: str) -> gr.Textbox:
    """
    Toggle reference text input visibility based on task type.
    
    Args:
        task: Selected task type
        
    Returns:
        Updated Textbox component
    """
    return gr.Textbox(visible=True) if task == "🔍 Locate Object by Reference" else gr.Textbox(visible=False)


def create_ui() -> gr.Blocks:
    """
    Create and configure Gradio user interface.
    
    Returns:
        Configured Gradio Blocks interface
    """
    with gr.Blocks(title="🐳DeepSeek-OCR🐳", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🐳 DeepSeek-OCR Complete Demo 🐳
            **💡 How to use:**
            1.  **Upload an image** using the upload box.
            2.  Select a **resolution**. `Gundam` is recommended for most documents.
            3.  Select a **task type**:
                - **📝 Free OCR**: Extract raw text from images.
                - **📄 Convert to Markdown**: Convert documents to Markdown while preserving structure.
                - **📈 Parse Chart**: Extract structured data from charts and graphs.
                - **🔍 Locate Object by Reference**: Find specific objects/text.
            4. If this tool is helpful, please give it a thumbs up! 🙏 ❤️
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="🖼️ Upload Image", sources=["upload", "clipboard"])
                model_size = gr.Dropdown(
                    choices=list(MODEL_SIZE_CONFIGS.keys()),
                    value=DEFAULT_MODEL_SIZE,
                    label="⚙️ Resolution Size"
                )
                task_type = gr.Dropdown(
                    choices=list(TASK_PROMPTS.keys()) + ["🔍 Locate Object by Reference"],
                    value=DEFAULT_TASK_TYPE,
                    label="🚀 Task Type"
                )
                ref_text_input = gr.Textbox(
                    label="📝 Reference Text (for localization task)",
                    placeholder="e.g.: teacher, 20-10, a red car...",
                    visible=False
                )
                submit_btn = gr.Button("Process Image", variant="primary")

            with gr.Column(scale=2):
                output_text = gr.Textbox(label="📄 Text Result", lines=15, show_copy_button=True)
                output_image = gr.Image(label="🖼️ Image Result (if any)", type="pil")

        # UI interaction logic
        task_type.change(fn=toggle_ref_text_visibility, inputs=task_type, outputs=ref_text_input)
        submit_btn.click(
            fn=process_ocr_task,
            inputs=[image_input, model_size, task_type, ref_text_input],
            outputs=[output_text, output_image]
        )

        # Example images and tasks
        gr.Examples(
            examples=[
                ["doc_markdown.png", "will upload", "📄 will upload", ""],
            ],
            inputs=[image_input, model_size, task_type, ref_text_input],
            outputs=[output_text, output_image],
            fn=process_ocr_task,
            cache_examples=False,  # Disable cache to ensure examples run every time
        )
    
    return demo


def main() -> None:
    """Main function to initialize and start the application."""
    # Load model at startup
    load_model_and_tokenizer()
    
    # Create examples directory if it doesn't exist
    if not os.path.exists("examples"):
        os.makedirs("examples")
    
    # Create and launch UI
    demo = create_ui()
    demo.queue(max_size=20).launch(share=True)


if __name__ == "__main__":
    main()