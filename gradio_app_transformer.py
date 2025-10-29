#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-OCR Gradio App using Transformers with BF16 and FlashAttention Support
Optimized version for modern GPUs with configurable precision and attention settings
"""

import os
import tempfile
import gradio as gr
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import json
import base64
from io import BytesIO

# =============================================================================
# CONFIGURATION SETTINGS - Modify these at the beginning
# =============================================================================

# GPU and CUDA settings
CUDA_DEVICE = '0'  # Change to '1', '2', etc. for different GPUs
USE_FLASH_ATTENTION = True  # Set to False if you have issues with FlashAttention
USE_BF16 = True  # Set to False to use FP32 instead
USE_SAFETENSORS = True  # Set to False to use regular tensors

# Model settings
MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'
TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

# Performance settings
CLEAR_CACHE_ON_INIT = True  # Clear GPU cache on initialization

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

# Configure FlashAttention
if USE_FLASH_ATTENTION:
    os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "0"
    print("✅ FlashAttention enabled")
else:
    os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
    print("⚠️ FlashAttention disabled")

# Clear GPU cache if requested
if CLEAR_CACHE_ON_INIT and torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("🧹 GPU cache cleared")

# Global variables
model = None
tokenizer = None

def initialize_model():
    """Initialize the DeepSeek-OCR model with BF16 and FlashAttention support"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        print("🚀 Initializing DeepSeek-OCR model with optimizations...")
    
        # Load tokenizer
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # Load model with optimizations
        print("🤖 Loading model...")
        model = AutoModel.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True, 
            use_safetensors=USE_SAFETENSORS,
            torch_dtype=TORCH_DTYPE,
            _attn_implementation='flash_attention_2' if USE_FLASH_ATTENTION else None
        )
        
        # Set model to evaluation mode
        model = model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"🎯 Model moved to GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ CUDA not available, using CPU")
                    
        # Clear cache after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Model initialization completed!")
        print(f"💾 Model memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
    
    return model, tokenizer

def process_ocr_transformers(image, task_type, custom_prompt, base_size, image_size, crop_mode, min_crops, max_crops):
    """Process OCR using Transformers with optimizations"""
    global model, tokenizer
    
    try:
        # Initialize model if not already done
        if model is None or tokenizer is None:
            initialize_model()
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Temporary directory: {temp_dir}")
            
            # Save uploaded image
            if isinstance(image, str):
                # If image is a file path
                image_path = image
            else:
                # If image is a PIL Image
                image_path = os.path.join(temp_dir, "input_image.jpg")
                image.save(image_path)
            
            # Define prompts based on task type
            prompts = {
                "OCR with Bounding Boxes": "<image>\n<|grounding|>OCR with bounding boxes. ",
                "Document to Markdown": "<image>\n<|grounding|>Convert the document to markdown. ",
                "Table Recognition": "<image>\n<|grounding|>Extract table structure and content. ",
                "Formula Recognition": "<image>\n<|grounding|>Recognize mathematical formulas. ",
                "Custom Task": f"<image>\n<|grounding|>{custom_prompt} ",
            }
            
            prompt = prompts.get(task_type, prompts["OCR with Bounding Boxes"])
            
            print(f"🎯 Task type: {task_type}")
            print(f"💬 Used prompt: {prompt}")
            
            
            # Call model inference
            print("🔄 Starting model inference...")
            result = model.infer(
                tokenizer, 
                prompt=prompt, 
                image_file=image_path, 
                output_path=temp_dir, 
                base_size=base_size, 
                image_size=image_size, 
                crop_mode=crop_mode, 
                save_results=True, 
                test_compress=True
            )
            
            # Process results
            print("📊 Processing results...")
            
            # Check for generated files in temp_dir
            result_mmd_path = os.path.join(temp_dir, "result.mmd")
            result_image_path = os.path.join(temp_dir, "result_with_boxes.jpg")
            
            # Read the markdown result
            output_text = ""
            result_image = None
            
            if os.path.exists(result_mmd_path):
                with open(result_mmd_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                    output_text = f"## 📝 OCR Results:\n\n{markdown_content}"
                print("✅ Markdown result loaded")
            else:
                # Fallback to original result processing
                if isinstance(result, dict):
                    # Extract text and other information
                    extracted_text = result.get('text', '')
                    bounding_boxes = result.get('boxes', [])
                    
                    # Format output
                    output_text = f"## 📝 Extracted Text:\n\n{extracted_text}\n\n"
                    
                    if bounding_boxes:
                        output_text += f"## 🎯 Detected Bounding Boxes ({len(bounding_boxes)} total):\n\n"
                        for i, box in enumerate(bounding_boxes[:10]):  # Show first 10 boxes
                            output_text += f"**Box {i+1}**: {box}\n"
                    
                elif isinstance(result, str):
                    # If result is a string, try to parse it
                    print(f"📄 String result: {result}")
                    
                    # Check if it contains bounding box information
                    if "<|ref|>" in result and "<|det|>" in result:
                        # Parse the structured output
                        lines = result.split('\n')
                        text_parts = []
                        boxes = []
                        
                        for line in lines:
                            if "<|ref|>" in line and "<|det|>" in line:
                                # Extract text and coordinates
                                ref_start = line.find("<|ref|>") + 7
                                ref_end = line.find("<|/ref|>")
                                det_start = line.find("<|det|>") + 7
                                det_end = line.find("<|/det|>")
                                
                                if ref_start > 6 and ref_end > ref_start and det_start > ref_end and det_end > det_start:
                                    text = line[ref_start:ref_end]
                                    coords = line[det_start:det_end]
                                    text_parts.append(text)
                                    boxes.append(coords)
                        
                        # Format the output
                        output_text = "## 📝 Extracted Text:\n\n"
                        if text_parts:
                            output_text += " ".join(text_parts) + "\n\n"
                        
                        if boxes:
                            output_text += f"## 🎯 Detected Bounding Boxes ({len(boxes)} total):\n\n"
                            for i, (text, box) in enumerate(zip(text_parts, boxes)):
                                output_text += f"**{text}**: {box}\n"
                        
                    else:
                        # Regular text result
                        output_text = f"## 📝 Recognition Results:\n\n{result}"
                else:
                    output_text = f"## 📝 Recognition Results:\n\n{str(result)}"
            
            # Load the visualization image if it exists
            if os.path.exists(result_image_path):
                try:
                    from PIL import Image
                    result_image = Image.open(result_image_path)
                    print(f"🖼️ Loaded visualization image: {result_image.size}")
                except Exception as e:
                    print(f"❌ Error loading visualization image: {e}")
                    result_image = None
            
            # Clear cache after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"💾 Memory after inference: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            return output_text, result_image
                
    except Exception as e:
        error_msg = f"❌ Error occurred during processing: {str(e)}"
        print(error_msg)
        return error_msg, None

def create_gradio_interface():
    """Create the Gradio interface with optimizations info"""
    
    # Custom CSS for friendly, readable fonts
    custom_css = """
    /* Import Google Fonts for better readability */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Apply friendly fonts throughout the interface */
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Make text more readable */
    .gradio-container * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Improve button readability */
    button {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em !important;
    }
    
    /* Improve input field readability */
    input, textarea, select {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-size: 14px !important;
        line-height: 1.5 !important;
    }
    
    /* Improve label readability */
    label {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    
    /* Improve markdown text readability */
    .markdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        line-height: 1.6 !important;
        font-size: 15px !important;
    }
    
    /* Improve textbox readability */
    .textbox {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        line-height: 1.5 !important;
    }
    
    /* Improve dropdown readability */
    .dropdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }
    
    /* Make headings more readable */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Status indicator styling */
    .status-indicator {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        display: inline-block;
        margin: 4px;
    }
    """
    
    with gr.Blocks(title="DeepSeek-OCR (BF16 + FlashAttention)", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.Markdown(
            f"""
            # 🔍 DeepSeek-OCR (Optimized Version)

            **💡 Tip**: Press Enter anywhere to start processing!
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Language selector
                language = gr.Dropdown(
                    choices=["中文", "English"],
                    value="English",
                    label="Language"
                )
                
                # Image input
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=300
                )
                
                # Task type selector
                task_type = gr.Dropdown(
                    choices=[
                        "Document to Markdown", 
                        "OCR with Bounding Boxes",
                        "Table Recognition",
                        "Formula Recognition",
                        "Custom Task"
                    ],
                    value="Document to Markdown",
                    label="Task Type"
                )
                
                # Custom prompt (only shown for custom task)
                custom_prompt = gr.Textbox(
                    label="Custom Prompt",
                    placeholder="Please enter your custom task description...",
                    visible=False
                )
                
                # Parameters
                with gr.Accordion("Advanced Parameters", open=False):
                    base_size = gr.Slider(
                        minimum=512,
                        maximum=2048,
                        value=1024,
                        step=64,
                        label="Base Size"
                    )
                    
                    image_size = gr.Slider(
                        minimum=320,
                        maximum=1280,
                        value=640,
                        step=32,
                        label="Image Size"
                    )
                    
                    crop_mode = gr.Checkbox(
                        value=True,
                        label="Crop Mode"
                    )
                    
                    min_crops = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=2,
                        step=1,
                        label="Min Crops"
                    )
                    
                    max_crops = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=6,
                        step=1,
                        label="Max Crops"
                    )
                
                # Process button with Enter key support
                process_btn = gr.Button(
                    "Start Processing",
                    variant="primary",
                    size="lg",
                    elem_id="start-processing-btn"
                )
                
            with gr.Column(scale=2):
                # Output image (for visualization)
                output_image = gr.Image(
                    label="Result Image",
                    height=400
                )

                # Output text
                output_text = gr.Textbox(
                    label="Recognition Results",
                    lines=20,
                    max_lines=20,
                    show_copy_button=True
                )
        
        # Event handlers
        def update_language(lang):
            """Update interface language"""
            if lang == "English":
                # Keep English labels (no change needed)
                return {
                    image_input: gr.Image(label="Upload Image"),
                    task_type: gr.Dropdown(label="Task Type"),
                    custom_prompt: gr.Textbox(label="Custom Prompt"),
                    process_btn: gr.Button("Start Processing"),
                    output_text: gr.Textbox(label="Recognition Results"),
                    output_image: gr.Image(label="Result Image")
                }
            else:
                # Change to Chinese labels
                return {
                    image_input: gr.Image(label="上传图片"),
                    task_type: gr.Dropdown(label="任务类型"),
                    custom_prompt: gr.Textbox(label="自定义提示词"),
                    process_btn: gr.Button("开始处理"),
                    output_text: gr.Textbox(label="识别结果"),
                    output_image: gr.Image(label="结果图像")
                }
        
        def toggle_custom_prompt(task):
            """Show/hide custom prompt based on task type"""
            return gr.Textbox(visible=(task == "Custom Task"))
        
        def process_wrapper(image, task_type, custom_prompt, base_size, image_size, crop_mode, min_crops, max_crops):
            """Wrapper function for processing"""
            if image is None:
                return "Please upload an image first!", None
            
            print(f"🔄 Processing with parameters: task={task_type}, base_size={base_size}, image_size={image_size}")
            
            result_text, result_image = process_ocr_transformers(
                image, task_type, custom_prompt, base_size, image_size, crop_mode, min_crops, max_crops
            )
            
            print(f"✅ Wrapper received: text_length={len(result_text) if result_text else 0}, image={result_image is not None}")
            
            return result_text, result_image
        
        # Event bindings
        language.change(
            fn=update_language,
            inputs=[language],
            outputs=[image_input, task_type, custom_prompt, process_btn, output_text, output_image]
        )
        
        task_type.change(
            fn=toggle_custom_prompt,
            inputs=[task_type],
            outputs=[custom_prompt]
        )
        
        process_btn.click(
            fn=process_wrapper,
            inputs=[image_input, task_type, custom_prompt, base_size, image_size, crop_mode, min_crops, max_crops],
            outputs=[output_text, output_image]
        )
        
        # Add JavaScript to handle Enter key globally
        demo.load(
            fn=None,
            inputs=[],
            outputs=[],
            js="""
            function() {
                document.addEventListener('keydown', function(event) {
                    if (event.key === 'Enter' && !event.target.matches('textarea, input[type="text"], input[type="search"]')) {
                        event.preventDefault();
                        // Find and click the process button by ID
                        const processBtn = document.getElementById('start-processing-btn');
                        if (processBtn) {
                            processBtn.click();
                        }
                    }
                });
            }
            """
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["Document to Markdown", "Convert document to Markdown format"],
                ["OCR with Bounding Boxes", "Detect and recognize all text in the image"],
                ["Table Recognition", "Recognize table structure and content"],
                ["Formula Recognition", "Recognize mathematical formulas"],
            ],
            inputs=[task_type, custom_prompt],
            label="Example Tasks"
        )
    
    return demo

def main():
    """Main function"""
    print("🚀 Starting DeepSeek-OCR Optimized version...")
    print(f"⚙️ Configuration:")
    print(f"   - CUDA Device: {CUDA_DEVICE}")
    print(f"   - Precision: {'BF16' if USE_BF16 else 'FP32'}")
    print(f"   - FlashAttention: {'Enabled' if USE_FLASH_ATTENTION else 'Disabled'}")
    print(f"   - SafeTensors: {'Enabled' if USE_SAFETENSORS else 'Disabled'}")
    
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,  # Different port to avoid conflicts
        share=False
    )

if __name__ == "__main__":
    main()
