import gradio as gr
import torch
from transformers import AutoModel, AutoTokenizer
import os
import tempfile
from PIL import Image, ImageDraw
import re
from typing import Tuple, Optional, Dict, Any

# --- 常量和配置 ---
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
MODEL_SIZE_CONFIGS = {
    "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "Gundam (推荐)": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}

TASK_PROMPTS = {
    "📝 自由OCR": "<image>\n自由OCR.",
    "📄 转换为Markdown": "<image>\n<|grounding|>将文档转换为markdown.",
    "📈 解析图表": "<image>\n解析图表.",
}

DEFAULT_MODEL_SIZE = "Gundam (推荐)"
DEFAULT_TASK_TYPE = "📄 转换为Markdown"
BOUNDING_BOX_PATTERN = re.compile(r"<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>")
BOUNDING_BOX_COLOR = "red"
BOUNDING_BOX_WIDTH = 3
NORMALIZATION_FACTOR = 1000

# --- 全局变量 ---
model = None
tokenizer = None
model_gpu = None


def load_model_and_tokenizer() -> None:
    """启动时加载DeepSeek-OCR模型和分词器。"""
    global model, tokenizer
    print("正在加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval()
    print("✅ 模型加载成功。")


def move_model_to_gpu() -> None:
    """如果模型尚未在GPU上，则将其移动到GPU。"""
    global model_gpu
    if model_gpu is None:
        print("🚀 正在将模型移动到GPU...")
        # 使用非阻塞传输以获得更好的性能
        model_gpu = model.cuda().to(torch.bfloat16, non_blocking=True)
        print("✅ 模型已在GPU上。")


def find_result_image(path: str) -> Optional[Image.Image]:
    """
    在指定路径中查找预生成的结果图像。
    
    Args:
        path: 搜索结果图像的目录路径
        
    Returns:
        如果找到则返回PIL图像，否则返回None
    """
    for filename in os.listdir(path):
        if "grounding" in filename or "result" in filename:
            try:
                image_path = os.path.join(path, filename)
                return Image.open(image_path)
            except Exception as e:
                print(f"打开结果图像 {filename} 时出错: {e}")
    return None


def build_prompt(task_type: str, ref_text: str) -> str:
    """
    根据任务类型和参考文本构建适当的提示。
    
    Args:
        task_type: OCR任务类型
        ref_text: 定位任务的参考文本
        
    Returns:
        格式化的提示字符串
    """
    if task_type == "🔍 通过参考定位对象":
        if not ref_text or ref_text.strip() == "":
            raise gr.Error("对于'定位'任务，您必须提供要查找的参考文本！")
        return f"<image>\n在图像中定位 <|ref|>{ref_text.strip()}<|/ref|>."
    
    return TASK_PROMPTS.get(task_type, TASK_PROMPTS["📝 自由OCR"])


def extract_and_draw_bounding_boxes(text_result: str, original_image: Image.Image) -> Optional[Image.Image]:
    """
    从文本结果中提取边界框坐标并在图像上绘制它们。
    
    Args:
        text_result: 包含边界框坐标的OCR文本结果
        original_image: 要绘制的原始PIL图像
        
    Returns:
        绘制了边界框的PIL图像，如果没有找到坐标则返回None
    """
    # 直接使用迭代器以避免不必要地创建列表
    matches = list(BOUNDING_BOX_PATTERN.finditer(text_result))
    
    if not matches:
        return None
    
    print(f"✅ 找到 {len(matches)} 个边界框。正在原始图像上绘制。")
    
    # 创建原始图像的副本以进行绘制
    image_with_bboxes = original_image.copy()
    draw = ImageDraw.Draw(image_with_bboxes)
    w, h = original_image.size
    
    # 预先计算缩放因子以获得更好的性能
    w_scale = w / NORMALIZATION_FACTOR
    h_scale = h / NORMALIZATION_FACTOR
    
    for match in matches:
        # 更有效地提取和缩放坐标
        coords = tuple(int(c) for c in match.groups())
        x1_norm, y1_norm, x2_norm, y2_norm = coords
        
        # 使用预先计算的因子缩放归一化坐标
        x1 = int(x1_norm * w_scale)
        y1 = int(y1_norm * h_scale)
        x2 = int(x2_norm * w_scale)
        y2 = int(y2_norm * h_scale)
        
        # 绘制矩形
        draw.rectangle([x1, y1, x2, y2], outline=BOUNDING_BOX_COLOR, width=BOUNDING_BOX_WIDTH)
    
    return image_with_bboxes


def run_inference(prompt: str, image_path: str, output_path: str, config: Dict[str, Any]) -> str:
    """
    使用给定参数运行模型推理。
    
    Args:
        prompt: 模型的格式化提示
        image_path: 输入图像的路径
        output_path: 输出文件的目录路径
        config: 模型配置字典
        
    Returns:
        模型的文本结果
    """
    print(f"🏃 使用提示运行推理: {prompt}")
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
    print(f"====\n📄 文本结果: {text_result}\n====")
    return text_result


def process_ocr_task(image: Optional[Image.Image], model_size: str, task_type: str, ref_text: str) -> Tuple[str, Optional[Image.Image]]:
    """
    使用DeepSeek-OCR处理图像以支持所有任务。
    
    Args:
        image: 输入PIL图像
        model_size: 模型大小配置
        task_type: OCR任务类型
        ref_text: 定位任务的参考文本
        
    Returns:
        (text_result, result_image) 元组
    """
    if image is None:
        return "请先上传图像。", None
    
    # 确保模型在GPU上
    move_model_to_gpu()
    
    # 根据任务类型构建提示
    prompt = build_prompt(task_type, ref_text)
    
    # 获取模型配置
    config = MODEL_SIZE_CONFIGS.get(model_size, MODEL_SIZE_CONFIGS[DEFAULT_MODEL_SIZE])
    
    with tempfile.TemporaryDirectory() as output_path:
        # 使用优化格式保存临时图像
        temp_image_path = os.path.join(output_path, "temp_image.png")
        # 使用optimize=True以获得更好的压缩
        image.save(temp_image_path, optimize=True)
        
        # 运行推理
        text_result = run_inference(prompt, temp_image_path, output_path, config)
        
        # 尝试从文本结果中提取并绘制边界框
        result_image = extract_and_draw_bounding_boxes(text_result, image)
        
        # 如果没有找到边界框，则回退到预生成的结果图像
        if result_image is None:
            print("⚠️ 在文本结果中未找到边界框坐标。回退到搜索结果图像文件。")
            result_image = find_result_image(output_path)
        
        return text_result, result_image


def toggle_ref_text_visibility(task: str) -> gr.Textbox:
    """
    根据任务类型切换参考文本输入的可见性。
    
    Args:
        task: 选定的任务类型
        
    Returns:
        更新的Textbox组件
    """
    return gr.Textbox(visible=True) if task == "🔍 通过参考定位对象" else gr.Textbox(visible=False)


def create_ui() -> gr.Blocks:
    """
    创建和配置Gradio用户界面。
    
    Returns:
        配置好的Gradio Blocks界面
    """
    with gr.Blocks(title="🐳DeepSeek-OCR🐳", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🐳 DeepSeek-OCR 完整演示 🐳
            **💡 使用方法:**
            1.  使用上传框**上传图像**。
            2.  选择一个**分辨率**。对于大多数文档，推荐使用`Gundam`。
            3.  选择一个**任务类型**:
                - **📝 自由OCR**: 从图像中提取原始文本。
                - **📄 转换为Markdown**: 将文档转换为Markdown，保留结构。
                - **📈 解析图表**: 从图表和图形中提取结构化数据。
                - **🔍 通过参考定位对象**: 查找特定对象/文本。
            4. 如果这个工具有帮助，请给它点个赞！ 🙏 ❤️
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="🖼️ 上传图像", sources=["upload", "clipboard"])
                model_size = gr.Dropdown(
                    choices=list(MODEL_SIZE_CONFIGS.keys()),
                    value=DEFAULT_MODEL_SIZE,
                    label="⚙️ 分辨率大小"
                )
                task_type = gr.Dropdown(
                    choices=list(TASK_PROMPTS.keys()) + ["🔍 通过参考定位对象"],
                    value=DEFAULT_TASK_TYPE,
                    label="🚀 任务类型"
                )
                ref_text_input = gr.Textbox(
                    label="📝 参考文本（用于定位任务）",
                    placeholder="例如：老师、20-10、一辆红色汽车...",
                    visible=False
                )
                submit_btn = gr.Button("处理图像", variant="primary")

            with gr.Column(scale=2):
                output_text = gr.Textbox(label="📄 文本结果", lines=15, show_copy_button=True)
                output_image = gr.Image(label="🖼️ 图像结果（如果有）", type="pil")

        # UI交互逻辑
        task_type.change(fn=toggle_ref_text_visibility, inputs=task_type, outputs=ref_text_input)
        submit_btn.click(
            fn=process_ocr_task,
            inputs=[image_input, model_size, task_type, ref_text_input],
            outputs=[output_text, output_image]
        )

        # 示例图像和任务
        gr.Examples(
            examples=[
                ["doc_markdown.png", "will upload", "📄 will upload", ""],
            ],
            inputs=[image_input, model_size, task_type, ref_text_input],
            outputs=[output_text, output_image],
            fn=process_ocr_task,
            cache_examples=False,  # 禁用缓存以确保示例每次都运行
        )
    
    return demo


def main() -> None:
    """初始化和启动应用程序的主函数。"""
    # 启动时加载模型
    load_model_and_tokenizer()
    
    # 如果示例目录不存在则创建
    if not os.path.exists("examples"):
        os.makedirs("examples")
    
    # 创建并启动UI
    demo = create_ui()
    demo.queue(max_size=20).launch(share=True)


if __name__ == "__main__":
    main()