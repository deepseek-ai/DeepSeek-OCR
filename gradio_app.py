import asyncio
import re
import os
import sys
from pathlib import Path
import threading

import torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

os.environ['VLLM_USE_V1'] = '0'

import gradio as gr
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
import time
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

# 添加DeepSeek-OCR-vllm到路径
sys.path.insert(0, str(Path(__file__).parent / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm"))

from deepseek_ocr import DeepseekOCRForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import MODEL_PATH

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# 全局变量存储引擎和事件循环
engine = None
processor = None
event_loop = None
loop_thread = None
engine_lock = threading.Lock()


def load_image(image_input):
    """加载并处理图像"""
    try:
        if isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            image = image_input

        # 自动旋转图像
        corrected_image = ImageOps.exif_transpose(image)
        return corrected_image.convert('RGB')
    except Exception as e:
        print(f"加载图像出错: {e}")
        return None


def re_match(text):
    """匹配grounding标记"""
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


def extract_coordinates_and_label(ref_text, image_width, image_height):
    """提取坐标和标签"""
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None
    return (label_type, cor_list)


def draw_bounding_boxes(image, refs):
    """在图像上绘制边界框"""
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    font = ImageFont.load_default()

    cropped_images = []

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                # 使用更深的颜色范围，提高可见度
                color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
                color_a = color + (60, )  # 增加半透明填充的不透明度

                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped_images.append(cropped)
                        except Exception as e:
                            print(e)

                    try:
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
                        pass
        except:
            continue

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw, cropped_images


def start_event_loop():
    """在后台线程启动事件循环"""
    global event_loop
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    event_loop.run_forever()


def get_or_create_event_loop():
    """获取或创建全局事件循环"""
    global event_loop, loop_thread

    if event_loop is None or not event_loop.is_running():
        with engine_lock:
            if event_loop is None or not event_loop.is_running():
                loop_thread = threading.Thread(target=start_event_loop, daemon=True)
                loop_thread.start()
                # 等待事件循环启动
                while event_loop is None:
                    time.sleep(0.1)

    return event_loop


async def initialize_engine(gpu_memory_utilization=0.75):
    """初始化vLLM引擎"""
    global engine, processor

    if engine is None:
        print("正在初始化推理引擎...")
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            max_model_len=8192,
            enforce_eager=False,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("推理引擎初始化完成！")

    if processor is None:
        processor = DeepseekOCRProcessor()

    return engine, processor


async def stream_generate(image_features=None, prompt='', max_tokens=8192):
    """流式生成文本"""
    global engine

    if engine is None:
        await initialize_engine()

    logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822})]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )

    request_id = f"request-{int(time.time() * 1000)}"

    print(f"提交请求: {request_id}")

    if image_features and '<image>' in prompt:
        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_features}
        }
    elif prompt:
        request = {
            "prompt": prompt
        }
    else:
        raise ValueError('提示词不能为空！')

    full_text = ""
    printed_length = 0

    try:
        async for request_output in engine.generate(request, sampling_params, request_id):
            if request_output.outputs:
                full_text = request_output.outputs[0].text
                new_text = full_text[printed_length:]
                if new_text:
                    print(f"已生成 {len(new_text)} 字符...", end='\r')
                printed_length = len(full_text)

        print(f"\n生成完成: 共 {len(full_text)} 字符")
    except Exception as e:
        print(f"生成过程出错: {e}")
        raise

    return full_text


def run_async_task(coro):
    """在全局事件循环中运行异步任务"""
    loop = get_or_create_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def process_ocr(image, task_type, custom_prompt, base_size, image_size, crop_mode,
                min_crops, max_crops, gpu_memory):
    """处理OCR任务"""
    if image is None:
        return "请先上传图片。 / Please upload an image first.", None, None, None

    print(f"\n{'='*50}")
    print(f"正在处理新请求: {task_type}")
    print(f"{'='*50}")

    # 创建临时目录保存裁剪的图片
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp(prefix="deepseek_ocr_")
    print(f"临时目录: {temp_dir}")

    # 设置全局配置
    import config
    config.BASE_SIZE = base_size
    config.IMAGE_SIZE = image_size
    config.CROP_MODE = crop_mode
    config.MIN_CROPS = min_crops
    config.MAX_CROPS = max_crops

    # 加载图像
    image_rgb = load_image(image)
    if image_rgb is None:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return "加载图片失败。 / Failed to load image.", None, None, None

    # Task name mapping (support both Chinese and English)
    task_map = {
        # Chinese
        "文档转Markdown": "document_to_markdown",
        "OCR带定位框": "ocr_with_grounding",
        "纯文本OCR": "plain_text_ocr",
        "图表解析": "figure_parsing",
        "图像描述": "image_description",
        "文本定位": "text_localization",
        "物体定位": "object_localization",
        "自定义": "custom",
        # English
        "Document to Markdown": "document_to_markdown",
        "OCR with Grounding": "ocr_with_grounding",
        "Plain Text OCR": "plain_text_ocr",
        "Figure Parsing": "figure_parsing",
        "Image Description": "image_description",
        "Text Localization": "text_localization",
        "Object Localization": "object_localization",
        "Custom": "custom"
    }

    # Get normalized task type
    normalized_task = task_map.get(task_type, "ocr_with_grounding")

    # 构建prompt
    prompt_templates = {
        "document_to_markdown": "<image>\n<|grounding|>Convert the document to markdown.",
        "ocr_with_grounding": "<image>\n<|grounding|>OCR this image.",
        "plain_text_ocr": "<image>\nFree OCR.",
        "figure_parsing": "<image>\nParse the figure.",
        "image_description": "<image>\nDescribe this image in detail.",
        "text_localization": "<image>\nLocate text in the image.",
        "object_localization": "<image>\nLocate <|ref|>{target}<|/ref|> in the image.",
        "custom": custom_prompt if custom_prompt else "<image>\nOCR this image."
    }

    # 特殊处理物体定位任务
    if normalized_task == "object_localization":
        if custom_prompt and custom_prompt.strip():
            # 用户输入的是要定位的目标物体
            prompt = f"<image>\nLocate <|ref|>{custom_prompt}<|/ref|> in the image."
        else:
            prompt = "<image>\nLocate objects in the image."
    else:
        prompt = prompt_templates.get(normalized_task, prompt_templates["ocr_with_grounding"])

    try:
        # 初始化引擎（使用全局事件循环）
        print("正在初始化引擎...")
        run_async_task(initialize_engine(gpu_memory_utilization=gpu_memory))

        # 处理图像
        if '<image>' in prompt:
            print("正在处理图像特征...")
            config.PROMPT = prompt
            image_features = processor.tokenize_with_images(
                images=[image_rgb],
                bos=True,
                eos=True,
                cropping=crop_mode
            )
        else:
            image_features = None

        # 生成结果（使用全局事件循环）
        print("正在生成文本...")
        result_text = run_async_task(stream_generate(image_features, prompt))

        # 处理结果
        result_with_boxes = None
        cropped_images = []
        clean_markdown = result_text

        if '<image>' in prompt and '<|ref|>' in result_text:
            # 提取匹配项
            print("正在处理定位标签...")
            matches_ref, matches_images, matches_other = re_match(result_text)

            # 绘制边界框
            if matches_ref:
                result_with_boxes, cropped_images = draw_bounding_boxes(image_rgb, matches_ref)

            # 清理markdown - 保留文本内容，只删除grounding标签
            clean_markdown = result_text

            # 处理图像匹配项 - 将图片转换为base64嵌入到Markdown中
            import base64
            from io import BytesIO

            for idx, match in enumerate(matches_images):
                if idx < len(cropped_images):
                    # 保存裁剪的图片到临时目录（用于备份）
                    image_path = os.path.join(temp_dir, f'image_{idx}.jpg')
                    cropped_images[idx].save(image_path, 'JPEG', quality=95)
                    print(f"保存图片: {image_path}")

                    # 将图片转换为base64编码
                    buffered = BytesIO()
                    cropped_images[idx].save(buffered, format="JPEG", quality=95)
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()

                    # 在Markdown中使用data URI嵌入图片
                    clean_markdown = clean_markdown.replace(match, f'![图片 {idx}](data:image/jpeg;base64,{img_base64})\n\n')
                else:
                    clean_markdown = clean_markdown.replace(match, f'![图片 {idx}](image_{idx}.jpg)\n')

            # 处理其他匹配项 - 提取ref标签中的文本内容
            for match in matches_other:
                # match是完整的标签: <|ref|>文本<|/ref|><|det|>坐标<|/det|>
                # 我们需要提取"文本"部分
                ref_pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>.*?<\|/det\|>'
                ref_match = re.search(ref_pattern, match)
                if ref_match:
                    text_content = ref_match.group(1)
                    clean_markdown = clean_markdown.replace(match, text_content)
                else:
                    clean_markdown = clean_markdown.replace(match, '')

            # 清理其他特殊符号和标记
            clean_markdown = clean_markdown.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

            # 清理多余的"text"标记 - 移除独立行上的"text"
            clean_markdown = re.sub(r'(?m)^text\s*$', '', clean_markdown)
            clean_markdown = re.sub(r'(?m)^text\n', '', clean_markdown)

            # 清理多余的空行（超过2个连续空行的情况）
            clean_markdown = re.sub(r'\n{3,}', '\n\n', clean_markdown)

        print("处理完成！")
        print(f"裁剪图片保存在: {temp_dir}")

        # 确保所有返回值都有效
        final_markdown = clean_markdown if clean_markdown else ""
        final_boxes_image = result_with_boxes
        final_cropped = cropped_images if cropped_images else []
        final_raw = result_text if result_text else ""

        print(f"准备返回结果...")
        print(f"  - Markdown: {len(final_markdown)} 字符")
        print(f"  - 标注图像: {'有' if final_boxes_image else '无'}")
        print(f"  - 裁剪图片: {len(final_cropped)} 个")
        print(f"  - 原始输出: {len(final_raw)} 字符")

        return final_markdown, final_boxes_image, final_cropped, final_raw

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(temp_dir, ignore_errors=True)
        return f"错误: {str(e)}", None, None, None


def create_gradio_interface():
    """创建Gradio界面"""

    # 双语文本字典
    texts = {
        'zh': {
            'title': 'DeepSeek-OCR: 高级视觉文档理解系统',
            'intro': '''本界面支持多种OCR和文档理解任务：
- **文档转Markdown**: 将文档转换为Markdown格式，保留结构和定位信息
- **OCR带定位框**: 提取文本并标注边界框
- **纯文本OCR**: 简单的文本提取，不保留布局信息
- **图表解析**: 分析和描述图表、图形和示意图
- **图像描述**: 生成详细的图像描述
- **文本定位**: 定位和识别文本区域
- **物体定位**: 🆕 定位图片中的指定物体（在输入框中填写物体名称，如：手掌、苹果、人脸）
- **自定义**: 使用您自己的自定义提示词

**注意**: 首次推理需要约30-60秒进行模型编译，后续运行会快很多！''',
            'input_section': '输入',
            'output_section': '结果',
            'upload_image': '上传图片',
            'task_type': '任务类型',
            'tasks': ['文档转Markdown', 'OCR带定位框', '纯文本OCR', '图表解析', '图像描述', '文本定位', '物体定位', '自定义'],
            'custom_prompt_label': '目标物体/自定义提示词',
            'custom_prompt_placeholder': '物体定位：输入要定位的物体名称（如：手掌、苹果、汽车）\n自定义：输入完整提示词',
            'advanced_settings': '高级设置',
            'model_presets': '''**模型尺寸预设:**
- Tiny: base=512, image=512, 不裁剪
- Small: base=640, image=640, 不裁剪
- Base: base=1024, image=1024, 不裁剪
- Large: base=1280, image=1280, 不裁剪
- Gundam: base=1024, image=640, 启用裁剪（推荐）''',
            'base_size': '基础尺寸',
            'image_size': '图像尺寸',
            'crop_mode': '启用动态裁剪',
            'min_crops': '最小裁剪数',
            'max_crops': '最大裁剪数',
            'gpu_memory': 'GPU显存利用率',
            'process_btn': '开始处理',
            'markdown_output': 'Markdown输出',
            'clean_markdown': '清理后的Markdown',
            'annotated_image': '标注图像',
            'boxes_image': '带边界框的图像',
            'extracted_images': '提取的图像',
            'cropped_regions': '裁剪区域',
            'raw_output': '原始输出',
            'raw_model_output': '原始模型输出',
        },
        'en': {
            'title': 'DeepSeek-OCR: Advanced Visual Document Understanding',
            'intro': '''This interface supports multiple OCR and document understanding tasks:
- **Document to Markdown**: Convert documents to Markdown format with structure and location info
- **OCR with Grounding**: Extract text with bounding box annotations
- **Plain Text OCR**: Simple text extraction without layout preservation
- **Figure Parsing**: Analyze and describe charts, diagrams, and schematics
- **Image Description**: Generate detailed image descriptions
- **Text Localization**: Locate and recognize text regions
- **Object Localization**: 🆕 Locate specified objects in images (enter object name in input box, e.g.: palm, apple, face)
- **Custom**: Use your own custom prompts

**Note**: First inference takes ~30-60s for model compilation, subsequent runs are much faster!''',
            'input_section': 'Input',
            'output_section': 'Output',
            'upload_image': 'Upload Image',
            'task_type': 'Task Type',
            'tasks': ['Document to Markdown', 'OCR with Grounding', 'Plain Text OCR', 'Figure Parsing', 'Image Description', 'Text Localization', 'Object Localization', 'Custom'],
            'custom_prompt_label': 'Target Object/Custom Prompt',
            'custom_prompt_placeholder': 'Object Localization: Enter object name (e.g.: palm, apple, car)\nCustom: Enter full prompt',
            'advanced_settings': 'Advanced Settings',
            'model_presets': '''**Model Size Presets:**
- Tiny: base=512, image=512, no cropping
- Small: base=640, image=640, no cropping
- Base: base=1024, image=1024, no cropping
- Large: base=1280, image=1280, no cropping
- Gundam: base=1024, image=640, cropping enabled (recommended)''',
            'base_size': 'Base Size',
            'image_size': 'Image Size',
            'crop_mode': 'Enable Dynamic Cropping',
            'min_crops': 'Minimum Crops',
            'max_crops': 'Maximum Crops',
            'gpu_memory': 'GPU Memory Utilization',
            'process_btn': 'Process',
            'markdown_output': 'Markdown Output',
            'clean_markdown': 'Cleaned Markdown',
            'annotated_image': 'Annotated Image',
            'boxes_image': 'Image with Bounding Boxes',
            'extracted_images': 'Extracted Images',
            'cropped_regions': 'Cropped Regions',
            'raw_output': 'Raw Output',
            'raw_model_output': 'Raw Model Output',
        }
    }

    # CSS for centered title
    custom_css = """
    <style>
    .title-center {
        text-align: center;
        margin-bottom: 1em;
    }
    .lang-selector {
        position: absolute;
        top: 20px;
        right: 20px;
    }
    </style>
    """

    with gr.Blocks(title="DeepSeek-OCR", theme=gr.themes.Soft(), css=custom_css) as demo:
        # Language state
        current_lang = gr.State('zh')

        # Language selector
        with gr.Row():
            with gr.Column(scale=4):
                title_md = gr.Markdown(f'<h1 class="title-center">{texts["zh"]["title"]}</h1>')
            with gr.Column(scale=1, elem_classes="lang-selector"):
                language = gr.Radio(
                    choices=["中文", "English"],
                    value="中文",
                    label="Language / 语言",
                    container=False
                )

        intro_md = gr.Markdown(texts['zh']['intro'])

        with gr.Row():
            with gr.Column(scale=1):
                # 输入区域
                input_section_md = gr.Markdown(f"### {texts['zh']['input_section']}")
                image_input = gr.Image(type="pil", label=texts['zh']['upload_image'])

                task_type = gr.Dropdown(
                    choices=texts['zh']['tasks'],
                    value=texts['zh']['tasks'][0],
                    label=texts['zh']['task_type']
                )

                custom_prompt = gr.Textbox(
                    label=texts['zh']['custom_prompt_label'],
                    placeholder=texts['zh']['custom_prompt_placeholder'],
                    lines=2
                )

                # 高级设置
                with gr.Accordion(texts['zh']['advanced_settings'], open=False) as advanced_accordion:
                    model_presets_md = gr.Markdown(texts['zh']['model_presets'])

                    base_size = gr.Slider(
                        minimum=512,
                        maximum=1536,
                        value=1024,
                        step=128,
                        label=texts['zh']['base_size']
                    )

                    image_size = gr.Slider(
                        minimum=512,
                        maximum=1536,
                        value=640,
                        step=128,
                        label=texts['zh']['image_size']
                    )

                    crop_mode = gr.Checkbox(
                        value=True,
                        label=texts['zh']['crop_mode']
                    )

                    min_crops = gr.Slider(
                        minimum=1,
                        maximum=9,
                        value=2,
                        step=1,
                        label=texts['zh']['min_crops']
                    )

                    max_crops = gr.Slider(
                        minimum=1,
                        maximum=9,
                        value=6,
                        step=1,
                        label=texts['zh']['max_crops']
                    )

                    gpu_memory = gr.Slider(
                        minimum=0.3,
                        maximum=0.95,
                        value=0.75,
                        step=0.05,
                        label=texts['zh']['gpu_memory']
                    )

                process_btn = gr.Button(texts['zh']['process_btn'], variant="primary", size="lg")

            with gr.Column(scale=2):
                # 输出区域
                output_section_md = gr.Markdown(f"### {texts['zh']['output_section']}")

                with gr.Tabs():
                    with gr.Tab(texts['zh']['markdown_output']) as tab_markdown:
                        markdown_output = gr.Markdown(
                            label=texts['zh']['clean_markdown'],
                            value=""
                        )

                    with gr.Tab(texts['zh']['annotated_image']) as tab_annotated:
                        image_with_boxes = gr.Image(
                            label=texts['zh']['boxes_image'],
                            type="pil",
                            height=600
                        )

                    with gr.Tab(texts['zh']['extracted_images']) as tab_extracted:
                        cropped_gallery = gr.Gallery(
                            label=texts['zh']['cropped_regions'],
                            columns=3,
                            height="auto"
                        )

                    with gr.Tab(texts['zh']['raw_output']) as tab_raw:
                        raw_output = gr.Textbox(
                            label=texts['zh']['raw_model_output'],
                            lines=20,
                            max_lines=30
                        )


        # 语言切换功能
        def update_language(lang_choice):
            lang = 'zh' if lang_choice == '中文' else 'en'
            t = texts[lang]

            return [
                f'<h1 class="title-center">{t["title"]}</h1>',  # title_md
                t['intro'],  # intro_md
                f"### {t['input_section']}",  # input_section_md
                gr.update(label=t['upload_image']),  # image_input
                gr.update(choices=t['tasks'], value=t['tasks'][0], label=t['task_type']),  # task_type
                gr.update(label=t['custom_prompt_label'], placeholder=t['custom_prompt_placeholder']),  # custom_prompt
                gr.update(label=t['advanced_settings']),  # advanced_accordion
                t['model_presets'],  # model_presets_md
                gr.update(label=t['base_size']),  # base_size
                gr.update(label=t['image_size']),  # image_size
                gr.update(label=t['crop_mode']),  # crop_mode
                gr.update(label=t['min_crops']),  # min_crops
                gr.update(label=t['max_crops']),  # max_crops
                gr.update(label=t['gpu_memory']),  # gpu_memory
                gr.update(value=t['process_btn']),  # process_btn
                f"### {t['output_section']}",  # output_section_md
                gr.update(label=t['markdown_output']),  # tab_markdown
                gr.update(label=t['clean_markdown']),  # markdown_output
                gr.update(label=t['annotated_image']),  # tab_annotated
                gr.update(label=t['boxes_image']),  # image_with_boxes
                gr.update(label=t['extracted_images']),  # tab_extracted
                gr.update(label=t['cropped_regions']),  # cropped_gallery
                gr.update(label=t['raw_output']),  # tab_raw
                gr.update(label=t['raw_model_output']),  # raw_output
            ]

        language.change(
            fn=update_language,
            inputs=[language],
            outputs=[
                title_md, intro_md, input_section_md, image_input, task_type, custom_prompt,
                advanced_accordion, model_presets_md, base_size, image_size, crop_mode,
                min_crops, max_crops, gpu_memory, process_btn, output_section_md,
                tab_markdown, markdown_output, tab_annotated, image_with_boxes,
                tab_extracted, cropped_gallery, tab_raw, raw_output
            ]
        )

        # 绑定处理函数
        process_btn.click(
            fn=process_ocr,
            inputs=[
                image_input, task_type, custom_prompt, base_size, image_size,
                crop_mode, min_crops, max_crops, gpu_memory
            ],
            outputs=[markdown_output, image_with_boxes, cropped_gallery, raw_output]
        )

    return demo


if __name__ == "__main__":
    # 预先创建事件循环
    get_or_create_event_loop()

    demo = create_gradio_interface()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
