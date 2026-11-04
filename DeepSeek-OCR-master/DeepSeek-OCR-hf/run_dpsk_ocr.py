from transformers import AutoModel, AutoTokenizer
import torch
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


model_name = 'deepseek-ai/DeepSeek-OCR'


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)



# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = './output'

# Issue #306: save_results=True 落盘前目录必须存在，上游 infer 内虽有
# makedirs，但示例脚本在此显式创建更稳妥；原占位 'your/output/dir' /
# 签名注释里的 output_path=' '（单个空格）易误导，已改为 './output'。
os.makedirs(output_path, exist_ok=True)
os.makedirs(f'{output_path}/images', exist_ok=True)



# infer(self, tokenizer, prompt='', image_file='', output_path='./output', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False):
# 注意 (issue #306):
# - 只有 save_results=True 才会写 result.mmd / result_with_boxes.jpg / images/*.jpg；
#   默认 False 时只走控制台 streamer 打印，不落盘（文档易误导，已在此注释说明）。
# - eval_mode=True 会提前 return 文本并跳过 save_results 落盘分支（上游 modeling_deepseekocr.py
#   的 `if ... and eval_mode: return outputs` 在 save 之前）；要“既拿返回值又落盘”，
#   请用 eval_mode=False + save_results=True 落盘，或另行手动写文件。
# - grounding prompt + crop_mode=True 会把子图裁成 images/*.jpg，result.mmd 里对应位置
#   只剩 ![](images/N.jpg)；想要全文 markdown 可试 crop_mode=False（见评论区 workaround）。

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)
