<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <a href="README.md">English</a> | <b>繁體中文</b> | <a href="README.es.md">Español</a>
</div>

<div align="center">
  <img src="assets/logo.svg" width="60%" alt="DeepSeek AI" />
</div>

<hr>
<div align="center">
  <a href="https://www.deepseek.com/" target="_blank">
    <img alt="首頁" src="assets/badge.svg" />
  </a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-OCR" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" />
  </a>
</div>

<div align="center">
  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" />
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" />
  </a>
</div>

<p align="center">
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-OCR"><b>📥 模型下載</b></a> |
  <a href="https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek_OCR_paper.pdf"><b>📄 論文連結</b></a> |
  <a href="https://arxiv.org/abs/2510.18234"><b>📄 Arxiv 論文連結</b></a> |
</p>

<h2>
<p align="center">
  <a href="">DeepSeek-OCR：上下文光學壓縮</a>
</p>
</h2>

<p align="center">
<img src="assets/fig1.png" style="width: 1000px" align=center>
</p>
<p align="center">
<a href="">探索視覺文字壓縮的邊界。</a>
</p>

## 發布
- [2025/10/23]🚀🚀🚀 DeepSeek-OCR 現在已在 [vLLM](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html#installing-vllm) 上游正式支援。感謝 [vLLM](https://github.com/vllm-project/vllm) 團隊的協助。
- [2025/10/20]🚀🚀🚀 我們發布了 DeepSeek-OCR，這是一個從以 LLM 為中心的角度研究視覺編碼器作用的模型。

## 目錄
- [安裝](#安裝)
- [vLLM 推理](#vllm-推理)
- [Transformers 推理](#transformers-推理)

## 安裝
>我們的環境是 cuda11.8+torch2.6.0。
1. 克隆此儲存庫並導航至 DeepSeek-OCR 文件夾
```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
```
2. Conda
```Shell
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```
3. 套件

- 下載 vllm-0.8.5 [whl](https://github.com/vllm-project/vllm/releases/tag/v0.8.5)
```Shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```
**注意：** 如果您希望 vLLM 和 transformers 代碼在相同的環境中運行，您無需擔心此安裝錯誤：vllm 0.8.5+cu118 需要 transformers>=4.51.1

## vLLM 推理
- VLLM:
>**注意：** 在 DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py 中更改 INPUT_PATH/OUTPUT_PATH 和其他設置
```Shell
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
```
1. 圖像：流式輸出
```Shell
python run_dpsk_ocr_image.py
```
2. pdf：並發 ~2500tokens/s(一個 A100-40G)
```Shell
python run_dpsk_ocr_pdf.py
```
3.基準測試的批量評估
```Shell
python run_dpsk_ocr_eval_batch.py
```

**[2025/10/23] 上游 [vLLM](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html#installing-vllm) 的版本：**

```shell
uv venv
source .venv/bin/activate
# 在 v0.11.1 發布之前，您需要從夜間構建安裝 vLLM
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

```python
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image

# 創建模型實例
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor]
)

# 使用您的圖像文件準備批處理輸入
image_1 = Image.open("path/to/your/image_1.png").convert("RGB")
image_2 = Image.open("path/to/your/image_2.png").convert("RGB")
prompt = "<image>\nFree OCR."

model_input = [
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image_1}
    },
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image_2}
    }
]

sampling_param = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            # ngram logit 處理器參數
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # 白名單: <td>, </td>
            ),
            skip_special_tokens=False,
        )
# 生成輸出
model_outputs = llm.generate(model_input, sampling_param)

# 打印輸出
for output in model_outputs:
    print(output.outputs[0].text)
```
## Transformers 推理
- Transformers
```python
from transformers import AutoModel, AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>將文件轉換為 markdown。 "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'

res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)
```
或者你可以
```Shell
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```
## 支持模式
當前開源模型支持以下模式：
- 本地分辨率：
  - 微型：512×512 （64 個視覺標記）✅
  - 小型：640×640 （100 個視覺標記）✅
  - 基礎：1024×1024 （256 個視覺標記）✅
  - 大型：1280×1280 （400 個視覺標記）✅
- 動態分辨率
  - 高達：n×640×640 + 1×1024×1024 ✅

## 提示示例
```python
# 文件：<image>\n<|grounding|>將文件轉換為 markdown。
# 其他圖像：<image>\n<|grounding|>對此圖像進行 OCR。
# 無佈局：<image>\nFree OCR。
# 文件中的圖：<image>\n解析該圖。
# 通用：<image>\n詳細描述此圖像。
# rec：<image>\n在圖像中定位 <|ref|>xxxx<|/ref|>。
# '先天下之忧而忧'
```

## 可視化
<table>
<tr>
<td><img src="assets/show1.jpg" style="width: 500px"></td>
<td><img src="assets/show2.jpg" style="width: 500px"></td>
</tr>
<tr>
<td><img src="assets/show3.jpg" style="width: 500px"></td>
<td><img src="assets/show4.jpg" style="width: 500px"></td>
</tr>
</table>

## 致謝

我們要感謝 [Vary](https://github.com/Ucas-HaoranWei/Vary/)、[GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/)、[MinerU](https://github.com/opendatalab/MinerU)、[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)、[OneChart](https://github.com/LingyvKong/OneChart)、[Slow Perception](https://github.com/Ucas-HaoranWei/Slow-Perception) 提供的寶貴模型和想法。

我們也感謝基準測試：[Fox](https://github.com/ucaslcl/Fox)、[OminiDocBench](https://github.com/opendatalab/OmniDocBench)。

## 引文

```bibtex
@article{wei2024deepseek-ocr,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2510.18234},
  year={2025}
}
```
