import argparse
from transformers import AutoModel, AutoTokenizer
import torch
import os

def main():
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR inference.")
    parser.add_argument("--image_file", type=str, default="your_image.jpg",
                        help="Path to the input image file.")
    parser.add_argument("--output_path", type=str, default="your/output/dir",
                        help="Directory to save the output.")
    parser.add_argument("--cuda_device", type=str, default="0",
                        help="CUDA_VISIBLE_DEVICES environment variable value.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    model_name = 'deepseek-ai/DeepSeek-OCR'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
    model = model.eval().cuda().to(torch.bfloat16)

    # prompt = "<image>\nFree OCR. "
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    if not os.path.exists(args.image_file):
        print(f"Error: Image file not found at {args.image_file}")
        return

    os.makedirs(args.output_path, exist_ok=True)

    res = model.infer(tokenizer, prompt=prompt, image_file=args.image_file, output_path = args.output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)

if __name__ == "__main__":
    main()
