from transformers import AutoModel, AutoTokenizer
import torch
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR Inference on AMD ROCm")
    parser.add_argument("image_file", type=str, help="Path to the input image file")
    parser.add_argument("--model_path", type=str, default="DeepSeek-OCR-model", help="Path to the DeepSeek-OCR model directory")
    parser.add_argument("--output_path", type=str, default="output", help="Path to the output directory")
    parser.add_argument("--prompt", type=str, default="<image>\n<|grounding|>Convert the document to markdown. ", help="Prompt for the model")
    args = parser.parse_args()

    # Create a device object
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # Remove flash attention and use device-agnostic .to(device)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, use_safetensors=True)
    model = model.eval().to(device).to(torch.bfloat16)

    res = model.infer(tokenizer, prompt=args.prompt, image_file=args.image_file, output_path=args.output_path, base_size=1024, image_size=640, crop_mode=True, save_results=True, test_compress=True)
    print(res)

if __name__ == "__main__":
    main()
