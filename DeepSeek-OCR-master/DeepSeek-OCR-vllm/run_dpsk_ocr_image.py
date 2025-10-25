import asyncio
import re
import os

import torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

os.environ['VLLM_USE_V1'] = '1'  # Updated to V1 for compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
import time
from deepseek_ocr import DeepseekOCRForCausalLM
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from tqdm import tqdm
# NOTE: VLLM V1 does not support per-request logits processors
# from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, CROP_MODE



ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

def load_image(image_path):

    try:
        image = Image.open(image_path)
        
        corrected_image = ImageOps.exif_transpose(image)

        return corrected_image
        
    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)


    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):


    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs):

    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    #     except IOError:
    font = ImageFont.load_default()

    img_idx = 0
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20, )
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{OUTPUT_PATH}/images/{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1
                        
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
    return img_draw


def process_image_with_refs(image, ref_texts):
    result_image = draw_bounding_boxes(image, ref_texts)
    return result_image




async def stream_generate(image=None, prompt=''):


    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,  
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # NOTE: VLLM V1 does not support per-request logits processors
    # logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90, whitelist_token_ids= {128821, 128822})] #whitelist: <td>, </td> 

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        # NOTE: Using repetition_penalty instead of custom logits processor for V1 compatibility
        repetition_penalty=1.05,
        skip_special_tokens=False,
        # ignore_eos=False,
        
    )
    
    request_id = f"request-{int(time.time())}"

    printed_length = 0  

    if image and '<image>' in prompt:
        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }
    elif prompt:
        request = {
            "prompt": prompt
        }
    else:
        assert False, f'prompt is none!!!'
    async for request_output in engine.generate(
        request, sampling_params, request_id
    ):
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            new_text = full_text[printed_length:]
            print(new_text, end='', flush=True)
            printed_length = len(full_text)
            final_output = full_text
    print('\n') 

    return final_output




if __name__ == "__main__":

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f'{OUTPUT_PATH}/images', exist_ok=True)

    image = load_image(INPUT_PATH).convert('RGB')

    
    if '<image>' in PROMPT:

        image_features = DeepseekOCRProcessor().tokenize_with_images(images = [image], bos=True, eos=True, cropping=CROP_MODE)
    else:
        image_features = ''

    prompt = PROMPT

    result_out = asyncio.run(stream_generate(image_features, prompt))


    save_results = 1

    if save_results and '<image>' in prompt:
        print('='*15 + 'save results:' + '='*15)

        image_draw = image.copy()

        outputs = result_out

        with open(f'{OUTPUT_PATH}/result_ori.mmd', 'w', encoding = 'utf-8') as afile:
            afile.write(outputs)

        matches_ref, matches_images, mathes_other = re_match(outputs)
        # print(matches_ref)
        result = process_image_with_refs(image_draw, matches_ref)


        for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
            outputs = outputs.replace(a_match_image, f'![](images/' + str(idx) + '.jpg)\n')

        for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
            outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

        # if 'structural formula' in conversation[0]['content']:
        #     outputs = '<smiles>' + outputs + '</smiles>'
        with open(f'{OUTPUT_PATH}/result.mmd', 'w', encoding = 'utf-8') as afile:
            afile.write(outputs)

        if 'line_type' in outputs:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            lines = eval(outputs)['Line']['line']

            line_type = eval(outputs)['Line']['line_type']
            # print(lines)

            endpoints = eval(outputs)['Line']['line_endpoint']

            fig, ax = plt.subplots(figsize=(3,3), dpi=200)
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)

            for idx, line in enumerate(lines):
                try:
                    p0 = eval(line.split(' -- ')[0])
                    p1 = eval(line.split(' -- ')[-1])

                    if line_type[idx] == '--':
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
                    else:
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth = 0.8, color = 'k')

                    ax.scatter(p0[0], p0[1], s=5, color = 'k')
                    ax.scatter(p1[0], p1[1], s=5, color = 'k')
                except:
                    pass

            for endpoint in endpoints:

                label = endpoint.split(': ')[0]
                (x, y) = eval(endpoint.split(': ')[1])
                ax.annotate(label, (x, y), xytext=(1, 1), textcoords='offset points', 
                            fontsize=5, fontweight='light')
            
            try:
                if 'Circle' in eval(outputs).keys():
                    circle_centers = eval(outputs)['Circle']['circle_center']
                    radius = eval(outputs)['Circle']['radius']

                    for center, r in zip(circle_centers, radius):
                        center = eval(center.split(': ')[1])
                        circle = Circle(center, radius=r, fill=False, edgecolor='black', linewidth=0.8)
                        ax.add_patch(circle)
            except:
                pass


            plt.savefig(f'{OUTPUT_PATH}/geo.jpg')
            plt.close()

        result.save(f'{OUTPUT_PATH}/result_with_boxes.jpg')
