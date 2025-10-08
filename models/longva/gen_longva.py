from longva.model.builder import load_pretrained_model
from longva.mm_utils import tokenizer_image_token, process_images
from longva.constants import IMAGE_TOKEN_INDEX
from PIL import Image
from decord import VideoReader, cpu
import torch
import numpy as np
import argparse


def load_model(model_name):
    if model_name == "longva_7b":
        model_path = "lmms-lab/LongVA-7B-DPO"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    torch.manual_seed(0)

    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda:0")

    return {"model": model, "image_processor": image_processor, "tokenizer": tokenizer}

def generate(model_info, video_path, query, max_new_tokens=1024):
    model = model_info["model"]
    image_processor = model_info["image_processor"]
    tokenizer = model_info["tokenizer"]

    max_frames_num = 16

    gen_kwargs = {"do_sample": True, "temperature": 0.5, "top_p": None, "num_beams": 1, "use_cache": True, "max_new_tokens": max_new_tokens}

    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{query}<|im_end|>\n<|im_start|>assistant\n"

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    video_tensor = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=[video_tensor],  modalities=["video"], **gen_kwargs)
    
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="longva_7b", type=str)
    parser.add_argument("--video-path", type=str, required=True)
    
    args = parser.parse_args()

    model_name = args.model_name
    video_path = args.video_path

    model_info = load_model(model_name)

    caption = generate(model_info, video_path, "What happens in this video? Please describe it in detail.")

    print(caption)

if __name__ == "__main__":
    main()