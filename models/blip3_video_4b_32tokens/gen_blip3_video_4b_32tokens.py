from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, LogitsProcessor
import torch
import numpy as np
import torchvision
import torchvision.io
import math
import argparse

def load_model(model_name):
    if model_name == "blip3_video_4b_32tokens":
        model_name_or_path = "Salesforce/xgen-mm-vid-phi3-mini-r-v1.5-32tokens-8frames"
    elif model_name == "blip3_video_4b_128tokens":
        model_name_or_path = "Salesforce/xgen-mm-vid-phi3-mini-r-v1.5-128tokens-8frames"  
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = model.update_special_tokens(tokenizer)

    model = model.to('cuda')
    model.eval()
    tokenizer.padding_side = "left"
    tokenizer.eos_token = "<|end|>"

    return {
        "model": model,
        "tokenizer": tokenizer,
        "image_processor": image_processor
    }

def sample_frames(vframes, num_frames):
    frame_indice = np.linspace(int(num_frames/2), len(vframes) - int(num_frames/2), num_frames, dtype=int)
    video = vframes[frame_indice]
    video_list = []
    for i in range(len(video)):
        video_list.append(torchvision.transforms.functional.to_pil_image(video[i]))
    return video_list

def generate(model_info, video_path, query, max_new_tokens=1024):

    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    image_processor = model_info["image_processor"]

    # predict
    vframes, _, _ = torchvision.io.read_video(
        filename=video_path, pts_unit="sec", output_format="TCHW"
    )
    total_frames = len(vframes)
    images = sample_frames(vframes, 8)

    prompt = ""
    prompt = prompt + "<image>\n"
    prompt = prompt + query
    messages = [{"role": "user", "content": prompt}]

    image_sizes = [image.size for image in images]
    image_tensor = [image_processor([img])["pixel_values"].to(model.device, dtype=torch.bfloat16) for img in images]

    image_tensor = torch.stack(image_tensor, dim=1)
    image_tensor = image_tensor.squeeze(2)
    inputs = {"pixel_values": image_tensor}

    full_conv = "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
    for msg in messages:
        msg_str = "<|{role}|>\n{content}<|end|>\n".format(
            role=msg["role"], content=msg["content"]
        )
        full_conv += msg_str

    full_conv += "<|assistant|>\n"
    print(full_conv)
    language_inputs = tokenizer([full_conv], return_tensors="pt")
    for name, value in language_inputs.items():
        language_inputs[name] = value.to(model.device)
    inputs.update(language_inputs)

    with torch.inference_mode():
        generated_text = model.generate(
            **inputs,
            image_size=[image_sizes],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.05,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            top_p=None,
            num_beams=1,
        )

    outputs = (
        tokenizer.decode(generated_text[0], skip_special_tokens=True)
        .split("<|end|>")[0]
        .strip()
    )
    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="blip3_video_4b_32tokens", type=str)
    parser.add_argument("--video-path", type=str, required=True)
    
    args = parser.parse_args()

    model_name = args.model_name
    video_path = args.video_path

    model_info = load_model(model_name)

    caption = generate(model_info, video_path, "What happens in this video? Please describe it in detail.")

    print(caption)

if __name__ == "__main__":
    main()